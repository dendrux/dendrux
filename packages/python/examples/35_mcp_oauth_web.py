"""Multi-user MCP OAuth in a web application, end to end.

A Starlette app exposes ``/connect`` (start sign-in), ``/oauth/callback``
(finish it), ``/ask`` (run an Agent that must call an MCP tool), and
``/revoke`` (sign out). An HTTP client plays the user's browser. The MCP
server and its OAuth authorization server are local demo servers
(``_mcp_oauth_demo.py``); the model is the real Anthropic API.

The walkthrough: connect -> consent -> callback handled by a *second*
``MCPOAuth`` instance (as another web worker would) -> Agent tool call ->
forced token expiry -> transparent refresh -> step-up to the ``admin``
scope -> sign-out -> ``not_signed_in``.

Prerequisites:
    - ``ANTHROPIC_API_KEY`` in the repository-root .env or environment
    - ``pip install "dendrux[anthropic,mcp]" python-dotenv uvicorn``

Run from ``packages/python``:

    python examples/35_mcp_oauth_web.py

The demo identifies the signed-in application user with an ``X-Demo-User``
header. A real application uses its own authenticated session: the
``MCPAuthKey`` passed to ``complete_authorization`` must come from the
principal of the callback request, never from the URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
from pathlib import Path

import httpx2
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import TYPE_CHECKING

from _mcp_oauth_demo import serve_demo  # noqa: E402 - the demo servers live beside this file

from dendrux import Agent  # noqa: E402
from dendrux.llm.anthropic import AnthropicProvider  # noqa: E402
from dendrux.mcp import (  # noqa: E402
    MCPAuthKey,
    MCPAuthorizationRequired,
    MCPOAuth,
    MCPOAuthClient,
    MCPOAuthFlowError,
    MCPOAuthMemoryStore,
    MCPOAuthRefreshed,
    MCPOAuthRefreshFailed,
    MCPOAuthRequired,
    MCPOAuthRevoked,
    MCPRuntime,
    MCPRuntimeEvent,
    MCPSource,
    MCPToolCallCompleted,
)

if TYPE_CHECKING:
    from starlette.requests import Request

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

CONNECTION_KEY = "demo-mcp"
# The scopes each tool needs. An application that calls tools directly can
# read them from MCPAuthorizationRequired.scopes instead.
TOOL_SCOPES = {"echo": [], "admin_echo": ["admin"]}


class EventPrinter:
    """Print value-free runtime events and remember OAuth verdicts."""

    def __init__(self) -> None:
        self.last_oauth_required: MCPOAuthRequired | None = None

    def on_event(self, event: MCPRuntimeEvent) -> None:
        """Report OAuth and tool-call events."""
        if isinstance(event, MCPOAuthRequired):
            self.last_oauth_required = event
            print(f"    event: {event!r}")
        elif isinstance(event, (MCPOAuthRefreshed, MCPOAuthRefreshFailed, MCPOAuthRevoked)):
            print(f"    event: {event!r}")
        elif isinstance(event, MCPToolCallCompleted):
            print(f"    event: MCP tool {event.tool} completed")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_app(*, rs_url: str, app_url: str, printer: EventPrinter) -> tuple[Starlette, list]:
    """Wire the four routes over one store shared by two MCPOAuth instances."""
    callback_url = f"{app_url}/oauth/callback"
    store = MCPOAuthMemoryStore()  # use an encrypted database table in production
    client = MCPOAuthClient.dynamic(client_name="Dendrux OAuth demo", redirect_uris=[callback_url])
    # Two instances over one store stand in for two web workers: sign-in
    # begins on one and completes on the other.
    connect_oauth = MCPOAuth(store=store.client(), client=client)
    worker_oauth = MCPOAuth(store=store.client(), client=client)
    runtime = MCPRuntime(observer=printer, idle_timeout=None)
    source = MCPSource.http("demo", rs_url)

    def principal(request: Request) -> MCPAuthKey:
        return MCPAuthKey(CONNECTION_KEY, tenant_key=request.headers["x-demo-user"])

    def connection_for(key: MCPAuthKey):  # type: ignore[no-untyped-def]
        return runtime.bind(
            connection_key=key.connection_key,
            tenant_key=key.tenant_key,
            source=source,
            oauth=worker_oauth,
        )

    async def connect(request: Request) -> Response:
        scopes = request.query_params.get("scopes")
        url = await connect_oauth.begin_authorization(
            principal(request),
            source,
            redirect_uri=callback_url,
            scopes=scopes.split(",") if scopes else None,
        )
        return RedirectResponse(url, status_code=302)

    async def callback(request: Request) -> Response:
        params = request.query_params
        key = principal(request)
        try:
            await worker_oauth.complete_authorization(
                key=key,
                state=params.get("state", ""),
                code=params.get("code"),
                iss=params.get("iss"),
                error=params.get("error"),
            )
        except MCPOAuthFlowError as error:
            return JSONResponse({"connected": False, "reason": error.reason}, status_code=400)
        # A step-up replaced the credential: retire the live transport so the
        # next call reconnects with the new token and re-reads the catalog.
        connection_for(key)
        await runtime.refresh_authorization(
            connection_key=key.connection_key, tenant_key=key.tenant_key
        )
        return JSONResponse({"connected": True})

    async def ask(request: Request) -> Response:
        key = principal(request)
        body = await request.json()
        tool = body["tool"]
        connection = connection_for(key)
        try:
            await connection.discover()
        except MCPAuthorizationRequired as error:
            return JSONResponse({"status": error.reason})
        printer.last_oauth_required = None
        async with Agent(
            name="OAuthDemoAgent",
            provider=AnthropicProvider(model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")),
            prompt=(
                f"You must call the demo__{tool} tool with the user's text before answering. "
                "Then reply with exactly the tool's result and nothing else. If the tool "
                "fails, reply with exactly: TOOL_FAILED"
            ),
            tool_sources=[connection.tools(allowed_tools=[tool])],
            max_iterations=4,
        ) as agent:
            result = await agent.run(f"Call the tool with text {json.dumps(body['text'])}.")
        verdict = printer.last_oauth_required
        if verdict is not None:
            status = await worker_oauth.status(key)
            granted = (status.scope or "").split()
            needed = [s for s in TOOL_SCOPES[tool] if s not in granted]
            return JSONResponse({"status": verdict.reason, "step_up_scopes": [*granted, *needed]})
        return JSONResponse({"status": "ok", "answer": result.answer})

    async def revoke(request: Request) -> Response:
        key = principal(request)
        connection_for(key)
        revocation = await runtime.revoke_authorization(
            connection_key=key.connection_key, tenant_key=key.tenant_key
        )
        return JSONResponse({"local": revocation.local, "remote": revocation.remote})

    app = Starlette(
        routes=[
            Route("/connect", connect),
            Route("/oauth/callback", callback),
            Route("/ask", ask, methods=["POST"]),
            Route("/revoke", revoke, methods=["POST"]),
        ]
    )
    return app, [runtime, connect_oauth, worker_oauth]


async def browser_sign_in(browser: httpx2.AsyncClient, scopes: str | None = None) -> dict:
    """Follow /connect -> authorization server consent -> /oauth/callback."""
    params = {"scopes": scopes} if scopes else None
    response = await browser.get("/connect", params=params)
    assert response.status_code == 302
    print("    browser: redirected to the authorization server")
    consent = await browser.get(response.headers["location"])
    assert consent.status_code == 302
    print("    browser: user consented; redirected back to /oauth/callback")
    callback = await browser.get(consent.headers["location"])
    return callback.json()


async def main() -> None:
    """Run the full walkthrough and print each step."""
    # Production recommendation: the HTTP client and MCP SDK log request URLs,
    # reason phrases, and session ids at INFO. Here the browser stand-in's
    # callback URL carries the authorization code.
    for name in ("httpx2", "httpx", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)
    printer = EventPrinter()
    async with serve_demo() as demo:
        port = _free_port()
        app_url = f"http://127.0.0.1:{port}"
        app, owned = build_app(rs_url=demo.rs_url, app_url=app_url, printer=printer)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        headers = {"X-Demo-User": "alice"}
        try:
            async with httpx2.AsyncClient(base_url=app_url, headers=headers, timeout=60) as browser:
                print("1. Connect: begin on worker A, consent, callback on worker B")
                print(f"    app: {await browser_sign_in(browser)}")

                print("2. Ask: the Agent calls demo__echo with the user's token")
                answer = (await browser.post("/ask", json={"tool": "echo", "text": "hello"})).json()
                print(f"    app: {answer}")
                assert answer == {"status": "ok", "answer": "hello"}, answer

                print("3. Force every access token to expire at the authorization server")
                demo.authorization_server.expire_access_tokens()
                answer = (await browser.post("/ask", json={"tool": "echo", "text": "again"})).json()
                print(f"    app: {answer}")
                assert answer["status"] == "ok", answer
                assert demo.authorization_server.counters["refresh"] == 1

                print("4. Step-up: demo__admin_echo needs the admin scope")
                answer = (
                    await browser.post("/ask", json={"tool": "admin_echo", "text": "root"})
                ).json()
                print(f"    app: {answer}")
                assert answer["status"] == "insufficient_scope", answer
                scopes = ",".join(answer["step_up_scopes"])
                print(f"    app: re-consent with scopes {scopes}")
                print(f"    app: {await browser_sign_in(browser, scopes)}")
                answer = (
                    await browser.post("/ask", json={"tool": "admin_echo", "text": "root"})
                ).json()
                print(f"    app: {answer}")
                assert answer == {"status": "ok", "answer": "admin:root"}, answer

                print("5. Sign out")
                print(f"    app: {(await browser.post('/revoke')).json()}")

                print("6. Ask again")
                answer = (await browser.post("/ask", json={"tool": "echo", "text": "x"})).json()
                print(f"    app: {answer}")
                assert answer == {"status": "not_signed_in"}, answer
        finally:
            runtime, *oauth_instances = owned
            await runtime.close()
            for oauth in oauth_instances:
                await oauth.aclose()
            server.should_exit = True
            await serving
    print("Walkthrough complete.")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to the repository-root .env "
            "or export it, then run this example again."
        )
    asyncio.run(main())
