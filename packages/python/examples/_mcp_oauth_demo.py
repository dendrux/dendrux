"""Loopback OAuth authorization server and MCP resource server for demos.

Used by ``35_mcp_oauth_web.py``. The authorization server is the MCP SDK's
own auth routes over an in-memory provider that approves every consent
automatically; the resource server is the SDK's ``MCPServer`` with bearer
auth, exposing ``echo`` (scope ``mcp``) and ``admin_echo`` (scope ``admin``,
answered with a 403 ``insufficient_scope`` challenge otherwise).

Demo only: tokens live in memory, consent is automatic, and both servers
listen on 127.0.0.1 over plain HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import socket
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import uvicorn
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    ProviderTokenVerifier,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.mcpserver import MCPServer
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class DemoAuthorizationServer:
    """In-memory OAuth provider: auto-consent, rotating refresh tokens."""

    issuer: str = ""
    clients: dict[str, OAuthClientInformationFull] = field(default_factory=dict)
    codes: dict[str, AuthorizationCode] = field(default_factory=dict)
    access_tokens: dict[str, AccessToken] = field(default_factory=dict)
    refresh_tokens: dict[str, RefreshToken] = field(default_factory=dict)
    families: dict[str, str] = field(default_factory=dict)
    revoked_families: set[str] = field(default_factory=set)
    counters: dict[str, int] = field(
        default_factory=lambda: {"authorize": 0, "refresh": 0, "revoke": 0}
    )

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: Any) -> str:
        self.counters["authorize"] += 1
        code = secrets.token_urlsafe(24)
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(
            str(params.redirect_uri), code=code, state=params.state, iss=self.issuer
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, code: str
    ) -> AuthorizationCode | None:
        return self.codes.get(code)

    def _mint(
        self, client_id: str, scopes: list[str], resource: str | None, family: str
    ) -> OAuthToken:
        access = "demo-at-" + secrets.token_urlsafe(16)
        refresh = "demo-rt-" + secrets.token_urlsafe(16)
        self.access_tokens[access] = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(time.time() + 3600),
            resource=resource,
        )
        self.refresh_tokens[refresh] = RefreshToken(
            token=refresh, client_id=client_id, scopes=scopes, resource=resource
        )
        self.families[access] = self.families[refresh] = family
        return OAuthToken(
            access_token=access,
            expires_in=3600,
            scope=" ".join(scopes) or None,
            refresh_token=refresh,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, code: AuthorizationCode
    ) -> OAuthToken:
        self.codes.pop(code.code, None)
        return self._mint(client.client_id, code.scopes, code.resource, secrets.token_hex(8))

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        token = self.refresh_tokens.get(refresh_token)
        if token is None or self.families.get(refresh_token) in self.revoked_families:
            return None
        return token

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        self.counters["refresh"] += 1
        self.refresh_tokens.pop(refresh_token.token, None)  # one-time use
        family = self.families.get(refresh_token.token, secrets.token_hex(8))
        return self._mint(client.client_id, scopes, refresh_token.resource, family)

    async def load_access_token(self, token: str) -> AccessToken | None:
        access = self.access_tokens.get(token)
        if access is None or self.families.get(token) in self.revoked_families:
            return None
        if access.expires_at is not None and access.expires_at < time.time():
            return None
        return access

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        family = self.families.get(token.token)
        if family is not None:
            self.revoked_families.add(family)

    def expire_access_tokens(self) -> None:
        """Force every issued access token to be expired (demo control)."""
        for access in self.access_tokens.values():
            access.expires_at = int(time.time()) - 1


class _Intercept:
    """ASGI wrapper answering a few routes itself before the wrapped app."""

    def __init__(self, app: Any, handle: Any) -> None:
        self.app = app
        self.handle = handle

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = []
        more = True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        if not await self.handle(scope, body, send):
            await self.app(scope, replay, send)


async def _json(send: Any, status: int, payload: Any, headers: list[Any] | None = None) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), *(headers or [])],
        }
    )
    await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


@dataclass
class DemoServers:
    """Running demo servers: ``as_url``, ``rs_url``, and the in-memory provider."""

    authorization_server: DemoAuthorizationServer
    as_url: str
    rs_url: str


def _as_app(provider: DemoAuthorizationServer, as_url: str) -> Any:
    routes = create_auth_routes(
        provider=provider,
        issuer_url=AnyHttpUrl(as_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=["mcp", "admin"], default_scopes=["mcp"]
        ),
        revocation_options=RevocationOptions(enabled=True),
    )

    async def handle(scope: dict[str, Any], body: bytes, send: Any) -> bool:
        if scope["path"] == "/.well-known/oauth-authorization-server":
            await _json(
                send,
                200,
                {
                    "issuer": as_url,
                    "authorization_endpoint": f"{as_url}/authorize",
                    "token_endpoint": f"{as_url}/token",
                    "registration_endpoint": f"{as_url}/register",
                    "revocation_endpoint": f"{as_url}/revoke",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                    "scopes_supported": ["mcp", "admin"],
                    "authorization_response_iss_parameter_supported": True,
                },
            )
            return True
        if scope["path"] == "/revoke":
            # The SDK's revocation request model requires client_secret; a
            # public client authenticates with client_id alone.
            provider.counters["revoke"] += 1
            form = {key: values[0] for key, values in parse_qs(body.decode()).items()}
            token = form.get("token", "")
            loaded = provider.refresh_tokens.get(token) or provider.access_tokens.get(token)
            if loaded is not None and loaded.client_id == form.get("client_id"):
                await provider.revoke_token(loaded)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return True
        return False

    return _Intercept(Starlette(routes=routes), handle)


def _rs_app(provider: DemoAuthorizationServer, as_url: str, rs_url: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        server = MCPServer(
            name="oauth-demo",
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(as_url),
                resource_server_url=AnyHttpUrl(rs_url),
                required_scopes=["mcp"],
                validate_token_resource=True,
            ),
            token_verifier=ProviderTokenVerifier(provider),  # type: ignore[arg-type]  # demo provider satisfies the protocol structurally
        )

    @server.tool()
    def echo(text: str) -> str:
        """Echo text back (requires scope mcp)."""
        return text

    @server.tool()
    def admin_echo(text: str) -> str:
        """Echo text with an admin prefix (requires scope admin)."""
        return f"admin:{text}"

    inner = server.streamable_http_app()

    async def handle(scope: dict[str, Any], body: bytes, send: Any) -> bool:
        if scope["path"] == "/.well-known/oauth-protected-resource/mcp":
            await _json(
                send,
                200,
                {
                    "resource": rs_url,
                    "authorization_servers": [as_url],
                    "scopes_supported": ["mcp"],
                },
            )
            return True
        if scope["path"] != "/mcp" or scope["method"] != "POST" or not body:
            return False
        try:
            message = json.loads(body)
        except ValueError:
            return False
        is_admin_call = (
            isinstance(message, dict)
            and message.get("method") == "tools/call"
            and message.get("params", {}).get("name") == "admin_echo"
        )
        if not is_admin_call:
            return False
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        token = headers.get("authorization", "")[7:]
        access = await provider.load_access_token(token)
        if access is not None and "admin" not in access.scopes:
            await _json(
                send,
                403,
                {"error": "insufficient_scope"},
                [(b"www-authenticate", b'Bearer error="insufficient_scope", scope="admin"')],
            )
            return True
        return False

    return _Intercept(inner, handle)


@contextlib.asynccontextmanager
async def serve_demo() -> AsyncIterator[DemoServers]:
    """Run the demo authorization and resource servers on loopback ports."""
    sockets = []
    for _ in range(2):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(64)
        sockets.append(sock)
    as_url = f"http://127.0.0.1:{sockets[0].getsockname()[1]}"
    rs_url = f"http://127.0.0.1:{sockets[1].getsockname()[1]}/mcp"
    provider = DemoAuthorizationServer(issuer=as_url)
    servers = []
    for app, sock in (
        (_as_app(provider, as_url), sockets[0]),
        (_rs_app(provider, as_url, rs_url), sockets[1]),
    ):
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
        servers.append((server, asyncio.create_task(server.serve(sockets=[sock])), sock))
    for server, task, _ in servers:
        while not server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
    try:
        yield DemoServers(provider, as_url, rs_url)
    finally:
        for server, _, _ in servers:
            server.should_exit = True
        for _, task, sock in servers:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=5.0)
            sock.close()
