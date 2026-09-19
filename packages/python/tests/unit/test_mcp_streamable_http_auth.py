"""Credential rejection through the real SDK Streamable HTTP stack.

The SDK turns a non-2xx reply to the initialize POST into a JSON-RPC error
with no status attached, so only an end-to-end run over a real transport
proves that a wrong bearer token reaches applications as
``MCPAuthenticationError``. The server is the SDK's own, wrapped in a
bearer check, served by uvicorn on a loopback port.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("uvicorn")
import uvicorn
from mcp.server.mcpserver import MCPServer as SDKServer

from dendrux import Agent
from dendrux.mcp import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPRuntime,
    MCPSource,
)
from dendrux.mcp._client import MCPClientAdapter

TOKEN = "good-token"
WRONG_TOKEN = "wrong-token"

ASGIApp = Callable[
    [dict[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]


def _bearer_app(rejection_status: int) -> ASGIApp:
    server = SDKServer(name="bearer-demo", instructions="demo server")

    @server.tool()
    def echo(text: str) -> str:
        return text

    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if headers.get("authorization") != f"Bearer {TOKEN}":
                await send(
                    {
                        "type": "http.response.start",
                        "status": rejection_status,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"rejected"})
                return
        await inner(scope, receive, send)

    return app


@asynccontextmanager
async def _serve(app: ASGIApp) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        sock.close()


def _source(url: str, token: str) -> MCPSource:
    return MCPSource.http("demo", url, headers={"Authorization": f"Bearer {token}"})


def _rendered_chain(error: BaseException) -> str:
    parts: list[str] = []
    node: BaseException | None = error
    while node is not None:
        parts.append(str(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_wrong_bearer_token_is_a_typed_rejection(status: int) -> None:
    async with _serve(_bearer_app(status)) as url:
        adapter = MCPClientAdapter(_source(url, WRONG_TOKEN))

        with pytest.raises(MCPAuthenticationError) as excinfo:
            await adapter.connect()

        error = excinfo.value
        assert error.status_code == status
        assert str(error) == (
            f"Failed to connect to MCP source 'demo': "
            f"the server rejected its credentials (HTTP {status})."
        )
        assert error.transport_detail == "MCPError: Server returned an error response"
        assert WRONG_TOKEN not in _rendered_chain(error)
        assert WRONG_TOKEN not in (error.transport_detail or "")
        await adapter.close()


@pytest.mark.asyncio
async def test_correct_bearer_token_connects_and_lists_tools() -> None:
    async with _serve(_bearer_app(401)) as url:
        adapter = MCPClientAdapter(_source(url, TOKEN))
        await adapter.connect()
        try:
            tools = await adapter.list_tools()
        finally:
            await adapter.close()

        assert [tool.name for tool in tools] == ["echo"]
        assert adapter.info is None  # cleared on close
        assert adapter._last_status is not None and adapter._last_status < 400


@pytest.mark.asyncio
async def test_wrong_path_stays_a_plain_connection_error() -> None:
    """A bad URL must not be reported as rejected credentials."""
    async with _serve(_bearer_app(401)) as url:
        adapter = MCPClientAdapter(_source(url.replace("/mcp", "/nope"), TOKEN))

        with pytest.raises(MCPConnectionError) as excinfo:
            await adapter.connect()

        error = excinfo.value
        assert not isinstance(error, MCPAuthenticationError)
        assert "TaskGroup" not in (error.transport_detail or "")


@pytest.mark.asyncio
async def test_managed_runtime_surfaces_the_rejection_to_the_agent() -> None:
    async with _serve(_bearer_app(401)) as url, MCPRuntime(shutdown_timeout=1.0) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url, WRONG_TOKEN))
        agent = Agent(prompt="test", tool_sources=[connection.tools()])

        with pytest.raises(MCPAuthenticationError) as excinfo:
            await agent.get_tool_lookups()

        assert excinfo.value.status_code == 401
        await agent.close()
