"""MCP session expiry, connection recovery and the read cap on a real SDK server.

D-004: after a stateful server forgets a session, the ``tools/call`` POST gets
HTTP 404 "Session not found". The call never ran, so the runtime opens a new
session and retries it once. A transport loss mid-call is different: the
outcome is unknown and nothing is retried. Stateless servers carry no session
and are never treated this way.

D-045: an Agent whose connection was lost moves to a replacement connection
on its next call instead of failing forever.

D-051: the MCP read cap stops reading an oversized ``tools/call`` response.

The server is the SDK's own (``MCPServer.streamable_http_app()``) on a loopback
socket via uvicorn. Session expiry is produced by the real SDK session manager:
the wrapper rewrites a forgotten session id so the manager answers with its
own 404.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("uvicorn")
import uvicorn
from mcp.server.mcpserver import MCPServer as SDKServer

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.mcp import (
    MCPConnectionStatus,
    MCPOutcomeUnknownError,
    MCPResultTooLargeError,
    MCPRuntime,
    MCPSessionExpiredError,
    MCPSource,
    MCPToolCallError,
)
from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._http import wire_cap
from dendrux.tools._limits import result_cap
from dendrux.types import Limits, LLMResponse, RunStatus, ToolCall

# Protocol versions of the initialize-handshake era (2024 and 2025 specs).
_HANDSHAKE_ERA = ("2024-", "2025-")

ASGIApp = Callable[
    [dict[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]


class Harness:
    """A real SDK server plus the knobs a test turns on its HTTP traffic."""

    def __init__(self, *, legacy: bool = True) -> None:
        # legacy=True: a 2025-era stateful server (initialize + Mcp-Session-Id),
        # the kind that forgets sessions. legacy=False: the SDK's own
        # 2026-07-28 stateless protocol, which has no session at all.
        self.legacy = legacy
        self.executions: list[str] = []
        self.sessions: set[str] = set()
        self.expired: set[str] = set()
        self.drop_next_call = False
        self.not_found_for_calls = False
        # Every tools/call finds its session forgotten (retry bound test).
        self.expire_always = False
        server = SDKServer(name="session-demo")

        @server.tool()
        def echo(text: str) -> str:
            self.executions.append(text)
            return f"echo:{text}"

        @server.tool()
        def blob(size: int) -> str:
            self.executions.append(f"blob:{size}")
            return "x" * size

        self.inner = server.streamable_http_app()

    def expire_sessions(self) -> None:
        """The server forgets every session it has handed out so far."""
        self.expired |= self.sessions

    async def app(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        headers = [(k, v) for k, v in scope.get("headers", [])]
        version = next(
            (v.decode() for k, v in headers if k.lower() == b"mcp-protocol-version"), None
        )
        if self.legacy and version is not None and not version.startswith(_HANDSHAKE_ERA):
            # A legacy-only server: the client's modern probe fails and it
            # falls back to the initialize handshake.
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"Bad Request"})
            return
        body = b""
        more = True
        messages: list[Any] = []
        while more:
            message = await receive()
            messages.append(message)
            body += message.get("body", b"")
            more = message.get("more_body", False)
        is_call = b'"tools/call"' in body
        if is_call and self.expire_always:
            self.expire_sessions()
        session = next((v.decode() for k, v in headers if k.lower() == b"mcp-session-id"), None)
        if session in self.expired:
            # The SDK manager no longer knows this id: it answers 404 itself.
            headers = [
                (k, b"forgotten-" + v if k.lower() == b"mcp-session-id" else v) for k, v in headers
            ]
            scope = {**scope, "headers": headers}

        if is_call and self.not_found_for_calls:
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"Not Found"})
            return

        replay = iter(messages)

        async def replay_receive() -> Any:
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        drop = is_call and self.drop_next_call
        if drop:
            self.drop_next_call = False

        async def watch_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                for key, value in message.get("headers", []):
                    if key.lower() == b"mcp-session-id":
                        self.sessions.add(value.decode())
            if drop and message["type"] == "http.response.body":
                # The tool ran, then the connection dies before any reply.
                raise ConnectionResetError("dropped mid-call")
            await send(message)

        await self.inner(scope, replay_receive, watch_send)


@asynccontextmanager
async def _serve(app: ASGIApp) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]

    # A plain function: uvicorn would take a bound method for an ASGI2 app.
    async def asgi(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await app(scope, receive, send)

    server = uvicorn.Server(uvicorn.Config(asgi, log_level="critical", lifespan="on"))
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


def _source(url: str, **kw: Any) -> MCPSource:
    return MCPSource.http("demo", url, **kw)


@asynccontextmanager
async def _runtime(url: str, **kw: Any) -> AsyncIterator[tuple[MCPRuntime, Any]]:
    async with MCPRuntime(shutdown_timeout=2.0, close_timeout=2.0) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url, **kw))
        yield runtime, connection


def _status(runtime: MCPRuntime) -> list[MCPConnectionStatus]:
    return [c.status for c in runtime.snapshot().connections]


@pytest.mark.asyncio
async def test_expired_session_is_reopened_and_the_call_retried_once() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url) as (runtime, connection):
        agent = Agent(prompt="p", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        echo = lookups.fn["demo__echo"]
        assert await echo(text="a") == {"result": "echo:a"}

        harness.expire_sessions()
        assert await echo(text="b") == {"result": "echo:b"}
        assert await echo(text="c") == {"result": "echo:c"}

        # The rejected attempt never ran: each call executed exactly once.
        assert harness.executions == ["a", "b", "c"]
        assert len(harness.sessions) == 2
        assert _status(runtime) == [MCPConnectionStatus.ACTIVE]
        await agent.close()


@pytest.mark.asyncio
async def test_fresh_agents_recover_after_session_expiry() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url) as (_, connection):
        first = Agent(prompt="p", tool_sources=[connection.tools()])
        assert await (await first.get_tool_lookups()).fn["demo__echo"](text="a") == {
            "result": "echo:a"
        }
        harness.expire_sessions()
        for n in range(3):
            agent = Agent(prompt="p", tool_sources=[connection.tools()])
            lookups = await agent.get_tool_lookups()
            assert await lookups.fn["demo__echo"](text=f"fresh-{n}") == {
                "result": f"echo:fresh-{n}"
            }
            await agent.close()
        assert harness.executions == ["a", "fresh-0", "fresh-1", "fresh-2"]
        await first.close()


@pytest.mark.asyncio
async def test_direct_call_tool_recovers_from_session_expiry() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url) as (_, connection):
        assert await connection.call_tool("echo", {"text": "a"}) == {"result": "echo:a"}
        harness.expire_sessions()
        assert await connection.call_tool("echo", {"text": "b"}) == {"result": "echo:b"}
        assert harness.executions == ["a", "b"]


@pytest.mark.asyncio
async def test_retry_is_bounded_to_one() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url) as (_, connection):
        agent = Agent(prompt="p", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        assert await lookups.fn["demo__echo"](text="a") == {"result": "echo:a"}

        harness.expire_sessions()
        harness.expire_always = True
        with pytest.raises(MCPSessionExpiredError):
            await lookups.fn["demo__echo"](text="never")
        assert harness.executions == ["a"]
        await agent.close()


@pytest.mark.asyncio
async def test_transport_loss_mid_call_is_never_retried_and_the_agent_recovers() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url) as (runtime, connection):
        agent = Agent(prompt="p", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        echo = lookups.fn["demo__echo"]
        assert await echo(text="a") == {"result": "echo:a"}

        harness.drop_next_call = True
        with pytest.raises(MCPOutcomeUnknownError):
            await echo(text="maybe")
        # The tool ran once and was not run again by the runtime.
        assert harness.executions == ["a", "maybe"]

        # D-045: the same Agent moves to a new connection instead of failing.
        assert await echo(text="after") == {"result": "echo:after"}
        assert harness.executions == ["a", "maybe", "after"]
        assert _status(runtime) == [MCPConnectionStatus.ACTIVE]
        await agent.close()


@pytest.mark.asyncio
async def test_adapter_classifies_only_session_carrying_404s() -> None:
    harness = Harness()
    async with _serve(harness.app) as url:
        adapter = MCPClientAdapter(_source(url))
        await adapter.connect()
        try:
            harness.expire_sessions()
            with pytest.raises(MCPSessionExpiredError) as excinfo:
                await adapter.call_tool("echo", {"text": "x"})
            assert excinfo.value.connection_lost is True
            assert excinfo.value.__context__ is None
        finally:
            await adapter.close()
        assert harness.executions == []


@pytest.mark.asyncio
async def test_stateless_server_404_is_an_ordinary_failure() -> None:
    harness = Harness(legacy=False)
    async with _serve(harness.app) as url, _runtime(url) as (runtime, connection):
        agent = Agent(prompt="p", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        assert await lookups.fn["demo__echo"](text="a") == {"result": "echo:a"}
        assert harness.sessions == set()

        harness.not_found_for_calls = True
        with pytest.raises(MCPToolCallError) as excinfo:
            await lookups.fn["demo__echo"](text="b")
        assert not isinstance(excinfo.value, MCPSessionExpiredError)
        # No fencing: the same connection keeps serving.
        harness.not_found_for_calls = False
        assert await lookups.fn["demo__echo"](text="c") == {"result": "echo:c"}
        assert len(runtime.snapshot().connections) == 1
        await agent.close()


@pytest.mark.asyncio
async def test_read_cap_stops_an_oversized_response() -> None:
    # Runs without image ingest get no image allowance: the cap is 4x the text cap.
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url, max_result_bytes=1000) as (_, conn):
        agent = Agent(prompt="p", tool_sources=[conn.tools()])
        lookups = await agent.get_tool_lookups()
        with pytest.raises(MCPResultTooLargeError) as excinfo:
            await lookups.fn["demo__blob"](size=500_000)
        assert excinfo.value.size is not None and excinfo.value.size < 200_000
        # The session is healthy: an ordinary failure, not a lost connection.
        assert await lookups.fn["demo__echo"](text="ok") == {"result": "echo:ok"}
        await agent.close()


def _blob_run(connection: Any, limits: Limits) -> tuple[Agent, MockLLM]:
    llm = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="demo__blob", params={"size": 5000})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(prompt="p", provider=llm, tool_sources=[connection.tools()], limits=limits)
    return agent, llm


def _tool_message(llm: MockLLM) -> Any:
    [message] = [m for m in llm.call_history[1]["messages"] if m.role.value == "tool"]
    return json.loads(message.content)


@pytest.mark.asyncio
async def test_agent_cap_applies_to_mcp_results() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url) as (_, connection):
        agent, llm = _blob_run(connection, Limits(max_tool_result_bytes=2000))
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        # A structured result over the Agent's cap is replaced, keeping valid JSON.
        body = _tool_message(llm)
        assert body["code"] == "size_limit" and body["bytes"] > 2000
        await agent.close()


@pytest.mark.asyncio
async def test_source_cap_in_a_run_is_a_size_limit_result() -> None:
    harness = Harness()
    async with _serve(harness.app) as url, _runtime(url, max_result_bytes=1000) as (_, conn):
        agent, llm = _blob_run(conn, Limits())
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        body = _tool_message(llm)
        assert body["code"] == "size_limit" and body["error"] == "result too large"
        assert body["bytes"] > 1000
        await agent.close()


def test_wire_cap_follows_limits_and_adds_image_room_only_with_images() -> None:
    assert wire_cap(1_000_000) == 4_000_000
    with result_cap(2000):
        assert wire_cap(1_000_000) == 8000
    with result_cap(1024 * 1024, image_bytes=33 * 1024 * 1024):
        assert wire_cap(1_000_000) == 4_000_000 + 33 * 1024 * 1024
