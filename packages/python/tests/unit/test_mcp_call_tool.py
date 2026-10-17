"""``MCPConnection.call_tool`` through the real SDK Streamable HTTP stack.

The server is the SDK's own, wrapped in a bearer check whose accepted token
the test can rotate, and served by uvicorn on a loopback port. The runtime
path under test is the one Agent-run calls take.
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
from mcp.types import CallToolResult, TextContent

from dendrux import Agent
from dendrux.mcp import (
    MCPAuthenticationError,
    MCPCallCapacityError,
    MCPRuntime,
    MCPRuntimeClosedError,
    MCPSource,
    MCPToolCallCompleted,
    MCPToolCallError,
    MCPToolCallFailed,
    MCPToolCallStarted,
    MCPToolNotFoundError,
    error_code_for,
)

SECRET = "super-secret-token"

ASGIApp = Callable[
    [dict[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]


class _State:
    def __init__(self) -> None:
        self.token = SECRET
        self.in_flight = 0
        self.peak = 0
        self.calls: list[str] = []
        self.rejections = 0


def _app(state: _State) -> ASGIApp:
    server = SDKServer(name="call-tool-demo")

    @server.tool()
    def echo(text: str) -> str:
        state.calls.append(text)
        return f"echo:{text}"

    @server.tool()
    def boom(text: str) -> CallToolResult:
        return CallToolResult(content=[TextContent(text=f"exploded with {text}")], is_error=True)

    @server.tool()
    async def slow(seconds: float) -> str:
        state.in_flight += 1
        state.peak = max(state.peak, state.in_flight)
        try:
            await asyncio.sleep(seconds)
        finally:
            state.in_flight -= 1
        return "slept"

    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if headers.get("authorization") != f"Bearer {state.token}":
                state.rejections += 1
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"rejected"})
                return
        await inner(scope, receive, send)

    return app


@asynccontextmanager
async def _serve(state: _State) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_app(state), log_level="warning", lifespan="on"))
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


def _source(url: str, **kwargs: Any) -> MCPSource:
    return MCPSource.http("demo", url, headers={"Authorization": f"Bearer {SECRET}"}, **kwargs)


def _chain(error: BaseException) -> str:
    parts: list[str] = []
    node: BaseException | None = error
    while node is not None:
        parts.append(str(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


class _Recorder:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def on_event(self, event: Any) -> None:
        self.events.append(event)


class _RotatingProvider:
    """Hands out the current server token; adopts the new one on rejection."""

    def __init__(self, state: _State) -> None:
        self._state = state
        self.token = state.token
        self.rejected = 0

    async def get_auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def on_auth_rejected(self) -> None:
        self.rejected += 1
        self.token = self._state.token


async def test_call_tool_returns_the_normalized_agent_payload() -> None:
    state = _State()
    recorder = _Recorder()
    async with _serve(state) as url, MCPRuntime(observer=recorder) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))

        result = await connection.call_tool("echo", {"text": "hi"})

        assert result == {"result": "echo:hi"}
        # The lease is returned: the connection stays warm and idle.
        assert runtime.snapshot().in_flight_calls == 0
        # Same payload an Agent-run call returns.
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        assert await lookups.fn["demo__echo"](text="hi") == result
        await agent.close()
    started = [e for e in recorder.events if isinstance(e, MCPToolCallStarted)]
    completed = [e for e in recorder.events if isinstance(e, MCPToolCallCompleted)]
    assert started[0].tool == "echo"
    assert completed[0].tool == "echo"


async def test_arguments_default_to_empty_and_inputs_are_validated() -> None:
    state = _State()
    async with _serve(state) as url, MCPRuntime() as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))
        for bad in ("", None, 3):
            with pytest.raises(ValueError, match="non-empty string"):
                await connection.call_tool(bad)  # type: ignore[arg-type]
        for bad_timeout in (0, -1, float("nan"), float("inf"), True, "1"):
            with pytest.raises(ValueError, match="timeout"):
                await connection.call_tool("echo", {"text": "x"}, timeout=bad_timeout)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="strings"):
            await connection.call_tool("echo", {1: "x"})  # type: ignore[dict-item]
        # Nothing reached the server or the runtime.
        assert state.calls == []
        assert runtime.snapshot().connections == ()


async def test_unknown_tool_raises_not_found_and_sends_nothing() -> None:
    state = _State()
    recorder = _Recorder()
    async with _serve(state) as url, MCPRuntime(observer=recorder) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))

        with pytest.raises(MCPToolNotFoundError) as excinfo:
            await connection.call_tool("demo__echo", {"text": "x"})

        assert excinfo.value.tool_name == "demo__echo"
        assert error_code_for(excinfo.value) == "tool_not_found"
        assert not any(isinstance(e, MCPToolCallStarted) for e in recorder.events)
        # The lease was returned: the same runtime still serves the real name.
        assert await connection.call_tool("echo", {"text": "ok"}) == {"result": "echo:ok"}
        assert SECRET not in _chain(excinfo.value)


async def test_server_error_result_raises_tool_call_error_with_redaction() -> None:
    state = _State()
    recorder = _Recorder()
    async with _serve(state) as url, MCPRuntime(observer=recorder) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))

        with pytest.raises(MCPToolCallError) as excinfo:
            await connection.call_tool("boom", {"text": f"Bearer {SECRET}"})

        assert "exploded" in str(excinfo.value)
        assert SECRET not in _chain(excinfo.value)
        assert SECRET not in repr(excinfo.value)
        assert any(isinstance(e, MCPToolCallFailed) for e in recorder.events)
        assert runtime.snapshot().in_flight_calls == 0
        # A tool that merely failed must not poison the shared connection.
        assert await connection.call_tool("echo", {"text": "after"}) == {"result": "echo:after"}


async def test_a_recoverable_401_reauthenticates_once_and_retries() -> None:
    state = _State()
    provider = _RotatingProvider(state)
    async with _serve(state) as url, MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", url, reauthenticate_on_401=True),
        )
        assert await connection.call_tool("echo", {"text": "one"}) == {"result": "echo:one"}

        state.token = "rotated-token"
        assert await connection.call_tool("echo", {"text": "two"}) == {"result": "echo:two"}

        assert provider.rejected == 1
        assert state.calls == ["one", "two"]
        assert runtime.snapshot().in_flight_calls == 0


async def test_an_unrecoverable_401_surfaces_a_typed_error_without_credentials() -> None:
    state = _State()
    async with _serve(state) as url, MCPRuntime() as runtime:
        # reauthenticate_on_401 is off: the rejection is terminal.
        connection = runtime.bind(connection_key="demo", source=_source(url))
        await connection.discover()
        state.token = "rotated-token"

        with pytest.raises(MCPAuthenticationError) as excinfo:
            await connection.call_tool("echo", {"text": "x"})

        assert excinfo.value.status_code == 401
        assert SECRET not in _chain(excinfo.value)
        assert runtime.snapshot().in_flight_calls == 0


async def test_concurrent_calls_respect_the_in_flight_cap() -> None:
    state = _State()
    async with _serve(state) as url, MCPRuntime(max_in_flight_calls=2) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))
        await connection.discover()

        results = await asyncio.gather(
            *(connection.call_tool("slow", {"seconds": 0.15}) for _ in range(6))
        )

        assert results == [{"result": "slept"}] * 6
        assert state.peak == 2
        assert runtime.snapshot().in_flight_calls == 0


async def test_a_call_that_cannot_get_a_slot_is_shed_with_capacity_error() -> None:
    state = _State()
    async with (
        _serve(state) as url,
        MCPRuntime(max_in_flight_calls=1, call_wait_timeout=0.2) as runtime,
    ):
        connection = runtime.bind(connection_key="demo", source=_source(url))
        await connection.discover()

        holder = asyncio.create_task(connection.call_tool("slow", {"seconds": 1.0}))
        while state.in_flight == 0:
            await asyncio.sleep(0.01)
        with pytest.raises(MCPCallCapacityError):
            await connection.call_tool("echo", {"text": "queued"})
        assert await holder == {"result": "slept"}
        assert state.calls == []


async def test_call_after_runtime_close_raises_closed_error() -> None:
    state = _State()
    async with _serve(state) as url:
        runtime = MCPRuntime()
        connection = runtime.bind(connection_key="demo", source=_source(url))
        assert await connection.call_tool("echo", {"text": "live"}) == {"result": "echo:live"}
        await runtime.close()

        with pytest.raises(MCPRuntimeClosedError):
            await connection.call_tool("echo", {"text": "late"})
        assert state.calls == ["live"]


async def test_calling_from_another_event_loop_is_a_usage_error() -> None:
    state = _State()
    async with _serve(state) as url, MCPRuntime() as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))
        await connection.discover()

        def other_loop() -> Any:
            return asyncio.run(connection.call_tool("echo", {"text": "x"}))

        with pytest.raises(RuntimeError, match="event loop"):
            await asyncio.to_thread(other_loop)
        assert state.calls == []
        # The owning loop is unaffected.
        assert await connection.call_tool("echo", {"text": "y"}) == {"result": "echo:y"}


async def test_timeout_is_an_outer_deadline_and_leaves_the_connection_usable() -> None:
    state = _State()
    recorder = _Recorder()
    async with _serve(state) as url, MCPRuntime(observer=recorder) as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))
        await connection.discover()

        with pytest.raises(TimeoutError, match="did not finish within") as excinfo:
            await connection.call_tool("slow", {"seconds": 5.0}, timeout=0.3)

        assert SECRET not in _chain(excinfo.value)
        assert runtime.snapshot().in_flight_calls == 0
        assert await connection.call_tool("echo", {"text": "again"}) == {"result": "echo:again"}
        started = [e for e in recorder.events if isinstance(e, MCPToolCallStarted)]
        terminal = [
            e
            for e in recorder.events
            if type(e).__name__
            in {
                "MCPToolCallCompleted",
                "MCPToolCallFailed",
                "MCPToolCallCancelled",
                "MCPToolCallOutcomeUnknown",
            }
        ]
        assert len(started) == len(terminal)


async def test_timeout_none_still_honors_the_source_call_timeout() -> None:
    state = _State()
    async with _serve(state) as url, MCPRuntime() as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url, call_timeout=0.3))
        await connection.discover()

        with pytest.raises(MCPToolCallError) as excinfo:
            await connection.call_tool("slow", {"seconds": 5.0})

        assert not isinstance(excinfo.value, TimeoutError)
        assert runtime.snapshot().in_flight_calls == 0


async def test_a_generous_timeout_does_not_disturb_a_fast_call() -> None:
    state = _State()
    async with _serve(state) as url, MCPRuntime() as runtime:
        connection = runtime.bind(connection_key="demo", source=_source(url))

        assert await connection.call_tool("echo", {"text": "q"}, timeout=30) == {"result": "echo:q"}
