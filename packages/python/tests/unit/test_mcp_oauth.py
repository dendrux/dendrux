"""Live authentication attribution, rotation, and diagnostic safety contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx2
import pytest
from mcp.server.mcpserver import MCPServer as SDKServer
from mcp.types import CallToolResult, TextContent, Tool

from dendrux.mcp import MCPAuthenticationError, MCPConnectionError, MCPError, MCPRuntime, MCPSource
from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._runtime import _ViewToolSource
from dendrux.mcp._server import create_mcp_executor
from tests.unit.test_mcp_streamable_http_auth import _serve


def test_connection_error_has_structured_origin_without_url_in_message():
    source = MCPSource.http("demo", "https://source.internal/mcp?secret=one")
    adapter = MCPClientAdapter(source)
    request = httpx2.Request("POST", "https://target.internal:8443/mcp?token=unknown-secret")
    error = adapter._safe_error("connect to", httpx2.ConnectError("refused", request=request))
    assert error.origin.host == "source.internal"
    assert error.destination.host == "target.internal"
    assert error.destination.port == 8443
    assert error.redirect_target == error.destination
    assert error.host == "target.internal"
    assert "internal" not in str(error)
    assert "internal" not in repr(error.destination)
    assert "token" not in repr(error)


async def test_transport_tool_errors_do_not_embed_unknown_urls():
    adapter = MCPClientAdapter(MCPSource.http("demo", "https://source.example"))
    adapter.call_tool = AsyncMock(
        side_effect=httpx2.ConnectError(
            "Cannot connect https://unknown.internal/mcp?token=not-configured"
        )
    )
    executor = create_mcp_executor(
        adapter, namespace="demo", mcp_tool_name="echo", max_result_bytes=1000
    )
    with pytest.raises(Exception) as caught:
        await executor()
    assert "unknown.internal" not in str(caught.value)
    assert "not-configured" not in str(caught.value)


def live_app():
    server = SDKServer(name="oauth-test")
    state = {"token": "one", "executions": 0, "rejects": 0, "status": 401}

    @server.tool()
    async def echo(text: str) -> str:
        state["executions"] += 1
        return text

    inner = server.streamable_http_app()

    async def app(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {state['token']}".encode():
                state["rejects"] += 1
                await send(
                    {"type": "http.response.start", "status": state["status"], "headers": []}
                )
                await send({"type": "http.response.body", "body": b"expired"})
                return
        await inner(scope, receive, send)

    return app, state


@pytest.mark.parametrize("status", [401, 403])
async def test_live_call_has_typed_authentication_error(status):
    app, state = live_app()
    async with _serve(app) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer one"})
        )
        await adapter.connect()
        try:
            state["token"] = "two"
            state["status"] = status
            with pytest.raises(MCPAuthenticationError) as caught:
                await adapter.call_tool("echo", {"text": "hello"})
            assert caught.value.status_code == status
            assert caught.value.request_rejected
            assert state["executions"] == 0
        finally:
            await adapter.close()


class Provider:
    def __init__(self, token="one"):
        self.token = token
        self.gets = 0
        self.rejections = 0

    async def get_auth(self):
        self.gets += 1
        return {"Authorization": f"Bearer {self.token}"}

    async def on_auth_rejected(self):
        self.rejections += 1
        self.token = "two"


async def test_opt_in_reauth_retries_rejected_write_once():
    app, state = live_app()
    provider = Provider()
    async with _serve(app) as url, MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", url, reauthenticate_on_401=True),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        state["token"] = "two"
        try:
            assert await executor(text="hello") == {"result": "hello"}
            assert state["executions"] == 1
            assert provider.gets == 2
            assert provider.rejections == 1
        finally:
            await view.close()


class Adapter:
    instances = []
    gate = None
    tools = [Tool(name="echo", input_schema={"type": "object"})]

    def __init__(self, source):
        self.source = source
        self.info = None
        self.calls = 0
        self.lists = 0
        self.closed = False
        self.started = asyncio.Event()
        self.instances.append(self)

    async def connect(self):
        pass

    async def list_tools(self):
        self.lists += 1
        return list(self.tools)

    async def call_tool(self, name, arguments):
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return CallToolResult(content=[TextContent(text=self.source.headers["Authorization"])])

    async def close(self):
        self.closed = True


@pytest.fixture
def adapters(monkeypatch):
    Adapter.instances = []
    Adapter.gate = None
    Adapter.tools = [Tool(name="echo", input_schema={"type": "object"})]
    monkeypatch.setattr("dendrux.mcp._runtime.MCPClientAdapter", Adapter)
    return Adapter


@pytest.mark.parametrize("preserve", [True, False])
async def test_rotation_preserves_handles_and_reconnects_lazily(adapters, preserve):
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            source=MCPSource.http("demo", "https://example.com"),
            credentials=Provider(),
            credential_identity="v1",
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        await runtime.rotate_credentials(
            connection_key="demo",
            credentials=Provider("two"),
            credential_identity="v2",
            preserve_catalog=preserve,
        )
        assert len(adapters.instances) == 1
        assert adapters.instances[0].closed
        assert view._entry.raw_tools
        await executor()
        assert len(adapters.instances) == 2
        assert adapters.instances[1].lists == (0 if preserve else 1)
        await connection.discover()
        await view.close()


async def test_rotation_drains_calls_and_blocks_new_admission(adapters):
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            source=MCPSource.http("demo", "https://example.com"),
            credentials=Provider(),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        first = adapters.instances[0]
        first.gate = asyncio.Event()
        executor = view._create_executor("echo")
        active = asyncio.create_task(executor())
        await first.started.wait()
        rotation = asyncio.create_task(
            runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        )
        await asyncio.sleep(0)
        queued = asyncio.create_task(executor())
        await asyncio.sleep(0)
        assert first.calls == 1
        assert not rotation.done()
        first.gate.set()
        await asyncio.gather(active, rotation, queued)
        assert first.calls == 1
        assert adapters.instances[1].calls == 1
        await view.close()


def rejected(status=401, correlated=True):
    error = MCPAuthenticationError("Credentials rejected.")
    error.status_code = status
    error.request_rejected = correlated
    return error


@pytest.mark.parametrize("second_rejected", [False, True])
async def test_concurrent_rejections_share_one_refresh(adapters, monkeypatch, second_rejected):
    count = 6
    all_started = asyncio.Event()
    old_calls = 0
    executions = 0
    events = []

    class Observer:
        def on_event(self, event):
            events.append(event)

    async def call(self, name, arguments):
        nonlocal old_calls, executions
        self.calls += 1
        if self.source.headers["Authorization"] == "Bearer one":
            old_calls += 1
            if old_calls == count:
                all_started.set()
            await all_started.wait()
            raise rejected()
        if second_rejected:
            raise rejected()
        executions += 1
        return CallToolResult(content=[TextContent(text="ok")])

    monkeypatch.setattr(adapters, "call_tool", call)
    provider = Provider()
    async with MCPRuntime(observer=Observer()) as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com", reauthenticate_on_401=True),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        results = await asyncio.wait_for(
            asyncio.gather(*(executor() for _ in range(count)), return_exceptions=True), timeout=2
        )
        assert provider.gets == 2
        assert provider.rejections == 1
        assert len(adapters.instances) == 2
        assert executions == (0 if second_rejected else count)
        if second_rejected:
            assert all(isinstance(result, MCPAuthenticationError) for result in results)
        else:
            assert results == ["ok"] * count
        assert runtime._in_flight_calls == 0
        starts = [e for e in events if type(e).__name__ == "MCPToolCallStarted"]
        terminals = [
            e for e in events if type(e).__name__ in ("MCPToolCallCompleted", "MCPToolCallFailed")
        ]
        assert len(starts) == len(terminals) == count
        assert {e.instance_id for e in starts} == {e.instance_id for e in terminals}
        await view.close()


@pytest.mark.parametrize(
    "failure",
    [
        rejected(403),
        rejected(correlated=False),
        RuntimeError("401 in arbitrary text"),
        httpx2.ReadTimeout("timeout"),
        httpx2.ConnectError("connection lost"),
    ],
)
async def test_unproven_or_non401_failures_are_never_replayed(adapters, monkeypatch, failure):
    call = AsyncMock(side_effect=failure)
    monkeypatch.setattr(adapters, "call_tool", call)
    provider = Provider()
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com", reauthenticate_on_401=True),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        with pytest.raises(MCPError):
            await view._create_executor("echo")()
        assert call.await_count == 1
        assert provider.rejections == 0
        assert provider.gets == 1
        await view.close()


async def test_default_does_not_reauthenticate(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "call_tool", AsyncMock(side_effect=rejected()))
    provider = Provider()
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com"),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        with pytest.raises(MCPAuthenticationError):
            await view._create_executor("echo")()
        assert provider.rejections == 0
        await view.close()


async def test_rotation_timeout_preserves_live_transport_and_credentials(adapters):
    provider = Provider()
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com"),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        first = adapters.instances[0]
        first.gate = asyncio.Event()
        active = asyncio.create_task(view._create_executor("echo")())
        await first.started.wait()
        with pytest.raises(TimeoutError):
            await runtime.rotate_credentials(
                connection_key="demo", credentials=Provider("two"), timeout=0.01
            )
        assert not first.closed
        assert runtime._registrations[connection.identity].credentials is provider
        assert view._entry.rotation_task is None
        first.gate.set()
        await active
        await view.close()


async def test_cancelled_rotation_waiter_does_not_cancel_shared_rotation(adapters):
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        first = adapters.instances[0]
        first.gate = asyncio.Event()
        active = asyncio.create_task(view._create_executor("echo")())
        await first.started.wait()
        waiter = asyncio.create_task(
            runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        )
        await asyncio.sleep(0)
        rotation = view._entry.rotation_task
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not rotation.cancelled()
        first.gate.set()
        await asyncio.gather(active, rotation)
        await view._discover()
        await view._create_executor("echo")()
        await view.close()


async def test_changed_catalog_rejects_existing_executor_before_sending(adapters):
    from dendrux.mcp import MCPStaleConnectionError

    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        await runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        adapters.tools = [Tool(name="echo", input_schema={"type": "object", "required": ["new"]})]
        with pytest.raises(MCPStaleConnectionError):
            await executor()
        assert adapters.instances[1].calls == 0
        await view.close()


async def test_close_failure_prevents_rotated_transport_publication(adapters, monkeypatch):
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        monkeypatch.setattr(
            adapters.instances[0], "close", AsyncMock(side_effect=RuntimeError("close failed"))
        )
        with pytest.raises(MCPConnectionError):
            await runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        assert len(adapters.instances) == 1
        assert connection.identity not in runtime._entries
        await view.close()


async def test_provider_rejection_failure_is_safe_and_shared(adapters, monkeypatch):
    from dendrux.mcp import MCPCredentialError

    monkeypatch.setattr(adapters, "call_tool", AsyncMock(side_effect=rejected()))
    provider = Provider()
    provider.on_auth_rejected = AsyncMock(side_effect=RuntimeError("secret-refresh-token"))
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com", reauthenticate_on_401=True),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        for _ in range(2):
            with pytest.raises(MCPCredentialError) as caught:
                await executor()
            assert "secret-refresh-token" not in str(caught.value)
            assert caught.value.__context__ is None
        assert provider.on_auth_rejected.await_count == 1
        assert provider.gets == 1
        await view.close()


def test_reauthentication_requires_rejection_callback():
    class OldProvider:
        async def get_auth(self):
            return {"Authorization": "Bearer one"}

    with pytest.raises(ValueError, match="MCPRefreshingCredentialProvider"):
        MCPRuntime().bind(
            connection_key="demo",
            credentials=OldProvider(),
            source=MCPSource.http("demo", "https://example.com", reauthenticate_on_401=True),
        )


async def test_background_get401_cannot_classify_an_unrelated_call():
    adapter = MCPClientAdapter(MCPSource.http("demo", "https://example.com"))

    async def call(*args, **kwargs):
        await adapter._observe_response(
            httpx2.Response(401, request=httpx2.Request("GET", "https://example.com"))
        )
        raise RuntimeError("unrelated failure")

    class Client:
        call_tool = staticmethod(call)

    adapter._client = Client()
    with pytest.raises(RuntimeError) as caught:
        await adapter.call_tool("echo", {})
    assert not isinstance(caught.value, MCPAuthenticationError)


async def test_concurrent_http_statuses_are_attributed_per_call():
    adapter = MCPClientAdapter(MCPSource.http("demo", "https://example.com"))
    barrier = asyncio.Event()
    arrivals = 0

    async def call(name, arguments, **kwargs):
        nonlocal arrivals
        request = httpx2.Request(
            "POST", "https://example.com", json={"method": "tools/call", "params": {"name": name}}
        )
        await adapter._observe_response(
            httpx2.Response(401 if name == "expired" else 500, request=request)
        )
        arrivals += 1
        if arrivals == 2:
            barrier.set()
        await barrier.wait()
        raise RuntimeError("SDK failure")

    class Client:
        call_tool = staticmethod(call)

    adapter._client = Client()
    expired, failed = await asyncio.gather(
        adapter.call_tool("expired", {}), adapter.call_tool("failed", {}), return_exceptions=True
    )
    assert isinstance(expired, MCPAuthenticationError)
    assert not isinstance(failed, MCPAuthenticationError)


async def test_rotate_unopened_connection_is_inert_and_keeps_handle(adapters):
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            credential_identity="one",
            source=MCPSource.http("demo", "https://example.com"),
        )
        newer = Provider("two")
        await runtime.rotate_credentials(
            connection_key="demo", credentials=newer, credential_identity="two"
        )
        assert not adapters.instances
        assert newer.gets == 0
        await connection.discover()
        assert newer.gets == 1
        assert adapters.instances[0].source.headers["Authorization"] == "Bearer two"


async def test_rotation_cannot_revive_a_handle_invalidated_by_bind(adapters):
    from dendrux.mcp import MCPStaleConnectionError

    async with MCPRuntime() as runtime:
        source = MCPSource.http("demo", "https://example.com")
        stale = runtime.bind(
            connection_key="demo", credentials=Provider(), credential_identity="one", source=source
        )
        current = runtime.bind(
            connection_key="demo",
            credentials=Provider("two"),
            credential_identity="two",
            source=source,
        )
        await runtime.rotate_credentials(
            connection_key="demo", credentials=Provider("three"), credential_identity="three"
        )
        with pytest.raises(MCPStaleConnectionError):
            await stale.discover()
        await current.discover()


async def test_rotation_during_connection_establishment(adapters, monkeypatch):
    started = asyncio.Event()
    gate = asyncio.Event()

    async def connect(self):
        started.set()
        await gate.wait()

    monkeypatch.setattr(adapters, "connect", connect)
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        discovery = asyncio.create_task(connection.discover())
        await started.wait()
        rotation = asyncio.create_task(
            runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        )
        await asyncio.sleep(0)
        assert not rotation.done()
        gate.set()
        await asyncio.gather(discovery, rotation)
        await connection.discover()
        assert adapters.instances[-1].source.headers["Authorization"] == "Bearer two"


async def test_concurrent_explicit_rotations_are_serialized(adapters):
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        await connection.discover()
        await asyncio.gather(
            runtime.rotate_credentials(connection_key="demo", credentials=Provider("two")),
            runtime.rotate_credentials(connection_key="demo", credentials=Provider("three")),
        )
        assert len(adapters.instances) == 1
        await connection.discover()
        assert adapters.instances[-1].source.headers["Authorization"] == "Bearer three"


async def test_rotation_close_timeout_fails_without_opening_a_replacement(adapters, monkeypatch):
    gate = asyncio.Event()

    async def close():
        await gate.wait()

    async with MCPRuntime(close_timeout=0.01) as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        await connection.discover()
        monkeypatch.setattr(adapters.instances[0], "close", close)
        with pytest.raises(MCPConnectionError):
            await runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        assert len(adapters.instances) == 1
        assert connection.identity not in runtime._entries
        gate.set()
        if runtime._abandoned_tasks:
            await asyncio.gather(*runtime._abandoned_tasks)


async def test_cancelling_one_reauth_waiter_keeps_other_call_recovery(adapters, monkeypatch):
    gate = asyncio.Event()
    started = asyncio.Event()
    provider = Provider()

    async def refresh():
        provider.rejections += 1
        started.set()
        await gate.wait()
        provider.token = "two"

    provider.on_auth_rejected = refresh

    async def call(self, name, arguments):
        if self.source.headers["Authorization"] == "Bearer one":
            raise rejected()
        return CallToolResult(content=[TextContent(text="ok")])

    monkeypatch.setattr(adapters, "call_tool", call)
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com", reauthenticate_on_401=True),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        first = asyncio.create_task(executor())
        await started.wait()
        second = asyncio.create_task(executor())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        assert await second == "ok"
        assert provider.rejections == 1
        assert provider.gets == 2
        assert runtime._in_flight_calls == 0
        await view.close()


async def test_eviction_during_refresh_does_not_resurrect_connection(adapters, monkeypatch):
    started = asyncio.Event()
    gate = asyncio.Event()
    provider = Provider()

    async def refresh():
        started.set()
        await gate.wait()

    provider.on_auth_rejected = refresh
    monkeypatch.setattr(adapters, "call_tool", AsyncMock(side_effect=rejected()))
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", "https://example.com", reauthenticate_on_401=True),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        call = asyncio.create_task(view._create_executor("echo")())
        await started.wait()
        await runtime.evict(connection_key="demo", mode="force")
        result = await asyncio.gather(call, return_exceptions=True)
        assert isinstance(result[0], BaseException)
        gate.set()
        assert connection.identity not in runtime._entries
        assert connection.identity not in runtime._registrations
        assert len(adapters.instances) == 1
        await view.close()


async def test_refresh_callback_timeout_is_sanitized(adapters, monkeypatch):
    from dendrux.mcp import MCPCredentialError

    provider = Provider()
    provider.on_auth_rejected = asyncio.Event().wait
    monkeypatch.setattr(adapters, "call_tool", AsyncMock(side_effect=rejected()))
    async with MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http(
                "demo", "https://example.com", reauthenticate_on_401=True, connect_timeout=0.01
            ),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        with pytest.raises(MCPCredentialError):
            await view._create_executor("echo")()
        assert runtime._in_flight_calls == 0
        await view.close()


async def test_rotation_validation_and_unknown_identity():
    from dendrux.mcp import MCPRuntimeClosedError, MCPStaleConnectionError

    runtime = MCPRuntime()
    with pytest.raises(MCPStaleConnectionError):
        await runtime.rotate_credentials(connection_key="missing", credentials=Provider())
    runtime.bind(connection_key="stdio", source=MCPSource.stdio("demo", ["echo"]))
    with pytest.raises(ValueError, match="HTTP"):
        await runtime.rotate_credentials(connection_key="stdio", credentials=Provider())
    with pytest.raises(ValueError, match="preserve_catalog"):
        await runtime.rotate_credentials(
            connection_key="stdio", credentials=Provider(), preserve_catalog=1
        )
    await runtime.close()
    with pytest.raises(MCPRuntimeClosedError):
        await runtime.rotate_credentials(connection_key="stdio", credentials=Provider())


async def test_call_telemetry_uses_current_instance_after_explicit_rotation(adapters):
    events = []

    class Observer:
        def on_event(self, event):
            events.append(event)

    async with MCPRuntime(observer=Observer()) as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=Provider(),
            source=MCPSource.http("demo", "https://example.com"),
        )
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        await executor()
        await runtime.rotate_credentials(connection_key="demo", credentials=Provider("two"))
        await executor()
        starts = [e.instance_id for e in events if type(e).__name__ == "MCPToolCallStarted"]
        ends = [e.instance_id for e in events if type(e).__name__ == "MCPToolCallCompleted"]
        assert starts == ends
        assert starts[0] != starts[1]
        await view.close()


async def test_cross_origin_401_is_typed_without_authorizing_replay():
    adapter = MCPClientAdapter(MCPSource.http("demo", "https://source.example"))

    async def call(name, arguments, **kwargs):
        request = httpx2.Request(
            "POST",
            "https://other.internal/mcp?secret=private",
            json={"method": "tools/call", "params": {"name": name}},
        )
        await adapter._observe_response(httpx2.Response(401, request=request))
        raise RuntimeError("SDK rejected request")

    class Client:
        call_tool = staticmethod(call)

    adapter._client = Client()
    with pytest.raises(MCPAuthenticationError) as caught:
        await adapter.call_tool("echo", {})
    assert caught.value.status_code == 401
    assert not caught.value.request_rejected
    assert caught.value.redirect_target.host == "other.internal"
    assert "other.internal" not in str(caught.value)
    assert "private" not in repr(caught.value)


def test_invalid_url_diagnostics_do_not_mask_connection_failure():
    adapter = MCPClientAdapter(MCPSource.http("demo", "https://example.com:not-a-port"))
    error = adapter._safe_error("connect to", httpx2.InvalidURL("Invalid port"))
    assert error.origin is None
    assert error.destination is None
    assert error.host is None
    assert "example.com" not in str(error)


async def test_cross_origin_redirect_is_blocked_and_reports_target_on_real_sdk():
    from dendrux.mcp import MCPConnectionError

    reached = []

    async def target(scope, receive, send):
        if scope["type"] != "http":
            return
        reached.append(scope)
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b"rejected"})

    async with _serve(target) as target_url:

        async def redirect(scope, receive, send):
            if scope["type"] != "http":
                return
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", (target_url + "?token=unknown-secret").encode())],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        async with _serve(redirect) as source_url:
            adapter = MCPClientAdapter(MCPSource.http("demo", source_url))
            with pytest.raises(MCPConnectionError) as caught:
                await adapter.connect()
            await adapter.close()
    error = caught.value
    assert error.origin.port == httpx2.URL(source_url).port
    assert not reached
    assert error.destination == error.origin
    assert error.redirect_target.port == httpx2.URL(target_url).port
    assert "127.0.0.1" not in str(error)
    assert "unknown-secret" not in (error.transport_detail or "")


@pytest.mark.parametrize("follow", [False, True])
async def test_sdk_same_origin_redirect_respects_source_setting(follow):
    from dendrux.mcp import MCPConnectionError

    reached = []

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        if scope["path"] == "/next":
            reached.append(scope["path"])
            status, headers = 401, []
        else:
            status, headers = 307, [(b"location", b"/next")]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    async with _serve(app) as url:
        adapter = MCPClientAdapter(MCPSource.http("demo", url, follow_redirects=follow))
        with pytest.raises(MCPAuthenticationError if follow else MCPConnectionError):
            await adapter.connect()
        await adapter.close()
    assert bool(reached) is follow


# ---------------------------------------------------------------------------
# Connect-time and discovery-time 401 recovery
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)

    def names(self):
        return [type(event).__name__ for event in self.events]


def _opening_events(recorder):
    terminals = {"MCPConnectionOpened", "MCPConnectionFailed", "MCPConnectionAborted"}
    names = recorder.names()
    return names.count("MCPConnectionOpening"), [n for n in names if n in terminals]


def _reject_stale(phase, failure=None):
    """Adapter method that rejects the stale token."""

    async def connect(self):
        if self.source.headers["Authorization"] == "Bearer one":
            raise failure or rejected()

    async def list_tools(self):
        self.lists += 1
        if self.source.headers["Authorization"] == "Bearer one":
            raise failure or rejected()
        return list(self.tools)

    return connect if phase == "connect" else list_tools


def _bind(runtime, provider, *, reauthenticate=True, connect_timeout=None):
    kwargs = {} if connect_timeout is None else {"connect_timeout": connect_timeout}
    return runtime.bind(
        connection_key="demo",
        credentials=provider,
        source=MCPSource.http(
            "demo", "https://example.com", reauthenticate_on_401=reauthenticate, **kwargs
        ),
    )


@pytest.mark.parametrize("phase", ["connect", "discovery"])
async def test_connect_time_401_recovers_once(adapters, monkeypatch, phase):
    method = "connect" if phase == "connect" else "list_tools"
    monkeypatch.setattr(adapters, method, _reject_stale(phase))
    recorder = Recorder()
    provider = Provider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        discovery = await connection.discover()
        assert [tool.name for tool in discovery.tools] == ["echo"]
        assert provider.rejections == 1
        assert provider.gets == 2
        assert len(adapters.instances) == 2
        assert adapters.instances[0].closed
        assert not adapters.instances[1].closed
        assert runtime._entries[connection.identity].adapter is adapters.instances[1]
        assert _opening_events(recorder) == (1, ["MCPConnectionOpened"])


async def test_connect_time_recovery_is_single_flight(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    recorder = Recorder()
    provider = Provider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        await asyncio.gather(*(connection.discover() for _ in range(5)))
        assert provider.rejections == 1
        assert provider.gets == 2
        assert len(adapters.instances) == 2
        assert _opening_events(recorder) == (1, ["MCPConnectionOpened"])


class PlainProvider:
    def __init__(self):
        self.gets = 0

    async def get_auth(self):
        self.gets += 1
        return {"Authorization": "Bearer one"}


@pytest.mark.parametrize("case", ["plain_provider", "opt_out", "forbidden", "cross_origin"])
async def test_connect_time_rejection_without_recovery(adapters, monkeypatch, case):
    failure = {"forbidden": rejected(403), "cross_origin": rejected(correlated=False)}.get(case)
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect", failure))
    recorder = Recorder()
    provider = PlainProvider() if case == "plain_provider" else Provider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider, reauthenticate=case in ("forbidden", "cross_origin"))
        with pytest.raises(MCPAuthenticationError):
            await connection.discover()
        assert getattr(provider, "rejections", 0) == 0
        assert provider.gets == 1
        assert len(adapters.instances) == 1
        assert adapters.instances[0].closed
        assert _opening_events(recorder) == (1, ["MCPConnectionFailed"])


async def test_second_connect_time_401_is_not_retried_again(adapters, monkeypatch):
    async def always_rejected(self):
        raise rejected()

    monkeypatch.setattr(adapters, "connect", always_rejected)
    recorder = Recorder()
    provider = Provider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        with pytest.raises(MCPAuthenticationError):
            await connection.discover()
        assert provider.rejections == 1
        assert provider.gets == 2
        assert len(adapters.instances) == 2
        assert all(adapter.closed for adapter in adapters.instances)
        assert _opening_events(recorder) == (1, ["MCPConnectionFailed"])
        assert runtime._registrations[connection.identity].circuit.consecutive_failures == 1


@pytest.mark.parametrize("failure", ["raises", "times_out"])
async def test_connect_time_callback_failure_is_safe(adapters, monkeypatch, failure):
    from dendrux.mcp import MCPCredentialError

    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    recorder = Recorder()
    provider = Provider()
    seen_entries = []
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider, connect_timeout=0.05)

        async def on_auth_rejected():
            seen_entries.append(runtime._entries[connection.identity])
            if failure == "raises":
                raise RuntimeError("secret-refresh-token")
            await asyncio.Event().wait()

        provider.on_auth_rejected = on_auth_rejected
        with pytest.raises(MCPCredentialError) as caught:
            await connection.discover()
        assert "secret-refresh-token" not in str(caught.value)
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        assert len(seen_entries) == 1
        assert seen_entries[0].refresh_failed
        assert provider.gets == 1
        assert len(adapters.instances) == 1
        assert adapters.instances[0].closed
        assert _opening_events(recorder) == (1, ["MCPConnectionFailed"])
        failed = [e for e in recorder.events if type(e).__name__ == "MCPConnectionFailed"]
        assert failed[0].cause == "MCPCredentialError"


async def test_real_sdk_connect_time_401_recovers_with_refreshing_provider():
    app, state = live_app()
    state["token"] = "two"
    provider = Provider("one")
    recorder = Recorder()
    async with _serve(app) as url, MCPRuntime(observer=recorder) as runtime:
        connection = runtime.bind(
            connection_key="demo",
            credentials=provider,
            source=MCPSource.http("demo", url, reauthenticate_on_401=True),
        )
        discovery = await connection.discover()
        assert [tool.name for tool in discovery.tools] == ["echo"]
        assert provider.rejections == 1
        assert provider.gets == 2
        assert state["rejects"] >= 1
        assert _opening_events(recorder) == (1, ["MCPConnectionOpened"])


async def test_rejected_adapter_close_failure_does_not_stop_recovery(adapters, monkeypatch, caplog):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))

    async def close(self):
        self.closed = True
        if self.source.headers["Authorization"] == "Bearer one":
            raise RuntimeError("close failed for Bearer one")

    monkeypatch.setattr(adapters, "close", close)
    recorder = Recorder()
    provider = Provider()
    caplog.set_level("DEBUG", logger="dendrux")
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        await connection.discover()
        assert provider.rejections == 1
        assert len(adapters.instances) == 2
        assert runtime._entries[connection.identity].adapter is adapters.instances[1]
        assert _opening_events(recorder) == (1, ["MCPConnectionOpened"])
    assert any(
        getattr(record, "code", None) == "mcp_cleanup_failed" and record.count == 1
        for record in caplog.records
    )
    # The provider-resolved token is a redaction needle in cleanup diagnostics.
    assert "Bearer one" not in caplog.text


class BlockingProvider(Provider):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def on_auth_rejected(self):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        await super().on_auth_rejected()


async def test_runtime_close_during_connect_recovery_aborts_cleanly(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    recorder = Recorder()
    provider = BlockingProvider()
    runtime = MCPRuntime(observer=recorder, shutdown_timeout=1.0)
    connection = _bind(runtime, provider)
    discover = asyncio.create_task(connection.discover())
    await asyncio.wait_for(provider.entered.wait(), timeout=2)
    await runtime.close()
    # Shutdown cancels the shared connect task; its waiter sees that cancellation.
    with pytest.raises(asyncio.CancelledError):
        await discover
    assert provider.cancelled
    assert provider.gets == 1
    assert len(adapters.instances) == 1
    assert adapters.instances[0].closed
    assert _opening_events(recorder) == (1, ["MCPConnectionAborted"])


async def test_cancelled_waiter_does_not_cancel_shared_connect_recovery(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    recorder = Recorder()
    provider = BlockingProvider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        first = asyncio.create_task(connection.discover())
        second = asyncio.create_task(connection.discover())
        await asyncio.wait_for(provider.entered.wait(), timeout=2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not provider.cancelled
        provider.release.set()
        discovery = await asyncio.wait_for(second, timeout=2)
        assert [tool.name for tool in discovery.tools] == ["echo"]
        assert provider.rejections == 1
        assert provider.gets == 2
        assert _opening_events(recorder) == (1, ["MCPConnectionOpened"])


async def test_eviction_during_connect_recovery_never_publishes(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    recorder = Recorder()
    provider = BlockingProvider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        discover = asyncio.create_task(connection.discover())
        await asyncio.wait_for(provider.entered.wait(), timeout=2)
        evict = asyncio.create_task(runtime.evict(connection_key="demo", timeout=1.0))
        await asyncio.sleep(0)
        provider.release.set()
        await asyncio.wait_for(evict, timeout=2)
        # The retried transport is established for a fenced entry, so it is
        # closed unpublished, exactly like any connect that loses its slot.
        with pytest.raises(MCPConnectionError, match="released before establishment"):
            await discover
        assert connection.identity not in runtime._entries
        assert provider.rejections == 1
        assert all(adapter.closed for adapter in adapters.instances)
        assert _opening_events(recorder) == (1, ["MCPConnectionAborted"])


async def test_hanging_rejected_adapter_close_fails_the_opening(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    release = asyncio.Event()

    async def close(self):
        self.closed = True
        if self.source.headers["Authorization"] == "Bearer one":
            await release.wait()

    monkeypatch.setattr(adapters, "close", close)
    recorder = Recorder()
    provider = Provider()
    runtime = MCPRuntime(observer=recorder, close_timeout=0.05)
    try:
        connection = _bind(runtime, provider)
        waiters = [asyncio.create_task(connection.discover()) for _ in range(3)]
        results = await asyncio.wait_for(
            asyncio.gather(*waiters, return_exceptions=True), timeout=2
        )
        for result in results:
            assert isinstance(result, MCPConnectionError)
            assert not isinstance(result, MCPAuthenticationError)
            assert "Bearer" not in str(result)
        # Recovery stops before notifying the provider or reconnecting.
        assert provider.rejections == 0
        assert provider.gets == 1
        assert len(adapters.instances) == 1
        assert _opening_events(recorder) == (1, ["MCPConnectionFailed"])
        failed = [e for e in recorder.events if type(e).__name__ == "MCPConnectionFailed"]
        assert failed[0].cause == "MCPConnectionError"
        assert runtime._registrations[connection.identity].circuit.consecutive_failures == 1
        assert connection.identity not in runtime._entries
    finally:
        release.set()
        await runtime.close()


async def test_cancel_while_closing_rejected_adapter_hands_close_to_background(
    adapters, monkeypatch
):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    closing = asyncio.Event()
    release = asyncio.Event()

    async def close(self):
        self.closed = True
        if self.source.headers["Authorization"] == "Bearer one":
            closing.set()
            await release.wait()

    monkeypatch.setattr(adapters, "close", close)
    recorder = Recorder()
    provider = Provider()
    runtime = MCPRuntime(observer=recorder, close_timeout=30.0)
    try:
        connection = _bind(runtime, provider)
        discover = asyncio.create_task(connection.discover())
        await asyncio.wait_for(closing.wait(), timeout=2)
        open_task = runtime._entries[connection.identity].task
        open_task.cancel()
        await asyncio.gather(open_task, return_exceptions=True)
        assert open_task.cancelled()
        # The still-running close is tracked, not leaked.
        assert len(runtime._abandoned_tasks) == 1
        assert provider.rejections == 0
        assert _opening_events(recorder) == (1, ["MCPConnectionAborted"])
        await asyncio.gather(discover, return_exceptions=True)
    finally:
        release.set()
        await asyncio.sleep(0)
        await runtime.close()
    assert not runtime._abandoned_tasks


async def test_hanging_unpublished_adapter_close_is_bounded(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    closing = asyncio.Event()
    release = asyncio.Event()

    async def close(self):
        self.closed = True
        if self.source.headers["Authorization"] == "Bearer two":
            closing.set()
            await release.wait()

    monkeypatch.setattr(adapters, "close", close)
    recorder = Recorder()
    provider = BlockingProvider()
    runtime = MCPRuntime(observer=recorder, close_timeout=0.05, shutdown_timeout=0.5)
    try:
        connection = _bind(runtime, provider)
        discover = asyncio.create_task(connection.discover())
        await asyncio.wait_for(provider.entered.wait(), timeout=2)
        open_task = runtime._entries[connection.identity].task
        evict = asyncio.create_task(runtime.evict(connection_key="demo", timeout=1.0))
        await asyncio.sleep(0)
        provider.release.set()
        await asyncio.wait_for(closing.wait(), timeout=2)
        # The open task finishes within close_timeout although close never returns.
        results = await asyncio.wait_for(
            asyncio.gather(open_task, return_exceptions=True), timeout=1
        )
        assert isinstance(results[0], MCPConnectionError)
        assert "released before establishment" in str(results[0])
        assert _opening_events(recorder) == (1, ["MCPConnectionAborted"])
        assert len(runtime._abandoned_tasks) == 1
        await asyncio.wait_for(asyncio.gather(evict, discover, return_exceptions=True), timeout=2)
    finally:
        release.set()
        await asyncio.sleep(0)
        await runtime.close()
    assert not runtime._abandoned_tasks


async def test_cancel_during_unpublished_adapter_close_emits_one_terminal(adapters, monkeypatch):
    monkeypatch.setattr(adapters, "connect", _reject_stale("connect"))
    closing = asyncio.Event()

    async def close(self):
        self.closed = True
        if self.source.headers["Authorization"] == "Bearer two":
            closing.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(adapters, "close", close)
    recorder = Recorder()
    provider = BlockingProvider()
    runtime = MCPRuntime(observer=recorder, close_timeout=0.05, shutdown_timeout=0.5)
    try:
        connection = _bind(runtime, provider)
        discover = asyncio.create_task(connection.discover())
        await asyncio.wait_for(provider.entered.wait(), timeout=2)
        open_task = runtime._entries[connection.identity].task
        evict = asyncio.create_task(runtime.evict(connection_key="demo", timeout=1.0))
        await asyncio.sleep(0)
        provider.release.set()
        # The retry succeeds for the fenced entry and is closed unpublished.
        await asyncio.wait_for(closing.wait(), timeout=2)
        open_task.cancel()
        await asyncio.gather(open_task, return_exceptions=True)
        assert _opening_events(recorder) == (1, ["MCPConnectionAborted"])
        await asyncio.wait_for(asyncio.gather(evict, discover, return_exceptions=True), timeout=2)
        assert _opening_events(recorder) == (1, ["MCPConnectionAborted"])
    finally:
        await runtime.close()


class CountingProvider:
    """Each rejection moves to the next token; every token is rejected."""

    def __init__(self):
        self.generation = 1
        self.gets = 0
        self.rejections = 0

    async def get_auth(self):
        self.gets += 1
        return {"Authorization": f"Bearer gen-{self.generation}"}

    async def on_auth_rejected(self):
        self.rejections += 1
        self.generation += 1


async def test_call_rejection_then_reconnect_rejection_notifies_provider_once(
    adapters, monkeypatch
):
    """One rejected operation spends one on_auth_rejected(), call path included."""

    async def connect(self):
        if self.source.headers["Authorization"] != "Bearer gen-1":
            raise rejected()

    monkeypatch.setattr(adapters, "connect", connect)
    monkeypatch.setattr(adapters, "call_tool", AsyncMock(side_effect=rejected()))
    recorder = Recorder()
    provider = CountingProvider()
    async with MCPRuntime(observer=recorder) as runtime:
        connection = _bind(runtime, provider)
        view = _ViewToolSource(connection.tools())
        await view._discover()
        executor = view._create_executor("echo")
        with pytest.raises(MCPAuthenticationError):
            await executor()
        assert provider.rejections == 1
        assert provider.gets == 2
        assert len(adapters.instances) == 2
        assert recorder.names().count("MCPConnectionOpening") == 2
        assert recorder.names().count("MCPConnectionFailed") == 1
        await view.close()


async def test_explicit_rotation_reconnect_keeps_connect_time_recovery(adapters, monkeypatch):
    """Only a rejection rotation spends the budget; a manual rotation does not."""
    provider = CountingProvider()
    async with MCPRuntime() as runtime:
        connection = _bind(runtime, Provider())
        view = _ViewToolSource(connection.tools())
        await view._discover()
        await runtime.rotate_credentials(connection_key="demo", credentials=provider)

        async def connect(self):
            if self.source.headers["Authorization"] == "Bearer gen-1":
                raise rejected()

        monkeypatch.setattr(adapters, "connect", connect)
        await view._create_executor("echo")()
        assert provider.rejections == 1
        assert provider.gets == 2
        await view.close()
