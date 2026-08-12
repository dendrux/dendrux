"""Live managed-runtime contracts: connection reuse and lease lifecycle.

These tests pin the Release 2 behavior of MCPRuntime before implementation:
one physical connection per (tenant_key, connection_key), single-flight
connect + discovery, per-view namespaced catalogs, per-Agent leases released
on Agent.close(), and drain-based runtime shutdown.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from dendrux.agent import Agent
from dendrux.mcp._client import MCPConnectionInfo
from dendrux.mcp._errors import (
    MCPAuthenticationError,
    MCPBindingConflictError,
    MCPCallCapacityError,
    MCPCapacityError,
    MCPConnectionCapacityError,
    MCPConnectionEvictingError,
    MCPConnectionLostError,
    MCPCredentialError,
    MCPOutcomeUnknownError,
    MCPRuntimeClosedError,
    MCPStaleConnectionError,
    MCPToolCallError,
)
from dendrux.mcp._runtime import (
    MCPConnection,
    MCPRuntime,
    MCPRuntimeState,
    _resolve_source_credentials,
    _ViewToolSource,
)
from dendrux.mcp._source import MCPSource


def _tool(name: str, *, read_only: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
        annotations=ToolAnnotations(read_only_hint=read_only, destructive_hint=not read_only),
    )


class _InstrumentedAdapter:
    """Fake MCPClientAdapter counting physical connections and closes."""

    instances: ClassVar[list[_InstrumentedAdapter]] = []
    tools: ClassVar[list[Tool]] = []
    connect_gate: ClassVar[asyncio.Event | None] = None
    close_gate: ClassVar[asyncio.Event | None] = None
    call_gate: ClassVar[asyncio.Event | None] = None
    call_cancel_seen: ClassVar[asyncio.Event | None] = None
    call_cancel_pause: ClassVar[asyncio.Event | None] = None
    connect_error: ClassVar[Exception | None] = None
    fail_urls: ClassVar[set[str]] = set()
    close_error: ClassVar[Exception | None] = None
    call_error: ClassVar[Exception | None] = None
    suppress_cancel: ClassVar[bool] = False
    suppress_call_cancel: ClassVar[bool] = False
    return_after_close: ClassVar[bool] = False
    # Call admission instrumentation: the order calls entered the transport,
    # and how many were ever inside it at once.
    call_log: ClassVar[list[str]] = []
    in_call: ClassVar[int] = 0
    peak_in_call: ClassVar[int] = 0

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self.connect_calls = 0
        self.list_tools_calls = 0
        self.close_calls = 0
        self.call_tool_calls = 0
        self.closed = False
        self.info = MCPConnectionInfo(
            protocol_version="2026-07-28",
            server_name="fake-server",
            server_version="1.0.0",
            instructions=None,
        )
        type(self).instances.append(self)

    async def connect(self) -> None:
        self.connect_calls += 1
        gate = type(self).connect_gate
        if gate is not None:
            if type(self).suppress_cancel:
                # Misbehaving transport: swallows cancellation and keeps going.
                while True:
                    try:
                        await gate.wait()
                        break
                    except asyncio.CancelledError:
                        continue
            else:
                await gate.wait()
        if self.source.url in type(self).fail_urls:
            raise ConnectionError(f"refused: {self.source.url}")
        error = type(self).connect_error
        if error is not None:
            raise error

    async def list_tools(self) -> list[Tool]:
        self.list_tools_calls += 1
        return list(type(self).tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        cls = type(self)
        self.call_tool_calls += 1
        cls.call_log.append(str(arguments.get("value", name)))
        cls.in_call += 1
        cls.peak_in_call = max(cls.peak_in_call, cls.in_call)
        try:
            gate = cls.call_gate
            if gate is not None:
                if cls.call_cancel_seen is not None:
                    try:
                        await gate.wait()
                    except asyncio.CancelledError:
                        cls.call_cancel_seen.set()
                        pause = cls.call_cancel_pause
                        if pause is not None:
                            await pause.wait()
                        raise
                elif cls.suppress_call_cancel:
                    while True:
                        try:
                            await gate.wait()
                            break
                        except asyncio.CancelledError:
                            continue
                else:
                    await gate.wait()
            if self.closed and not cls.return_after_close:
                # A real transport fails an in-flight call once it is closed.
                raise ConnectionError("transport closed")
            error = cls.call_error
            if error is not None:
                raise error
            return CallToolResult(content=[TextContent(text=f"{name}:ok")])
        finally:
            cls.in_call -= 1

    async def close(self) -> None:
        self.close_calls += 1
        gate = type(self).close_gate
        if gate is not None:
            await gate.wait()
        error = type(self).close_error
        if error is not None:
            raise error
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_adapter() -> Any:
    _InstrumentedAdapter.instances = []
    _InstrumentedAdapter.tools = [_tool("read", read_only=True), _tool("write")]
    _InstrumentedAdapter.connect_gate = None
    _InstrumentedAdapter.close_gate = None
    _InstrumentedAdapter.call_gate = None
    _InstrumentedAdapter.call_cancel_seen = None
    _InstrumentedAdapter.call_cancel_pause = None
    _InstrumentedAdapter.connect_error = None
    _InstrumentedAdapter.fail_urls = set()
    _InstrumentedAdapter.close_error = None
    _InstrumentedAdapter.call_error = None
    _InstrumentedAdapter.suppress_cancel = False
    _InstrumentedAdapter.suppress_call_cancel = False
    _InstrumentedAdapter.return_after_close = False
    _InstrumentedAdapter.call_log = []
    _InstrumentedAdapter.in_call = 0
    _InstrumentedAdapter.peak_in_call = 0
    with patch("dendrux.mcp._runtime.MCPClientAdapter", _InstrumentedAdapter):
        yield


class _Credentials:
    """Instrumented credential provider returning a scripted value sequence."""

    def __init__(self, *values: Any) -> None:
        self._values = list(values) or [{"Authorization": "Bearer must-not-leak"}]
        self.calls = 0
        self.gate: asyncio.Event | None = None
        self.error: BaseException | None = None
        self.fail_once = False
        self.cancelled = False

    async def get_auth(self) -> Any:
        self.calls += 1
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.error is not None:
            error = self.error
            if self.fail_once:
                self.error = None
            raise error
        return self._values[min(self.calls - 1, len(self._values) - 1)]


def _chain_text(error: BaseException) -> str:
    """Every message a monitoring SDK would collect from the chain."""
    parts: list[str] = []
    node: BaseException | None = error
    while node is not None:
        parts.append(str(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


def _bind(
    runtime: MCPRuntime,
    *,
    tenant_key: str | None = None,
    connection_key: str = "github-1",
    credentials: Any = None,
    credential_identity: str | None = None,
) -> MCPConnection:
    return runtime.bind(
        connection_key=connection_key,
        tenant_key=tenant_key,
        source=MCPSource.http("github", "https://mcp.example.com"),
        credentials=credentials,
        credential_identity=credential_identity,
    )


def _total_connects() -> int:
    return sum(adapter.connect_calls for adapter in _InstrumentedAdapter.instances)


def _total_calls() -> int:
    return sum(adapter.call_tool_calls for adapter in _InstrumentedAdapter.instances)


class TestAgentConsumesToolViews:
    def test_agent_accepts_a_tool_view_without_any_io(self) -> None:
        runtime = MCPRuntime()
        view = _bind(runtime).tools()

        agent = Agent(prompt="test", tool_sources=[view])

        assert agent is not None
        assert _InstrumentedAdapter.instances == []

    def test_two_views_with_the_same_namespace_are_rejected(self) -> None:
        connection = _bind(MCPRuntime())

        with pytest.raises(ValueError, match="duplicate"):
            Agent(
                prompt="test",
                tool_sources=[
                    connection.tools(allowed_tools=["read"]),
                    connection.tools(allowed_tools=["write"]),
                ],
            )


class TestConnectionReuse:
    @pytest.mark.asyncio
    async def test_first_discovery_connects_once_and_namespaces_the_catalog(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _bind(runtime).tools(namespace="gh")
            agent = Agent(prompt="test", tool_sources=[view])

            lookups = await agent.get_tool_lookups()

            assert len(_InstrumentedAdapter.instances) == 1
            assert _total_connects() == 1
            assert sorted(lookups.fn) == ["gh__read", "gh__write"]
            assert await lookups.fn["gh__read"](value=1) == "read:ok"

    @pytest.mark.asyncio
    async def test_view_policy_filters_tools_and_forces_serial_execution(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            view = connection.tools(allowed_tools=["read"], force_serial_tools=["read"])
            agent = Agent(prompt="test", tool_sources=[view])

            await agent.get_tool_lookups()
            tool_defs = {td.name: td for td in agent.get_all_tool_defs()}

            assert sorted(tool_defs) == ["github__read"]
            assert tool_defs["github__read"].parallel is False

    @pytest.mark.asyncio
    async def test_unknown_allowed_tool_names_fail_discovery(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _bind(runtime).tools(allowed_tools=["read", "no_such_tool"])
            agent = Agent(prompt="test", tool_sources=[view])

            with pytest.raises(ValueError, match="no_such_tool"):
                await agent.get_tool_lookups()

    @pytest.mark.asyncio
    async def test_two_agents_share_one_physical_connection_and_catalog(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(
                prompt="second",
                tool_sources=[connection.tools(namespace="gh", allowed_tools=["read"])],
            )

            first_lookups = await first.get_tool_lookups()
            second_lookups = await second.get_tool_lookups()

            assert len(_InstrumentedAdapter.instances) == 1
            assert _total_connects() == 1
            assert _InstrumentedAdapter.instances[0].list_tools_calls == 1
            assert sorted(first_lookups.fn) == ["github__read", "github__write"]
            assert sorted(second_lookups.fn) == ["gh__read"]

    @pytest.mark.asyncio
    async def test_concurrent_discovery_is_single_flight(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])

            first_task = asyncio.create_task(first.get_tool_lookups())
            second_task = asyncio.create_task(second.get_tool_lookups())
            await asyncio.sleep(0.05)

            assert len(_InstrumentedAdapter.instances) == 1
            assert not first_task.done()
            assert not second_task.done()

            _InstrumentedAdapter.connect_gate.set()
            await asyncio.gather(first_task, second_task)

            assert _total_connects() == 1

    @pytest.mark.asyncio
    async def test_one_agent_can_hold_two_views_over_one_connection(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            agent = Agent(
                prompt="test",
                tool_sources=[
                    connection.tools(namespace="gh-read", allowed_tools=["read"]),
                    connection.tools(namespace="gh-admin"),
                ],
            )

            lookups = await agent.get_tool_lookups()

            assert len(_InstrumentedAdapter.instances) == 1
            assert sorted(lookups.fn) == [
                "gh-admin__read",
                "gh-admin__write",
                "gh-read__read",
            ]

    @pytest.mark.asyncio
    async def test_tenants_never_share_a_physical_connection(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view_a = _bind(runtime, tenant_key="tenant-a").tools()
            view_b = _bind(runtime, tenant_key="tenant-b").tools()

            await Agent(prompt="a", tool_sources=[view_a]).get_tool_lookups()
            await Agent(prompt="b", tool_sources=[view_b]).get_tool_lookups()

            assert len(_InstrumentedAdapter.instances) == 2

    @pytest.mark.asyncio
    async def test_connect_failure_reaches_every_waiter_and_is_not_sticky(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.connect_error = ConnectionError("boom")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])

            first_task = asyncio.create_task(first.get_tool_lookups())
            second_task = asyncio.create_task(second.get_tool_lookups())
            await asyncio.sleep(0.05)
            _InstrumentedAdapter.connect_gate.set()
            results = await asyncio.gather(first_task, second_task, return_exceptions=True)

            assert all(isinstance(result, ConnectionError) for result in results)

            _InstrumentedAdapter.connect_gate = None
            _InstrumentedAdapter.connect_error = None
            retry = Agent(prompt="retry", tool_sources=[connection.tools()])
            lookups = await retry.get_tool_lookups()

            assert sorted(lookups.fn) == ["github__read", "github__write"]
            assert len(_InstrumentedAdapter.instances) == 2


class TestCredentialResolution:
    @pytest.mark.asyncio
    async def test_credentials_resolve_lazily_on_first_connection(self) -> None:
        credentials = _Credentials({"Authorization": "Bearer live-token"})
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)
            assert credentials.calls == 0  # binding alone must not resolve

            await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert credentials.calls == 1
            # Resolved headers join a connect-scoped copy of the source; the
            # configured source is never mutated.
            assert dict(_InstrumentedAdapter.instances[0].source.headers) == {
                "Authorization": "Bearer live-token"
            }
            assert dict(connection.source.headers) == {}

    @pytest.mark.asyncio
    async def test_concurrent_discovery_resolves_credentials_once(self) -> None:
        credentials = _Credentials()
        credentials.gate = asyncio.Event()
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])

            first_task = asyncio.create_task(first.get_tool_lookups())
            second_task = asyncio.create_task(second.get_tool_lookups())
            await asyncio.sleep(0.05)

            # Both Agents wait on one in-flight resolution; neither starts its own.
            assert credentials.calls == 1
            credentials.gate.set()
            await asyncio.gather(first_task, second_task)

            assert credentials.calls == 1
            assert _total_connects() == 1

    @pytest.mark.asyncio
    async def test_live_connection_reuse_skips_credential_lookup(self) -> None:
        credentials = _Credentials()
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)
            await Agent(prompt="first", tool_sources=[connection.tools()]).get_tool_lookups()
            await Agent(
                prompt="second",
                tool_sources=[connection.tools(namespace="gh")],
            ).get_tool_lookups()

            assert credentials.calls == 1

    @pytest.mark.asyncio
    async def test_retirement_reconnect_resolves_fresh_credentials(self) -> None:
        credentials = _Credentials(
            {"Authorization": "Bearer token-1"},
            {"Authorization": "Bearer token-2"},
        )
        async with MCPRuntime(idle_timeout=0, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)
            agent = Agent(prompt="first", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()
            await agent.close()  # idle_timeout=0 retires immediately
            await asyncio.sleep(0.05)
            assert _InstrumentedAdapter.instances[0].closed is True

            await Agent(prompt="second", tool_sources=[connection.tools()]).get_tool_lookups()

            assert credentials.calls == 2
            assert dict(_InstrumentedAdapter.instances[1].source.headers) == {
                "Authorization": "Bearer token-2"
            }

    @pytest.mark.asyncio
    async def test_eviction_then_rebind_resolves_fresh_credentials(self) -> None:
        credentials = _Credentials(
            {"Authorization": "Bearer token-1"},
            {"Authorization": "Bearer token-2"},
        )
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            first = _bind(runtime, credentials=credentials)
            agent = Agent(prompt="first", tool_sources=[first.tools()])
            await agent.get_tool_lookups()
            await agent.close()
            await runtime.evict(connection_key="github-1")

            rebound = _bind(runtime, credentials=credentials)
            await Agent(prompt="second", tool_sources=[rebound.tools()]).get_tool_lookups()

            assert credentials.calls == 2
            assert dict(_InstrumentedAdapter.instances[1].source.headers) == {
                "Authorization": "Bearer token-2"
            }

    @pytest.mark.asyncio
    async def test_http_mapping_credentials_merge_into_headers(self) -> None:
        credentials = _Credentials({"Authorization": "Bearer resolved-token"})
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github",
                    "https://mcp.example.com",
                    headers={"X-Static": "configured"},
                ),
                credentials=credentials,
            )
            await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            adapter = _InstrumentedAdapter.instances[0]
            assert dict(adapter.source.headers) == {
                "X-Static": "configured",
                "Authorization": "Bearer resolved-token",
            }
            assert dict(connection.source.headers) == {"X-Static": "configured"}

    @pytest.mark.asyncio
    async def test_stdio_mapping_credentials_merge_into_env(self) -> None:
        credentials = _Credentials({"MCP_TOKEN": "resolved-secret"})
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = runtime.bind(
                connection_key="local-1",
                source=MCPSource.stdio("local", ["mcp-server"], env={"BASE": "1"}),
                credentials=credentials,
            )
            await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            adapter = _InstrumentedAdapter.instances[0]
            assert dict(adapter.source.env) == {"BASE": "1", "MCP_TOKEN": "resolved-secret"}

    @pytest.mark.asyncio
    async def test_stdio_non_mapping_credentials_fail_typed(self) -> None:
        credentials = _Credentials("bearer-style-token")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = runtime.bind(
                connection_key="local-1",
                source=MCPSource.stdio("local", ["mcp-server"]),
                credentials=credentials,
            )
            with pytest.raises(MCPCredentialError, match="environment") as excinfo:
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert "bearer-style-token" not in _chain_text(excinfo.value)
            assert _InstrumentedAdapter.instances == []  # never reached the transport

    @pytest.mark.asyncio
    async def test_invalid_resolved_header_values_fail_without_leaking_them(self) -> None:
        credentials = _Credentials({"Authorization": "Bearer bad\r\nX-Injected: 1"})
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)

            with pytest.raises(MCPCredentialError, match="ValueError") as excinfo:
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert "X-Injected" not in _chain_text(excinfo.value)
            assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_provider_returning_none_fails_typed(self) -> None:
        credentials = _Credentials(None)
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)

            with pytest.raises(MCPCredentialError, match="None"):
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_provider_failure_is_typed_reaches_waiters_and_is_not_sticky(self) -> None:
        credentials = _Credentials({"Authorization": "Bearer recovered-token"})
        credentials.error = RuntimeError("refresh failed for token must-not-leak")
        credentials.fail_once = True
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])

            results = await asyncio.gather(
                first.get_tool_lookups(),
                second.get_tool_lookups(),
                return_exceptions=True,
            )

            for result in results:
                assert isinstance(result, MCPCredentialError)
                assert str(result) == (
                    "Failed to resolve credentials for MCP source 'github' (RuntimeError)."
                )
                # Provider exception text is application-authored and may
                # embed the very secret being refreshed; it must be detached
                # from the chain, not merely suppressed in tracebacks.
                assert result.__context__ is None
                assert "must-not-leak" not in _chain_text(result)
            assert _InstrumentedAdapter.instances == []

            retry = Agent(prompt="retry", tool_sources=[connection.tools()])
            lookups = await retry.get_tool_lookups()
            assert sorted(lookups.fn) == ["github__read", "github__write"]
            assert credentials.calls == 2
            assert dict(_InstrumentedAdapter.instances[-1].source.headers) == {
                "Authorization": "Bearer recovered-token"
            }

    @pytest.mark.asyncio
    async def test_provider_hang_is_bounded_by_connect_timeout(self) -> None:
        credentials = _Credentials()
        credentials.gate = asyncio.Event()  # never set
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github",
                    "https://mcp.example.com",
                    connect_timeout=0.05,
                ),
                credentials=credentials,
            )
            with pytest.raises(MCPCredentialError, match="TimeoutError"):
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_a_pending_resolution_cleanly(self) -> None:
        credentials = _Credentials()
        credentials.gate = asyncio.Event()  # never released
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime, credentials=credentials).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.02)
        assert credentials.calls == 1

        await asyncio.wait_for(runtime.close(), timeout=1.0)
        await asyncio.wait([discovery], timeout=1.0)

        assert discovery.done()
        assert discovery.cancelled() or discovery.exception() is not None
        assert credentials.cancelled is True
        assert _InstrumentedAdapter.instances == []  # never reached the transport
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_tenants_resolve_credentials_independently(self) -> None:
        credentials_a = _Credentials({"Authorization": "Bearer token-a"})
        credentials_b = _Credentials({"Authorization": "Bearer token-b"})
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view_a = _bind(runtime, tenant_key="tenant-a", credentials=credentials_a).tools()
            view_b = _bind(runtime, tenant_key="tenant-b", credentials=credentials_b).tools()
            await Agent(prompt="a", tool_sources=[view_a]).get_tool_lookups()
            await Agent(prompt="b", tool_sources=[view_b]).get_tool_lookups()

            assert credentials_a.calls == 1
            assert credentials_b.calls == 1
            tokens = {
                dict(adapter.source.headers).get("Authorization")
                for adapter in _InstrumentedAdapter.instances
            }
            assert tokens == {"Bearer token-a", "Bearer token-b"}

    @pytest.mark.asyncio
    async def test_http_non_mapping_credentials_fail_typed(self) -> None:
        """Bare tokens and auth objects are rejected: only mapping values give
        the runtime exact strings it can redact from every error surface."""
        credentials = _Credentials("raw-token-string")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)

            with pytest.raises(MCPCredentialError, match="request headers") as excinfo:
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert "raw-token-string" not in _chain_text(excinfo.value)
            assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty", [{}, ""])
    async def test_empty_resolved_credentials_fail_typed(self, empty: Any) -> None:
        """{} passes an isinstance Mapping check but merges nothing; accepting
        it would connect silently unauthenticated, the outcome the None guard
        exists to prevent."""
        credentials = _Credentials(empty)
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)

            with pytest.raises(MCPCredentialError, match="empty"):
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_hostile_mapping_failure_is_wrapped_and_detached(self) -> None:
        """A lazy secret-backed mapping can raise while it is being merged;
        that happens after get_auth() returned, and its text needs the same
        detachment as a provider exception."""

        class _ExplodingMapping(Mapping):
            def __getitem__(self, key: str) -> str:
                raise RuntimeError("decrypt failed for MAP-SECRET")

            def __iter__(self) -> Any:
                return iter(["Authorization"])

            def __len__(self) -> int:
                return 1

        credentials = _Credentials(_ExplodingMapping())
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)

            with pytest.raises(MCPCredentialError) as excinfo:
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert str(excinfo.value) == (
                "Failed to resolve credentials for MCP source 'github' (RuntimeError)."
            )
            assert excinfo.value.__context__ is None
            assert "MAP-SECRET" not in _chain_text(excinfo.value)
            assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_hostile_mapping_value_error_is_wrapped_and_detached(self) -> None:
        """Application-owned mappings do not become trusted merely because
        they raise the same exception type as MCPSource validation."""

        class _ExplodingMapping(Mapping):
            def __getitem__(self, key: str) -> str:
                raise ValueError("decrypt failed for VALUE-SECRET")

            def __iter__(self) -> Any:
                return iter(["Authorization"])

            def __len__(self) -> int:
                return 1

        credentials = _Credentials(_ExplodingMapping())
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)

            with pytest.raises(MCPCredentialError) as excinfo:
                await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            assert str(excinfo.value) == (
                "Failed to resolve credentials for MCP source 'github' (ValueError)."
            )
            assert excinfo.value.__context__ is None
            assert "VALUE-SECRET" not in _chain_text(excinfo.value)
            assert _InstrumentedAdapter.instances == []

    def test_managed_runtime_rejects_opaque_static_auth(self) -> None:
        """A managed connection cannot promise output redaction when an auth
        object's current secret values are unknowable."""

        class _OpaqueAuth:
            token = "OPAQUE-SECRET"

        runtime = MCPRuntime()
        source = MCPSource.http(
            "github",
            "https://mcp.example.com",
            auth=_OpaqueAuth(),
        )

        with pytest.raises(ValueError, match="opaque") as excinfo:
            runtime.bind(connection_key="github-1", source=source)

        assert "OPAQUE-SECRET" not in str(excinfo.value)
        assert runtime._registrations == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("signal", [SystemExit(3), KeyboardInterrupt()])
    async def test_process_exit_signals_pass_through_unwrapped(
        self,
        signal: BaseException,
    ) -> None:
        """SystemExit and KeyboardInterrupt are process-level control flow,
        not credential failures; converting them would swallow a shutdown.
        Exercised at the boundary directly because asyncio deliberately
        re-raises them from a task into the event loop itself."""
        credentials = _Credentials()
        credentials.error = signal
        source = MCPSource.http("github", "https://mcp.example.com")

        with pytest.raises(type(signal)):
            await _resolve_source_credentials(source, credentials)


class TestLeaseLifecycle:
    @pytest.mark.asyncio
    async def test_agent_close_releases_its_lease_but_keeps_the_connection(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            await first.get_tool_lookups()

            await first.close()

            adapter = _InstrumentedAdapter.instances[0]
            assert adapter.closed is False

            second = Agent(prompt="second", tool_sources=[connection.tools()])
            await second.get_tool_lookups()

            assert len(_InstrumentedAdapter.instances) == 1
            assert adapter.list_tools_calls == 1

    @pytest.mark.asyncio
    async def test_retained_executor_is_invalid_after_agent_close(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        lookups = await agent.get_tool_lookups()
        retained_executor = lookups.fn["github__write"]

        await agent.close()

        with pytest.raises(MCPToolCallError, match="no longer active"):
            await retained_executor(value="must-not-run")
        assert _InstrumentedAdapter.instances[0].call_tool_calls == 0
        await runtime.close()

    @pytest.mark.asyncio
    async def test_in_flight_call_keeps_runtime_draining_after_agent_close(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime(shutdown_timeout=1.0)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        lookups = await agent.get_tool_lookups()

        call = asyncio.create_task(lookups.fn["github__write"](value="running"))
        for _ in range(100):
            if _InstrumentedAdapter.instances[0].call_tool_calls:
                break
            await asyncio.sleep(0.01)
        await agent.close()
        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)

        assert close_task.done() is False

        _InstrumentedAdapter.call_gate.set()
        assert await call == "write:ok"
        await asyncio.wait_for(close_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_agent_close_is_idempotent_per_lease(self) -> None:
        async with MCPRuntime() as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])
            await first.get_tool_lookups()
            await second.get_tool_lookups()

            await first.close()
            await first.close()

            close_task = asyncio.create_task(runtime.close())
            await asyncio.sleep(0.05)

            assert not close_task.done()

            await second.close()
            await asyncio.wait_for(close_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_runtime_close_with_no_leases_closes_connections(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        await agent.close()

        await runtime.close()

        assert _InstrumentedAdapter.instances[0].closed is True
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_runtime_close_drains_active_leases_before_closing(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()

        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)

        assert not close_task.done()
        assert runtime.state is MCPRuntimeState.DRAINING
        assert _InstrumentedAdapter.instances[0].closed is False
        with pytest.raises(RuntimeError):
            _bind(runtime, connection_key="late-binding")

        await agent.close()
        await asyncio.wait_for(close_task, timeout=1.0)

        assert _InstrumentedAdapter.instances[0].closed is True
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_shutdown_timeout_force_closes_undrained_connections(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()

        await asyncio.wait_for(runtime.close(), timeout=1.0)

        assert _InstrumentedAdapter.instances[0].closed is True
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_discovery_after_runtime_close_is_rejected(self) -> None:
        runtime = MCPRuntime()
        view = _bind(runtime).tools()
        await runtime.close()

        agent = Agent(prompt="test", tool_sources=[view])

        with pytest.raises(RuntimeError, match="closed"):
            await agent.get_tool_lookups()


class TestEventLoopAffinity:
    def test_runtime_is_bound_to_its_first_event_loop(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)

        asyncio.run(Agent(prompt="first", tool_sources=[connection.tools()]).get_tool_lookups())

        follow_up = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])
        with pytest.raises(RuntimeError, match="event loop"):
            asyncio.run(follow_up.get_tool_lookups())


class TestCancellationSafety:
    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_leak_a_lease(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        runtime = MCPRuntime()
        connection = _bind(runtime)
        first = Agent(prompt="first", tool_sources=[connection.tools()])
        second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])

        first_task = asyncio.create_task(first.get_tool_lookups())
        second_task = asyncio.create_task(second.get_tool_lookups())
        await asyncio.sleep(0.05)

        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task

        _InstrumentedAdapter.connect_gate.set()
        await first_task
        await first.close()

        # A leaked lease would make the default 30s drain time this out.
        await asyncio.wait_for(runtime.close(), timeout=1.0)
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_cancelling_one_discovery_does_not_poison_the_connection(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            second = Agent(prompt="second", tool_sources=[connection.tools(namespace="gh")])

            first_task = asyncio.create_task(first.get_tool_lookups())
            second_task = asyncio.create_task(second.get_tool_lookups())
            await asyncio.sleep(0.05)

            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task

            _InstrumentedAdapter.connect_gate.set()
            lookups = await asyncio.wait_for(second_task, timeout=1.0)

            assert sorted(lookups.fn) == ["gh__read", "gh__write"]
            assert _total_connects() == 1

    @pytest.mark.asyncio
    async def test_forced_shutdown_during_blocked_connect_leaves_no_live_adapter(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()  # never released
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        assert len(_InstrumentedAdapter.instances) == 1
        assert _InstrumentedAdapter.instances[0].closed is False

        await asyncio.wait_for(runtime.close(), timeout=1.0)
        await asyncio.wait([discovery], timeout=1.0)

        assert discovery.done()
        assert discovery.cancelled() or discovery.exception() is not None
        assert _InstrumentedAdapter.instances[0].closed is True
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_close_callers_finish_together(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()

        close_a = asyncio.create_task(runtime.close())
        close_b = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)

        assert not close_a.done()
        assert not close_b.done()

        await agent.close()
        await asyncio.wait_for(asyncio.gather(close_a, close_b), timeout=1.0)

        assert runtime.state is MCPRuntimeState.CLOSED
        assert _InstrumentedAdapter.instances[0].close_calls == 1


class TestReleaseOwnership:
    @pytest.mark.asyncio
    async def test_release_of_a_discarded_entry_does_not_affect_its_replacement(self) -> None:
        from dendrux.mcp._runtime import _ConnectionEntry

        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()

        # A stale waiter unwinding after its failed entry was discarded must
        # not decrement the replacement entry now live under the same identity.
        discarded = _ConnectionEntry(connection.source)
        discarded.leases = 1
        runtime._release(connection.identity, discarded)

        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)

        assert not close_task.done()  # the real lease is still counted

        await agent.close()
        await asyncio.wait_for(close_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_late_finishing_connect_closes_its_adapter_instead_of_publishing(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.suppress_cancel = True
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)

        # Leave no Agent waiter to retrieve the shared task's eventual
        # late-eviction error. The runtime must retain and consume it.
        discovery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discovery

        # Close must stay bounded even though the transport swallows
        # cancellation and its connect task cannot be interrupted.
        await asyncio.wait_for(runtime.close(), timeout=1.0)
        assert runtime.state is MCPRuntimeState.CLOSED
        assert len(runtime._abandoned_tasks) == 1

        # Let the misbehaving connect finish late: it must close its adapter
        # itself rather than publish it into the closed runtime.
        _InstrumentedAdapter.connect_gate.set()
        for _ in range(100):
            if not runtime._abandoned_tasks:
                break
            await asyncio.sleep(0.01)

        assert runtime._abandoned_tasks == set()
        assert _InstrumentedAdapter.instances[0].closed is True

    @pytest.mark.asyncio
    async def test_forced_connect_cancellation_uses_one_shared_grace_period(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.suppress_cancel = True
        runtime = MCPRuntime(shutdown_timeout=0.01)
        discoveries = []
        for index in range(3):
            connection = _bind(runtime, connection_key=f"github-{index}")
            agent = Agent(prompt="test", tool_sources=[connection.tools()])
            discoveries.append(asyncio.create_task(agent.get_tool_lookups()))
        await asyncio.sleep(0.05)

        loop = asyncio.get_running_loop()
        started = loop.time()
        with patch("dendrux.mcp._runtime._FORCED_CANCEL_GRACE", 0.2):
            await runtime.close()
        elapsed = loop.time() - started

        # Sequential grace would take >= 0.6s for three stuck tasks.
        assert elapsed < 0.45
        assert len(runtime._abandoned_tasks) == 3

        _InstrumentedAdapter.connect_gate.set()
        await asyncio.gather(*discoveries, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_established_adapters_close_concurrently(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agents = []
        for index in range(3):
            connection = _bind(runtime, connection_key=f"github-{index}")
            agent = Agent(prompt="test", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()
            agents.append(agent)
        for agent in agents:
            await agent.close()

        _InstrumentedAdapter.close_gate = asyncio.Event()
        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)
        close_calls = [adapter.close_calls for adapter in _InstrumentedAdapter.instances]

        _InstrumentedAdapter.close_gate.set()
        await asyncio.wait_for(close_task, timeout=1.0)

        assert close_calls == [1, 1, 1]


class TestCrossLoopSafety:
    def test_close_from_a_different_loop_is_rejected(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)

        asyncio.run(Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups())

        with pytest.raises(RuntimeError, match="event loop"):
            asyncio.run(runtime.close())

    def test_lease_release_from_a_different_loop_is_rejected(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        source = agent._tool_sources[0]

        with asyncio.Runner() as owner:
            owner.run(agent.get_tool_lookups())

            with pytest.raises(RuntimeError, match="event loop"):
                asyncio.run(source.close())

            # Rejection must not discard the view's only lease token.
            assert source._leased is True
            assert source._entry is not None

            owner.run(source.close())
            assert source._leased is False
            owner.run(runtime.close())


class TestBindingDeterminism:
    @pytest.mark.asyncio
    async def test_changed_auth_object_requires_eviction_while_live(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        first = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github",
                "https://mcp.example.com",
                auth=["user", "password-one"],
            ),
        )
        agent = Agent(prompt="test", tool_sources=[first.tools()])
        await agent.get_tool_lookups()

        with pytest.raises(MCPBindingConflictError) as excinfo:
            runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github",
                    "https://mcp.example.com",
                    auth=["user", "password-two"],
                ),
            )

        assert excinfo.value.mismatch == "configuration"
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_same_mutable_auth_object_reuses_live_connection(self) -> None:
        auth = ["user", "mutable-password"]
        runtime = MCPRuntime(shutdown_timeout=0.05)
        first = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http("github", "https://mcp.example.com", auth=auth),
        )
        first_agent = Agent(prompt="first", tool_sources=[first.tools()])
        await first_agent.get_tool_lookups()

        rebound = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http("github", "https://mcp.example.com", auth=auth),
        )
        second_agent = Agent(prompt="second", tool_sources=[rebound.tools()])
        await second_agent.get_tool_lookups()

        assert _total_connects() == 1
        await first_agent.close()
        await second_agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_per_request_provider_instances_rebind_freely(self) -> None:
        """Applications naturally construct a provider per request; without a
        declared credential identity the newest bind silently becomes
        canonical instead of conflicting with the live connection."""
        async with MCPRuntime(idle_timeout=0, shutdown_timeout=0.05) as runtime:
            first_provider = _Credentials({"Authorization": "Bearer first"})
            second_provider = _Credentials({"Authorization": "Bearer second"})
            first = _bind(runtime, credentials=first_provider)
            first_agent = Agent(prompt="first", tool_sources=[first.tools()])
            await first_agent.get_tool_lookups()

            second = _bind(runtime, credentials=second_provider)  # no conflict
            second_agent = Agent(prompt="second", tool_sources=[second.tools()])
            await second_agent.get_tool_lookups()

            assert _total_connects() == 1  # live reuse, no fresh lookup
            assert second_provider.calls == 0

            # The first handle stays valid too: no staleness without identity.
            await first_agent.close()
            await second_agent.close()
            await asyncio.sleep(0.05)  # idle_timeout=0 retires the connection

            retry = Agent(prompt="retry", tool_sources=[first.tools()])
            await retry.get_tool_lookups()

            # The reconnect resolved through the newest bound provider.
            assert first_provider.calls == 1
            assert second_provider.calls == 1
            assert dict(_InstrumentedAdapter.instances[1].source.headers) == {
                "Authorization": "Bearer second"
            }

    @pytest.mark.asyncio
    async def test_live_rebind_cannot_drop_the_credential_provider(self) -> None:
        credentials = _Credentials({"Authorization": "Bearer live"})
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=credentials)
            agent = Agent(prompt="authenticated", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()

            with pytest.raises(MCPBindingConflictError) as excinfo:
                _bind(runtime)

            assert excinfo.value.mismatch == "configuration"
            await agent.close()

    @pytest.mark.asyncio
    async def test_idle_rebind_without_credentials_supersedes_authenticated_handle(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            credentials = _Credentials({"Authorization": "Bearer old"})
            authenticated = _bind(runtime, credentials=credentials)
            anonymous = _bind(runtime)

            with pytest.raises(MCPStaleConnectionError, match="superseded"):
                await Agent(
                    prompt="stale",
                    tool_sources=[authenticated.tools()],
                ).get_tool_lookups()

            await Agent(prompt="anonymous", tool_sources=[anonymous.tools()]).get_tool_lookups()
            assert credentials.calls == 0
            assert dict(_InstrumentedAdapter.instances[0].source.headers) == {}

    @pytest.mark.asyncio
    async def test_changed_credential_identity_requires_eviction_while_live(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            first = _bind(runtime, credentials=_Credentials(), credential_identity="cred-v1")
            agent = Agent(prompt="test", tool_sources=[first.tools()])
            await agent.get_tool_lookups()

            with pytest.raises(MCPBindingConflictError) as excinfo:
                _bind(runtime, credentials=_Credentials(), credential_identity="cred-v2")

            assert excinfo.value.mismatch == "configuration"
            await agent.close()

    @pytest.mark.asyncio
    async def test_same_credential_identity_refreshes_the_provider_in_place(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            first = _bind(runtime, credentials=_Credentials(), credential_identity="cred-v1")
            first_agent = Agent(prompt="first", tool_sources=[first.tools()])
            await first_agent.get_tool_lookups()

            rebound = _bind(runtime, credentials=_Credentials(), credential_identity="cred-v1")
            second_agent = Agent(prompt="second", tool_sources=[rebound.tools()])
            await second_agent.get_tool_lookups()

            assert _total_connects() == 1

    @pytest.mark.asyncio
    async def test_same_provider_rebind_reuses_live_connection(self) -> None:
        credentials = _Credentials()
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            first = _bind(runtime, credentials=credentials)
            first_agent = Agent(prompt="first", tool_sources=[first.tools()])
            await first_agent.get_tool_lookups()

            rebound = _bind(runtime, credentials=credentials)
            second_agent = Agent(prompt="second", tool_sources=[rebound.tools()])
            await second_agent.get_tool_lookups()

            assert _total_connects() == 1
            assert credentials.calls == 1

    @pytest.mark.asyncio
    async def test_idle_credential_identity_rebind_supersedes_old_handle(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            old_credentials = _Credentials({"Authorization": "Bearer old-token"})
            new_credentials = _Credentials({"Authorization": "Bearer new-token"})
            stale = _bind(runtime, credentials=old_credentials, credential_identity="cred-v1")
            current = _bind(runtime, credentials=new_credentials, credential_identity="cred-v2")

            with pytest.raises(RuntimeError, match="superseded"):
                await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()

            await Agent(prompt="current", tool_sources=[current.tools()]).get_tool_lookups()

            assert old_credentials.calls == 0
            assert new_credentials.calls == 1
            assert dict(_InstrumentedAdapter.instances[0].source.headers) == {
                "Authorization": "Bearer new-token"
            }

    @pytest.mark.asyncio
    async def test_idle_auth_rebind_supersedes_old_handle(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        stale = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github",
                "https://mcp.example.com",
                auth=["user", "password-one"],
            ),
        )
        current = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github",
                "https://mcp.example.com",
                auth=["user", "password-two"],
            ),
        )

        with pytest.raises(RuntimeError, match="superseded"):
            await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()

        current_agent = Agent(prompt="current", tool_sources=[current.tools()])
        await current_agent.get_tool_lookups()
        await current_agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_changed_live_configuration_requires_eviction(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github",
                    "https://mcp.example.com",
                    headers={"Authorization": "Bearer old-token"},
                ),
            )
            await Agent(prompt="test", tool_sources=[connection.tools()]).get_tool_lookups()

            with pytest.raises(MCPBindingConflictError) as excinfo:
                runtime.bind(
                    connection_key="github-1",
                    source=MCPSource.http(
                        "github",
                        "https://mcp.example.com",
                        headers={"Authorization": "Bearer new-token"},
                    ),
                )

            assert excinfo.value.mismatch == "configuration"

            # The source name is presentation (the default namespace), not
            # connection configuration — rebinding it while live is fine.
            renamed = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github_alias",
                    "https://mcp.example.com",
                    headers={"Authorization": "Bearer old-token"},
                ),
            )
            assert renamed.source.name == "github_alias"

    @pytest.mark.asyncio
    async def test_superseded_handle_cannot_open_the_connection(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            stale = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github",
                    "https://mcp.example.com",
                    headers={"Authorization": "Bearer old-token"},
                ),
            )
            current = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http(
                    "github",
                    "https://mcp.example.com",
                    headers={"Authorization": "Bearer new-token"},
                ),
            )

            with pytest.raises(RuntimeError, match="superseded"):
                await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()

            lookups = await Agent(
                prompt="current",
                tool_sources=[current.tools()],
            ).get_tool_lookups()
            assert sorted(lookups.fn) == ["github__read", "github__write"]


class TestGovernanceEvents:
    @pytest.mark.asyncio
    async def test_aliased_views_report_their_tools(self) -> None:
        from dendrux.runtime.runner import _emit_init_events

        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            agent = Agent(
                prompt="test",
                tool_sources=[
                    connection.tools(namespace="github_work", allowed_tools=["read"]),
                ],
            )

            with patch(
                "dendrux.runtime.runner._emit_init_governance_event",
                new_callable=AsyncMock,
            ) as emitted:
                await _emit_init_events(agent, None, None, "run-1")

        connected = [
            call.args[4] for call in emitted.await_args_list if "tool_count" in call.args[4]
        ]
        assert connected == [
            {
                "source_name": "github",
                "namespace": "github_work",
                "tool_count": 1,
                "tool_names": ["github_work__read"],
            }
        ]


async def _await_first_call(adapter: _InstrumentedAdapter) -> None:
    """Wait until the fake transport has entered its gated tool call."""
    for _ in range(100):
        if adapter.call_tool_calls:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("tool call never reached the transport")


class TestExplicitEviction:
    @pytest.mark.asyncio
    async def test_drain_waits_for_active_leases_and_calls(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        call = asyncio.create_task(lookups.fn["github__write"](value="running"))
        await _await_first_call(adapter)
        evicting = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
        )
        await asyncio.sleep(0.05)

        assert not evicting.done()
        assert adapter.closed is False

        _InstrumentedAdapter.call_gate.set()
        assert await call == "write:ok"
        await asyncio.sleep(0.05)

        assert not evicting.done()  # the Agent still holds its lease

        await agent.close()
        await asyncio.wait_for(evicting, timeout=1.0)

        assert adapter.closed is True
        await runtime.close()

    @pytest.mark.asyncio
    async def test_new_leases_and_calls_are_rejected_while_draining(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        lookups = await agent.get_tool_lookups()
        retained = lookups.fn["github__write"]
        adapter = _InstrumentedAdapter.instances[0]

        evicting = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
        )
        await asyncio.sleep(0.05)
        assert not evicting.done()

        late = Agent(prompt="late", tool_sources=[connection.tools(namespace="late")])
        with pytest.raises(RuntimeError, match="evict"):
            await late.get_tool_lookups()
        with pytest.raises(MCPConnectionEvictingError):
            await retained(value="must-not-run")
        assert adapter.call_tool_calls == 0

        await agent.close()
        await asyncio.wait_for(evicting, timeout=1.0)
        await runtime.close()

    @pytest.mark.asyncio
    async def test_retained_call_raises_stale_after_eviction_completes(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        retained = (await agent.get_tool_lookups()).fn["github__write"]

        await runtime.evict(connection_key="github-1", mode="force")

        with pytest.raises(MCPStaleConnectionError) as excinfo:
            await retained(value="must-not-run")
        assert excinfo.value.identity == (None, "github-1")
        assert not isinstance(excinfo.value, MCPToolCallError)
        assert _total_calls() == 0
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_drain_timeout_escalates_to_a_forced_close(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()

        # The Agent never releases its lease: drain must escalate, not hang.
        await asyncio.wait_for(
            runtime.evict(connection_key="github-1", mode="drain", timeout=0.05),
            timeout=1.0,
        )

        assert _InstrumentedAdapter.instances[0].closed is True
        assert runtime.state is MCPRuntimeState.OPEN

        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_force_returns_after_fencing_when_transport_close_resists(self) -> None:
        runtime = MCPRuntime()
        stale = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[stale.tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]
        _InstrumentedAdapter.close_gate = asyncio.Event()

        await asyncio.wait_for(
            runtime.evict(connection_key="github-1", mode="force"),
            timeout=1.0,
        )

        assert adapter.close_calls == 1
        assert adapter.closed is False
        assert len(runtime._abandoned_tasks) == 1

        rebound = _bind(runtime)
        current = Agent(prompt="current", tool_sources=[rebound.tools()])
        await current.get_tool_lookups()
        assert len(_InstrumentedAdapter.instances) == 2

        _InstrumentedAdapter.close_gate.set()
        for _ in range(100):
            if adapter.closed:
                break
            await asyncio.sleep(0.01)
        assert adapter.closed is True

        await agent.close()
        await current.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_force_cancels_establishment_and_closes_exactly_once(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()  # never released
        runtime = MCPRuntime()
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        assert len(_InstrumentedAdapter.instances) == 1

        await asyncio.wait_for(
            runtime.evict(connection_key="github-1", mode="force"),
            timeout=1.0,
        )
        await asyncio.wait([discovery], timeout=1.0)

        assert discovery.done()
        assert discovery.cancelled() or discovery.exception() is not None
        assert _InstrumentedAdapter.instances[0].close_calls == 1

        await runtime.close()
        assert _InstrumentedAdapter.instances[0].close_calls == 1

    @pytest.mark.asyncio
    async def test_force_tracks_connect_task_that_ignores_eviction_cancellation(
        self,
    ) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.suppress_cancel = True
        runtime = MCPRuntime()
        stale = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[stale.tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)

        await asyncio.wait_for(
            runtime.evict(connection_key="github-1", mode="force"),
            timeout=1.0,
        )

        assert len(runtime._abandoned_tasks) == 1
        rebound = _bind(runtime)
        assert rebound.generation != stale.generation

        _InstrumentedAdapter.connect_gate.set()
        await asyncio.gather(discovery, return_exceptions=True)
        for _ in range(100):
            if not runtime._abandoned_tasks:
                break
            await asyncio.sleep(0.01)

        assert runtime._abandoned_tasks == set()
        assert _InstrumentedAdapter.instances[0].closed is True

        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_rebind_cannot_bypass_eviction_during_cancelled_establishment(
        self,
    ) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()  # never released
        _InstrumentedAdapter.close_gate = asyncio.Event()  # hold cancelled-task cleanup open
        runtime = MCPRuntime()
        stale = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[stale.tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        evicting = asyncio.create_task(runtime.evict(connection_key="github-1", mode="force"))
        await asyncio.sleep(0.01)

        assert not evicting.done()
        with pytest.raises(RuntimeError, match="being evicted"):
            runtime.bind(
                connection_key="github-1",
                source=MCPSource.http("github", "https://mcp.example.com/v2"),
            )
        late = Agent(prompt="late", tool_sources=[stale.tools(namespace="late")])
        with pytest.raises(RuntimeError, match="evict"):
            await late.get_tool_lookups()

        await asyncio.wait_for(evicting, timeout=1.0)
        _InstrumentedAdapter.close_gate.set()
        await asyncio.gather(discovery, return_exceptions=True)

        rebound = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http("github", "https://mcp.example.com/v2"),
        )
        assert rebound.generation != stale.generation

        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_forced_interruption_reports_unknown_outcome(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime()
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        lookups = await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        call = asyncio.create_task(lookups.fn["github__write"](value="running"))
        await _await_first_call(adapter)

        await asyncio.wait_for(
            runtime.evict(connection_key="github-1", mode="force"),
            timeout=1.0,
        )
        assert adapter.closed is True

        _InstrumentedAdapter.call_gate.set()
        with pytest.raises(MCPOutcomeUnknownError):
            await call

        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_rebinding_after_eviction_accepts_a_new_endpoint(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="first", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        await agent.close()

        # A changed endpoint conflicts while the connection is registered.
        with pytest.raises(MCPBindingConflictError):
            runtime.bind(
                connection_key="github-1",
                source=MCPSource.http("github", "https://mcp.example.com/v2"),
            )

        await runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)

        rebound = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http("github", "https://mcp.example.com/v2"),
        )
        second = Agent(prompt="second", tool_sources=[rebound.tools()])
        lookups = await second.get_tool_lookups()

        assert sorted(lookups.fn) == ["github__read", "github__write"]
        assert len(_InstrumentedAdapter.instances) == 2
        assert _InstrumentedAdapter.instances[1].source.url == "https://mcp.example.com/v2"
        await second.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_stale_handles_cannot_reach_the_rebound_connection(self) -> None:
        runtime = MCPRuntime()
        stale = _bind(runtime)
        agent = Agent(prompt="first", tool_sources=[stale.tools()])
        lookups = await agent.get_tool_lookups()
        retained = lookups.fn["github__write"]
        await agent.close()

        await runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)

        # Rebinding with identical configuration must still invalidate handles
        # issued before the eviction.
        fresh = _bind(runtime)
        current = Agent(prompt="second", tool_sources=[fresh.tools()])
        await current.get_tool_lookups()

        with pytest.raises(RuntimeError, match="superseded"):
            await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()
        with pytest.raises(MCPToolCallError):
            await retained(value="must-not-run")

        assert len(_InstrumentedAdapter.instances) == 2
        await current.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_eviction_is_scoped_to_one_tenant_partition(self) -> None:
        runtime = MCPRuntime()
        agent_a = Agent(prompt="a", tool_sources=[_bind(runtime, tenant_key="tenant-a").tools()])
        agent_b = Agent(prompt="b", tool_sources=[_bind(runtime, tenant_key="tenant-b").tools()])
        await agent_a.get_tool_lookups()
        lookups_b = await agent_b.get_tool_lookups()
        adapter_a, adapter_b = _InstrumentedAdapter.instances

        await runtime.evict(tenant_key="tenant-a", connection_key="github-1", mode="force")

        assert adapter_a.closed is True
        assert adapter_b.closed is False
        assert await lookups_b.fn["github__read"](value=1) == "read:ok"

        await agent_a.close()
        await agent_b.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_concurrent_and_repeated_evictions_share_one_operation(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        first = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
        )
        second = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
        )
        await asyncio.sleep(0.05)

        assert not first.done()
        assert not second.done()

        await agent.close()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)

        assert adapter.close_calls == 1

        # Repeating an eviction after it finished is a no-op, not a re-close.
        await runtime.evict(connection_key="github-1", mode="force")
        assert adapter.close_calls == 1
        await runtime.close()

    @pytest.mark.asyncio
    async def test_force_escalates_an_in_flight_drain(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        draining = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=10.0)
        )
        await asyncio.sleep(0.05)
        assert not draining.done()

        # Credentials compromised mid-drain: force must not wait for the lease.
        await asyncio.wait_for(
            runtime.evict(connection_key="github-1", mode="force"),
            timeout=1.0,
        )
        await asyncio.wait_for(draining, timeout=1.0)

        assert adapter.closed is True
        assert adapter.close_calls == 1
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_shutdown_and_eviction_do_not_double_close(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=1.0)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        evicting = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
        )
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)

        await agent.close()
        await asyncio.wait_for(asyncio.gather(evicting, closing), timeout=2.0)

        assert adapter.close_calls == 1
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_forced_shutdown_completes_a_long_running_drain_eviction(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        evicting = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=10.0)
        )
        await asyncio.sleep(0.01)
        closing = asyncio.create_task(runtime.close())

        await asyncio.wait_for(asyncio.gather(evicting, closing), timeout=1.0)

        assert adapter.close_calls == 1
        assert runtime.state is MCPRuntimeState.CLOSED
        assert runtime._evictions == {}

        await agent.close()

    @pytest.mark.asyncio
    async def test_force_eviction_escalates_shutdown_already_draining(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=10.0)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        closing = asyncio.create_task(runtime.close())
        for _ in range(100):
            if runtime.state is MCPRuntimeState.DRAINING:
                break
            await asyncio.sleep(0.01)
        assert runtime.state is MCPRuntimeState.DRAINING

        evicting = asyncio.create_task(runtime.evict(connection_key="github-1", mode="force"))
        await asyncio.wait_for(asyncio.gather(evicting, closing), timeout=1.0)

        assert adapter.close_calls == 1
        assert runtime.state is MCPRuntimeState.CLOSED

        await agent.close()

    @pytest.mark.asyncio
    async def test_forced_shutdown_also_reports_unknown_outcome(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        lookups = await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        call = asyncio.create_task(lookups.fn["github__write"](value="running"))
        await _await_first_call(adapter)

        # The drain budget expires while the call is still running, so the
        # transport closes underneath it exactly as a forced eviction would.
        await asyncio.wait_for(runtime.close(), timeout=1.0)

        _InstrumentedAdapter.call_gate.set()
        with pytest.raises(MCPOutcomeUnknownError):
            await call

    @pytest.mark.asyncio
    async def test_evicting_an_unknown_connection_is_a_no_op(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            await runtime.evict(connection_key="never-bound", mode="drain", timeout=1.0)
            await runtime.evict(tenant_key="tenant-a", connection_key="never-bound", mode="force")

            assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_evicting_a_bound_but_unopened_connection_invalidates_its_handle(
        self,
    ) -> None:
        runtime = MCPRuntime()
        stale = _bind(runtime)

        await runtime.evict(connection_key="github-1", mode="drain")

        with pytest.raises(RuntimeError, match="no longer registered"):
            await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()
        assert _InstrumentedAdapter.instances == []

        await runtime.close()

    @pytest.mark.asyncio
    async def test_evicting_after_runtime_close_is_a_no_op(self) -> None:
        runtime = MCPRuntime()
        await runtime.close()

        await runtime.evict(connection_key="github-1", mode="force")

        assert runtime.state is MCPRuntimeState.CLOSED
        assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_invalid_eviction_arguments_are_rejected(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            with pytest.raises(ValueError, match="mode"):
                await runtime.evict(connection_key="github-1", mode="stop")  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="timeout"):
                await runtime.evict(connection_key="github-1", timeout=0)
            with pytest.raises(ValueError, match="connection"):
                await runtime.evict(connection_key="  ")


class TestTypedFailureModes:
    """Operational failures are typed so callers can branch without parsing."""

    @pytest.mark.asyncio
    async def test_ordinary_tool_failure_is_not_reported_as_unknown_outcome(self) -> None:
        _InstrumentedAdapter.call_error = RuntimeError("server exploded")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()

            # A plain server-side failure has a KNOWN outcome: it failed.
            with pytest.raises(MCPToolCallError) as excinfo:
                await lookups.fn["github__write"](value=1)
            assert not isinstance(excinfo.value, MCPOutcomeUnknownError)

            entry = agent._tool_sources[0]._entry
            assert entry is not None
            assert entry.active_calls == 0

            # No sticky state: a second failure behaves identically.
            with pytest.raises(MCPToolCallError):
                await lookups.fn["github__write"](value=2)
            assert entry.active_calls == 0

            await agent.close()

    @pytest.mark.asyncio
    async def test_closed_runtime_raises_a_typed_error(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        view = _bind(runtime).tools()
        await runtime.close()

        with pytest.raises(MCPRuntimeClosedError):
            _bind(runtime, connection_key="late")
        with pytest.raises(MCPRuntimeClosedError):
            await Agent(prompt="test", tool_sources=[view]).get_tool_lookups()

    @pytest.mark.asyncio
    async def test_evicting_identity_raises_a_typed_transient_error(self) -> None:
        runtime = MCPRuntime()
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()

        evicting = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
        )
        await asyncio.sleep(0.05)

        with pytest.raises(MCPConnectionEvictingError) as bind_error:
            _bind(runtime)
        with pytest.raises(MCPConnectionEvictingError) as lease_error:
            await Agent(
                prompt="late", tool_sources=[connection.tools(namespace="late")]
            ).get_tool_lookups()

        assert bind_error.value.identity == (None, "github-1")
        assert lease_error.value.identity == (None, "github-1")

        await agent.close()
        await asyncio.wait_for(evicting, timeout=1.0)
        await runtime.close()

    @pytest.mark.asyncio
    async def test_stale_handles_raise_a_typed_permanent_error(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        evicted_before_use = _bind(runtime, connection_key="never-leased")
        await runtime.evict(connection_key="never-leased", mode="drain", timeout=1.0)

        # Unregistered: the identity was evicted before it was ever leased.
        with pytest.raises(MCPStaleConnectionError) as unregistered:
            await Agent(
                prompt="stale", tool_sources=[evicted_before_use.tools()]
            ).get_tool_lookups()
        assert unregistered.value.identity == (None, "never-leased")

        # Superseded: an idle rebind rotated the configuration.
        stale = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github", "https://mcp.example.com", headers={"Authorization": "Bearer old"}
            ),
        )
        runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github", "https://mcp.example.com", headers={"Authorization": "Bearer new"}
            ),
        )
        with pytest.raises(MCPStaleConnectionError) as superseded:
            await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()
        assert superseded.value.identity == (None, "github-1")

        await runtime.close()

    @pytest.mark.asyncio
    async def test_executor_before_discovery_is_rejected(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            source = _ViewToolSource(_bind(runtime).tools())

            with pytest.raises(RuntimeError, match="before discovery"):
                source._create_executor("read")


class TestTransportMisbehaviour:
    """A transport that fails during cleanup must not corrupt the runtime."""

    @pytest.mark.asyncio
    async def test_close_failure_does_not_break_shutdown(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        await agent.get_tool_lookups()
        await agent.close()
        _InstrumentedAdapter.close_error = OSError("close failed")

        await asyncio.wait_for(runtime.close(), timeout=1.0)

        assert runtime.state is MCPRuntimeState.CLOSED
        assert _InstrumentedAdapter.instances[0].close_calls == 1

    @pytest.mark.asyncio
    async def test_close_failure_does_not_log_resolved_credentials(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "RUNTIME-CLOSE-SECRET"
        runtime = MCPRuntime(shutdown_timeout=0.05)
        connection = _bind(
            runtime,
            credentials=_Credentials({"Authorization": f"Bearer {secret}"}),
        )
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        await agent.close()
        _InstrumentedAdapter.close_error = OSError(f"close echoed {secret}")

        with caplog.at_level("WARNING", logger="dendrux.mcp._runtime"):
            await runtime.close()

        assert secret not in caplog.text
        assert "[redacted]" in caplog.text

    @pytest.mark.asyncio
    async def test_close_failure_does_not_break_eviction(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        await agent.get_tool_lookups()
        await agent.close()
        _InstrumentedAdapter.close_error = OSError("close failed")

        await asyncio.wait_for(runtime.evict(connection_key="github-1", mode="force"), timeout=1.0)

        # The identity is still fully released despite the failed cleanup.
        _InstrumentedAdapter.close_error = None
        rebound = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http("github", "https://mcp.example.com/v2"),
        )
        assert rebound.generation != connection.generation
        await runtime.close()

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_the_connect_error(self) -> None:
        _InstrumentedAdapter.connect_error = ConnectionError("handshake refused")
        _InstrumentedAdapter.close_error = OSError("close failed too")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

            # The original connect failure wins over the cleanup failure.
            with pytest.raises(ConnectionError, match="handshake refused"):
                await agent.get_tool_lookups()

    @pytest.mark.asyncio
    async def test_zero_tool_server_warns_and_yields_an_empty_catalog(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _InstrumentedAdapter.tools = []
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

            with caplog.at_level("WARNING", logger="dendrux.mcp._runtime"):
                lookups = await agent.get_tool_lookups()

            assert not lookups.fn
            assert "discovered zero tools" in caplog.text
            await agent.close()


class TestRuntimeConfigurationValidation:
    def test_bind_rejects_a_non_source_and_bad_credentials(self) -> None:
        runtime = MCPRuntime()

        with pytest.raises(ValueError, match="must be an MCPSource"):
            runtime.bind(connection_key="github-1", source="https://mcp.example.com")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="get_auth"):
            runtime.bind(
                connection_key="github-1",
                source=MCPSource.http("github", "https://mcp.example.com"),
                credentials=object(),  # type: ignore[arg-type]
            )

    def test_credential_identity_requires_a_provider_and_valid_text(self) -> None:
        runtime = MCPRuntime()
        source = MCPSource.http("github", "https://mcp.example.com")

        with pytest.raises(ValueError, match="requires a credentials provider"):
            runtime.bind(
                connection_key="github-1",
                source=source,
                credential_identity="cred-v1",
            )
        for invalid in ("", "   ", "cred\n1"):
            with pytest.raises(ValueError, match="credential identity"):
                runtime.bind(
                    connection_key="github-1",
                    source=source,
                    credentials=_Credentials(),
                    credential_identity=invalid,
                )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("idle_timeout", "300"),
            ("shutdown_timeout", None),
            ("max_connections", 1.5),
            ("max_in_flight_calls", True),
        ],
    )
    def test_non_numeric_tuning_is_rejected(self, field: str, value: Any) -> None:
        with pytest.raises(ValueError, match=field):
            MCPRuntime(**{field: value})

    def test_identity_keys_and_policy_inputs_are_validated(self) -> None:
        runtime = MCPRuntime()
        source = MCPSource.http("github", "https://mcp.example.com")

        with pytest.raises(ValueError, match="connection key is required"):
            runtime.bind(connection_key=None, source=source)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="collection of tool names"):
            # A bare string would silently expose one tool per character.
            runtime.bind(connection_key="github-1", source=source).tools(allowed_tools="read")

    @pytest.mark.asyncio
    async def test_evict_validates_its_identity_keys(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            with pytest.raises(ValueError, match="connection key is required"):
                await runtime.evict(connection_key=None)  # type: ignore[arg-type]


class TestDoubleFailurePaths:
    """Cleanup that fails while handling an earlier failure stays contained."""

    @pytest.mark.asyncio
    async def test_late_establishment_cleanup_failure_is_contained(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.suppress_cancel = True  # ignores eviction cancellation
        _InstrumentedAdapter.close_error = OSError("close failed too")
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        await asyncio.wait_for(runtime.evict(connection_key="github-1", mode="force"), timeout=1.0)

        # The connect finishes late into an evicted entry and its own cleanup
        # also fails; neither may surface as an unobserved task exception.
        _InstrumentedAdapter.connect_gate.set()
        await asyncio.wait([discovery], timeout=1.0)

        assert discovery.done()
        assert discovery.cancelled() or discovery.exception() is not None
        assert runtime._abandoned_tasks == set()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_abandoned_connect_failure_is_observed(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.suppress_cancel = True
        _InstrumentedAdapter.connect_error = ConnectionError("handshake refused")
        runtime = MCPRuntime(shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        await asyncio.wait_for(runtime.close(), timeout=1.0)

        assert runtime.state is MCPRuntimeState.CLOSED
        assert len(runtime._abandoned_tasks) == 1

        # The abandoned task finishes with a plain transport error: the runtime
        # must consume it and drop its tracking entry.
        _InstrumentedAdapter.connect_gate.set()
        await asyncio.wait([discovery], timeout=1.0)
        await asyncio.sleep(0.05)

        assert runtime._abandoned_tasks == set()

    @pytest.mark.asyncio
    async def test_an_expired_budget_never_waits(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        expired = asyncio.get_running_loop().time() - 1.0

        # A deadline already in the past must report "not drained" immediately
        # rather than blocking on a waiter that no one will ever set.
        assert await runtime._wait_for_quiet(expired, lambda: False) is False
        assert runtime._drain_waiters == set()


async def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    """Poll until predicate() is true, or fail the test."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was never reached")


class TestIdleRetirement:
    """Idle retirement closes the socket but keeps the identity bindable.

    Unlike eviction, the registration and generation survive, so the same
    MCPConnection handle transparently reconnects on next use.
    """

    @pytest.mark.asyncio
    async def test_idle_connection_is_closed_after_its_timeout(self) -> None:
        async with MCPRuntime(idle_timeout=0.05, shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            await agent.get_tool_lookups()
            adapter = _InstrumentedAdapter.instances[0]

            await agent.close()
            assert adapter.closed is False  # still warm immediately after release

            await _wait_until(lambda: adapter.closed)
            assert runtime._entries == {}

    @pytest.mark.asyncio
    async def test_the_same_handle_reconnects_after_retirement(self) -> None:
        async with MCPRuntime(idle_timeout=0.05, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            await first.get_tool_lookups()
            await first.close()
            await _wait_until(lambda: _InstrumentedAdapter.instances[0].closed)

            # The registration survived: no rebind, no stale-handle error.
            second = Agent(prompt="second", tool_sources=[connection.tools()])
            lookups = await second.get_tool_lookups()

            assert sorted(lookups.fn) == ["github__read", "github__write"]
            assert await lookups.fn["github__read"](value=1) == "read:ok"
            assert len(_InstrumentedAdapter.instances) == 2
            await second.close()

    @pytest.mark.asyncio
    async def test_a_new_lease_cancels_pending_retirement(self) -> None:
        async with MCPRuntime(idle_timeout=0.2, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            await first.get_tool_lookups()
            await first.close()

            second = Agent(prompt="second", tool_sources=[connection.tools()])
            await second.get_tool_lookups()
            await asyncio.sleep(0.3)  # well past the original idle deadline

            adapter = _InstrumentedAdapter.instances[0]
            assert adapter.closed is False
            assert len(_InstrumentedAdapter.instances) == 1
            await second.close()

    @pytest.mark.asyncio
    async def test_an_in_flight_call_defers_retirement(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        async with MCPRuntime(idle_timeout=0.05, shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            adapter = _InstrumentedAdapter.instances[0]

            call = asyncio.create_task(lookups.fn["github__write"](value="running"))
            await _await_first_call(adapter)
            await agent.close()  # lease released, call still running
            await asyncio.sleep(0.15)

            assert adapter.closed is False  # never closed under a live call

            _InstrumentedAdapter.call_gate.set()
            assert await call == "write:ok"
            await _wait_until(lambda: adapter.closed)

    @pytest.mark.asyncio
    async def test_zero_idle_timeout_retires_immediately(self) -> None:
        async with MCPRuntime(idle_timeout=0, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            agent = Agent(prompt="test", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()
            await agent.close()

            await _wait_until(lambda: _InstrumentedAdapter.instances[0].closed)

            # Still reusable: retirement is not eviction.
            reused = Agent(prompt="reused", tool_sources=[connection.tools()])
            await reused.get_tool_lookups()
            assert len(_InstrumentedAdapter.instances) == 2
            await reused.close()

    @pytest.mark.asyncio
    async def test_tenants_retire_independently(self) -> None:
        async with MCPRuntime(idle_timeout=0.05, shutdown_timeout=0.05) as runtime:
            idle_agent = Agent(
                prompt="a", tool_sources=[_bind(runtime, tenant_key="tenant-a").tools()]
            )
            busy_agent = Agent(
                prompt="b", tool_sources=[_bind(runtime, tenant_key="tenant-b").tools()]
            )
            await idle_agent.get_tool_lookups()
            busy_lookups = await busy_agent.get_tool_lookups()
            adapter_a, adapter_b = _InstrumentedAdapter.instances

            await idle_agent.close()
            await _wait_until(lambda: adapter_a.closed)

            assert adapter_b.closed is False
            assert await busy_lookups.fn["github__read"](value=1) == "read:ok"
            await busy_agent.close()

    @pytest.mark.asyncio
    async def test_a_lease_racing_retirement_reconnects_transparently(self) -> None:
        _InstrumentedAdapter.close_gate = asyncio.Event()  # retirement stalls mid-close
        async with MCPRuntime(idle_timeout=0.01, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            await first.get_tool_lookups()
            await first.close()
            await _wait_until(lambda: _InstrumentedAdapter.instances[0].close_calls == 1)

            # Lease arrives while retirement is in flight: it must wait for the
            # retirement to finish and then reconnect, never raise stale-handle.
            second = Agent(prompt="second", tool_sources=[connection.tools()])
            lookups = await asyncio.wait_for(second.get_tool_lookups(), timeout=1.0)

            assert sorted(lookups.fn) == ["github__read", "github__write"]
            assert len(_InstrumentedAdapter.instances) == 2
            _InstrumentedAdapter.close_gate.set()
            await second.close()

    @pytest.mark.asyncio
    async def test_eviction_during_retirement_still_invalidates_the_handle(self) -> None:
        _InstrumentedAdapter.close_gate = asyncio.Event()
        async with MCPRuntime(idle_timeout=0.01, shutdown_timeout=0.05) as runtime:
            stale = _bind(runtime)
            agent = Agent(prompt="test", tool_sources=[stale.tools()])
            await agent.get_tool_lookups()
            await agent.close()
            await _wait_until(lambda: _InstrumentedAdapter.instances[0].close_calls == 1)

            await asyncio.wait_for(
                runtime.evict(connection_key="github-1", mode="force"), timeout=1.0
            )
            _InstrumentedAdapter.close_gate.set()

            # Eviction outranks retirement: the handle is now permanently stale.
            with pytest.raises(MCPStaleConnectionError):
                await Agent(prompt="stale", tool_sources=[stale.tools()]).get_tool_lookups()

            rebound = _bind(runtime)
            assert rebound.generation != stale.generation

    @pytest.mark.asyncio
    async def test_shutdown_cancels_timers_and_closes_once(self) -> None:
        runtime = MCPRuntime(idle_timeout=0.05, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]
        await agent.close()

        # Shutdown races the pending idle timer: exactly one close either way.
        await asyncio.wait_for(runtime.close(), timeout=1.0)
        await asyncio.sleep(0.15)

        assert adapter.close_calls == 1
        assert runtime.state is MCPRuntimeState.CLOSED
        assert runtime._entries == {}

    @pytest.mark.asyncio
    async def test_resistant_retirement_stays_bounded_and_tracked(self) -> None:
        _InstrumentedAdapter.close_gate = asyncio.Event()  # never released here
        runtime = MCPRuntime(idle_timeout=0.01, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        await agent.get_tool_lookups()
        await agent.close()

        await _wait_until(lambda: len(runtime._abandoned_tasks) == 1)
        assert runtime._entries == {}  # released despite the stuck transport

        _InstrumentedAdapter.close_gate.set()
        await _wait_until(lambda: runtime._abandoned_tasks == set())
        await asyncio.wait_for(runtime.close(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_retirement_cycles_leak_no_tasks_or_timers(self) -> None:
        before = len(asyncio.all_tasks())
        runtime = MCPRuntime(idle_timeout=0.02, shutdown_timeout=0.05)
        connection = _bind(runtime)

        for index in range(5):
            agent = Agent(prompt=f"run-{index}", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()
            await agent.close()
            await _wait_until(lambda: runtime._entries == {})

        assert len(_InstrumentedAdapter.instances) == 5
        assert all(adapter.closed for adapter in _InstrumentedAdapter.instances)

        await asyncio.wait_for(runtime.close(), timeout=1.0)
        await asyncio.sleep(0.05)

        assert runtime._entries == {}
        assert runtime._evictions == {}
        assert runtime._abandoned_tasks == set()
        assert runtime._drain_waiters == set()
        assert len(asyncio.all_tasks()) <= before + 1  # only this test's task

    @pytest.mark.asyncio
    async def test_retirement_cancels_an_abandoned_establishment(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()  # never released
        runtime = MCPRuntime(idle_timeout=0.01, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        assert len(_InstrumentedAdapter.instances) == 1

        # The only requester walks away mid-handshake: the half-open socket
        # must be reclaimed rather than left connecting forever.
        discovery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discovery

        await _wait_until(lambda: runtime._entries == {})
        assert _InstrumentedAdapter.instances[0].close_calls == 1
        await asyncio.wait_for(runtime.close(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_retirement_defers_to_a_concurrent_eviction(self) -> None:
        async with MCPRuntime(idle_timeout=10.0, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            agent = Agent(prompt="test", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()
            entry = agent._tool_sources[0]._entry
            assert entry is not None
            await agent.close()

            # An eviction fences the entry after its idle timer fired but
            # before the retirement body runs. Retirement owns neither the
            # entry nor its transport any more and must not touch either.
            entry.fenced = True
            await runtime._retire(connection.identity, entry)

            assert runtime._entries.get(connection.identity) is entry
            assert _InstrumentedAdapter.instances[0].closed is False
            assert entry.retire_task is None

    @pytest.mark.asyncio
    async def test_retirement_is_bounded_when_establishment_resists_cancellation(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.suppress_cancel = True  # ignores idle cancellation
        runtime = MCPRuntime(idle_timeout=0.01, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        discovery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discovery

        # Retirement cannot interrupt the handshake, so it hands the task off
        # rather than blocking: the identity is released either way.
        await _wait_until(lambda: len(runtime._abandoned_tasks) == 1)
        assert runtime._entries == {}

        _InstrumentedAdapter.connect_gate.set()
        await _wait_until(lambda: runtime._abandoned_tasks == set())
        assert _InstrumentedAdapter.instances[0].closed is True
        await asyncio.wait_for(runtime.close(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_failed_establishment_cancels_its_pending_idle_timer(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.connect_error = ConnectionError("handshake refused")
        runtime = MCPRuntime(idle_timeout=300.0, shutdown_timeout=0.05)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        entry = runtime._entries[connection.identity]

        # The sole waiter walks away mid-handshake, so its release arms an
        # idle timer against an entry that never finished establishing.
        discovery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert entry.idle_handle is not None

        # The shared handshake then fails and discards the entry. A surviving
        # timer would pin the entry, its source and its credentials in the
        # event loop for the whole idle window.
        _InstrumentedAdapter.connect_gate.set()
        await _wait_until(lambda: runtime._entries == {})

        assert entry.idle_handle is None
        await asyncio.wait_for(runtime.close(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_idle_timeout_none_keeps_connections_warm(self) -> None:
        async with MCPRuntime(idle_timeout=None, shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            agent = Agent(prompt="test", tool_sources=[connection.tools()])
            await agent.get_tool_lookups()
            entry = runtime._entries[connection.identity]
            adapter = _InstrumentedAdapter.instances[0]

            await agent.close()
            await asyncio.sleep(0.1)

            assert entry.idle_handle is None  # no timer is ever armed
            assert adapter.closed is False
            assert runtime._entries != {}

            # Explicit eviction is unaffected by disabling idle retirement.
            await runtime.evict(connection_key="github-1", mode="drain", timeout=1.0)
            assert adapter.closed is True

    @pytest.mark.asyncio
    async def test_idle_timeout_none_still_closes_at_shutdown(self) -> None:
        runtime = MCPRuntime(idle_timeout=None, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]
        await agent.close()
        await asyncio.sleep(0.05)

        assert adapter.closed is False

        await asyncio.wait_for(runtime.close(), timeout=1.0)
        assert adapter.closed is True
        assert runtime._entries == {}


class TestConcurrentAgentTeardown:
    """An Agent torn down mid-discovery must not pin its connection."""

    @pytest.mark.asyncio
    async def test_closing_an_agent_during_discovery_releases_its_lease(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        runtime = MCPRuntime(idle_timeout=0.05, shutdown_timeout=0.3)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])

        discovery = asyncio.create_task(agent.get_tool_lookups())
        await asyncio.sleep(0.05)
        entry = runtime._entries[connection.identity]

        # Another task tears the Agent down while the handshake is running.
        # close() cannot release a lease the view has not taken yet, so the
        # view must hand it back as soon as discovery acquires it.
        await agent.close()
        _InstrumentedAdapter.connect_gate.set()
        with pytest.raises(RuntimeError, match="closed during discovery"):
            await discovery

        assert entry.leases == 0
        # A pinned lease would block retirement forever and burn the whole
        # shutdown budget; the connection must retire on schedule instead.
        await _wait_until(lambda: runtime._entries == {})
        await asyncio.wait_for(runtime.close(), timeout=1.0)
        assert _InstrumentedAdapter.instances[0].closed is True

    @pytest.mark.asyncio
    async def test_refresh_can_rediscover_a_managed_view(self) -> None:
        runtime = MCPRuntime(idle_timeout=None)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])

        first = await agent.get_tool_lookups()
        assert sorted(first.fn) == ["github__read", "github__write"]

        await agent.refresh()
        assert runtime._entries[connection.identity].leases == 0

        second = await agent.get_tool_lookups()
        assert sorted(second.fn) == ["github__read", "github__write"]
        assert _total_connects() == 1

        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_transient_discovery_failure_can_retry_the_same_view(self) -> None:
        runtime = MCPRuntime(idle_timeout=None)
        connection = _bind(runtime)
        agent = Agent(prompt="test", tool_sources=[connection.tools()])
        _InstrumentedAdapter.connect_error = ConnectionError("temporary handshake failure")

        with pytest.raises(ConnectionError, match="temporary handshake failure"):
            await agent.get_tool_lookups()

        _InstrumentedAdapter.connect_error = None
        lookups = await agent.get_tool_lookups()

        assert sorted(lookups.fn) == ["github__read", "github__write"]
        assert _total_connects() == 2
        assert _InstrumentedAdapter.instances[0].closed is True

        await agent.close()
        await runtime.close()


def _bind_key(
    runtime: MCPRuntime,
    key: str,
    *,
    tenant_key: str | None = None,
) -> MCPConnection:
    """Bind one identity whose URL identifies it in adapter assertions."""
    return runtime.bind(
        connection_key=key,
        tenant_key=tenant_key,
        source=MCPSource.http("github", f"https://mcp.example.com/{key}"),
    )


def _opened_keys() -> list[str]:
    return [str(a.source.url).rsplit("/", 1)[-1] for a in _InstrumentedAdapter.instances]


class TestConnectionCapacity:
    """max_connections caps live physical entries, not registrations."""

    @pytest.mark.asyncio
    async def test_registrations_consume_no_capacity(self) -> None:
        async with MCPRuntime(max_connections=1, shutdown_timeout=0.05) as runtime:
            for index in range(50):
                _bind_key(runtime, f"idle-{index}")

            assert runtime._entries == {}

            agent = Agent(prompt="test", tool_sources=[_bind_key(runtime, "live").tools()])
            await agent.get_tool_lookups()

            assert len(runtime._entries) == 1
            await agent.close()

    @pytest.mark.asyncio
    async def test_cap_limits_concurrent_physical_connections(self) -> None:
        async with MCPRuntime(
            max_connections=2, connection_wait_timeout=0.05, shutdown_timeout=0.05
        ) as runtime:
            held = []
            for index in range(2):
                agent = Agent(
                    prompt=f"a{index}",
                    tool_sources=[_bind_key(runtime, f"c{index}").tools()],
                )
                await agent.get_tool_lookups()
                held.append(agent)

            assert len(_InstrumentedAdapter.instances) == 2

            third = Agent(prompt="third", tool_sources=[_bind_key(runtime, "c2").tools()])
            with pytest.raises(MCPConnectionCapacityError) as excinfo:
                await third.get_tool_lookups()

            assert excinfo.value.identity == (None, "c2")
            assert excinfo.value.limit == 2
            assert excinfo.value.timeout == 0.05
            assert len(_InstrumentedAdapter.instances) == 2
            for agent in held:
                await agent.close()

    @pytest.mark.asyncio
    async def test_zero_wait_timeout_fails_fast(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=0, shutdown_timeout=0.05
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises(MCPConnectionCapacityError):
                await Agent(
                    prompt="second", tool_sources=[_bind_key(runtime, "c1").tools()]
                ).get_tool_lookups()

            assert loop.time() - started < 0.05  # never waited
            await busy.close()

    @pytest.mark.asyncio
    async def test_existing_identity_reuse_bypasses_the_queue(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            shared = _bind_key(runtime, "c0")
            first = Agent(prompt="first", tool_sources=[shared.tools()])
            await first.get_tool_lookups()

            queued = asyncio.create_task(
                Agent(
                    prompt="queued", tool_sources=[_bind_key(runtime, "c1").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)
            assert not queued.done()

            # The same identity needs no slot, so it must not wait behind c1.
            second = Agent(prompt="second", tool_sources=[shared.tools(namespace="gh")])
            lookups = await asyncio.wait_for(second.get_tool_lookups(), timeout=0.5)
            assert sorted(lookups.fn) == ["gh__read", "gh__write"]
            assert len(_InstrumentedAdapter.instances) == 1

            # Releasing every lease makes c0 reclaimable and admits c1.
            await first.close()
            await second.close()
            await asyncio.wait_for(queued, timeout=2.0)
            assert _opened_keys() == ["c0", "c1"]

    @pytest.mark.asyncio
    async def test_many_waiters_for_one_identity_share_one_position(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            shared = _bind_key(runtime, "c1")
            agents = [Agent(prompt=f"w{i}", tool_sources=[shared.tools()]) for i in range(5)]
            waiters = [asyncio.create_task(a.get_tool_lookups()) for a in agents]
            await asyncio.sleep(0.05)
            later = asyncio.create_task(
                Agent(
                    prompt="later", tool_sources=[_bind_key(runtime, "c2").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)

            await busy.close()
            results = await asyncio.wait_for(asyncio.gather(*waiters), timeout=2.0)

            assert all(sorted(r.fn) == ["github__read", "github__write"] for r in results)
            assert _opened_keys() == ["c0", "c1"]  # five waiters, one connection
            assert not later.done()  # c2 is still behind the busy c1

            for agent in agents:
                await agent.close()
            await asyncio.wait_for(later, timeout=2.0)
            assert _opened_keys() == ["c0", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_admission_is_fifo(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            first_waiter = asyncio.create_task(
                Agent(
                    prompt="first", tool_sources=[_bind_key(runtime, "first").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)
            second_waiter = asyncio.create_task(
                Agent(
                    prompt="second", tool_sources=[_bind_key(runtime, "second").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)

            await busy.close()
            await asyncio.wait_for(first_waiter, timeout=2.0)

            assert _opened_keys() == ["c0", "first"]
            assert not second_waiter.done()  # strictly behind the first
            second_waiter.cancel()
            await asyncio.gather(second_waiter, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_a_new_identity_cannot_barge_past_the_queue(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            queued = asyncio.create_task(
                Agent(
                    prompt="queued", tool_sources=[_bind_key(runtime, "queued").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)

            # Freeing the slot and racing a brand-new identity against the
            # queue head: the head must still win.
            await busy.close()
            latecomer = asyncio.create_task(
                Agent(
                    prompt="late", tool_sources=[_bind_key(runtime, "late").tools()]
                ).get_tool_lookups()
            )
            await asyncio.wait_for(queued, timeout=2.0)

            assert _opened_keys() == ["c0", "queued"]
            assert not latecomer.done()
            latecomer.cancel()
            await asyncio.gather(latecomer, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_cancelled_head_does_not_block_the_queue(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            head = asyncio.create_task(
                Agent(
                    prompt="head", tool_sources=[_bind_key(runtime, "head").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)
            follower = asyncio.create_task(
                Agent(
                    prompt="follower", tool_sources=[_bind_key(runtime, "follower").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)

            head.cancel()
            await asyncio.gather(head, return_exceptions=True)
            await busy.close()
            await asyncio.wait_for(follower, timeout=2.0)

            assert _opened_keys() == ["c0", "follower"]
            assert runtime._admission_queue == deque()

    @pytest.mark.asyncio
    async def test_timed_out_head_does_not_block_the_queue(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=0.3, shutdown_timeout=0.05
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            head = asyncio.create_task(
                Agent(
                    prompt="head", tool_sources=[_bind_key(runtime, "head").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.15)
            follower = asyncio.create_task(
                Agent(
                    prompt="follower", tool_sources=[_bind_key(runtime, "follower").tools()]
                ).get_tool_lookups()
            )

            with pytest.raises(MCPConnectionCapacityError):
                await head
            await busy.close()
            await asyncio.wait_for(follower, timeout=2.0)

            assert _opened_keys() == ["c0", "follower"]

    @pytest.mark.asyncio
    async def test_last_release_triggers_pressure_retirement(self) -> None:
        async with MCPRuntime(
            max_connections=1,
            connection_wait_timeout=5.0,
            idle_timeout=None,  # capacity pressure must retire regardless
            shutdown_timeout=0.05,
        ) as runtime:
            busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
            await busy.get_tool_lookups()

            waiter = asyncio.create_task(
                Agent(
                    prompt="waiter", tool_sources=[_bind_key(runtime, "c1").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)
            assert not waiter.done()

            await busy.close()
            await asyncio.wait_for(waiter, timeout=2.0)

            assert _InstrumentedAdapter.instances[0].closed is True  # c0 retired
            assert _opened_keys() == ["c0", "c1"]

    @pytest.mark.asyncio
    async def test_pressure_retires_the_oldest_idle_connection(self) -> None:
        async with MCPRuntime(
            max_connections=2,
            connection_wait_timeout=5.0,
            idle_timeout=None,
            shutdown_timeout=0.05,
        ) as runtime:
            older = Agent(prompt="older", tool_sources=[_bind_key(runtime, "older").tools()])
            newer = Agent(prompt="newer", tool_sources=[_bind_key(runtime, "newer").tools()])
            await older.get_tool_lookups()
            await newer.get_tool_lookups()

            await older.close()  # idle first
            await asyncio.sleep(0.01)
            await newer.close()  # idle second

            await asyncio.wait_for(
                Agent(
                    prompt="third", tool_sources=[_bind_key(runtime, "third").tools()]
                ).get_tool_lookups(),
                timeout=2.0,
            )

            by_key = dict(zip(_opened_keys(), _InstrumentedAdapter.instances, strict=False))
            assert by_key["older"].closed is True
            assert by_key["newer"].closed is False

    @pytest.mark.asyncio
    async def test_pressure_retirement_preserves_the_registration(self) -> None:
        async with MCPRuntime(
            max_connections=1,
            connection_wait_timeout=5.0,
            idle_timeout=None,
            shutdown_timeout=0.05,
        ) as runtime:
            retired = _bind_key(runtime, "c0")
            first = Agent(prompt="first", tool_sources=[retired.tools()])
            await first.get_tool_lookups()
            await first.close()

            pressure = Agent(prompt="pressure", tool_sources=[_bind_key(runtime, "c1").tools()])
            await asyncio.wait_for(pressure.get_tool_lookups(), timeout=2.0)
            await pressure.close()

            # Retirement is not eviction: the original handle still works.
            reused = Agent(prompt="reused", tool_sources=[retired.tools()])
            lookups = await asyncio.wait_for(reused.get_tool_lookups(), timeout=2.0)

            assert sorted(lookups.fn) == ["github__read", "github__write"]
            await reused.close()

    @pytest.mark.asyncio
    async def test_a_failed_connection_releases_its_slot(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        _InstrumentedAdapter.fail_urls = {"https://mcp.example.com/broken"}
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            failing = asyncio.create_task(
                Agent(
                    prompt="failing", tool_sources=[_bind_key(runtime, "broken").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(
                Agent(
                    prompt="queued", tool_sources=[_bind_key(runtime, "healthy").tools()]
                ).get_tool_lookups()
            )
            await asyncio.sleep(0.05)
            assert not queued.done()

            _InstrumentedAdapter.connect_gate.set()
            with pytest.raises(ConnectionError):
                await failing

            # The slot the doomed handshake held must free immediately.
            await asyncio.wait_for(queued, timeout=2.0)
            assert _opened_keys() == ["broken", "healthy"]

    @pytest.mark.asyncio
    async def test_wait_timeout_excludes_the_handshake(self) -> None:
        _InstrumentedAdapter.connect_gate = asyncio.Event()
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=0.05, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="slow", tool_sources=[_bind_key(runtime, "c0").tools()])
            discovery = asyncio.create_task(agent.get_tool_lookups())

            # Admitted immediately; the slow handshake must not be charged
            # against the capacity wait budget.
            await asyncio.sleep(0.2)
            assert not discovery.done()

            _InstrumentedAdapter.connect_gate.set()
            lookups = await asyncio.wait_for(discovery, timeout=1.0)
            assert sorted(lookups.fn) == ["github__read", "github__write"]
            await agent.close()

    @pytest.mark.asyncio
    async def test_tenants_hold_independent_physical_entries(self) -> None:
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=0.05, shutdown_timeout=0.05
        ) as runtime:
            tenant_a = Agent(
                prompt="a",
                tool_sources=[_bind_key(runtime, "github", tenant_key="tenant-a").tools()],
            )
            await tenant_a.get_tool_lookups()

            # Same connection key, different tenant: a separate connection that
            # cannot reuse tenant-a's, so it must queue and time out.
            tenant_b = Agent(
                prompt="b",
                tool_sources=[_bind_key(runtime, "github", tenant_key="tenant-b").tools()],
            )
            with pytest.raises(MCPConnectionCapacityError) as excinfo:
                await tenant_b.get_tool_lookups()

            assert excinfo.value.identity == ("tenant-b", "github")
            assert len(_InstrumentedAdapter.instances) == 1
            await tenant_a.close()

    @pytest.mark.asyncio
    async def test_resistant_cleanup_releases_logical_capacity(self) -> None:
        _InstrumentedAdapter.close_gate = asyncio.Event()  # never released
        runtime = MCPRuntime(
            max_connections=1,
            connection_wait_timeout=5.0,
            idle_timeout=None,
            shutdown_timeout=0.05,
        )
        busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
        await busy.get_tool_lookups()
        await busy.close()

        waiter = Agent(prompt="waiter", tool_sources=[_bind_key(runtime, "c1").tools()])

        # Pressure retirement hands the stuck close to background tracking;
        # the logical slot must free at handoff, not at socket close.
        await asyncio.wait_for(waiter.get_tool_lookups(), timeout=2.0)

        assert len(runtime._abandoned_tasks) == 1
        assert _opened_keys() == ["c0", "c1"]
        _InstrumentedAdapter.close_gate.set()
        await waiter.close()
        await asyncio.wait_for(runtime.close(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_pressure_retirement_does_not_arm_an_idle_timer_too(self) -> None:
        _InstrumentedAdapter.close_gate = asyncio.Event()
        runtime = MCPRuntime(
            max_connections=1,
            connection_wait_timeout=5.0,
            idle_timeout=300.0,
            shutdown_timeout=0.05,
        )
        first = Agent(prompt="first", tool_sources=[_bind_key(runtime, "first").tools()])
        await first.get_tool_lookups()
        entry = runtime._entries[(None, "first")]
        waiting_agent = Agent(
            prompt="waiting",
            tool_sources=[_bind_key(runtime, "waiting").tools()],
        )
        waiting = asyncio.create_task(waiting_agent.get_tool_lookups())
        await _wait_until(lambda: bool(runtime._admission_queue))

        await first.close()
        await _wait_until(lambda: entry.retire_task is not None)

        assert entry.idle_handle is None
        _InstrumentedAdapter.close_gate.set()
        await asyncio.wait_for(waiting, timeout=2.0)
        await waiting_agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_evicting_a_queued_identity_wakes_it_immediately(self) -> None:
        runtime = MCPRuntime(
            max_connections=1,
            connection_wait_timeout=5.0,
            shutdown_timeout=0.05,
        )
        busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "busy").tools()])
        await busy.get_tool_lookups()
        queued_agent = Agent(
            prompt="queued",
            tool_sources=[_bind_key(runtime, "queued").tools()],
        )
        queued = asyncio.create_task(queued_agent.get_tool_lookups())
        await _wait_until(lambda: bool(runtime._admission_queue))

        await runtime.evict(connection_key="queued", mode="force")

        with pytest.raises(MCPStaleConnectionError):
            await asyncio.wait_for(queued, timeout=0.5)
        assert runtime._admission_queue == deque()
        assert runtime._admissions == {}
        await queued_agent.close()
        await busy.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_live_eviction_releases_capacity_for_the_fifo_head(self) -> None:
        runtime = MCPRuntime(
            max_connections=1,
            connection_wait_timeout=5.0,
            shutdown_timeout=0.05,
        )
        busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "busy").tools()])
        await busy.get_tool_lookups()
        queued_agent = Agent(
            prompt="queued",
            tool_sources=[_bind_key(runtime, "queued").tools()],
        )
        queued = asyncio.create_task(queued_agent.get_tool_lookups())
        await _wait_until(lambda: bool(runtime._admission_queue))

        evicting = asyncio.create_task(
            runtime.evict(connection_key="busy", mode="drain", timeout=1.0)
        )
        await _wait_until(lambda: (None, "busy") in runtime._evictions)
        assert not queued.done()
        await busy.close()
        await asyncio.wait_for(evicting, timeout=2.0)
        lookups = await asyncio.wait_for(queued, timeout=2.0)

        assert sorted(lookups.fn) == ["github__read", "github__write"]
        assert _opened_keys() == ["busy", "queued"]
        await queued_agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_shutdown_wakes_capacity_waiters(self) -> None:
        runtime = MCPRuntime(max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05)
        busy = Agent(prompt="busy", tool_sources=[_bind_key(runtime, "c0").tools()])
        await busy.get_tool_lookups()

        waiter = asyncio.create_task(
            Agent(
                prompt="waiter", tool_sources=[_bind_key(runtime, "c1").tools()]
            ).get_tool_lookups()
        )
        await asyncio.sleep(0.05)
        assert not waiter.done()

        closing = asyncio.create_task(runtime.close())
        with pytest.raises(MCPRuntimeClosedError):
            await asyncio.wait_for(waiter, timeout=2.0)

        await busy.close()
        await asyncio.wait_for(closing, timeout=2.0)
        assert runtime._admission_queue == deque()

    @pytest.mark.asyncio
    async def test_shutdown_fences_new_admission_before_it_yields(self) -> None:
        runtime = MCPRuntime(max_connections=1, shutdown_timeout=0.05)
        connection = _bind_key(runtime, "late")
        start = asyncio.Event()

        async def close_runtime() -> None:
            await start.wait()
            await runtime.close()

        async def open_connection() -> Any:
            await start.wait()
            return await Agent(
                prompt="late",
                tool_sources=[connection.tools()],
            ).get_tool_lookups()

        closing = asyncio.create_task(close_runtime())
        opening = asyncio.create_task(open_connection())
        await asyncio.sleep(0)
        start.set()
        closed_result, open_result = await asyncio.gather(
            closing,
            opening,
            return_exceptions=True,
        )

        assert closed_result is None
        assert isinstance(open_result, MCPRuntimeClosedError)
        assert _InstrumentedAdapter.instances == []

    @pytest.mark.asyncio
    async def test_concurrent_opens_never_exceed_the_cap(self) -> None:
        async with MCPRuntime(
            max_connections=3, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            peak = 0

            async def one(index: int) -> None:
                nonlocal peak
                agent = Agent(
                    prompt=f"a{index}",
                    tool_sources=[_bind_key(runtime, f"c{index}").tools()],
                )
                await agent.get_tool_lookups()
                peak = max(peak, len(runtime._entries))
                assert len(runtime._entries) <= 3
                await agent.close()

            await asyncio.wait_for(asyncio.gather(*(one(i) for i in range(12))), timeout=10.0)

            assert peak == 3  # the cap is actually reached, not merely respected

    def test_connection_wait_timeout_is_validated(self) -> None:
        for invalid in (-1.0, float("inf"), "5"):
            with pytest.raises(ValueError, match="connection_wait_timeout"):
                MCPRuntime(connection_wait_timeout=invalid)  # type: ignore[arg-type]


class TestCallConcurrency:
    """max_in_flight_calls caps executing tool calls across the whole runtime."""

    @pytest.mark.asyncio
    async def test_cap_limits_simultaneously_executing_calls(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=2, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            call = lookups.fn["github__read"]

            pending = [asyncio.create_task(call(value=index)) for index in range(5)]
            await _wait_until(lambda: len(runtime._call_queue) == 3)

            assert runtime._in_flight_calls == 2
            assert _total_calls() == 2  # three never reached the transport

            gate.set()
            await asyncio.wait_for(asyncio.gather(*pending), timeout=2.0)

            assert _InstrumentedAdapter.peak_in_call == 2
            assert _total_calls() == 5
            assert runtime._in_flight_calls == 0
            assert runtime._call_queue == deque()
            await agent.close()

    @pytest.mark.asyncio
    async def test_queued_calls_start_in_fifo_order(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            call = lookups.fn["github__read"]

            pending = [asyncio.create_task(call(value="first"))]
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            for position, label in enumerate(("second", "third", "fourth"), start=1):
                pending.append(asyncio.create_task(call(value=label)))
                await _wait_until(lambda queued=position: len(runtime._call_queue) == queued)

            gate.set()
            await asyncio.wait_for(asyncio.gather(*pending), timeout=2.0)

            assert _InstrumentedAdapter.call_log == ["first", "second", "third", "fourth"]
            await agent.close()

    @pytest.mark.asyncio
    async def test_wait_timeout_raises_a_typed_capacity_error(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=0.05, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            busy = asyncio.create_task(lookups.fn["github__read"](value="busy"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)

            with pytest.raises(MCPCallCapacityError) as excinfo:
                await lookups.fn["github__write"](value="shed")

            assert excinfo.value.identity == (None, "github-1")
            assert excinfo.value.tool == "github__write"
            assert excinfo.value.limit == 1
            assert excinfo.value.timeout == 0.05
            assert isinstance(excinfo.value, MCPCapacityError)
            # Shedding is not a failed call: nothing was attempted server-side.
            assert not isinstance(excinfo.value, MCPToolCallError)
            assert _total_calls() == 1
            assert runtime._call_queue == deque()
            assert runtime._in_flight_calls == 1  # the busy call kept its slot

            entry = agent._tool_sources[0]._entry
            assert entry is not None
            assert entry.active_calls == 1  # the shed call never counted

            gate.set()
            await asyncio.wait_for(busy, timeout=1.0)
            await agent.close()

    @pytest.mark.asyncio
    async def test_zero_call_wait_timeout_sheds_immediately(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            busy = asyncio.create_task(lookups.fn["github__read"](value="busy"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)

            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises(MCPCallCapacityError):
                await lookups.fn["github__write"](value="shed")

            assert loop.time() - started < 0.05  # never queued
            gate.set()
            await asyncio.wait_for(busy, timeout=1.0)
            await agent.close()

    @pytest.mark.asyncio
    async def test_a_failed_call_releases_its_slot(self) -> None:
        _InstrumentedAdapter.call_error = RuntimeError("server exploded")
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()

            # With a single slot and zero wait budget, one leaked slot would
            # turn the next call into a capacity error instead of a retry.
            for _ in range(3):
                with pytest.raises(MCPToolCallError):
                    await lookups.fn["github__write"](value="boom")
                assert runtime._in_flight_calls == 0

            _InstrumentedAdapter.call_error = None
            assert await lookups.fn["github__write"](value="ok") == "write:ok"
            assert runtime._in_flight_calls == 0
            await agent.close()

    @pytest.mark.asyncio
    async def test_cancelling_a_queued_call_frees_its_position(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            call = lookups.fn["github__read"]

            busy = asyncio.create_task(call(value="busy"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            abandoned = asyncio.create_task(call(value="abandoned"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)
            follower = asyncio.create_task(call(value="follower"))
            await _wait_until(lambda: len(runtime._call_queue) == 2)

            abandoned.cancel()
            await asyncio.gather(abandoned, return_exceptions=True)
            await _wait_until(lambda: len(runtime._call_queue) == 1)

            gate.set()
            await asyncio.wait_for(asyncio.gather(busy, follower), timeout=2.0)

            # The cancelled position neither ran nor wedged the one behind it.
            assert _InstrumentedAdapter.call_log == ["busy", "follower"]
            assert runtime._in_flight_calls == 0
            await agent.close()

    @pytest.mark.asyncio
    async def test_cancelling_an_executing_call_releases_its_slot(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            call = lookups.fn["github__read"]

            running = asyncio.create_task(call(value="running"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            queued = asyncio.create_task(call(value="queued"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)

            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

            gate.set()
            assert await asyncio.wait_for(queued, timeout=1.0) == "read:ok"
            assert runtime._in_flight_calls == 0
            entry = agent._tool_sources[0]._entry
            assert entry is not None
            assert entry.active_calls == 0
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_close_invalidates_its_queued_call(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            connection = _bind(runtime)
            busy_agent = Agent(prompt="busy", tool_sources=[connection.tools()])
            busy_lookups = await busy_agent.get_tool_lookups()
            queued_agent = Agent(prompt="queued", tool_sources=[connection.tools(namespace="gh")])
            queued_lookups = await queued_agent.get_tool_lookups()

            busy = asyncio.create_task(busy_lookups.fn["github__read"](value="busy"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            queued = asyncio.create_task(queued_lookups.fn["gh__read"](value="queued"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)

            await queued_agent.close()

            with pytest.raises(MCPToolCallError):
                await asyncio.wait_for(queued, timeout=1.0)
            assert runtime._call_queue == deque()

            gate.set()
            await asyncio.wait_for(busy, timeout=1.0)
            assert _InstrumentedAdapter.call_log == ["busy"]
            await busy_agent.close()

    @pytest.mark.asyncio
    async def test_agent_close_wakes_only_its_own_shared_connection_waiters(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            connection = _bind(runtime)
            busy_agent = Agent(prompt="busy", tool_sources=[connection.tools(namespace="busy")])
            closing_agent = Agent(
                prompt="closing", tool_sources=[connection.tools(namespace="closing")]
            )
            surviving_agent = Agent(
                prompt="surviving", tool_sources=[connection.tools(namespace="surviving")]
            )
            busy_call = (await busy_agent.get_tool_lookups()).fn["busy__read"]
            closing_call = (await closing_agent.get_tool_lookups()).fn["closing__read"]
            surviving_call = (await surviving_agent.get_tool_lookups()).fn["surviving__read"]

            busy = asyncio.create_task(busy_call(value="busy"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            doomed = asyncio.create_task(closing_call(value="closing"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)
            survivor = asyncio.create_task(surviving_call(value="surviving"))
            await _wait_until(lambda: len(runtime._call_queue) == 2)
            doomed_waiter, surviving_waiter = runtime._call_queue

            await closing_agent.close()

            assert doomed_waiter.event.is_set()
            assert not surviving_waiter.event.is_set()
            with pytest.raises(MCPToolCallError):
                await asyncio.wait_for(doomed, timeout=1.0)
            assert list(runtime._call_queue) == [surviving_waiter]
            assert not surviving_waiter.event.is_set()

            gate.set()
            await asyncio.wait_for(asyncio.gather(busy, survivor), timeout=2.0)
            assert _InstrumentedAdapter.call_log == ["busy", "surviving"]
            await busy_agent.close()
            await surviving_agent.close()

    @pytest.mark.asyncio
    async def test_eviction_rejects_only_its_own_queued_calls(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_connections=5,
            max_in_flight_calls=1,
            call_wait_timeout=5.0,
            shutdown_timeout=0.05,
        ) as runtime:
            agents = []
            calls = {}
            for key in ("busy", "evicted", "spared"):
                agent = Agent(prompt=key, tool_sources=[_bind_key(runtime, key).tools()])
                lookups = await agent.get_tool_lookups()
                agents.append(agent)
                calls[key] = lookups.fn["github__read"]

            busy = asyncio.create_task(calls["busy"](value="busy"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            doomed = asyncio.create_task(calls["evicted"](value="evicted"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)
            spared = asyncio.create_task(calls["spared"](value="spared"))
            await _wait_until(lambda: len(runtime._call_queue) == 2)

            await asyncio.wait_for(
                runtime.evict(connection_key="evicted", mode="force", timeout=0.5),
                timeout=2.0,
            )

            with pytest.raises((MCPConnectionEvictingError, MCPStaleConnectionError)) as excinfo:
                await asyncio.wait_for(doomed, timeout=1.0)
            # It never started, so its outcome is known: it did not happen.
            assert not isinstance(excinfo.value, MCPOutcomeUnknownError)
            assert not isinstance(excinfo.value, MCPToolCallError)

            gate.set()
            await asyncio.wait_for(asyncio.gather(busy, spared), timeout=2.0)

            assert _InstrumentedAdapter.call_log == ["busy", "spared"]
            for agent in agents:
                await agent.close()

    @pytest.mark.asyncio
    async def test_shutdown_wakes_queued_calls_with_a_closed_error(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        runtime = MCPRuntime(max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        lookups = await agent.get_tool_lookups()
        call = lookups.fn["github__read"]

        busy = asyncio.create_task(call(value="busy"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)
        queued = asyncio.create_task(call(value="queued"))
        await _wait_until(lambda: len(runtime._call_queue) == 1)

        closing = asyncio.create_task(runtime.close())
        with pytest.raises(MCPRuntimeClosedError):
            await asyncio.wait_for(queued, timeout=2.0)

        gate.set()
        await asyncio.gather(busy, return_exceptions=True)
        await asyncio.wait_for(closing, timeout=2.0)

        assert runtime._call_queue == deque()
        assert _InstrumentedAdapter.call_log == ["busy"]
        await agent.close()

    @pytest.mark.asyncio
    async def test_call_slots_are_global_not_per_connection(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_connections=4,
            max_in_flight_calls=1,
            call_wait_timeout=5.0,
            shutdown_timeout=0.05,
        ) as runtime:
            agents = []
            calls = {}
            for tenant, key in (("tenant-a", "alpha"), ("tenant-b", "beta")):
                agent = Agent(
                    prompt=key,
                    tool_sources=[_bind_key(runtime, key, tenant_key=tenant).tools()],
                )
                lookups = await agent.get_tool_lookups()
                agents.append(agent)
                calls[key] = lookups.fn["github__read"]

            first = asyncio.create_task(calls["alpha"](value="alpha"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            second = asyncio.create_task(calls["beta"](value="beta"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)

            # Two live connections, one shared call budget.
            assert len(runtime._entries) == 2
            assert _total_calls() == 1

            gate.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=2.0)

            assert _InstrumentedAdapter.peak_in_call == 1
            for agent in agents:
                await agent.close()

    @pytest.mark.asyncio
    async def test_force_eviction_reclaims_slot_from_a_resistant_call(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        old_gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = old_gate
        _InstrumentedAdapter.suppress_call_cancel = True
        runtime = MCPRuntime(
            max_connections=2,
            max_in_flight_calls=1,
            call_wait_timeout=0.1,
            shutdown_timeout=0.05,
        )
        old_agent = Agent(
            prompt="old",
            tool_sources=[
                _bind_key(
                    runtime,
                    "private-connection-key",
                    tenant_key="alice@example.com",
                ).tools()
            ],
        )
        old_call = (await old_agent.get_tool_lookups()).fn["github__write"]
        resistant = asyncio.create_task(old_call(value="old"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        await asyncio.wait_for(
            runtime.evict(
                tenant_key="alice@example.com",
                connection_key="private-connection-key",
                mode="force",
            ),
            timeout=1.0,
        )

        assert not resistant.done()
        assert runtime._in_flight_calls == 0
        assert runtime._active_call_permits == set()
        warnings = [
            record.getMessage()
            for record in caplog.records
            if "ignored forced cancellation" in record.getMessage()
        ]
        assert warnings == [
            "1 MCP tool call(s) on source(s) github ignored forced "
            "cancellation; releasing their runtime slots while their callers remain "
            "responsible for eventual task completion."
        ]
        assert "alice@example.com" not in warnings[0]
        assert "private-connection-key" not in warnings[0]

        _InstrumentedAdapter.call_gate = None
        new_agent = Agent(prompt="new", tool_sources=[_bind_key(runtime, "new").tools()])
        new_call = (await new_agent.get_tool_lookups()).fn["github__read"]
        assert await new_call(value="new") == "read:ok"
        assert runtime._in_flight_calls == 0

        old_gate.set()
        with pytest.raises(MCPOutcomeUnknownError):
            await resistant
        await asyncio.sleep(0)
        assert runtime._in_flight_calls == 0
        assert runtime._abandoned_tasks == set()
        await old_agent.close()
        await new_agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_resistant_success_after_force_restores_cancellation_state(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        _InstrumentedAdapter.suppress_call_cancel = True
        _InstrumentedAdapter.return_after_close = True
        runtime = MCPRuntime(max_in_flight_calls=1, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        running = asyncio.create_task(call(value="mutating"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        await runtime.evict(connection_key="github-1", mode="force")
        assert not running.done()
        assert running.cancelling() == 1
        gate.set()

        assert await running == "write:ok"
        assert running.cancelling() == 0
        assert runtime._in_flight_calls == 0
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_application_cancel_wins_when_resistant_call_returns(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        _InstrumentedAdapter.suppress_call_cancel = True
        _InstrumentedAdapter.return_after_close = True
        runtime = MCPRuntime(max_in_flight_calls=1, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        running = asyncio.create_task(call(value="mutating"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        await runtime.evict(connection_key="github-1", mode="force")
        running.cancel()  # independent application request
        gate.set()

        with pytest.raises(asyncio.CancelledError):
            await running
        assert running.cancelling() == 1
        assert runtime._in_flight_calls == 0
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_eviction_racing_shutdown_interrupts_a_call_only_once(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        _InstrumentedAdapter.suppress_call_cancel = True
        runtime = MCPRuntime(shutdown_timeout=0.2)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        running = asyncio.create_task(call(value="mutating"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        # Forced eviction and forced shutdown both reach this permit while it
        # is still held, so both would cancel it without the interrupt guard.
        evicting = asyncio.create_task(
            runtime.evict(connection_key="github-1", mode="force", timeout=0.5)
        )
        await asyncio.sleep(0)  # let the eviction reach its grace window
        closing = asyncio.create_task(runtime.close())
        await asyncio.wait_for(asyncio.gather(evicting, closing), timeout=2.0)

        # Read before releasing the gate: the count only drops once the
        # executor unwinds and calls uncancel().
        cancel_requests = running.cancelling()
        gate.set()  # the stubborn transport finally returns

        # Nothing above may assert: this transport swallows cancellation, so a
        # failure before the gate opens would spin the loop during teardown
        # instead of reporting. Capture, then judge.
        outcome: BaseException | None = None
        try:
            await asyncio.wait_for(asyncio.shield(running), timeout=1.0)
        except BaseException as exc:  # noqa: BLE001
            outcome = exc

        assert cancel_requests == 1  # exactly one runtime-issued request
        # A second cancellation surfaces as CancelledError, telling the caller
        # nothing happened when the server may already have applied it.
        assert isinstance(outcome, MCPOutcomeUnknownError), f"got {outcome!r}"
        await agent.close()

    @pytest.mark.asyncio
    async def test_forced_call_restores_the_caller_cancellation_count(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime(max_in_flight_calls=1, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        converted = asyncio.Event()
        observed: dict[str, int] = {}

        async def caller() -> None:
            task = asyncio.current_task()
            assert task is not None
            async with asyncio.timeout(0.3):
                observed["before"] = task.cancelling()
                try:
                    await call(value="running")
                except MCPOutcomeUnknownError:
                    observed["after"] = task.cancelling()
                    converted.set()
                    await asyncio.sleep(10)

        task = asyncio.create_task(caller())
        await _wait_until(lambda: runtime._in_flight_calls == 1)
        await runtime.evict(connection_key="github-1", mode="force")
        await asyncio.wait_for(converted.wait(), timeout=1.0)

        assert observed == {"before": 0, "after": 0}
        with pytest.raises(TimeoutError):
            await task
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_application_cancellation_is_not_converted_during_force_state(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime(max_in_flight_calls=1, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        task = asyncio.create_task(call(value="running"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)
        entry = agent._tool_sources[0]._entry
        assert entry is not None

        # The connection can become forced immediately before an unrelated
        # application cancellation reaches the executor. Without an explicit
        # runtime-interruption marker, that cancellation must remain one.
        with runtime._lock:
            entry.force_evicted = True
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime._in_flight_calls == 0
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_application_cancellation_survives_runtime_interruption(self) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        _InstrumentedAdapter.call_cancel_seen = asyncio.Event()
        _InstrumentedAdapter.call_cancel_pause = asyncio.Event()
        runtime = MCPRuntime(max_in_flight_calls=1, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        task = asyncio.create_task(call(value="running"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        evicting = asyncio.create_task(runtime.evict(connection_key="github-1", mode="force"))
        assert _InstrumentedAdapter.call_cancel_seen is not None
        await asyncio.wait_for(_InstrumentedAdapter.call_cancel_seen.wait(), timeout=1.0)
        task.cancel()  # application cancellation, independent of the runtime's

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(evicting, timeout=1.0)
        assert task.cancelling() == 1
        assert runtime._in_flight_calls == 0
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_force_waits_for_permit_release_not_caller_completion(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _InstrumentedAdapter.call_gate = asyncio.Event()
        runtime = MCPRuntime(max_in_flight_calls=1, shutdown_timeout=0.05)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        converted = asyncio.Event()
        finish_caller = asyncio.Event()

        async def caller() -> None:
            try:
                await call(value="running")
            except MCPOutcomeUnknownError:
                converted.set()
                await finish_caller.wait()

        task = asyncio.create_task(caller())
        await _wait_until(lambda: runtime._in_flight_calls == 1)
        await runtime.evict(connection_key="github-1", mode="force")
        await asyncio.wait_for(converted.wait(), timeout=1.0)

        assert not task.done()
        assert task.cancelling() == 0
        assert runtime._in_flight_calls == 0
        assert runtime._abandoned_tasks == set()
        assert not any(
            "tool call ignored forced cancellation" in record.getMessage()
            for record in caplog.records
        )

        finish_caller.set()
        await task
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_late_abandoned_call_cannot_release_a_newer_slot(self) -> None:
        old_gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = old_gate
        _InstrumentedAdapter.suppress_call_cancel = True
        runtime = MCPRuntime(
            max_connections=2,
            max_in_flight_calls=1,
            call_wait_timeout=1.0,
            shutdown_timeout=0.05,
        )
        old_agent = Agent(prompt="old", tool_sources=[_bind_key(runtime, "old").tools()])
        old_call = (await old_agent.get_tool_lookups()).fn["github__write"]
        abandoned = asyncio.create_task(old_call(value="old"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)
        await runtime.evict(connection_key="old", mode="force")

        new_gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = new_gate
        new_agent = Agent(prompt="new", tool_sources=[_bind_key(runtime, "new").tools()])
        new_call = (await new_agent.get_tool_lookups()).fn["github__read"]
        current = asyncio.create_task(new_call(value="new"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        old_gate.set()
        with pytest.raises(MCPOutcomeUnknownError):
            await abandoned
        assert runtime._in_flight_calls == 1
        assert not current.done()

        new_gate.set()
        assert await current == "read:ok"
        assert runtime._in_flight_calls == 0
        await old_agent.close()
        await new_agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_forced_shutdown_abandons_resistant_call_permits(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        _InstrumentedAdapter.suppress_call_cancel = True
        runtime = MCPRuntime(
            max_in_flight_calls=1,
            call_wait_timeout=1.0,
            shutdown_timeout=0.01,
        )
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        call = (await agent.get_tool_lookups()).fn["github__write"]
        resistant = asyncio.create_task(call(value="running"))
        await _wait_until(lambda: runtime._in_flight_calls == 1)

        await asyncio.wait_for(runtime.close(), timeout=1.0)

        assert not resistant.done()
        assert runtime._in_flight_calls == 0
        assert runtime._active_call_permits == set()
        gate.set()
        with pytest.raises(MCPOutcomeUnknownError):
            await resistant
        await asyncio.sleep(0)
        assert runtime._in_flight_calls == 0
        assert runtime._abandoned_tasks == set()
        await agent.close()

    @pytest.mark.asyncio
    async def test_the_wait_budget_does_not_bound_the_call_itself(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=2, call_wait_timeout=0.02, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()

            slow = asyncio.create_task(lookups.fn["github__read"](value="slow"))
            await _wait_until(lambda: runtime._in_flight_calls == 1)
            await asyncio.sleep(0.1)  # far past the admission budget

            assert not slow.done()
            gate.set()
            assert await asyncio.wait_for(slow, timeout=1.0) == "read:ok"
            await agent.close()

    @pytest.mark.asyncio
    async def test_concurrent_calls_never_exceed_the_cap(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=3, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            call = lookups.fn["github__read"]

            pending = [asyncio.create_task(call(value=index)) for index in range(24)]
            await _wait_until(lambda: len(runtime._call_queue) == 21)

            gate.set()
            await asyncio.wait_for(asyncio.gather(*pending), timeout=5.0)

            assert _InstrumentedAdapter.peak_in_call == 3  # reached, never exceeded
            assert _total_calls() == 24
            assert runtime._in_flight_calls == 0
            assert runtime._call_queue == deque()
            await agent.close()

    def test_call_wait_timeout_is_validated(self) -> None:
        for invalid in (-1.0, float("nan"), "5"):
            with pytest.raises(ValueError, match="call_wait_timeout"):
                MCPRuntime(call_wait_timeout=invalid)  # type: ignore[arg-type]


class TestBrokenConnectionRecovery:
    """A dead transport is fenced once, retired, and replaced lazily.

    The failed in-flight call reports an unknown outcome — the server may
    already have received it — and is never retried by the runtime. Recovery
    only clears the way for future work: the registration and generation
    survive, so the same handle reconnects with freshly resolved credentials
    on its next acquisition.
    """

    @pytest.mark.asyncio
    async def test_transport_loss_mid_call_reports_unknown_outcome(self) -> None:
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()

            with pytest.raises(MCPOutcomeUnknownError) as excinfo:
                await lookups.fn["github__write"](value=1)

            assert "must not be retried" in str(excinfo.value)
            assert isinstance(excinfo.value.__cause__, MCPToolCallError)
            assert runtime._in_flight_calls == 0
            await agent.close()

    @pytest.mark.asyncio
    async def test_next_acquisition_reconnects_after_recovery(self) -> None:
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            lookups = await first.get_tool_lookups()
            broken = _InstrumentedAdapter.instances[0]

            with pytest.raises(MCPOutcomeUnknownError):
                await lookups.fn["github__write"](value=1)

            await _wait_until(lambda: broken.closed and runtime._entries == {})
            _InstrumentedAdapter.call_error = None

            second = Agent(prompt="second", tool_sources=[connection.tools()])
            fresh = await second.get_tool_lookups()
            assert await fresh.fn["github__write"](value=2) == "write:ok"
            assert len(_InstrumentedAdapter.instances) == 2

            # The first Agent's view is still bound to the dead transport:
            # its retained executor fails typed, and nothing was ever sent.
            with pytest.raises(MCPConnectionLostError) as excinfo:
                await lookups.fn["github__write"](value=3)
            assert excinfo.value.identity == (None, "github-1")
            assert not isinstance(excinfo.value, MCPToolCallError)
            assert broken.call_tool_calls == 1

            await first.close()
            await second.close()

    @pytest.mark.asyncio
    async def test_recovery_resolves_fresh_credentials_for_the_same_handle(self) -> None:
        provider = _Credentials(
            {"Authorization": "Bearer token-1"},
            {"Authorization": "Bearer token-2"},
        )
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime, credentials=provider)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            lookups = await first.get_tool_lookups()
            opened = _InstrumentedAdapter.instances[0]
            assert opened.source.headers["Authorization"] == "Bearer token-1"

            with pytest.raises(MCPOutcomeUnknownError):
                await lookups.fn["github__write"](value=1)
            await _wait_until(lambda: runtime._entries == {})
            _InstrumentedAdapter.call_error = None

            second = Agent(prompt="second", tool_sources=[connection.tools()])
            await second.get_tool_lookups()

            assert provider.calls == 2
            replacement = _InstrumentedAdapter.instances[1]
            assert replacement.source.headers["Authorization"] == "Bearer token-2"
            await first.close()
            await second.close()

    @pytest.mark.asyncio
    async def test_concurrent_failures_share_one_recovery(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            adapter = _InstrumentedAdapter.instances[0]

            calls = [
                asyncio.create_task(lookups.fn["github__write"](value=index)) for index in range(10)
            ]
            await _wait_until(lambda: _InstrumentedAdapter.in_call == 10)
            _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
            gate.set()

            results = await asyncio.gather(*calls, return_exceptions=True)
            assert all(isinstance(result, MCPOutcomeUnknownError) for result in results)

            await _wait_until(lambda: adapter.closed and runtime._entries == {})
            assert adapter.close_calls == 1
            assert len(_InstrumentedAdapter.instances) == 1
            await agent.close()

    @pytest.mark.asyncio
    async def test_recovery_interrupts_other_in_flight_calls_as_outcome_unknown(self) -> None:
        """Closing one shared SDK client can cancel its other request waiters;
        those calls were admitted and must not leak bare cancellation."""

        class _SiblingAdapter(_InstrumentedAdapter):
            sibling_started = asyncio.Event()

            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any],
            ) -> CallToolResult:
                self.call_tool_calls += 1
                if arguments["value"] == "dies":
                    await self.sibling_started.wait()
                    raise ConnectionResetError("socket died")
                self.sibling_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        with patch("dendrux.mcp._runtime.MCPClientAdapter", _SiblingAdapter):
            async with MCPRuntime(shutdown_timeout=0.05) as runtime:
                agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
                lookups = await agent.get_tool_lookups()

                sibling = asyncio.create_task(lookups.fn["github__write"](value="sibling"))
                await _SiblingAdapter.sibling_started.wait()
                failing = asyncio.create_task(lookups.fn["github__write"](value="dies"))

                results = await asyncio.gather(failing, sibling, return_exceptions=True)

                assert all(isinstance(result, MCPOutcomeUnknownError) for result in results)
                assert all("lost connection" in str(result) for result in results)
                assert runtime._in_flight_calls == 0
                await agent.close()

    @pytest.mark.asyncio
    async def test_recovery_reclaims_a_resistant_sibling_call_permit(self) -> None:
        """A request that ignores the recovery cancellation cannot retain a
        global call slot or delay opening the replacement connection."""

        class _ResistantSiblingAdapter(_InstrumentedAdapter):
            sibling_started = asyncio.Event()
            release_sibling = asyncio.Event()

            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any],
            ) -> CallToolResult:
                self.call_tool_calls += 1
                if arguments["value"] == "dies":
                    await self.sibling_started.wait()
                    raise ConnectionResetError("socket died")
                self.sibling_started.set()
                while True:
                    try:
                        await self.release_sibling.wait()
                        return CallToolResult(content=[TextContent(text="late:ok")])
                    except asyncio.CancelledError:
                        continue

        with patch("dendrux.mcp._runtime.MCPClientAdapter", _ResistantSiblingAdapter):
            runtime = MCPRuntime(max_in_flight_calls=2, shutdown_timeout=0.05)
            connection = _bind(runtime)
            agent = Agent(prompt="test", tool_sources=[connection.tools()])
            lookups = await agent.get_tool_lookups()
            sibling = asyncio.create_task(lookups.fn["github__write"](value="sibling"))
            await _ResistantSiblingAdapter.sibling_started.wait()

            with pytest.raises(MCPOutcomeUnknownError):
                await lookups.fn["github__write"](value="dies")

            await _wait_until(lambda: runtime._entries == {})
            assert runtime._in_flight_calls == 0

            replacement = Agent(prompt="replacement", tool_sources=[connection.tools()])
            await asyncio.wait_for(replacement.get_tool_lookups(), timeout=1.0)

            _ResistantSiblingAdapter.release_sibling.set()
            assert await asyncio.wait_for(sibling, timeout=1.0) == "late:ok"
            await agent.close()
            await replacement.close()
            await runtime.close()

    @pytest.mark.asyncio
    async def test_recovery_restores_the_caller_cancellation_count(self) -> None:
        """Absorbing the runtime's cancellation must not poison an outer
        timeout or TaskGroup owned by the application."""

        class _SiblingAdapter(_InstrumentedAdapter):
            sibling_started = asyncio.Event()

            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any],
            ) -> CallToolResult:
                self.call_tool_calls += 1
                if arguments["value"] == "dies":
                    await self.sibling_started.wait()
                    raise ConnectionResetError("socket died")
                self.sibling_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        with patch("dendrux.mcp._runtime.MCPClientAdapter", _SiblingAdapter):
            runtime = MCPRuntime(shutdown_timeout=0.05)
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            call = (await agent.get_tool_lookups()).fn["github__write"]
            converted = asyncio.Event()
            finish = asyncio.Event()
            observed: dict[str, int] = {}

            async def caller() -> None:
                task = asyncio.current_task()
                assert task is not None
                observed["before"] = task.cancelling()
                with pytest.raises(MCPOutcomeUnknownError, match="lost connection"):
                    await call(value="sibling")
                observed["after"] = task.cancelling()
                converted.set()
                await finish.wait()

            sibling = asyncio.create_task(caller())
            await _SiblingAdapter.sibling_started.wait()
            with pytest.raises(MCPOutcomeUnknownError):
                await call(value="dies")
            await asyncio.wait_for(converted.wait(), timeout=1.0)

            assert observed == {"before": 0, "after": 0}
            finish.set()
            await sibling
            await agent.close()
            await runtime.close()

    @pytest.mark.asyncio
    async def test_application_cancellation_survives_recovery_interruption(self) -> None:
        """A user cancellation racing the runtime's recovery cancellation
        remains cancellation rather than becoming an MCP tool error."""

        class _SiblingAdapter(_InstrumentedAdapter):
            sibling_started = asyncio.Event()
            runtime_cancel_seen = asyncio.Event()

            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any],
            ) -> CallToolResult:
                self.call_tool_calls += 1
                if arguments["value"] == "dies":
                    await self.sibling_started.wait()
                    raise ConnectionResetError("socket died")
                self.sibling_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.runtime_cancel_seen.set()
                    await asyncio.Event().wait()
                    raise

        with patch("dendrux.mcp._runtime.MCPClientAdapter", _SiblingAdapter):
            runtime = MCPRuntime(shutdown_timeout=0.05)
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            call = (await agent.get_tool_lookups()).fn["github__write"]
            sibling = asyncio.create_task(call(value="sibling"))
            await _SiblingAdapter.sibling_started.wait()

            with pytest.raises(MCPOutcomeUnknownError):
                await call(value="dies")
            await asyncio.wait_for(_SiblingAdapter.runtime_cancel_seen.wait(), timeout=1.0)
            sibling.cancel()

            with pytest.raises(asyncio.CancelledError):
                await sibling
            assert sibling.cancelling() == 1
            assert runtime._in_flight_calls == 0
            await agent.close()
            await runtime.close()

    @pytest.mark.asyncio
    async def test_queued_calls_never_enter_the_broken_transport(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(
            max_in_flight_calls=1, call_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            adapter = _InstrumentedAdapter.instances[0]

            busy = asyncio.create_task(lookups.fn["github__write"](value="busy"))
            await _await_first_call(adapter)
            queued = asyncio.create_task(lookups.fn["github__write"](value="queued"))
            await _wait_until(lambda: len(runtime._call_queue) == 1)

            _InstrumentedAdapter.call_error = BrokenPipeError("pipe closed")
            gate.set()

            with pytest.raises(MCPOutcomeUnknownError):
                await busy
            # Well inside the 5s wait budget: the fence wakes the queue.
            with pytest.raises(MCPConnectionLostError):
                await asyncio.wait_for(queued, timeout=1.0)
            assert adapter.call_tool_calls == 1
            await agent.close()

    @pytest.mark.asyncio
    async def test_ordinary_tool_failures_do_not_poison_the_connection(self) -> None:
        _InstrumentedAdapter.call_error = RuntimeError("tool exploded")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()

            with pytest.raises(MCPToolCallError) as excinfo:
                await lookups.fn["github__write"](value=1)
            assert not isinstance(excinfo.value, MCPOutcomeUnknownError)

            entry = runtime._entries[(None, "github-1")]
            assert entry.broken is False
            assert entry.fenced is False

            _InstrumentedAdapter.call_error = None
            assert await lookups.fn["github__write"](value=2) == "write:ok"
            assert len(_InstrumentedAdapter.instances) == 1
            await agent.close()

    @pytest.mark.asyncio
    async def test_credential_rejection_mid_call_does_not_reconnect(self) -> None:
        class _RejectedError(RuntimeError):
            status_code = 401

        _InstrumentedAdapter.call_error = _RejectedError("401 Unauthorized")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()

            # A rejection arrives over a WORKING transport: refreshing
            # credentials is the application's move, not reconnection.
            for attempt in (1, 2):
                with pytest.raises(MCPToolCallError) as excinfo:
                    await lookups.fn["github__write"](value=attempt)
                assert not isinstance(excinfo.value, MCPOutcomeUnknownError)

            assert len(_InstrumentedAdapter.instances) == 1
            assert _InstrumentedAdapter.instances[0].closed is False
            assert _total_connects() == 1
            await agent.close()

    @pytest.mark.asyncio
    async def test_connect_time_rejection_does_not_reconnect_in_a_loop(self) -> None:
        _InstrumentedAdapter.connect_error = MCPAuthenticationError(
            "Failed to connect to MCP source 'github': "
            "the server rejected its credentials (HTTP 401)."
        )
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            for _ in range(2):
                agent = Agent(prompt="test", tool_sources=[connection.tools()])
                with pytest.raises(MCPAuthenticationError):
                    await agent.get_tool_lookups()

            # One connect attempt per explicit acquisition; the runtime never
            # retries a credential rejection on its own.
            assert _total_connects() == 2
            assert runtime._entries == {}

    @pytest.mark.asyncio
    async def test_application_cancellation_does_not_mark_the_connection_broken(self) -> None:
        gate = asyncio.Event()
        _InstrumentedAdapter.call_gate = gate
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            adapter = _InstrumentedAdapter.instances[0]

            call = asyncio.create_task(lookups.fn["github__write"](value="cancel-me"))
            await _await_first_call(adapter)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call

            entry = runtime._entries[(None, "github-1")]
            assert entry.broken is False
            assert entry.fenced is False

            gate.set()
            assert await lookups.fn["github__write"](value="next") == "write:ok"
            assert len(_InstrumentedAdapter.instances) == 1
            await agent.close()

    @pytest.mark.asyncio
    async def test_recovery_frees_the_connection_slot(self) -> None:
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        async with MCPRuntime(
            max_connections=1, connection_wait_timeout=5.0, shutdown_timeout=0.05
        ) as runtime:
            github = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[github.tools()])
            lookups = await first.get_tool_lookups()

            with pytest.raises(MCPOutcomeUnknownError):
                await lookups.fn["github__write"](value=1)
            _InstrumentedAdapter.call_error = None

            # The first Agent still holds its lease on the dead entry, yet
            # the slot frees for a different identity once recovery discards
            # the broken transport.
            jira = _bind(runtime, connection_key="jira-1")
            second = Agent(prompt="second", tool_sources=[jira.tools(namespace="jira")])
            fresh = await asyncio.wait_for(second.get_tool_lookups(), timeout=1.0)
            assert await fresh.fn["jira__write"](value=2) == "write:ok"
            await first.close()
            await second.close()

    @pytest.mark.asyncio
    async def test_recovery_racing_eviction_does_not_double_close(self) -> None:
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
            lookups = await agent.get_tool_lookups()
            adapter = _InstrumentedAdapter.instances[0]

            with pytest.raises(MCPOutcomeUnknownError):
                await lookups.fn["github__write"](value=1)
            await asyncio.wait_for(
                runtime.evict(connection_key="github-1", mode="force"),
                timeout=1.0,
            )
            await _wait_until(lambda: adapter.closed and not runtime._evictions)

            assert adapter.close_calls == 1
            assert runtime._entries == {}
            assert (None, "github-1") not in runtime._registrations
            with pytest.raises(MCPStaleConnectionError):
                await lookups.fn["github__write"](value="after-eviction")

            replacement_connection = _bind(runtime)
            replacement = Agent(
                prompt="replacement",
                tool_sources=[replacement_connection.tools()],
            )
            await replacement.get_tool_lookups()
            with pytest.raises(MCPStaleConnectionError):
                await lookups.fn["github__write"](value="after-rebind")
            await replacement.close()
            await agent.close()

    @pytest.mark.asyncio
    async def test_recovery_racing_shutdown_does_not_double_close(self) -> None:
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        runtime = MCPRuntime(shutdown_timeout=0.5)
        agent = Agent(prompt="test", tool_sources=[_bind(runtime).tools()])
        lookups = await agent.get_tool_lookups()
        adapter = _InstrumentedAdapter.instances[0]

        with pytest.raises(MCPOutcomeUnknownError):
            await lookups.fn["github__write"](value=1)
        await agent.close()
        await asyncio.wait_for(runtime.close(), timeout=1.0)

        assert adapter.close_calls == 1
        assert runtime.state is MCPRuntimeState.CLOSED

    @pytest.mark.asyncio
    async def test_resistant_transport_does_not_block_recovery(self) -> None:
        close_gate = asyncio.Event()
        _InstrumentedAdapter.close_gate = close_gate
        _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _bind(runtime)
            first = Agent(prompt="first", tool_sources=[connection.tools()])
            lookups = await first.get_tool_lookups()

            with pytest.raises(MCPOutcomeUnknownError):
                await lookups.fn["github__write"](value=1)

            # The broken transport never finishes closing, yet the slot frees
            # after the bounded grace and a replacement can open.
            await _wait_until(lambda: runtime._entries == {})
            _InstrumentedAdapter.call_error = None
            second = Agent(prompt="second", tool_sources=[connection.tools()])
            fresh = await asyncio.wait_for(second.get_tool_lookups(), timeout=1.0)
            assert await fresh.fn["github__write"](value=2) == "write:ok"

            close_gate.set()  # release the abandoned cleanup before shutdown
            await first.close()
            await second.close()

    @pytest.mark.asyncio
    async def test_recovery_is_scoped_to_one_tenant_partition(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            alpha = _bind(runtime, tenant_key="tenant-alpha")
            beta = _bind(runtime, tenant_key="tenant-beta")
            agent_alpha = Agent(prompt="alpha", tool_sources=[alpha.tools()])
            agent_beta = Agent(prompt="beta", tool_sources=[beta.tools()])
            lookups_alpha = await agent_alpha.get_tool_lookups()
            lookups_beta = await agent_beta.get_tool_lookups()

            _InstrumentedAdapter.call_error = ConnectionResetError("socket died")
            with pytest.raises(MCPOutcomeUnknownError):
                await lookups_alpha.fn["github__write"](value=1)
            _InstrumentedAdapter.call_error = None

            assert await lookups_beta.fn["github__write"](value=2) == "write:ok"
            await _wait_until(lambda: ("tenant-alpha", "github-1") not in runtime._entries)
            assert ("tenant-beta", "github-1") in runtime._entries
            assert _InstrumentedAdapter.instances[1].close_calls == 0

            replacement = Agent(prompt="alpha-2", tool_sources=[alpha.tools()])
            fresh = await replacement.get_tool_lookups()
            assert await fresh.fn["github__write"](value=3) == "write:ok"
            assert len(_InstrumentedAdapter.instances) == 3

            await agent_alpha.close()
            await agent_beta.close()
            await replacement.close()
