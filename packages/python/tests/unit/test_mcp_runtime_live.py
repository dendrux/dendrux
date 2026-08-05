"""Live managed-runtime contracts: connection reuse and lease lifecycle.

These tests pin the Release 2 behavior of MCPRuntime before implementation:
one physical connection per (tenant_key, connection_key), single-flight
connect + discovery, per-view namespaced catalogs, per-Agent leases released
on Agent.close(), and drain-based runtime shutdown.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from dendrux.agent import Agent
from dendrux.mcp._client import MCPConnectionInfo
from dendrux.mcp._errors import (
    MCPBindingConflictError,
    MCPConnectionEvictingError,
    MCPOutcomeUnknownError,
    MCPRuntimeClosedError,
    MCPStaleConnectionError,
    MCPToolCallError,
)
from dendrux.mcp._runtime import (
    MCPConnection,
    MCPRuntime,
    MCPRuntimeState,
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
    connect_error: ClassVar[Exception | None] = None
    close_error: ClassVar[Exception | None] = None
    call_error: ClassVar[Exception | None] = None
    suppress_cancel: ClassVar[bool] = False

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
        error = type(self).connect_error
        if error is not None:
            raise error

    async def list_tools(self) -> list[Tool]:
        self.list_tools_calls += 1
        return list(type(self).tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.call_tool_calls += 1
        gate = type(self).call_gate
        if gate is not None:
            await gate.wait()
        if self.closed:
            # A real transport fails an in-flight call once it is closed.
            raise ConnectionError("transport closed")
        error = type(self).call_error
        if error is not None:
            raise error
        return CallToolResult(content=[TextContent(text=f"{name}:ok")])

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
    _InstrumentedAdapter.connect_error = None
    _InstrumentedAdapter.close_error = None
    _InstrumentedAdapter.call_error = None
    _InstrumentedAdapter.suppress_cancel = False
    with patch("dendrux.mcp._runtime.MCPClientAdapter", _InstrumentedAdapter):
        yield


class _Credentials:
    def __init__(self) -> None:
        self.calls = 0

    async def get_auth(self) -> Any:
        self.calls += 1
        return "must-not-leak"


def _bind(
    runtime: MCPRuntime,
    *,
    tenant_key: str | None = None,
    connection_key: str = "github-1",
    credentials: Any = None,
) -> MCPConnection:
    return runtime.bind(
        connection_key=connection_key,
        tenant_key=tenant_key,
        source=MCPSource.http("github", "https://mcp.example.com"),
        credentials=credentials,
    )


def _total_connects() -> int:
    return sum(adapter.connect_calls for adapter in _InstrumentedAdapter.instances)


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

    @pytest.mark.asyncio
    async def test_credential_provider_stays_dormant_in_this_slice(self) -> None:
        credentials = _Credentials()
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _bind(runtime, credentials=credentials).tools()

            await Agent(prompt="test", tool_sources=[view]).get_tool_lookups()

            assert credentials.calls == 0


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
                auth=object(),
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
                    auth=object(),
                ),
            )

        assert excinfo.value.mismatch == "configuration"
        await agent.close()
        await runtime.close()

    @pytest.mark.asyncio
    async def test_same_mutable_auth_object_reuses_live_connection(self) -> None:
        auth = object()
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
    async def test_idle_auth_rebind_supersedes_old_handle(self) -> None:
        runtime = MCPRuntime(shutdown_timeout=0.05)
        stale = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github",
                "https://mcp.example.com",
                auth=object(),
            ),
        )
        current = runtime.bind(
            connection_key="github-1",
            source=MCPSource.http(
                "github",
                "https://mcp.example.com",
                auth=object(),
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
        with pytest.raises(MCPToolCallError):
            await retained(value="must-not-run")
        assert adapter.call_tool_calls == 0

        await agent.close()
        await asyncio.wait_for(evicting, timeout=1.0)
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
        _InstrumentedAdapter.call_error = ConnectionError("server exploded")
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
