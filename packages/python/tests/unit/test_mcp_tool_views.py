"""MCP tool views: bare names, renames, per-tool skipping, and stable codes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.mcp import (
    MCPAuthenticationError,
    MCPAuthKey,
    MCPAuthorizationRequired,
    MCPCallCapacityError,
    MCPCircuitOpenError,
    MCPConnectionCapacityError,
    MCPConnectionError,
    MCPConnectionEvictingError,
    MCPConnectionLostError,
    MCPConnectionStateError,
    MCPCredentialError,
    MCPDestinationDeniedError,
    MCPOAuthRefreshError,
    MCPRuntime,
    MCPRuntimeEvent,
    MCPSkippedTool,
    MCPSource,
    MCPStaleConnectionError,
    MCPToolCallError,
    MCPToolCallStarted,
    MCPToolNameCollisionError,
    MCPToolPolicy,
    error_code_for,
)
from dendrux.mcp._client import MCPConnectionInfo
from dendrux.mcp._server import build_mcp_tool_defs
from dendrux.runtime.runner import _emit_init_events
from dendrux.skills import Skill
from dendrux.tool import tool
from dendrux.types import GovernanceEventType, LLMResponse, ToolCall

LONG_70 = "t" * 70
LONG_60 = "u" * 60


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
        annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False),
    )


class _FakeAdapter:
    """Fake MCPClientAdapter serving a scripted catalog per source name."""

    catalogs: ClassVar[dict[str, list[Tool]]] = {}
    connect_errors: ClassVar[dict[str, Exception]] = {}
    calls: ClassVar[list[str]] = []

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self.info = MCPConnectionInfo(
            protocol_version="2026-07-28",
            server_name="fake",
            server_version="1.0.0",
            instructions=None,
        )

    async def connect(self) -> None:
        error = type(self).connect_errors.get(self.source.name)
        if error is not None:
            raise error

    async def list_tools(self) -> list[Tool]:
        return list(type(self).catalogs.get(self.source.name, []))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        type(self).calls.append(name)
        return CallToolResult(content=[TextContent(text=f"{name}:ok")])

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _fake_adapter() -> Any:
    _FakeAdapter.catalogs = {"github": [_tool("read"), _tool("write")]}
    _FakeAdapter.connect_errors = {}
    _FakeAdapter.calls = []
    with patch("dendrux.mcp._runtime.MCPClientAdapter", _FakeAdapter):
        yield


def _connection(
    runtime: MCPRuntime,
    name: str = "github",
    *,
    key: str | None = None,
    failure_mode: str = "strict",
) -> Any:
    return runtime.bind(
        connection_key=key or f"{name}-1",
        source=MCPSource.http(
            name,
            f"https://{name}.example.com/mcp",
            failure_mode=failure_mode,  # type: ignore[arg-type]
        ),
    )


async def _discover(view: Any) -> tuple[Agent, dict[str, Any]]:
    agent = Agent(prompt="test", tool_sources=[view])
    await agent.get_tool_lookups()
    return agent, {td.name: td for td in agent.get_all_tool_defs()}


class _Recorder(BaseNotifier):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        self.events.append((event_type, data))


class _Observer:
    def __init__(self) -> None:
        self.events: list[MCPRuntimeEvent] = []

    def on_event(self, event: MCPRuntimeEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


class TestExposedNames:
    async def test_default_view_keeps_the_namespace_prefix(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent, defs = await _discover(_connection(runtime).tools())
            assert sorted(defs) == ["github__read", "github__write"]
            assert defs["github__read"].meta["exposed_name"] == "github__read"
            await agent.close()

    async def test_prefix_false_exposes_bare_sanitized_names(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool("read"), _tool("get.file")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent, defs = await _discover(_connection(runtime).tools(prefix=False))
            assert sorted(defs) == ["get_file", "read"]
            meta = defs["get_file"].meta
            assert meta["exposed_name"] == "get_file"
            assert meta["mcp_tool_name"] == "get.file"
            assert meta["namespace"] == "github"
            assert meta["source_name"] == "github"
            await agent.close()

    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [(True, ["github__fetch", "github__write"]), (False, ["fetch", "write"])],
    )
    async def test_rename_applies_in_both_modes(self, prefix: bool, expected: list[str]) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(prefix=prefix, rename={"read": "fetch"})
            agent, defs = await _discover(view)
            assert sorted(defs) == expected
            renamed = defs[expected[0]]
            assert renamed.meta["mcp_tool_name"] == "read"
            assert renamed.meta["exposed_name"] == expected[0]
            await agent.close()

    async def test_namespace_is_still_the_view_identity_in_bare_mode(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(namespace="gh", prefix=False)
            assert view.namespace == "gh"
            agent, defs = await _discover(view)
            assert defs["read"].meta["namespace"] == "gh"
            await agent.close()


class TestPolicyValidation:
    def test_defaults(self) -> None:
        policy = MCPToolPolicy()
        assert policy.prefix is True
        assert dict(policy.rename) == {}
        assert policy.strict is False

    def test_rename_is_an_immutable_copy(self) -> None:
        source = {"read": "fetch"}
        policy = MCPToolPolicy(rename=source)
        source["write"] = "put"
        assert dict(policy.rename) == {"read": "fetch"}
        with pytest.raises(TypeError):
            policy.rename["x"] = "y"  # type: ignore[index]

    def test_equal_policies_compare_and_hash_equal(self) -> None:
        first = MCPToolPolicy(rename={"read": "fetch"}, prefix=False)
        second = MCPToolPolicy(rename={"read": "fetch"}, prefix=False)
        assert first == second
        assert hash(first) == hash(second)
        assert first != MCPToolPolicy(rename={"read": "fetch"})

    @pytest.mark.parametrize(
        "value",
        ["", "has space", "dot.name", "x" * 65, "ünï"],
    )
    def test_unsafe_rename_value_raises(self, value: str) -> None:
        with pytest.raises(ValueError, match="rename"):
            MCPToolPolicy(rename={"read": value})

    @pytest.mark.parametrize("key", ["", "   ", "bad\nkey"])
    def test_invalid_rename_key_raises(self, key: str) -> None:
        with pytest.raises(ValueError, match="rename"):
            MCPToolPolicy(rename={key: "fine"})

    def test_non_string_rename_entries_raise(self) -> None:
        with pytest.raises(ValueError, match="rename"):
            MCPToolPolicy(rename={"read": 3})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="rename"):
            MCPToolPolicy(rename=["read"])  # type: ignore[arg-type]

    def test_duplicate_rename_values_raise(self) -> None:
        with pytest.raises(ValueError, match="rename"):
            MCPToolPolicy(rename={"read": "same", "write": "same"})

    @pytest.mark.parametrize("field", ["prefix", "strict"])
    def test_flags_must_be_bool(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            MCPToolPolicy(**{field: "yes"})  # type: ignore[arg-type]

    def test_over_long_prefixed_rename_raises_at_view_construction(self) -> None:
        connection = _connection(MCPRuntime())
        with pytest.raises(ValueError, match="rename"):
            connection.tools(rename={"read": LONG_60})

    def test_long_rename_is_fine_in_bare_mode(self) -> None:
        view = _connection(MCPRuntime()).tools(prefix=False, rename={"read": LONG_60})
        assert view.policy.rename["read"] == LONG_60

    def test_view_carries_the_new_options(self) -> None:
        view = _connection(MCPRuntime()).tools(prefix=False, rename={"read": "r"}, strict=True)
        assert view.policy == MCPToolPolicy(prefix=False, rename={"read": "r"}, strict=True)


# ---------------------------------------------------------------------------
# Per-tool skipping
# ---------------------------------------------------------------------------


def _skipped(reason: str, tool: str, namespace: str = "github") -> dict[str, str]:
    return {"source_name": "github", "namespace": namespace, "tool": tool, "reason": reason}


class TestPerToolSkipping:
    async def test_over_long_name_is_skipped_and_the_rest_load(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool("read"), _tool(LONG_70), _tool("write")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent, defs = await _discover(_connection(runtime).tools(prefix=False))
            assert sorted(defs) == ["read", "write"]
            assert agent._mcp_skipped_tools == [_skipped("name_unsafe", LONG_70)]
            await agent.close()

    async def test_prefix_alone_can_push_a_name_over_the_limit(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool(LONG_60)]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            connection = _connection(runtime)
            agent, defs = await _discover(connection.tools())
            assert defs == {}
            assert agent._mcp_skipped_tools == [_skipped("name_unsafe", LONG_60)]
            await agent.close()

            bare, bare_defs = await _discover(connection.tools(prefix=False))
            assert sorted(bare_defs) == [LONG_60]
            assert bare._mcp_skipped_tools == []
            await bare.close()

    async def test_rename_rescues_an_over_long_name(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool(LONG_70)]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(rename={LONG_70: "short"})
            agent, defs = await _discover(view)
            assert sorted(defs) == ["github__short"]
            assert agent._mcp_skipped_tools == []
            await agent.close()

    async def test_sanitize_duplicate_skips_the_later_tool(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool("a.b"), _tool("a_b"), _tool("read")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent, defs = await _discover(_connection(runtime).tools(prefix=False))
            assert sorted(defs) == ["a_b", "read"]
            assert defs["a_b"].meta["mcp_tool_name"] == "a.b"
            assert agent._mcp_skipped_tools == [_skipped("duplicate_name", "a_b")]
            await agent.close()

    async def test_rename_onto_an_existing_name_skips_the_later_tool(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(prefix=False, rename={"write": "read"})
            agent, defs = await _discover(view)
            assert sorted(defs) == ["read"]
            assert defs["read"].meta["mcp_tool_name"] == "read"
            assert agent._mcp_skipped_tools == [_skipped("duplicate_name", "write")]
            await agent.close()

    @pytest.mark.parametrize(
        "options",
        [
            {"allowed_tools": ["read", "missing"]},
            {"force_serial_tools": ["missing"]},
            {"rename": {"missing": "m"}},
        ],
    )
    async def test_unknown_names_are_reported_and_the_rest_load(
        self, options: dict[str, Any]
    ) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent, defs = await _discover(_connection(runtime).tools(**options))
            assert "github__read" in defs
            assert agent._mcp_skipped_tools == [_skipped("unknown_tool", "missing")]
            await agent.close()

    async def test_a_name_unknown_to_several_fields_is_reported_once(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(
                allowed_tools=["read", "gone"],
                force_serial_tools=["gone"],
                rename={"gone": "g"},
            )
            agent, defs = await _discover(view)
            assert sorted(defs) == ["github__read"]
            assert agent._mcp_skipped_tools == [_skipped("unknown_tool", "gone")]
            await agent.close()

    async def test_unknown_entries_come_before_naming_entries(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool(LONG_70), _tool("read")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(rename={"missing": "m"})
            agent, _ = await _discover(view)
            assert [entry["reason"] for entry in agent._mcp_skipped_tools] == [
                "unknown_tool",
                "name_unsafe",
            ]
            await agent.close()

    @pytest.mark.parametrize(
        ("catalog", "options"),
        [
            ([LONG_70, "read"], {}),
            (["a.b", "a_b"], {}),
            (["read"], {"allowed_tools": ["read", "missing"]}),
            (["read"], {"force_serial_tools": ["missing"]}),
            (["read"], {"rename": {"missing": "m"}}),
        ],
    )
    async def test_strict_raises_value_error_in_each_case(
        self, catalog: list[str], options: dict[str, Any]
    ) -> None:
        _FakeAdapter.catalogs["github"] = [_tool(name) for name in catalog]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(strict=True, **options)
            agent = Agent(prompt="test", tool_sources=[view])
            with pytest.raises(ValueError) as caught:
                await agent.get_tool_lookups()
            assert type(caught.value) is ValueError
            await agent.close()

    @pytest.mark.parametrize("prefix", [True, False])
    async def test_empty_base_name_is_unsafe_in_both_modes(self, prefix: bool) -> None:
        _FakeAdapter.catalogs["github"] = [_tool(""), _tool("read")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent, defs = await _discover(_connection(runtime).tools(prefix=prefix))
            assert sorted(defs) == (["github__read"] if prefix else ["read"])
            assert agent._mcp_skipped_tools == [_skipped("name_unsafe", "")]
            await agent.close()

    async def test_empty_base_name_raises_in_strict_prefixed_mode(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool("")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(prompt="test", tool_sources=[_connection(runtime).tools(strict=True)])
            with pytest.raises(ValueError) as caught:
                await agent.get_tool_lookups()
            assert type(caught.value) is ValueError
            await agent.close()

    def test_build_returns_defs_and_typed_skips(self) -> None:
        source = SimpleNamespace(name="srv", transport="stdio", call_timeout=30.0)
        tools = [_tool("ok"), _tool(LONG_70)]
        defs, skipped = build_mcp_tool_defs(
            source=source,  # type: ignore[arg-type]
            namespace="srv",
            tools=tools,
            connection_info=None,
            prefix=False,
        )
        assert [td.name for td in defs] == ["ok"]
        assert skipped == [
            MCPSkippedTool(source_name="srv", namespace="srv", tool=LONG_70, reason="name_unsafe")
        ]

    def test_skipped_tool_is_immutable(self) -> None:
        entry = MCPSkippedTool(source_name="s", namespace="n", tool="t", reason="duplicate_name")
        with pytest.raises(AttributeError):
            entry.tool = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Surfacing: RunResult.meta, governance events, stable codes
# ---------------------------------------------------------------------------


class TestSurfacing:
    async def test_run_result_and_governance_event_list_skipped_tools(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool("read"), _tool(LONG_70)]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(prefix=False, allowed_tools=["read", LONG_70, "gone"])
            agent = Agent(
                provider=MockLLM(
                    [
                        LLMResponse(tool_calls=[ToolCall(name="read", params={})]),
                        LLMResponse(text="done"),
                    ]
                ),
                prompt="test",
                tool_sources=[view],
            )
            notifier = _Recorder()
            try:
                result = await agent.run("hi", notifier=notifier)
            finally:
                await agent.close()

        assert result.answer == "done"
        assert _FakeAdapter.calls == ["read"]
        expected = [_skipped("unknown_tool", "gone"), _skipped("name_unsafe", LONG_70)]
        assert result.meta["mcp_skipped_tools"] == expected
        assert result.meta["mcp_skipped_sources"] == []
        skipped_events = [
            data for kind, data in notifier.events if kind == GovernanceEventType.MCP_TOOLS_SKIPPED
        ]
        assert skipped_events == [{"skipped": expected}]

    async def test_no_event_and_empty_list_when_nothing_is_skipped(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="test",
                tool_sources=[_connection(runtime).tools()],
            )
            notifier = _Recorder()
            try:
                result = await agent.run("hi", notifier=notifier)
            finally:
                await agent.close()

        assert result.meta["mcp_skipped_tools"] == []
        assert GovernanceEventType.MCP_TOOLS_SKIPPED not in [kind for kind, _ in notifier.events]

    async def test_agent_without_sources_still_reports_empty_lists(self) -> None:
        agent = Agent(provider=MockLLM([LLMResponse(text="done")]), prompt="test")
        result = await agent.run("hi")
        assert result.meta["mcp_skipped_sources"] == []
        assert result.meta["mcp_skipped_tools"] == []

    async def test_skipped_tools_from_a_best_effort_source_are_kept_with_other_sources(
        self,
    ) -> None:
        _FakeAdapter.catalogs["gitlab"] = [_tool(LONG_70), _tool("issues")]
        _FakeAdapter.connect_errors["broken"] = MCPConnectionError("down")
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="test",
                tool_sources=[
                    _connection(runtime, "gitlab").tools(prefix=False),
                    _connection(runtime, "broken", failure_mode="best_effort").tools(),
                ],
            )
            try:
                result = await agent.run("hi")
            finally:
                await agent.close()

        assert result.meta["mcp_skipped_tools"] == [
            {
                "source_name": "gitlab",
                "namespace": "gitlab",
                "tool": LONG_70,
                "reason": "name_unsafe",
            }
        ]
        (entry,) = result.meta["mcp_skipped_sources"]
        assert entry["code"] == "connection_failed"
        assert "reason" not in entry

    async def test_skipped_source_and_mcp_error_carry_code_and_oauth_reason(self) -> None:
        _FakeAdapter.connect_errors["github"] = MCPAuthorizationRequired(
            MCPAuthKey("github-1"), "expired"
        )
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime, failure_mode="best_effort").tools(namespace="gh")
            agent = Agent(prompt="test", tool_sources=[view])
            with patch(
                "dendrux.runtime.runner._emit_init_governance_event",
                new_callable=AsyncMock,
            ) as emitted:
                await _emit_init_events(agent, None, None, "run-1")
            await agent.close()

        (entry,) = [
            call.args[4]
            for call in emitted.await_args_list
            if call.args[3] == GovernanceEventType.MCP_ERROR
        ]
        assert entry["code"] == "authorization_required"
        assert entry["reason"] == "expired"
        assert entry["error_type"] == "MCPAuthorizationRequired"

    async def test_best_effort_skip_entry_carries_code_and_reason(self) -> None:
        _FakeAdapter.connect_errors["github"] = MCPAuthorizationRequired(
            MCPAuthKey("github-1"), "not_signed_in"
        )
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="test",
                tool_sources=[_connection(runtime, failure_mode="best_effort").tools()],
            )
            try:
                result = await agent.run("hi")
            finally:
                await agent.close()

        (entry,) = result.meta["mcp_skipped_sources"]
        assert entry["code"] == "authorization_required"
        assert entry["reason"] == "not_signed_in"

    async def test_fatal_mcp_error_payload_carries_code(self) -> None:
        _FakeAdapter.connect_errors["github"] = MCPAuthorizationRequired(
            MCPAuthKey("github-1"), "revoked"
        )
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="test",
                tool_sources=[_connection(runtime).tools()],
            )
            notifier = _Recorder()
            try:
                with pytest.raises(MCPAuthorizationRequired):
                    await agent.run("hi", notifier=notifier)
            finally:
                await agent.close()

        (payload,) = [d for kind, d in notifier.events if kind == GovernanceEventType.MCP_ERROR]
        assert payload["code"] == "authorization_required"
        assert payload["reason"] == "revoked"
        assert payload["error_type"] == "MCPAuthorizationRequired"

    async def test_fatal_strict_naming_error_payload_code(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool(LONG_70)]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="test",
                tool_sources=[_connection(runtime).tools(strict=True)],
            )
            notifier = _Recorder()
            try:
                with pytest.raises(ValueError):
                    await agent.run("hi", notifier=notifier)
            finally:
                await agent.close()

        (payload,) = [d for kind, d in notifier.events if kind == GovernanceEventType.MCP_ERROR]
        assert payload["code"] == "invalid_tool_names"
        assert "reason" not in payload

    async def test_console_renders_skipped_tools(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dendrux.notifiers.console import ConsoleNotifier

        await ConsoleNotifier().on_governance_event(
            "r1",
            GovernanceEventType.MCP_TOOLS_SKIPPED,
            0,
            {"skipped": [_skipped("name_unsafe", LONG_70)]},
        )
        assert "1 tool(s)" in capsys.readouterr().out

    def test_dashboard_knows_the_event(self) -> None:
        from dendrux.dashboard.normalizer import GOVERNANCE_EVENT_META

        severity, _ = GOVERNANCE_EVENT_META[GovernanceEventType.MCP_TOOLS_SKIPPED]
        assert severity == "warning"
        assert GovernanceEventType.MCP_TOOLS_SKIPPED == "mcp.tools_skipped"


_IDENTITY = (None, "c")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (MCPAuthorizationRequired(MCPAuthKey("c"), "expired"), "authorization_required"),
        (MCPAuthenticationError("rejected"), "authentication_failed"),
        (MCPCredentialError("bad"), "credential_error"),
        (MCPOAuthRefreshError("contended"), "credential_error"),
        (MCPDestinationDeniedError("denied"), "destination_denied"),
        (
            MCPCircuitOpenError(_IDENTITY, retry_after=1.0, failure_count=3, last_failure=None),
            "circuit_open",
        ),
        (MCPConnectionCapacityError(_IDENTITY, limit=1, timeout=0.1), "capacity"),
        (MCPCallCapacityError(_IDENTITY, tool="t", limit=1, timeout=0.1), "capacity"),
        (MCPConnectionLostError(_IDENTITY, "lost"), "connection_failed"),
        (MCPConnectionError("down"), "connection_failed"),
        (MCPStaleConnectionError(_IDENTITY, "stale"), "connection_state"),
        (MCPConnectionEvictingError(_IDENTITY, "evicting"), "connection_state"),
        (MCPConnectionStateError(_IDENTITY, "other"), "connection_state"),
        (MCPToolNameCollisionError("clash"), "tool_name_collision"),
        (ValueError("names"), "invalid_tool_names"),
        (MCPToolCallError("failed"), "unknown"),
        (ConnectionError("refused"), "unknown"),
        (RuntimeError("boom"), "unknown"),
    ],
)
def test_error_code_for_maps_by_class(error: BaseException, code: str) -> None:
    assert error_code_for(error) == code


# ---------------------------------------------------------------------------
# Telemetry and execution
# ---------------------------------------------------------------------------


class TestTelemetry:
    @pytest.mark.parametrize(
        ("options", "exposed"),
        [
            ({"prefix": False}, "read"),
            ({"prefix": False, "rename": {"read": "fetch"}}, "fetch"),
            ({"rename": {"read": "fetch"}}, "github__fetch"),
        ],
    )
    async def test_call_telemetry_uses_the_exposed_name(
        self, options: dict[str, Any], exposed: str
    ) -> None:
        observer = _Observer()
        async with MCPRuntime(shutdown_timeout=0.05, observer=observer) as runtime:
            agent = Agent(prompt="test", tool_sources=[_connection(runtime).tools(**options)])
            lookups = await agent.get_tool_lookups()
            assert await lookups.fn[exposed]() == "read:ok"
            await agent.close()

        started = [event for event in observer.events if isinstance(event, MCPToolCallStarted)]
        assert [event.tool for event in started] == [exposed]
        assert _FakeAdapter.calls == ["read"]

    async def test_stale_tool_check_keys_on_the_server_name(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            view = _connection(runtime).tools(prefix=False, rename={"read": "fetch"})
            agent = Agent(prompt="test", tool_sources=[view])
            lookups = await agent.get_tool_lookups()
            # The unchanged catalog matches: the renamed tool resolves its own
            # definition by server name and reaches the server.
            assert await lookups.fn["fetch"]() == "read:ok"
            assert _FakeAdapter.calls == ["read"]

            entry = agent._tool_sources[0]._entry  # type: ignore[attr-defined]
            changed = _tool("read").model_copy(update={"description": "changed"})
            entry.raw_tools = [changed, _tool("write")]
            with pytest.raises(MCPStaleConnectionError):
                await lookups.fn["fetch"]()
            assert _FakeAdapter.calls == ["read"]

            # A different tool keyed on its own server name is unaffected.
            assert await lookups.fn["write"]() == "write:ok"
            assert _FakeAdapter.calls == ["read", "write"]
            await agent.close()


# ---------------------------------------------------------------------------
# Agent-level collisions stay fatal
# ---------------------------------------------------------------------------


@tool(target="server")
def read() -> str:
    """Local tool clashing with a bare MCP name."""
    return "local"


class TestCollisionsStillRaise:
    async def test_bare_names_from_two_sources_clash(self) -> None:
        _FakeAdapter.catalogs["gitlab"] = [_tool("read")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                prompt="test",
                tool_sources=[
                    _connection(runtime).tools(prefix=False),
                    _connection(runtime, "gitlab").tools(prefix=False),
                ],
            )
            with pytest.raises(ValueError, match="collides"):
                await agent.get_tool_lookups()
            await agent.close()

    async def test_bare_name_clashing_with_a_local_tool_raises(self) -> None:
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                prompt="test",
                tools=[read],
                tool_sources=[_connection(runtime).tools(prefix=False)],
            )
            with pytest.raises(ValueError, match="collides"):
                await agent.get_tool_lookups()
            await agent.close()

    async def test_bare_use_skill_still_collides_with_the_reserved_name(self) -> None:
        _FakeAdapter.catalogs["github"] = [_tool("use_skill")]
        skill = Skill(name="helper", description="d", body="b", path=Path("."))
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                prompt="test",
                skills=[skill],
                tool_sources=[_connection(runtime).tools(prefix=False)],
            )
            with pytest.raises(ValueError, match="use_skill"):
                await agent.get_tool_lookups()
            await agent.close()

    async def test_rename_resolves_a_cross_source_clash(self) -> None:
        _FakeAdapter.catalogs["gitlab"] = [_tool("read")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            agent = Agent(
                prompt="test",
                tool_sources=[
                    _connection(runtime).tools(prefix=False),
                    _connection(runtime, "gitlab").tools(prefix=False, rename={"read": "gl_read"}),
                ],
            )
            lookups = await agent.get_tool_lookups()
            assert {"read", "gl_read"} <= set(lookups.fn)
            await agent.close()

    @pytest.mark.parametrize("clash", ["other_source", "local_tool", "use_skill"])
    async def test_collisions_report_a_distinct_code(self, clash: str) -> None:
        _FakeAdapter.catalogs["gitlab"] = [_tool("read")]
        if clash == "use_skill":
            _FakeAdapter.catalogs["github"] = [_tool("use_skill")]
        async with MCPRuntime(shutdown_timeout=0.05) as runtime:
            sources = [_connection(runtime).tools(prefix=False)]
            if clash == "other_source":
                sources.append(_connection(runtime, "gitlab").tools(prefix=False))
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="test",
                tools=[read] if clash == "local_tool" else [],
                skills=(
                    [Skill(name="helper", description="d", body="b", path=Path("."))]
                    if clash == "use_skill"
                    else None
                ),
                tool_sources=sources,
            )
            notifier = _Recorder()
            try:
                with pytest.raises(MCPToolNameCollisionError) as caught:
                    await agent.run("hi", notifier=notifier)
            finally:
                await agent.close()

        assert isinstance(caught.value, ValueError)
        assert error_code_for(caught.value) == "tool_name_collision"
        (payload,) = [d for kind, d in notifier.events if kind == GovernanceEventType.MCP_ERROR]
        assert payload["code"] == "tool_name_collision"
        assert payload["error_type"] == "MCPToolNameCollisionError"
