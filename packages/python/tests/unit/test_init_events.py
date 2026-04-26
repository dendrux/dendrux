"""Tests for skill and MCP init governance events.

Covers:
- GovernanceEventType enum additions (4 new members)
- Runner helper _emit_init_events() emitting skill.registered, skill.denied,
  mcp.connected after run.started inside the try block
- MCP discovery failure → mcp.error + run error cleanup
- Console notifier rendering of all 4 new event types
- Normalizer mapping for all 4 new event types
- SingleCall + tool_sources runtime validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

from dendrux.agent import Agent
from dendrux.tool import tool
from dendrux.types import GovernanceEventType

if TYPE_CHECKING:
    from datetime import datetime

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@tool(target="server")
def dummy_tool(x: int) -> int:
    """A dummy tool for tests."""
    return x


class RecordingRecorder:
    """Minimal recorder that captures governance events."""

    def __init__(self) -> None:
        self.governance_events: list[tuple[str, int, dict[str, Any], str | None]] = []

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.governance_events.append((event_type, iteration, data, correlation_id))

    # Stubs for other recorder methods the runner calls
    async def on_message_appended(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def on_llm_call_completed(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def on_tool_completed(self, *args: Any, **kwargs: Any) -> None:
        pass


class RecordingNotifier:
    """Minimal notifier that captures governance events."""

    def __init__(self) -> None:
        self.governance_events: list[tuple[str, int, dict[str, Any], str | None]] = []

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.governance_events.append((event_type, iteration, data, correlation_id))

    async def on_message_appended(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def on_llm_call_completed(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def on_tool_completed(self, *args: Any, **kwargs: Any) -> None:
        pass


# ------------------------------------------------------------------
# GovernanceEventType enum
# ------------------------------------------------------------------


class TestGovernanceEventTypeEnum:
    """New event type members exist and compare correctly."""

    def test_skill_registered_member(self) -> None:
        assert GovernanceEventType.SKILL_REGISTERED == "skill.registered"

    def test_skill_denied_member(self) -> None:
        assert GovernanceEventType.SKILL_DENIED == "skill.denied"

    def test_skill_invoked_member(self) -> None:
        assert GovernanceEventType.SKILL_INVOKED == "skill.invoked"

    def test_mcp_connected_member(self) -> None:
        assert GovernanceEventType.MCP_CONNECTED == "mcp.connected"

    def test_mcp_error_member(self) -> None:
        assert GovernanceEventType.MCP_ERROR == "mcp.error"

    def test_provider_retry_member(self) -> None:
        assert GovernanceEventType.PROVIDER_RETRY == "provider.retry"

    def test_total_member_count(self) -> None:
        """8 original + 5 init + 1 transport + 1 unmapped-placeholder = 15 total."""
        assert len(GovernanceEventType) == 15

    def test_guardrail_unmapped_placeholder_member(self) -> None:
        assert (
            GovernanceEventType.GUARDRAIL_UNMAPPED_PLACEHOLDER == "guardrail.unmapped_placeholder"
        )


# ------------------------------------------------------------------
# _emit_init_events helper
# ------------------------------------------------------------------


class TestEmitInitEvents:
    """Runner helper emits skill and MCP governance events."""

    async def test_emits_skill_loaded_events(self) -> None:
        """Each loaded skill produces a skill.registered event at iteration=0."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])
        agent._ensure_skills_loaded()

        recorder = RecordingRecorder()
        notifier = RecordingNotifier()
        await _emit_init_events(agent, recorder, notifier)

        loaded = [
            (et, it, d)
            for et, it, d, _ in recorder.governance_events
            if et == GovernanceEventType.SKILL_REGISTERED
        ]
        assert len(loaded) == 1
        assert loaded[0][1] == 0  # iteration=0
        assert loaded[0][2]["skill_name"] == "pdf-processing"
        assert "description" in loaded[0][2]

    async def test_emits_skill_denied_events(self) -> None:
        """Each denied skill produces a skill.denied event."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            skills=[skill],
            deny_skills=["pdf-processing"],
        )
        agent._ensure_skills_loaded()

        recorder = RecordingRecorder()
        notifier = RecordingNotifier()
        await _emit_init_events(agent, recorder, notifier)

        denied = [
            (et, it, d)
            for et, it, d, _ in recorder.governance_events
            if et == GovernanceEventType.SKILL_DENIED
        ]
        assert len(denied) == 1
        assert denied[0][1] == 0
        assert denied[0][2]["skill_name"] == "pdf-processing"
        assert denied[0][2]["reason"] == "denied_by_policy"

    async def test_notifier_receives_skill_events(self) -> None:
        """Notifier also receives the same skill events."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])
        agent._ensure_skills_loaded()

        recorder = RecordingRecorder()
        notifier = RecordingNotifier()
        await _emit_init_events(agent, recorder, notifier)

        assert len(notifier.governance_events) == 1
        assert notifier.governance_events[0][0] == GovernanceEventType.SKILL_REGISTERED

    async def test_no_skills_no_events(self) -> None:
        """Agent without skills emits no skill events."""
        from dendrux.runtime.runner import _emit_init_events

        agent = Agent(prompt="Test.", tools=[dummy_tool])
        agent._ensure_skills_loaded()

        recorder = RecordingRecorder()
        notifier = RecordingNotifier()
        await _emit_init_events(agent, recorder, notifier)

        assert len(recorder.governance_events) == 0

    async def test_recorder_none_ok(self) -> None:
        """Works when recorder is None (no persistence)."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])
        agent._ensure_skills_loaded()

        notifier = RecordingNotifier()
        await _emit_init_events(agent, None, notifier)

        # Notifier still gets events even without persistence
        assert len(notifier.governance_events) == 1

    async def test_notifier_none_ok(self) -> None:
        """Works when notifier is None."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])
        agent._ensure_skills_loaded()

        recorder = RecordingRecorder()
        await _emit_init_events(agent, recorder, None)

        assert len(recorder.governance_events) == 1

    async def test_mcp_connected_events(self) -> None:
        """MCP source discovery emits mcp.connected grouped by source_name."""
        from dendrux.runtime.runner import _emit_init_events
        from dendrux.types import ToolDef, ToolTarget

        agent = Agent(prompt="Test.", tools=[dummy_tool])
        agent._ensure_skills_loaded()

        # Simulate MCP sources with .name attribute
        fs_source = AsyncMock()
        fs_source.name = "filesystem"
        db_source = AsyncMock()
        db_source.name = "database"

        agent._tool_sources = [fs_source, db_source]
        agent._discovered_tool_defs = [
            ToolDef(
                name="fs__read_file",
                description="Read a file",
                parameters={},
                target=ToolTarget.SERVER,
                meta={"source_name": "filesystem"},
            ),
            ToolDef(
                name="fs__write_file",
                description="Write a file",
                parameters={},
                target=ToolTarget.SERVER,
                meta={"source_name": "filesystem"},
            ),
            ToolDef(
                name="db__query",
                description="Run a query",
                parameters={},
                target=ToolTarget.SERVER,
                meta={"source_name": "database"},
            ),
        ]

        # Patch get_tool_lookups to avoid real MCP discovery
        with patch.object(agent, "get_tool_lookups", new_callable=AsyncMock):
            recorder = RecordingRecorder()
            notifier = RecordingNotifier()
            await _emit_init_events(agent, recorder, notifier)

        mcp_events = [
            (et, d)
            for et, _, d, _ in recorder.governance_events
            if et == GovernanceEventType.MCP_CONNECTED
        ]
        assert len(mcp_events) == 2
        source_names = {e[1]["source_name"] for e in mcp_events}
        assert source_names == {"filesystem", "database"}

        # filesystem has 2 tools
        fs_event = next(e for e in mcp_events if e[1]["source_name"] == "filesystem")
        assert fs_event[1]["tool_count"] == 2
        assert set(fs_event[1]["tool_names"]) == {"fs__read_file", "fs__write_file"}

        # database has 1 tool
        db_event = next(e for e in mcp_events if e[1]["source_name"] == "database")
        assert db_event[1]["tool_count"] == 1

    async def test_mcp_connected_zero_tool_source(self) -> None:
        """MCP source with zero tools still emits mcp.connected."""
        from dendrux.runtime.runner import _emit_init_events
        from dendrux.types import ToolDef, ToolTarget

        agent = Agent(prompt="Test.", tools=[dummy_tool])
        agent._ensure_skills_loaded()

        # Two sources, but only one has tools
        active_source = AsyncMock()
        active_source.name = "filesystem"
        empty_source = AsyncMock()
        empty_source.name = "empty-server"

        agent._tool_sources = [active_source, empty_source]
        agent._discovered_tool_defs = [
            ToolDef(
                name="fs__read_file",
                description="Read a file",
                parameters={},
                target=ToolTarget.SERVER,
                meta={"source_name": "filesystem"},
            ),
        ]

        with patch.object(agent, "get_tool_lookups", new_callable=AsyncMock):
            recorder = RecordingRecorder()
            await _emit_init_events(agent, recorder, None)

        mcp_events = [
            d
            for et, _, d, _ in recorder.governance_events
            if et == GovernanceEventType.MCP_CONNECTED
        ]
        assert len(mcp_events) == 2

        empty_event = next(e for e in mcp_events if e["source_name"] == "empty-server")
        assert empty_event["tool_count"] == 0
        assert empty_event["tool_names"] == []

    async def test_no_mcp_sources_no_discovery(self) -> None:
        """Agent without tool_sources does not call get_tool_lookups."""
        from dendrux.runtime.runner import _emit_init_events

        agent = Agent(prompt="Test.", tools=[dummy_tool])
        agent._ensure_skills_loaded()

        with patch.object(agent, "get_tool_lookups", new_callable=AsyncMock) as mock:
            await _emit_init_events(agent, RecordingRecorder(), RecordingNotifier())
            mock.assert_not_called()

    async def test_mcp_discovery_failure_raises_wrapped(self) -> None:
        """MCP discovery failure raises _MCPDiscoveryError wrapping the original."""
        from dendrux.runtime.runner import _emit_init_events, _MCPDiscoveryError

        agent = Agent(prompt="Test.", tools=[dummy_tool])
        agent._ensure_skills_loaded()
        agent._tool_sources = [AsyncMock()]

        with (
            patch.object(
                agent,
                "get_tool_lookups",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Connection refused"),
            ),
            pytest.raises(_MCPDiscoveryError, match="Connection refused") as exc_info,
        ):
            await _emit_init_events(agent, RecordingRecorder(), RecordingNotifier())

        # Original cause is preserved
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    async def test_skill_emission_failure_not_misclassified_as_mcp(self) -> None:
        """Skill emission failure does NOT raise _MCPDiscoveryError even with tool_sources."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events, _MCPDiscoveryError
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])
        agent._ensure_skills_loaded()
        agent._tool_sources = [AsyncMock()]  # has MCP sources

        # Make the recorder throw during skill.registered emission
        bad_recorder = RecordingRecorder()
        bad_recorder.on_governance_event = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("DB write failed")
        )

        with pytest.raises(RuntimeError, match="DB write failed"):
            await _emit_init_events(agent, bad_recorder, None)

        # Should NOT be wrapped in _MCPDiscoveryError — it's a skill emission failure
        try:
            await _emit_init_events(agent, bad_recorder, None)
        except _MCPDiscoveryError:
            pytest.fail("Skill emission failure was misclassified as _MCPDiscoveryError")
        except RuntimeError:
            pass  # expected

    async def test_all_events_use_iteration_zero(self) -> None:
        """All init events are emitted with iteration=0."""
        from pathlib import Path

        from dendrux.runtime.runner import _emit_init_events
        from dendrux.skills import Skill

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            skills=[skill],
            deny_skills=["pdf-processing"],
        )
        agent._ensure_skills_loaded()

        recorder = RecordingRecorder()
        await _emit_init_events(agent, recorder, None)

        for _, iteration, _, _ in recorder.governance_events:
            assert iteration == 0


# ------------------------------------------------------------------
# SingleCall + tool_sources runtime guard
# ------------------------------------------------------------------


class TestSingleCallToolSourcesGuard:
    """Runtime loop override cannot bypass SingleCall + tool_sources check."""

    def test_singlecall_rejects_tool_sources_at_runtime(self) -> None:
        from dendrux.loops.single import SingleCall
        from dendrux.runtime.runner import _validate_loop_skill_compat

        agent = Agent(prompt="Test.", tools=[dummy_tool])
        # Simulate tool_sources added (normally via constructor)
        agent._tool_sources = [AsyncMock()]

        with pytest.raises(ValueError, match="(?i)single.*call"):
            _validate_loop_skill_compat(agent, SingleCall())


# ------------------------------------------------------------------
# Console notifier rendering
# ------------------------------------------------------------------


class TestConsoleNotifierInitEvents:
    """Console notifier renders the 5 new event types."""

    async def test_skill_registered_rendering(self, capsys) -> None:
        from dendrux.notifiers.console import ConsoleNotifier

        notifier = ConsoleNotifier()
        await notifier.on_governance_event(
            GovernanceEventType.SKILL_REGISTERED,
            0,
            {"skill_name": "pdf-processing", "description": "Extract text from PDFs"},
        )
        captured = capsys.readouterr()
        assert "pdf-processing" in captured.out

    async def test_skill_denied_rendering(self, capsys) -> None:
        from dendrux.notifiers.console import ConsoleNotifier

        notifier = ConsoleNotifier()
        await notifier.on_governance_event(
            GovernanceEventType.SKILL_DENIED,
            0,
            {"skill_name": "organize-notes", "reason": "denied_by_policy"},
        )
        captured = capsys.readouterr()
        assert "organize-notes" in captured.out

    async def test_mcp_connected_rendering(self, capsys) -> None:
        from dendrux.notifiers.console import ConsoleNotifier

        notifier = ConsoleNotifier()
        await notifier.on_governance_event(
            GovernanceEventType.MCP_CONNECTED,
            0,
            {
                "source_name": "filesystem",
                "tool_count": 3,
                "tool_names": ["read_file", "write_file", "list_dir"],
            },
        )
        captured = capsys.readouterr()
        assert "filesystem" in captured.out
        assert "3" in captured.out

    async def test_skill_invoked_rendering(self, capsys) -> None:
        from dendrux.notifiers.console import ConsoleNotifier

        notifier = ConsoleNotifier()
        await notifier.on_governance_event(
            GovernanceEventType.SKILL_INVOKED,
            1,
            {"skill_name": "pdf-processing"},
        )
        captured = capsys.readouterr()
        assert "pdf-processing" in captured.out
        assert "invoked" in captured.out

    async def test_mcp_error_rendering(self, capsys) -> None:
        from dendrux.notifiers.console import ConsoleNotifier

        notifier = ConsoleNotifier()
        await notifier.on_governance_event(
            GovernanceEventType.MCP_ERROR,
            0,
            {"error": "Connection refused"},
        )
        captured = capsys.readouterr()
        assert "Connection refused" in captured.out


# ------------------------------------------------------------------
# Normalizer mapping
# ------------------------------------------------------------------


class TestNormalizerInitEvents:
    """GOVERNANCE_EVENT_META has entries for 5 new event types."""

    def test_skill_registered_in_meta(self) -> None:
        from dendrux.dashboard.normalizer import GOVERNANCE_EVENT_META

        severity, title = GOVERNANCE_EVENT_META[GovernanceEventType.SKILL_REGISTERED]
        assert severity == "info"
        assert "skill" in title.lower() or "loaded" in title.lower()

    def test_skill_denied_in_meta(self) -> None:
        from dendrux.dashboard.normalizer import GOVERNANCE_EVENT_META

        severity, title = GOVERNANCE_EVENT_META[GovernanceEventType.SKILL_DENIED]
        assert severity == "warning"
        assert "skill" in title.lower() or "denied" in title.lower()

    def test_mcp_connected_in_meta(self) -> None:
        from dendrux.dashboard.normalizer import GOVERNANCE_EVENT_META

        severity, title = GOVERNANCE_EVENT_META[GovernanceEventType.MCP_CONNECTED]
        assert severity == "info"
        assert "mcp" in title.lower() or "connected" in title.lower()

    def test_mcp_error_in_meta(self) -> None:
        from dendrux.dashboard.normalizer import GOVERNANCE_EVENT_META

        severity, title = GOVERNANCE_EVENT_META[GovernanceEventType.MCP_ERROR]
        assert severity == "error"
        assert "mcp" in title.lower() or "error" in title.lower()

    def test_skill_invoked_in_meta(self) -> None:
        from dendrux.dashboard.normalizer import GOVERNANCE_EVENT_META

        severity, title = GOVERNANCE_EVENT_META[GovernanceEventType.SKILL_INVOKED]
        assert severity == "info"
        assert "skill" in title.lower() or "invoked" in title.lower()

    def test_governance_event_types_frozenset_updated(self) -> None:
        """The frozenset used for node dispatch includes new types."""
        from dendrux.dashboard.normalizer import _GOVERNANCE_EVENT_TYPES

        assert GovernanceEventType.SKILL_REGISTERED in _GOVERNANCE_EVENT_TYPES
        assert GovernanceEventType.SKILL_DENIED in _GOVERNANCE_EVENT_TYPES
        assert GovernanceEventType.SKILL_INVOKED in _GOVERNANCE_EVENT_TYPES
        assert GovernanceEventType.MCP_CONNECTED in _GOVERNANCE_EVENT_TYPES
        assert GovernanceEventType.MCP_ERROR in _GOVERNANCE_EVENT_TYPES

    async def test_normalizer_creates_governance_node_for_skill_loaded(self) -> None:
        """Normalizer converts skill.registered DB event into GovernanceEventNode."""
        from dendrux.dashboard.normalizer import (
            GovernanceEventNode,
            normalize_timeline,
        )

        @dataclass
        class _Event:
            id: str = ""
            event_type: str = ""
            sequence_index: int = 0
            iteration_index: int = 0
            correlation_id: str | None = None
            data: dict[str, Any] | None = None
            created_at: datetime | None = None

        @dataclass
        class _Run:
            id: str = ""
            agent_name: str = ""
            status: str = "success"
            input_data: dict[str, Any] | None = None
            output_data: dict[str, Any] | None = None
            answer: str | None = None
            error: str | None = None
            iteration_count: int = 0
            model: str | None = None
            strategy: str | None = None
            parent_run_id: str | None = None
            delegation_level: int = 0
            total_input_tokens: int = 0
            total_output_tokens: int = 0
            total_cost_usd: float | None = None
            total_cache_read_tokens: int = 0
            total_cache_creation_tokens: int = 0
            meta: dict[str, Any] | None = None
            created_at: datetime | None = None
            updated_at: datetime | None = None

        @dataclass
        class MockStore:
            _run: _Run | None = None
            _events: list[_Event] = field(default_factory=list)
            _traces: list[Any] = field(default_factory=list)
            _tool_calls: list[Any] = field(default_factory=list)

            async def get_run(self, run_id: str) -> _Run | None:
                return self._run if self._run and self._run.id == run_id else None

            async def get_run_events(self, run_id: str) -> list[_Event]:
                return self._events

            async def get_traces(self, run_id: str) -> list[Any]:
                return self._traces

            async def get_tool_calls(self, run_id: str) -> list[Any]:
                return self._tool_calls

        store = MockStore(
            _run=_Run(id="r1", agent_name="A", status="success"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "A"},
                ),
                _Event(
                    id="e1",
                    event_type="skill.registered",
                    sequence_index=1,
                    iteration_index=0,
                    data={
                        "skill_name": "pdf-processing",
                        "description": "Extract text from PDFs",
                    },
                ),
                _Event(
                    id="e2",
                    event_type="run.completed",
                    sequence_index=2,
                    data={"status": "success"},
                ),
            ],
        )

        result = await normalize_timeline("r1", store)
        assert result is not None
        # run.started + governance + finish = 3 nodes
        assert len(result.nodes) == 3
        gov_node = result.nodes[1]
        assert isinstance(gov_node, GovernanceEventNode)
        assert gov_node.event_type == "skill.registered"
        assert gov_node.severity == "info"
        assert gov_node.data["skill_name"] == "pdf-processing"


# ------------------------------------------------------------------
# skill.invoked emission in ReAct loop
# ------------------------------------------------------------------


class TestSkillInvokedEmission:
    """skill.invoked is emitted when LLM calls use_skill during the loop."""

    async def test_use_skill_emits_skill_invoked(self) -> None:
        """Successful use_skill tool call emits skill.invoked governance event."""
        from pathlib import Path

        from dendrux.llm.mock import MockLLM
        from dendrux.skills import Skill
        from dendrux.types import LLMResponse
        from dendrux.types import ToolCall as ToolCallType

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        provider = MockLLM(
            responses=[
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCallType(name="use_skill", params={"name": "pdf-processing"})],
                ),
                LLMResponse(text="Done."),
            ]
        )

        agent = Agent(
            prompt="Test.",
            provider=provider,
            tools=[dummy_tool],
            skills=[skill],
        )

        notifier = RecordingNotifier()
        await agent.run("Use the skill.", notifier=notifier)

        invoked = [
            (et, it, d)
            for et, it, d, _ in notifier.governance_events
            if et == GovernanceEventType.SKILL_INVOKED
        ]
        assert len(invoked) == 1
        assert invoked[0][2]["skill_name"] == "pdf-processing"
        assert invoked[0][1] == 1  # iteration 1 (first loop iteration)

    async def test_use_skill_unknown_still_emits_invoked(self) -> None:
        """use_skill with unknown name still emits skill.invoked (LLM invoked the mechanism)."""
        from pathlib import Path

        from dendrux.llm.mock import MockLLM
        from dendrux.skills import Skill
        from dendrux.types import LLMResponse
        from dendrux.types import ToolCall as ToolCallType

        fixtures = Path(__file__).parent.parent / "fixtures" / "skills"
        skill = Skill.from_dir(fixtures / "pdf-processing")

        provider = MockLLM(
            responses=[
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCallType(name="use_skill", params={"name": "nonexistent"})],
                ),
                LLMResponse(text="Done."),
            ]
        )

        agent = Agent(
            prompt="Test.",
            provider=provider,
            tools=[dummy_tool],
            skills=[skill],
        )

        notifier = RecordingNotifier()
        await agent.run("Try a skill.", notifier=notifier)

        invoked = [
            (et, d)
            for et, _, d, _ in notifier.governance_events
            if et == GovernanceEventType.SKILL_INVOKED
        ]
        assert len(invoked) == 1
        assert invoked[0][1]["skill_name"] == "nonexistent"
