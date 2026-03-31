"""Tests for the dashboard timeline normalizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from dendrux.dashboard.normalizer import (
    CancelledNode,
    ErrorNode,
    FinishNode,
    LLMCallNode,
    PauseSegmentNode,
    RunStartedNode,
    ToolCallNode,
    normalize_timeline,
    timeline_to_dict,
)

# ------------------------------------------------------------------
# Mock state store for normalizer tests
# ------------------------------------------------------------------

_T0 = datetime(2026, 3, 15, 10, 0, 0)


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
    meta: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class _Trace:
    id: str = ""
    role: str = ""
    content: str = ""
    order_index: int = 0
    meta: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass
class _ToolCall:
    id: str = ""
    tool_call_id: str = ""
    provider_tool_call_id: str | None = None
    tool_name: str = ""
    target: str = "server"
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    success: bool = True
    duration_ms: int | None = None
    iteration_index: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None


@dataclass
class NormalizerMockStore:
    """Mock store with pre-set data for normalizer tests."""

    _run: _Run | None = None
    _events: list[_Event] = field(default_factory=list)
    _traces: list[_Trace] = field(default_factory=list)
    _tool_calls: list[_ToolCall] = field(default_factory=list)

    async def get_run(self, run_id: str) -> _Run | None:
        return self._run if self._run and self._run.id == run_id else None

    async def get_run_events(self, run_id: str) -> list[_Event]:
        return self._events

    async def get_traces(self, run_id: str) -> list[_Trace]:
        return self._traces

    async def get_tool_calls(self, run_id: str) -> list[_ToolCall]:
        return self._tool_calls


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestNormalizeTimeline:
    async def test_returns_none_for_nonexistent_run(self) -> None:
        store = NormalizerMockStore()
        result = await normalize_timeline("nonexistent", store)
        assert result is None

    async def test_simple_run_started_completed(self) -> None:
        """Minimal run: started → llm → completed."""
        store = NormalizerMockStore(
            _run=_Run(
                id="r1",
                agent_name="TestAgent",
                status="success",
                input_data={"input": "hello"},
                answer="world",
                iteration_count=1,
                total_input_tokens=100,
                total_output_tokens=50,
                model="claude-sonnet",
                created_at=_T0,
            ),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "TestAgent"},
                    created_at=_T0,
                ),
                _Event(
                    id="e1",
                    event_type="llm.completed",
                    sequence_index=1,
                    iteration_index=1,
                    data={"input_tokens": 100, "output_tokens": 50, "model": "claude-sonnet"},
                    created_at=_T0 + timedelta(seconds=1),
                ),
                _Event(
                    id="e2",
                    event_type="run.completed",
                    sequence_index=2,
                    data={"status": "success"},
                    created_at=_T0 + timedelta(seconds=2),
                ),
            ],
            _traces=[
                _Trace(role="user", content="hello", order_index=0, meta={"iteration": 0}),
                _Trace(role="assistant", content="world", order_index=1, meta={"iteration": 1}),
            ],
        )

        result = await normalize_timeline("r1", store)
        assert result is not None
        assert result.summary.run_id == "r1"
        assert result.summary.agent_name == "TestAgent"
        assert result.summary.answer == "world"
        assert len(result.nodes) == 3

        assert isinstance(result.nodes[0], RunStartedNode)
        assert result.nodes[0].agent_name == "TestAgent"

        assert isinstance(result.nodes[1], LLMCallNode)
        assert result.nodes[1].input_tokens == 100
        assert result.nodes[1].iteration == 1
        assert result.nodes[1].assistant_text == "world"

        assert isinstance(result.nodes[2], FinishNode)
        assert result.nodes[2].status == "success"

    async def test_tool_call_enrichment(self) -> None:
        """Tool events are enriched with params/result from tool_calls table."""
        store = NormalizerMockStore(
            _run=_Run(id="r2", agent_name="A", status="success"),
            _events=[
                _Event(
                    id="e0",
                    event_type="tool.completed",
                    sequence_index=0,
                    iteration_index=1,
                    correlation_id="tc_123",
                    data={
                        "tool_name": "lookup",
                        "target": "server",
                        "success": True,
                        "duration_ms": 23,
                    },
                ),
            ],
            _tool_calls=[
                _ToolCall(
                    tool_call_id="tc_123",
                    tool_name="lookup",
                    params={"ticker": "AAPL"},
                    result={"price": 227.50},
                    success=True,
                    duration_ms=23,
                ),
            ],
        )

        result = await normalize_timeline("r2", store)
        assert result is not None
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert isinstance(node, ToolCallNode)
        assert node.tool_name == "lookup"
        assert node.params == {"ticker": "AAPL"}
        assert node.result == {"price": 227.50}
        assert node.duration_ms == 23

    async def test_pause_resume_segment(self) -> None:
        """Pause and resume events are merged into a single PauseSegmentNode."""
        paused_at = _T0 + timedelta(seconds=5)
        resumed_at = _T0 + timedelta(seconds=9, milliseconds=200)

        store = NormalizerMockStore(
            _run=_Run(id="r3", agent_name="A", status="success"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.paused",
                    sequence_index=0,
                    data={
                        "status": "waiting_client_tool",
                        "pending_tool_calls": [
                            {"id": "tc1", "name": "read_excel", "target": "client"},
                        ],
                    },
                    created_at=paused_at,
                ),
                _Event(
                    id="e1",
                    event_type="run.resumed",
                    sequence_index=1,
                    data={
                        "resumed_from": "waiting_client_tool",
                        "submitted_results": [
                            {"call_id": "tc1", "name": "read_excel", "success": True},
                        ],
                    },
                    created_at=resumed_at,
                ),
            ],
        )

        result = await normalize_timeline("r3", store)
        assert result is not None
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert isinstance(node, PauseSegmentNode)
        assert node.pause_status == "waiting_client_tool"
        assert len(node.pending_tool_calls) == 1
        assert node.pending_tool_calls[0].tool_name == "read_excel"
        assert node.paused_at == paused_at
        assert node.resumed_at == resumed_at
        assert node.wait_duration_ms == 4200
        assert len(node.submitted_results) == 1
        assert node.submitted_results[0].call_id == "tc1"

    async def test_pause_without_resume_still_pending(self) -> None:
        """A pause with no resume (run still waiting) produces a segment with no resume data."""
        store = NormalizerMockStore(
            _run=_Run(id="r4", agent_name="A", status="waiting_client_tool"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.paused",
                    sequence_index=0,
                    data={
                        "status": "waiting_client_tool",
                        "pending_tool_calls": [
                            {"id": "tc1", "name": "read_excel", "target": "client"},
                        ],
                    },
                    created_at=_T0,
                ),
            ],
        )

        result = await normalize_timeline("r4", store)
        assert result is not None
        node = result.nodes[0]
        assert isinstance(node, PauseSegmentNode)
        assert node.resumed_at is None
        assert node.wait_duration_ms is None
        assert len(node.submitted_results) == 0

    async def test_error_node(self) -> None:
        store = NormalizerMockStore(
            _run=_Run(id="r5", agent_name="A", status="error", error="LLM exploded"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.error",
                    sequence_index=0,
                    data={"error": "LLM exploded"},
                    created_at=_T0,
                ),
            ],
        )

        result = await normalize_timeline("r5", store)
        assert result is not None
        node = result.nodes[0]
        assert isinstance(node, ErrorNode)
        assert node.error == "LLM exploded"

    async def test_cancelled_node(self) -> None:
        """Cancelled runs produce a CancelledNode."""
        store = NormalizerMockStore(
            _run=_Run(id="r_can", agent_name="A", status="cancelled"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "A"},
                    created_at=_T0,
                ),
                _Event(
                    id="e1",
                    event_type="run.cancelled",
                    sequence_index=1,
                    created_at=_T0,
                ),
            ],
        )

        result = await normalize_timeline("r_can", store)
        assert result is not None
        assert len(result.nodes) == 2
        assert isinstance(result.nodes[0], RunStartedNode)
        assert isinstance(result.nodes[1], CancelledNode)

    async def test_full_pause_resume_lifecycle(self) -> None:
        """Full lifecycle: started → llm → tool → pause → resume → llm → completed."""
        store = NormalizerMockStore(
            _run=_Run(
                id="r6",
                agent_name="SpreadsheetAnalyst",
                status="success",
                input_data={"input": "analyze"},
                answer="Done",
                iteration_count=2,
                model="claude-sonnet",
            ),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "SpreadsheetAnalyst"},
                    created_at=_T0,
                ),
                _Event(
                    id="e1",
                    event_type="llm.completed",
                    sequence_index=1,
                    iteration_index=1,
                    data={"input_tokens": 500, "output_tokens": 200, "has_tool_calls": True},
                    created_at=_T0 + timedelta(seconds=1),
                ),
                _Event(
                    id="e2",
                    event_type="tool.completed",
                    sequence_index=2,
                    iteration_index=1,
                    correlation_id="tc_srv",
                    data={
                        "tool_name": "lookup_price",
                        "target": "server",
                        "success": True,
                        "duration_ms": 23,
                    },
                    created_at=_T0 + timedelta(seconds=1, milliseconds=50),
                ),
                _Event(
                    id="e3",
                    event_type="run.paused",
                    sequence_index=3,
                    data={
                        "status": "waiting_client_tool",
                        "pending_tool_calls": [
                            {"id": "tc_cli", "name": "read_excel", "target": "client"},
                        ],
                    },
                    created_at=_T0 + timedelta(seconds=2),
                ),
                _Event(
                    id="e4",
                    event_type="run.resumed",
                    sequence_index=4,
                    data={
                        "resumed_from": "waiting_client_tool",
                        "submitted_results": [
                            {"call_id": "tc_cli", "name": "read_excel", "success": True},
                        ],
                    },
                    created_at=_T0 + timedelta(seconds=6),
                ),
                _Event(
                    id="e5",
                    event_type="llm.completed",
                    sequence_index=5,
                    iteration_index=2,
                    data={"input_tokens": 300, "output_tokens": 100},
                    created_at=_T0 + timedelta(seconds=7),
                ),
                _Event(
                    id="e6",
                    event_type="run.completed",
                    sequence_index=6,
                    data={"status": "success"},
                    created_at=_T0 + timedelta(seconds=8),
                ),
            ],
            _traces=[
                _Trace(
                    role="assistant",
                    content="Let me look up...",
                    order_index=1,
                    meta={"iteration": 1},
                ),
                _Trace(role="assistant", content="Done", order_index=3, meta={"iteration": 2}),
            ],
            _tool_calls=[
                _ToolCall(
                    tool_call_id="tc_srv",
                    tool_name="lookup_price",
                    params={"ticker": "AAPL"},
                    result={"raw": "AAPL: $227.50"},
                    success=True,
                    duration_ms=23,
                ),
            ],
        )

        result = await normalize_timeline("r6", store)
        assert result is not None
        assert result.summary.agent_name == "SpreadsheetAnalyst"
        assert len(result.nodes) == 6

        # Verify node types in order
        assert isinstance(result.nodes[0], RunStartedNode)
        assert isinstance(result.nodes[1], LLMCallNode)
        assert isinstance(result.nodes[2], ToolCallNode)
        assert isinstance(result.nodes[3], PauseSegmentNode)
        assert isinstance(result.nodes[4], LLMCallNode)
        assert isinstance(result.nodes[5], FinishNode)

        # Verify pause segment
        pause = result.nodes[3]
        assert isinstance(pause, PauseSegmentNode)
        assert pause.wait_duration_ms == 4000
        assert len(pause.pending_tool_calls) == 1
        assert pause.pending_tool_calls[0].tool_name == "read_excel"
        assert len(pause.submitted_results) == 1

        # Verify tool enrichment
        tool = result.nodes[2]
        assert isinstance(tool, ToolCallNode)
        assert tool.params == {"ticker": "AAPL"}

        # Verify LLM enrichment
        llm1 = result.nodes[1]
        assert isinstance(llm1, LLMCallNode)
        assert llm1.assistant_text == "Let me look up..."

    async def test_system_prompt_from_run_started_event(self) -> None:
        """System prompt is extracted from run.started event data, not traces."""
        store = NormalizerMockStore(
            _run=_Run(id="r_sp", agent_name="A", status="success"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "A", "system_prompt": "You are a helpful assistant."},
                ),
            ],
        )

        result = await normalize_timeline("r_sp", store)
        assert result is not None
        assert result.system_prompt == "You are a helpful assistant."

    async def test_system_prompt_none_without_event_data(self) -> None:
        """system_prompt is None when run.started has no prompt."""
        store = NormalizerMockStore(
            _run=_Run(id="r_np", agent_name="A", status="success"),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "A"},
                ),
            ],
        )

        result = await normalize_timeline("r_np", store)
        assert result is not None
        assert result.system_prompt is None

    async def test_messages_by_iteration(self) -> None:
        """Messages are grouped by iteration for payload inspection."""
        store = NormalizerMockStore(
            _run=_Run(id="r7", agent_name="A", status="success"),
            _events=[],
            _traces=[
                _Trace(role="user", content="hello", order_index=0, meta={"iteration": 0}),
                _Trace(role="assistant", content="hi", order_index=1, meta={"iteration": 1}),
                _Trace(role="tool", content="{}", order_index=2, meta={"iteration": 1}),
                _Trace(role="assistant", content="done", order_index=3, meta={"iteration": 2}),
            ],
        )

        result = await normalize_timeline("r7", store)
        assert result is not None
        assert 0 in result.messages_by_iteration
        assert 1 in result.messages_by_iteration
        assert 2 in result.messages_by_iteration
        assert len(result.messages_by_iteration[0]) == 1
        assert len(result.messages_by_iteration[1]) == 2
        assert len(result.messages_by_iteration[2]) == 1

    async def test_sequence_index_preserved_on_nodes(self) -> None:
        """Each node carries its sequence_index for frontend ordering."""
        store = NormalizerMockStore(
            _run=_Run(id="r8", agent_name="A", status="success"),
            _events=[
                _Event(id="e0", event_type="run.started", sequence_index=0),
                _Event(id="e1", event_type="llm.completed", sequence_index=1, iteration_index=1),
                _Event(id="e2", event_type="run.completed", sequence_index=2),
            ],
        )

        result = await normalize_timeline("r8", store)
        assert result is not None
        assert result.nodes[0].sequence_index == 0
        assert result.nodes[1].sequence_index == 1
        assert result.nodes[2].sequence_index == 2


class TestTimelineToDict:
    async def test_serialization_roundtrip(self) -> None:
        """timeline_to_dict produces JSON-safe output."""
        store = NormalizerMockStore(
            _run=_Run(
                id="r_ser",
                agent_name="A",
                status="success",
                created_at=_T0,
            ),
            _events=[
                _Event(
                    id="e0",
                    event_type="run.started",
                    sequence_index=0,
                    data={"agent_name": "A"},
                    created_at=_T0,
                ),
            ],
        )

        result = await normalize_timeline("r_ser", store)
        assert result is not None
        d = timeline_to_dict(result)

        assert d["summary"]["run_id"] == "r_ser"
        assert d["summary"]["agent_name"] == "A"
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["type"] == "run_started"

        # Should be JSON-serializable
        import json

        json.dumps(d)  # Raises if not serializable
