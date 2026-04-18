"""Tests for resume/resume_with_input (Sprint 3, Group 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import resume, resume_with_input, run
from dendrux.tool import tool
from dendrux.types import (
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunStatus,
    ToolCall,
    ToolResult,
    UsageStats,
)
from tests._helpers.state_store_mocks import CancellationStubsMixin

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def server_add(a: int, b: int) -> int:
    """Server-side add."""
    return a + b


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Client-side tool."""
    return "should never run"


def _make_agent(**overrides) -> Agent:
    defaults = {
        "prompt": "You are a test agent.",
        "tools": [server_add, read_range],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


# ------------------------------------------------------------------
# Mock StateStore for resume tests
# ------------------------------------------------------------------


@dataclass
class RecordingStateStore(CancellationStubsMixin):
    """Fake StateStore that records calls and supports pause/resume."""

    created_runs: list[dict[str, Any]] = field(default_factory=list)
    finalized_runs: list[dict[str, Any]] = field(default_factory=list)
    paused_runs: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_saved: list[dict[str, Any]] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)
    _pause_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    _run_status: dict[str, str] = field(default_factory=dict)
    _claimed: set[str] = field(default_factory=set)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def create_run(self, run_id: str, agent_name: str, **kwargs: Any):
        self.created_runs.append({"run_id": run_id, "agent_name": agent_name, **kwargs})
        self._run_status[run_id] = "running"
        from dendrux.types import CreateRunResult, RunStatus

        return CreateRunResult(run_id=run_id, outcome="created", status=RunStatus.RUNNING)

    async def save_trace(
        self, run_id: str, role: str, content: str, *, order_index: int, meta: Any = None
    ) -> None:
        self.traces.append(
            {
                "run_id": run_id,
                "role": role,
                "content": content,
                "order_index": order_index,
                "meta": meta,
            }
        )

    async def save_tool_call(self, run_id: str, **kwargs: Any) -> None:
        self.tool_calls_saved.append({"run_id": run_id, **kwargs})

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        self.usages.append({"run_id": run_id, **kwargs})

    async def finalize_run(self, run_id: str, **kwargs: Any) -> bool:
        expected = kwargs.pop("expected_current_status", None)
        if expected is not None and self._run_status.get(run_id) != expected:
            return False
        self.finalized_runs.append({"run_id": run_id, **kwargs})
        self._run_status[run_id] = kwargs.get("status", "success")
        self._pause_data.pop(run_id, None)
        return True

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
    ) -> None:
        self.paused_runs.append(
            {
                "run_id": run_id,
                "status": status,
                "pause_data": pause_data,
                "iteration_count": iteration_count,
            }
        )
        self._pause_data[run_id] = pause_data
        self._run_status[run_id] = status

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None:
        return self._pause_data.get(run_id)

    async def get_pii_mapping(self, run_id: str) -> dict[str, str] | None:
        return None

    async def claim_paused_run(self, run_id: str, *, expected_status: str) -> bool:
        if self._run_status.get(run_id) != expected_status:
            return False
        if run_id in self._claimed:
            return False
        self._claimed.add(run_id)
        self._run_status[run_id] = "running"
        return True

    async def get_run(self, run_id: str) -> Any:
        return None

    async def get_traces(self, run_id: str) -> list[Any]:
        """Return trace records for order_index calculation."""

        @dataclass
        class _Trace:
            order_index: int

        return [_Trace(order_index=t["order_index"]) for t in self.traces if t["run_id"] == run_id]

    async def get_tool_calls(self, run_id: str) -> list[Any]:
        return []

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self._events.setdefault(run_id, []).append(kwargs)

    async def get_run_events(self, run_id: str) -> list[Any]:
        @dataclass
        class _Event:
            sequence_index: int
            event_type: str = ""
            iteration_index: int = 0
            correlation_id: str | None = None
            data: dict[str, Any] | None = None

        return [
            _Event(
                sequence_index=e.get("sequence_index", 0),
                event_type=e.get("event_type", ""),
                iteration_index=e.get("iteration_index", 0),
                correlation_id=e.get("correlation_id"),
                data=e.get("data"),
            )
            for e in self._events.get(run_id, [])
        ]

    async def list_runs(self, **kwargs: Any) -> list[Any]:
        return []


# ------------------------------------------------------------------
# Resume tests
# ------------------------------------------------------------------


class TestResume:
    async def test_resume_completes_paused_run(self) -> None:
        """Pause on client tool, resume with result, finish SUCCESS."""
        # Phase 1: run until pause
        tc_client = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm_pause = MockLLM([LLMResponse(tool_calls=[tc_client])])
        agent = _make_agent()
        store = RecordingStateStore()

        result1 = await run(
            agent,
            provider=llm_pause,
            user_input="Read sheet",
            state_store=store,
        )
        assert result1.status == RunStatus.WAITING_CLIENT_TOOL
        assert len(store.paused_runs) == 1

        # Phase 2: resume with tool result
        pending = result1.meta["pause_state"].pending_tool_calls
        tool_result = ToolResult(
            name="read_range",
            call_id=pending[0].id,
            payload='{"data": [1, 2, 3]}',
            success=True,
        )
        llm_finish = MockLLM([LLMResponse(text="The data is [1, 2, 3]")])

        result2 = await resume(
            result1.run_id,
            [tool_result],
            state_store=store,
            agent=agent,
            provider=llm_finish,
        )
        assert result2.status == RunStatus.SUCCESS
        assert result2.answer == "The data is [1, 2, 3]"
        assert len(store.finalized_runs) == 1

    async def test_resume_persists_client_tool_call_records(self) -> None:
        """Client tool results are persisted via on_tool_completed, not just on_message_appended."""
        tc_client = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm_pause = MockLLM([LLMResponse(tool_calls=[tc_client])])
        agent = _make_agent()
        store = RecordingStateStore()

        r1 = await run(agent, provider=llm_pause, user_input="Read", state_store=store)
        assert r1.status == RunStatus.WAITING_CLIENT_TOOL

        # Clear tool_calls from the initial run to isolate resume's records
        store.tool_calls_saved.clear()

        pending = r1.meta["pause_state"].pending_tool_calls
        tr = ToolResult(
            name="read_range",
            call_id=pending[0].id,
            payload='{"data": "ok"}',
            success=True,
            duration_ms=100,
        )
        llm_finish = MockLLM([LLMResponse(text="Done")])
        await resume(r1.run_id, [tr], state_store=store, agent=agent, provider=llm_finish)

        # on_tool_completed should have been called for the client tool
        assert len(store.tool_calls_saved) >= 1
        client_tc = store.tool_calls_saved[0]
        assert client_tc["tool_name"] == "read_range"
        assert client_tc["success"] is True
        assert client_tc["result_payload"] == '{"data": "ok"}'

    async def test_resume_rejects_wrong_tool_results(self) -> None:
        """Providing results for wrong call_ids raises ValueError."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _make_agent()
        store = RecordingStateStore()

        result = await run(agent, provider=llm, user_input="Read", state_store=store)
        assert result.status == RunStatus.WAITING_CLIENT_TOOL

        wrong_result = ToolResult(
            name="read_range",
            call_id="wrong_id",
            payload="{}",
            success=True,
        )
        with pytest.raises(ValueError, match="do not match"):
            await resume(
                result.run_id,
                [wrong_result],
                state_store=store,
                agent=agent,
                provider=MockLLM([]),
            )

    async def test_validation_failure_does_not_claim_run(self) -> None:
        """Wrong call_ids should NOT transition the run to RUNNING."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _make_agent()
        store = RecordingStateStore()

        result = await run(agent, provider=llm, user_input="Read", state_store=store)
        assert result.status == RunStatus.WAITING_CLIENT_TOOL

        wrong_result = ToolResult(
            name="read_range",
            call_id="wrong_id",
            payload="{}",
            success=True,
        )
        with pytest.raises(ValueError, match="do not match"):
            await resume(
                result.run_id,
                [wrong_result],
                state_store=store,
                agent=agent,
                provider=MockLLM([]),
            )

        # Run should still be in WAITING status — not claimed/stuck as RUNNING
        assert store._run_status[result.run_id] == "waiting_client_tool"

    async def test_resume_rejects_non_paused_run(self) -> None:
        """Resuming a run that's not in WAITING status raises ValueError."""
        store = RecordingStateStore()
        # No pause data exists for this run
        with pytest.raises(ValueError, match="no pause state"):
            await resume(
                "nonexistent_run",
                [],
                state_store=store,
                agent=_make_agent(),
                provider=MockLLM([]),
            )

    async def test_double_pause_resume(self) -> None:
        """Agent pauses twice in sequence, both resume correctly."""
        agent = _make_agent()
        store = RecordingStateStore()

        # Phase 1: first pause
        tc1 = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm1 = MockLLM([LLMResponse(tool_calls=[tc1])])
        r1 = await run(agent, provider=llm1, user_input="Step 1", state_store=store)
        assert r1.status == RunStatus.WAITING_CLIENT_TOOL

        # Phase 2: resume, then pause again on a second client tool call
        pending1 = r1.meta["pause_state"].pending_tool_calls
        tr1 = ToolResult(name="read_range", call_id=pending1[0].id, payload='"data1"', success=True)
        tc2 = ToolCall(name="read_range", params={"sheet": "S2"}, provider_tool_call_id="t2")
        llm2 = MockLLM([LLMResponse(tool_calls=[tc2])])

        # Need to clear the claim so the second pause can be claimed
        store._claimed.clear()
        r2 = await resume(r1.run_id, [tr1], state_store=store, agent=agent, provider=llm2)
        assert r2.status == RunStatus.WAITING_CLIENT_TOOL

        # Phase 3: resume second pause to completion
        pending2 = r2.meta["pause_state"].pending_tool_calls
        tr2 = ToolResult(name="read_range", call_id=pending2[0].id, payload='"data2"', success=True)
        llm3 = MockLLM([LLMResponse(text="Got both: data1 and data2")])

        store._claimed.clear()
        r3 = await resume(r1.run_id, [tr2], state_store=store, agent=agent, provider=llm3)
        assert r3.status == RunStatus.SUCCESS
        assert r3.answer == "Got both: data1 and data2"

    async def test_iteration_count_continues(self) -> None:
        """After resume, iteration count continues from where it paused."""
        agent = _make_agent()
        store = RecordingStateStore()

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm1 = MockLLM([LLMResponse(tool_calls=[tc])])
        r1 = await run(agent, provider=llm1, user_input="Go", state_store=store)
        assert r1.iteration_count == 1

        pending = r1.meta["pause_state"].pending_tool_calls
        tr = ToolResult(name="read_range", call_id=pending[0].id, payload='"ok"', success=True)
        llm2 = MockLLM([LLMResponse(text="Done")])

        r2 = await resume(r1.run_id, [tr], state_store=store, agent=agent, provider=llm2)
        assert r2.iteration_count == 2  # continued from 1

    async def test_final_run_result_has_all_steps(self) -> None:
        """Steps span the full run including pre-pause steps."""
        agent = _make_agent()
        store = RecordingStateStore()

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm1 = MockLLM([LLMResponse(tool_calls=[tc])])
        r1 = await run(agent, provider=llm1, user_input="Go", state_store=store)
        assert len(r1.steps) == 1  # one step before pause

        pending = r1.meta["pause_state"].pending_tool_calls
        tr = ToolResult(name="read_range", call_id=pending[0].id, payload='"ok"', success=True)
        llm2 = MockLLM([LLMResponse(text="Done")])

        r2 = await resume(r1.run_id, [tr], state_store=store, agent=agent, provider=llm2)
        # Should have both pre-pause and post-resume steps
        assert len(r2.steps) == 2

    async def test_recorder_order_index_continues(self) -> None:
        """Trace order_index continues from max existing index after resume."""
        agent = _make_agent()
        store = RecordingStateStore()

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm1 = MockLLM([LLMResponse(tool_calls=[tc])])
        await run(agent, provider=llm1, user_input="Go", state_store=store)

        pre_pause_indices = [t["order_index"] for t in store.traces]
        max_pre = max(pre_pause_indices)

        pending_data = store.paused_runs[0]["pause_data"]
        ps = PauseState.from_dict(pending_data)
        tr = ToolResult(
            name="read_range", call_id=ps.pending_tool_calls[0].id, payload='"ok"', success=True
        )
        llm2 = MockLLM([LLMResponse(text="Done")])

        await resume(
            store.paused_runs[0]["run_id"], [tr], state_store=store, agent=agent, provider=llm2
        )

        post_resume_indices = [t["order_index"] for t in store.traces if t["order_index"] > max_pre]
        assert all(i > max_pre for i in post_resume_indices)
        # No duplicate indices
        all_indices = [t["order_index"] for t in store.traces]
        assert len(all_indices) == len(set(all_indices))

    async def test_atomic_claim_prevents_double_resume(self) -> None:
        """Second claim returns False — cannot resume twice."""
        agent = _make_agent()
        store = RecordingStateStore()

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        r = await run(agent, provider=llm, user_input="Go", state_store=store)

        # First claim succeeds
        claimed1 = await store.claim_paused_run(r.run_id, expected_status="waiting_client_tool")
        assert claimed1 is True

        # Second claim fails (already claimed)
        claimed2 = await store.claim_paused_run(r.run_id, expected_status="waiting_client_tool")
        assert claimed2 is False

    async def test_pause_data_cleared_on_finalize(self) -> None:
        """pause_data is removed from store after successful finalize."""
        agent = _make_agent()
        store = RecordingStateStore()

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm1 = MockLLM([LLMResponse(tool_calls=[tc])])
        r = await run(agent, provider=llm1, user_input="Go", state_store=store)

        # Pause data exists
        assert await store.get_pause_state(r.run_id) is not None

        # Resume to completion
        pending = r.meta["pause_state"].pending_tool_calls
        tr = ToolResult(name="read_range", call_id=pending[0].id, payload='"ok"', success=True)
        llm2 = MockLLM([LLMResponse(text="Done")])
        await resume(r.run_id, [tr], state_store=store, agent=agent, provider=llm2)

        # Pause data cleared
        assert await store.get_pause_state(r.run_id) is None


class TestResumeWithInput:
    async def test_resume_with_input_via_manual_pause_state(self) -> None:
        """WAITING_HUMAN_INPUT → user input → SUCCESS (manually constructed state)."""
        agent = _make_agent(tools=[server_add])
        store = RecordingStateStore()

        pause_state = PauseState(
            agent_name=agent.name,
            pending_tool_calls=[],
            history=[
                Message(role=Role.USER, content="What file?"),
                Message(role=Role.ASSISTANT, content="Which file should I analyze?"),
            ],
            steps=[],
            iteration=1,
            trace_order_offset=2,
            usage=UsageStats(input_tokens=50, output_tokens=20, total_tokens=70),
        )
        run_id = "test_clarification_run"
        store._pause_data[run_id] = pause_state.to_dict()
        store._run_status[run_id] = "waiting_human_input"

        llm = MockLLM([LLMResponse(text="I'll analyze report.csv")])
        result = await resume_with_input(
            run_id,
            "report.csv",
            state_store=store,
            agent=agent,
            provider=llm,
        )
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "I'll analyze report.csv"

        call_msgs = llm.call_history[0]["messages"]
        user_msgs = [m for m in call_msgs if m.role == Role.USER]
        assert any("report.csv" in m.content for m in user_msgs)

    async def test_clarification_end_to_end_through_runner(self) -> None:
        """Full path: run() → Clarification → resume_with_input() → SUCCESS.

        Exercises the real loop → runner → state store → resume path
        without manually constructing PauseState.
        """
        from dendrux.strategies.base import Strategy
        from dendrux.types import AgentStep
        from dendrux.types import Clarification as Clar

        class ClarifyStrategy(Strategy):
            """Always returns Clarification."""

            def build_messages(self, *, system_prompt, history, tool_defs):  # type: ignore[override]
                msgs = [Message(role=Role.SYSTEM, content=system_prompt), *history]
                return msgs, tool_defs or None

            def parse_response(self, response):  # type: ignore[override]
                return AgentStep(reasoning=response.text, action=Clar(question="Which file?"))

            def format_tool_result(self, result):  # type: ignore[override]
                return Message(
                    role=Role.TOOL, content=result.payload, name=result.name, call_id=result.call_id
                )

        agent = _make_agent(tools=[server_add])
        store = RecordingStateStore()

        # Phase 1: run() with ClarifyStrategy → WAITING_HUMAN_INPUT
        llm1 = MockLLM([LLMResponse(text="I need to ask")])
        r1 = await run(
            agent,
            provider=llm1,
            user_input="Analyze something",
            state_store=store,
            strategy=ClarifyStrategy(),
        )
        assert r1.status == RunStatus.WAITING_HUMAN_INPUT
        assert r1.answer == "Which file?"
        assert len(store.paused_runs) == 1
        assert store.paused_runs[0]["status"] == "waiting_human_input"

        # Verify pause_state was persisted (not manually created)
        raw_pause = await store.get_pause_state(r1.run_id)
        assert raw_pause is not None
        assert raw_pause["agent_name"] == agent.name

        # Phase 2: resume_with_input() with NativeToolCalling (text → Finish)
        llm2 = MockLLM([LLMResponse(text="Analyzing report.csv now")])
        r2 = await resume_with_input(
            r1.run_id,
            "report.csv",
            state_store=store,
            agent=agent,
            provider=llm2,
            # NativeToolCalling is the default — plain text LLM response → Finish
        )
        assert r2.status == RunStatus.SUCCESS
        assert r2.answer == "Analyzing report.csv now"

        # Steps should span both phases
        assert len(r2.steps) >= 2  # clarification step + finish step

    async def test_resume_with_input_rejects_non_waiting_human(self) -> None:
        """resume_with_input on WAITING_CLIENT_TOOL raises ValueError."""
        agent = _make_agent()
        store = RecordingStateStore()

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        r = await run(agent, provider=llm, user_input="Go", state_store=store)
        assert r.status == RunStatus.WAITING_CLIENT_TOOL

        with pytest.raises(ValueError, match="not in status"):
            await resume_with_input(
                r.run_id,
                "some input",
                state_store=store,
                agent=agent,
                provider=MockLLM([]),
            )


class TestAgentIdentityOnResume:
    """H-003: resume must verify agent_name matches the paused run."""

    async def test_mismatched_agent_name_raises(self) -> None:
        """Resume with a different agent name fails closed."""
        agent_a = _make_agent(name="AgentA")
        store = RecordingStateStore()

        # Create a paused run under AgentA
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        r = await run(agent_a, provider=llm, user_input="Go", state_store=store)
        assert r.status == RunStatus.WAITING_CLIENT_TOOL

        # Try to resume with AgentB
        agent_b = _make_agent(name="AgentB")
        tr = ToolResult(
            name="read_range",
            call_id=r.meta["pause_state"].pending_tool_calls[0].id,
            payload='"data"',
        )
        with pytest.raises(ValueError, match="Agent name mismatch"):
            await resume(
                r.run_id,
                [tr],
                state_store=store,
                agent=agent_b,
                provider=MockLLM([LLMResponse(text="done")]),
            )
