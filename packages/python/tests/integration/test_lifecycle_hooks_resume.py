"""Lifecycle hook coverage for resume / submit_* paths.

PR A added run-level lifecycle hooks (on_run_started / on_run_finished /
on_run_failed) and tool_started. These tests verify the hooks fire on the
resume path too — not just on the initial agent.run() — so OTel root spans
opened on submit_* operations close cleanly.

Also verifies that a client-tool result submitted via submit_tool_results
fires on_tool_started before on_tool_completed, matching the start/complete
pairing contract documented on LoopNotifier.
"""

from __future__ import annotations

from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@tool()
async def refund(order_id: int) -> str:
    return f"refunded {order_id}"


@pytest.fixture
def db_store(engine):
    return SQLAlchemyStateStore(engine)


class CapturingNotifier(BaseNotifier):
    """Records every hook call name + run_id."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def on_run_started(self, run_id, *, agent_name=None, agent_model=None) -> None:
        self.calls.append(("on_run_started", (run_id,)))

    async def on_run_finished(self, run_id, result) -> None:
        self.calls.append(("on_run_finished", (run_id, result)))

    async def on_run_failed(self, run_id, error, *, iteration=None) -> None:
        self.calls.append(("on_run_failed", (run_id, error)))

    async def on_message_appended(self, run_id, message, iteration) -> None:
        self.calls.append(("on_message_appended", (run_id,)))

    async def on_llm_call_started(
        self, run_id, iteration, *, semantic_messages=None, semantic_tools=None
    ) -> None:
        self.calls.append(("on_llm_call_started", (run_id,)))

    async def on_llm_call_completed(self, run_id, response, iteration, **kwargs) -> None:
        self.calls.append(("on_llm_call_completed", (run_id,)))

    async def on_llm_call_failed(self, run_id, iteration, error, *, duration_ms=None) -> None:
        self.calls.append(("on_llm_call_failed", (run_id,)))

    async def on_tool_started(self, run_id, tool_call, iteration) -> None:
        self.calls.append(("on_tool_started", (run_id, tool_call.name)))

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
        self.calls.append(("on_tool_completed", (run_id, tool_call.name)))

    async def on_governance_event(
        self, run_id, event_type, iteration, data, *, correlation_id=None
    ) -> None:
        self.calls.append(("on_governance_event", (run_id, event_type)))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


def _client_tool_agent(llm: MockLLM, store: SQLAlchemyStateStore) -> Agent:
    return Agent(
        provider=llm,
        prompt="Test agent.",
        tools=[read_range],
        state_store=store,
    )


def _approval_agent(llm: MockLLM, store: SQLAlchemyStateStore) -> Agent:
    return Agent(
        provider=llm,
        prompt="Support agent.",
        tools=[refund],
        require_approval=["refund"],
        state_store=store,
    )


# ---------------------------------------------------------------------------
# submit_tool_results — resume after client-tool pause
# ---------------------------------------------------------------------------


class TestSubmitToolResultsLifecycle:
    async def test_fires_run_started_finished_on_resume(self, db_store) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)

        paused = await agent.run("Read the sheet")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL

        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        n = CapturingNotifier()
        result = await agent.submit_tool_results(
            paused.run_id,
            [ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
            notifier=n,
        )
        assert result.status == RunStatus.SUCCESS

        names = n.names()
        assert "on_run_started" in names, (
            "submit_tool_results goes through _resume_core which never fires "
            "on_run_started — OTel root span will never open for this turn."
        )
        assert "on_run_finished" in names, (
            "_resume_core never fires on_run_finished after _persist_loop_outcome."
        )
        assert names.index("on_run_started") < names.index("on_run_finished")

    async def test_fires_tool_started_for_submitted_client_tool(self, db_store) -> None:
        """Client-tool result must arrive on the recorder/notifier as a
        start/complete pair, not just complete. Otherwise OTel notifiers
        cannot wrap a span around the client-tool execution."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)

        paused = await agent.run("Read")
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        n = CapturingNotifier()
        await agent.submit_tool_results(
            paused.run_id,
            [ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
            notifier=n,
        )

        names = n.names()
        starts = [c for c in n.calls if c[0] == "on_tool_started" and c[1][1] == "read_range"]
        completes = [c for c in n.calls if c[0] == "on_tool_completed" and c[1][1] == "read_range"]
        assert starts, (
            "Client-tool result has no matching on_tool_started — start/complete "
            "pairing is broken for client tools."
        )
        assert completes
        assert names.index("on_tool_started") < names.index("on_tool_completed")


# ---------------------------------------------------------------------------
# submit_approval — approval-approve resume path
# ---------------------------------------------------------------------------


class TestSubmitApprovalLifecycle:
    async def test_fires_run_started_finished_on_approve(self, db_store) -> None:
        tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="processed")])
        agent = _approval_agent(llm, db_store)

        paused = await agent.run("refund 7")
        assert paused.status == RunStatus.WAITING_APPROVAL

        n = CapturingNotifier()
        result = await agent.submit_approval(paused.run_id, approved=True, notifier=n)
        assert result.status == RunStatus.SUCCESS

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names
        assert names.index("on_run_started") < names.index("on_run_finished")


# ---------------------------------------------------------------------------
# resume_stream — streaming resume path
# ---------------------------------------------------------------------------


class TestResumeStreamLifecycle:
    async def test_fires_run_started_finished(self, db_store) -> None:
        from dendrux.types import RunEventType

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)

        paused = await agent.run("Read")
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        n = CapturingNotifier()
        events = []
        async for ev in agent.resume_stream(
            paused.run_id,
            tool_results=[ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
            notifier=n,
        ):
            events.append(ev)

        terminal = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert terminal, "Stream must reach RUN_COMPLETED"

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names
        assert names.index("on_run_started") < names.index("on_run_finished")


# ---------------------------------------------------------------------------
# Lifecycle pairing under failure during resume preparation
# ---------------------------------------------------------------------------


class _FailingNotifier(BaseNotifier):
    """Notifier that records every hook AND raises on a chosen one. Used
    to inject a failure inside _prepare_resume's injected-history recording.
    """

    def __init__(self, fail_on: str) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on

    async def on_run_started(self, run_id, *, agent_name=None, agent_model=None) -> None:
        self.calls.append("on_run_started")
        if self._fail_on == "on_run_started":
            raise RuntimeError("forced run_started failure")

    async def on_run_finished(self, run_id, result) -> None:
        self.calls.append("on_run_finished")

    async def on_run_failed(self, run_id, error, *, iteration=None) -> None:
        self.calls.append("on_run_failed")

    async def on_message_appended(self, run_id, message, iteration) -> None:
        self.calls.append("on_message_appended")
        if self._fail_on == "on_message_appended":
            raise RuntimeError("forced message_appended failure")

    async def on_tool_started(self, run_id, tool_call, iteration) -> None:
        self.calls.append("on_tool_started")

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
        self.calls.append("on_tool_completed")
        if self._fail_on == "on_tool_completed":
            raise RuntimeError("forced tool_completed failure")

    async def on_llm_call_started(
        self, run_id, iteration, *, semantic_messages=None, semantic_tools=None
    ) -> None:
        self.calls.append("on_llm_call_started")

    async def on_llm_call_completed(self, run_id, response, iteration, **kwargs) -> None:
        self.calls.append("on_llm_call_completed")

    async def on_llm_call_failed(self, run_id, iteration, error, *, duration_ms=None) -> None:
        self.calls.append("on_llm_call_failed")

    async def on_governance_event(
        self, run_id, event_type, iteration, data, *, correlation_id=None
    ) -> None:
        self.calls.append("on_governance_event")


class TestResumeLifecycleFailureBalancing:
    """When something between on_run_started and run completion fails during
    a resume, on_run_failed must fire so the OTel root span closes cleanly."""

    async def test_run_failed_fires_when_recorder_raises_during_history_replay(
        self, db_store, monkeypatch
    ) -> None:
        """A recorder write that raises while replaying injected client-tool
        history (after on_run_started has fired but before loop re-entry)
        must still pair with on_run_failed.

        This is the exact scenario the previous review flagged: previously
        _prepare_resume fired on_run_started THEN recorded injected messages
        outside any try/except that could fire on_run_failed. After the fix,
        the firing + recording sit inside the caller's try block and
        on_run_failed fires on any failure between them.
        """
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)

        paused = await agent.run("Read")
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        # Make PersistenceRecorder.on_tool_completed raise on the injected
        # client-tool result. Patch on the class so the per-run instance
        # picks it up. Subsequent loop work would never run.
        from dendrux.runtime.persistence import PersistenceRecorder

        async def _boom(self, *args, **kwargs):
            raise RuntimeError("recorder boom")

        monkeypatch.setattr(PersistenceRecorder, "on_tool_completed", _boom)

        n = CapturingNotifier()
        with pytest.raises(RuntimeError, match="recorder boom"):
            await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
                notifier=n,
            )

        names = n.names()
        assert "on_run_started" in names, (
            "Lifecycle started must fire before injected-history replay "
            "(otherwise OTel notifiers can't open the root span before "
            "on_message_appended / on_tool_completed events arrive)."
        )
        assert "on_run_failed" in names, (
            "If something between on_run_started and run completion fails, "
            "on_run_failed must fire to close the lifecycle pair. Otherwise "
            "OTel root spans leak on every failed resume."
        )
        assert names.index("on_run_started") < names.index("on_run_failed")
        # And on_run_finished must NOT fire — the pair is started → failed
        assert "on_run_finished" not in names
