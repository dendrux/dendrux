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

    async def test_fires_tool_started_for_approved_server_tool(self, db_store) -> None:
        """Approved server-tool execution must arrive on the notifier as a
        start/complete pair, not just complete. Otherwise OTel notifiers
        can't open a span before tool execution begins."""
        tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="processed")])
        agent = _approval_agent(llm, db_store)

        paused = await agent.run("refund 7")
        assert paused.status == RunStatus.WAITING_APPROVAL

        n = CapturingNotifier()
        await agent.submit_approval(paused.run_id, approved=True, notifier=n)

        starts = [c for c in n.calls if c[0] == "on_tool_started" and c[1][1] == "refund"]
        completes = [c for c in n.calls if c[0] == "on_tool_completed" and c[1][1] == "refund"]
        assert starts, (
            "Approved server-tool execution has no matching on_tool_started — "
            "start/complete pairing is broken on the approval-approve path."
        )
        assert completes
        # Find indices for the refund pair specifically
        first_start = next(
            i for i, c in enumerate(n.calls) if c[0] == "on_tool_started" and c[1][1] == "refund"
        )
        first_complete = next(
            i for i, c in enumerate(n.calls) if c[0] == "on_tool_completed" and c[1][1] == "refund"
        )
        assert first_start < first_complete


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


# ---------------------------------------------------------------------------
# Stream cleanup cancellation must close the lifecycle pair
# ---------------------------------------------------------------------------


class _BlockingLLM:
    """Mock provider that blocks on the first complete() call until released.

    Lets a test reach RUN_STARTED on the wire, then break out of the stream
    while the run is still in RUNNING status — exercising the cleanup path
    that CAS-cancels the run.
    """

    def __init__(self) -> None:
        import asyncio

        self.model = "blocking-mock"
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    async def complete(self, *args, **kwargs):
        self.entered.set()
        await self.release.wait()
        from dendrux.types import LLMResponse

        return LLMResponse(text="never reached")

    async def complete_stream(self, *args, **kwargs):
        # Async-iterable + aclose. Must yield at least one event so the
        # streaming loop awaits it; we block on `release` after entry.
        self.entered.set()
        await self.release.wait()
        from dendrux.types import StreamEvent, StreamEventType

        yield StreamEvent(type=StreamEventType.DONE)

    async def aclose(self) -> None:
        return None


class TestStreamCleanupCancellationLifecycle:
    """When a stream consumer abandons the iteration after on_run_started
    fires, the cleanup callback CAS-cancels the run. It must also fire
    on_run_finished with status=CANCELLED so the lifecycle pair stays
    balanced — otherwise OTel root spans leak on every abandoned stream.
    """

    async def test_run_stream_cleanup_fires_run_finished_cancelled(self, db_store) -> None:
        from dendrux.types import RunEventType

        provider = _BlockingLLM()
        agent = Agent(provider=provider, prompt="Test agent.", state_store=db_store)

        n = CapturingNotifier()
        run_id: str | None = None
        async with agent.stream("hi", notifier=n) as stream:
            async for ev in stream:
                if ev.type == RunEventType.RUN_STARTED:
                    run_id = ev.run_id
                    break  # Abandon the stream — cleanup CAS-cancels the run.

        assert run_id is not None

        # Release the LLM block so the now-cancelled generator can drain.
        provider.release.set()

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names, (
            "Stream cleanup CAS-cancelled the run but never fired "
            "on_run_finished — lifecycle pair is leaked, OTel root span "
            "stays open forever."
        )
        # Pair order
        assert names.index("on_run_started") < names.index("on_run_finished")
        # Status on the cancelled finish must be CANCELLED
        finished = next(c for c in n.calls if c[0] == "on_run_finished")
        result = finished[1][1]
        assert result.status == RunStatus.CANCELLED

    async def test_resume_stream_cleanup_fires_run_finished_cancelled(self, db_store) -> None:
        from dendrux.types import RunEventType

        # First, get a real paused run via the normal (non-blocking) path.
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        setup_llm = MockLLM([LLMResponse(tool_calls=[tc])])
        setup_agent = _client_tool_agent(setup_llm, db_store)

        paused = await setup_agent.run("Read")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        # Now resume_stream with a blocking provider so we can break early
        # while the run is still RUNNING.
        provider = _BlockingLLM()
        resume_agent = Agent(
            provider=provider,
            prompt="Test agent.",
            tools=[read_range],
            state_store=db_store,
        )

        n = CapturingNotifier()
        async with resume_agent.resume_stream(
            paused.run_id,
            tool_results=[ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
            notifier=n,
        ) as stream:
            async for ev in stream:
                if ev.type == RunEventType.RUN_RESUMED:
                    break  # Abandon — cleanup CAS-cancels the run.

        provider.release.set()

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names, (
            "resume_stream cleanup CAS-cancelled the run but never fired "
            "on_run_finished — lifecycle pair is leaked on every abandoned "
            "resume stream."
        )
        assert names.index("on_run_started") < names.index("on_run_finished")
        finished = next(c for c in n.calls if c[0] == "on_run_finished")
        result = finished[1][1]
        assert result.status == RunStatus.CANCELLED

    async def test_run_stream_no_store_cleanup_fires_run_finished(self) -> None:
        """Non-persisted streams still emit on_run_started before yielding
        RUN_STARTED. If the consumer abandons such a stream, cleanup must
        still close the lifecycle pair on the local notifier — otherwise
        OTel root spans leak even in the no-persistence path."""
        from dendrux.types import RunEventType

        provider = _BlockingLLM()
        agent = Agent(provider=provider, prompt="Test agent.")  # NO state_store

        n = CapturingNotifier()
        async with agent.stream("hi", notifier=n) as stream:
            async for ev in stream:
                if ev.type == RunEventType.RUN_STARTED:
                    break

        provider.release.set()

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names, (
            "Non-persisted stream cleanup never closed the lifecycle pair — "
            "OTel root span leaks because cleanup early-returned on "
            "store is None without firing on_run_finished."
        )
        finished = next(c for c in n.calls if c[0] == "on_run_finished")
        result = finished[1][1]
        assert result.status == RunStatus.CANCELLED

    async def test_run_stream_cleanup_fires_finished_when_cas_loses(self, db_store) -> None:
        """If the run was already finalized externally before cleanup, the
        CAS loses but the local notifier still needs an on_run_finished
        for span hygiene."""
        from dendrux.types import RunEventType

        provider = _BlockingLLM()
        agent = Agent(provider=provider, prompt="Test agent.", state_store=db_store)

        n = CapturingNotifier()
        run_id: str | None = None
        async with agent.stream("hi", notifier=n) as stream:
            async for ev in stream:
                if ev.type == RunEventType.RUN_STARTED:
                    run_id = ev.run_id
                    # Externally finalize the run BEFORE cleanup runs so the
                    # CAS-guarded cancellation in cleanup loses.
                    await db_store.finalize_run(
                        run_id,
                        status=RunStatus.SUCCESS.value,
                        expected_current_status="running",
                    )
                    break

        provider.release.set()
        assert run_id is not None

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names, (
            "Cleanup CAS lost (run already finalized elsewhere) but the "
            "local notifier never saw on_run_finished — OTel root span "
            "leaks on the local consumer."
        )

    async def test_resume_stream_cleanup_fires_finished_when_cas_loses(self, db_store) -> None:
        """Same gap as run_stream but for resume_stream cleanup."""
        from dendrux.types import RunEventType

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        setup_llm = MockLLM([LLMResponse(tool_calls=[tc])])
        setup_agent = _client_tool_agent(setup_llm, db_store)

        paused = await setup_agent.run("Read")
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        provider = _BlockingLLM()
        resume_agent = Agent(
            provider=provider,
            prompt="Test agent.",
            tools=[read_range],
            state_store=db_store,
        )

        n = CapturingNotifier()
        async with resume_agent.resume_stream(
            paused.run_id,
            tool_results=[ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
            notifier=n,
        ) as stream:
            async for ev in stream:
                if ev.type == RunEventType.RUN_RESUMED:
                    # Finalize externally so cleanup CAS loses.
                    await db_store.finalize_run(
                        paused.run_id,
                        status=RunStatus.SUCCESS.value,
                        expected_current_status="running",
                    )
                    break

        provider.release.set()

        names = n.names()
        assert "on_run_started" in names
        assert "on_run_finished" in names, (
            "resume_stream cleanup CAS lost (run already finalized) but the "
            "local notifier never saw on_run_finished."
        )


# ---------------------------------------------------------------------------
# Approval-rejection governance event coverage
# ---------------------------------------------------------------------------


class _GovEventCapturer(BaseNotifier):
    """Captures (event_type, data) for every governance event."""

    def __init__(self) -> None:
        self.gov_events: list[tuple[str, dict[str, Any]]] = []

    async def on_governance_event(
        self, run_id, event_type, iteration, data, *, correlation_id=None
    ) -> None:
        self.gov_events.append((event_type, dict(data)))


class TestSubmitApprovalRejectionGovernance:
    """submit_approval(approved=False) must fire approval.decided=rejected.

    The streaming variant (resume_stream with rejection ToolResults) already
    fires this because it re-derives expected_status from the actual pause
    status. The sync variant goes through resume_claimed → _resume_core
    with expected_status="running", which historically failed the rejection
    gate at runner.py:2002 and silently dropped the governance event.
    """

    async def test_sync_rejection_fires_approval_decided(self, db_store) -> None:
        tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="acknowledged rejection")])
        agent = _approval_agent(llm, db_store)

        paused = await agent.run("refund 7")
        assert paused.status == RunStatus.WAITING_APPROVAL

        n = _GovEventCapturer()
        await agent.submit_approval(
            paused.run_id,
            approved=False,
            rejection_reason="Manager declined.",
            notifier=n,
        )

        decided = [(etype, data) for etype, data in n.gov_events if etype == "approval.decided"]
        assert decided, (
            "submit_approval(approved=False) silently skipped approval.decided. "
            "OTel/audit consumers never see the rejection signal."
        )
        assert decided[0][1].get("decision") == "rejected", decided[0][1]

    async def test_sync_approval_still_fires_approval_decided(self, db_store) -> None:
        """Regression: ensure the approve path still emits approval.decided."""
        tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="processed")])
        agent = _approval_agent(llm, db_store)

        paused = await agent.run("refund 7")
        n = _GovEventCapturer()
        await agent.submit_approval(paused.run_id, approved=True, notifier=n)

        decided = [(etype, data) for etype, data in n.gov_events if etype == "approval.decided"]
        assert decided
        assert decided[0][1].get("decision") == "approved"
