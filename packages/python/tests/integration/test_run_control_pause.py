"""Durable cooperative pause and its linearizability with cancel and approvals.

SQLite + Postgres via the ``store`` matrix fixture. Covers the state CAS
primitives, the top-of-iteration checkpoint, resume of PAUSED through every
entry point, cancel-wins-over-pause, the pre-pause point clearing a request,
the resume claim clearing a stale flag, and the approval-claim barrier
(pause/cancel between approved calls, pause landing right after the claim).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import update

from dendrux.agent import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import PauseStatusMismatchError, RunNotFoundError
from dendrux.runtime.context import get_delegation_context
from dendrux.tool import tool
from dendrux.types import (
    ApprovalDecision,
    LLMResponse,
    PauseState,
    RunEventType,
    RunStatus,
    ToolResult,
)
from tests._helpers.scripted_provider import ScriptedLLM, tool_call, tool_calls

pytestmark = pytest.mark.asyncio

# cancel_run from inside the run's own task interrupts it in-process: the
# CancelledError can land mid-query, and SQLAlchemy then discards that
# connection. An in-memory SQLite database lives on its one connection, so
# these tests need backends with independent connections.
_INTERRUPTIBLE = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


def _current_run_id() -> str:
    ctx = get_delegation_context()
    assert ctx is not None
    return ctx.run_id


async def _set_row(store: Any, run_id: str, **values: Any) -> None:
    async with store._session_factory() as session:
        await session.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))
        await session.commit()


async def _event_types(store: Any, run_id: str) -> list[str]:
    return [e.event_type for e in await store.get_run_events(run_id)]


async def _events(store: Any, run_id: str, event_type: str) -> list[Any]:
    return [e for e in await store.get_run_events(run_id) if e.event_type == event_type]


# ---------------------------------------------------------------------------
# State primitives
# ---------------------------------------------------------------------------


class TestPauseStatePrimitives:
    async def test_request_pause_latches_only_on_running(self, store) -> None:
        await store.create_run("r1", "a")
        assert await store.request_pause("r1", reason="why") is True
        record = await store.get_run("r1")
        assert record.pause_requested is True
        assert record.meta["dendrux.pause_request"] == {"reason": "why"}

        await store.create_run("r2", "a")
        await store.pause_run("r2", status="waiting_client_tool", pause_data={"x": 1})
        assert await store.request_pause("r2") is False
        assert (await store.get_run("r2")).pause_requested is False

        await store.create_run("r3", "a")
        await store.finalize_run("r3", status="success")
        assert await store.request_pause("r3") is False
        assert await store.request_pause("missing") is False

    async def test_try_pause_run_is_a_cas_that_loses_to_cancel(self, store) -> None:
        await store.create_run("r1", "a")
        await store.request_pause("r1")
        assert await store.try_pause_run("r1", status="paused", pause_data={"p": 1}) is True
        record = await store.get_run("r1")
        assert record.status == "paused"
        assert record.pause_requested is False  # any durable pause satisfies a request

        await store.create_run("r2", "a")
        await store.request_cancel("r2")
        assert await store.try_pause_run("r2", status="paused", pause_data={"p": 1}) is False
        assert (await store.get_run("r2")).status == "running"

        # Not running (already paused) → CAS loses too.
        assert await store.try_pause_run("r1", status="paused", pause_data={"p": 2}) is False

    async def test_claim_clears_pause_flag_and_finalize_clears_both(self, store) -> None:
        await store.create_run("r1", "a")
        await store.try_pause_run("r1", status="paused", pause_data={"p": 1})
        await _set_row(store, "r1", pause_requested=True)  # stale flag
        assert (await store.claim_paused_run("r1", expected_status="paused")).claim_epoch == 1
        assert (await store.get_run("r1")).pause_requested is False

        await store.request_pause("r1")
        await store.request_cancel("r1")
        await store.finalize_run("r1", status="cancelled", expected_current_status="running")
        record = await store.get_run("r1")
        assert record.pause_requested is False
        assert record.cancel_requested is False

    async def test_submit_and_claim_clears_pause_flag(self, store) -> None:
        # This fault injection / pause blob exercises the format-1 write path.
        from tests._fixtures.storage import pin_storage_format

        await pin_storage_format(store._engine, 1)
        await store.create_run("r1", "a")
        await store.try_pause_run("r1", status="paused", pause_data={"p": 1})
        await _set_row(store, "r1", pause_requested=True)
        assert await store.submit_and_claim("r1", expected_status="paused", submitted_data={"k": 1})
        assert (await store.get_run("r1")).pause_requested is False

    async def test_get_run_control(self, store) -> None:
        assert (await store.get_run_control("missing")).pause_requested is False
        await store.create_run("r1", "a")
        control = await store.get_run_control("r1")
        assert (control.cancel_requested, control.pause_requested) == (False, False)
        assert control.has_pending_notes is False
        await store.request_pause("r1")
        await store.request_cancel("r1")
        await store.add_note("r1", note_id="n1", text="hi")
        control = await store.get_run_control("r1")
        assert control.cancel_requested and control.pause_requested
        assert control.has_pending_notes is True
        await store.mark_notes_delivered("r1", ["n1"], iteration=1)
        assert (await store.get_run_control("r1")).has_pending_notes is False

    async def test_sweep_abandoned_includes_paused(self, store) -> None:
        import datetime as dt

        await store.create_run("r1", "a")
        await store.try_pause_run("r1", status="paused", pause_data={"p": 1})
        old = dt.datetime.now(dt.UTC) - dt.timedelta(hours=3)
        await _set_row(store, "r1", status_changed_at=old, updated_at=old)
        swept = await store.sweep_abandoned_runs(older_than=dt.timedelta(hours=1))
        assert [(s.run_id, s.previous_status) for s in swept] == [("r1", "paused")]
        assert (await store.get_run("r1")).status == "error"


# ---------------------------------------------------------------------------
# Top-of-iteration checkpoint and resume
# ---------------------------------------------------------------------------


def _pausing_agent(
    store: Any, *, extra: list[LLMResponse] | None = None
) -> tuple[Agent, ScriptedLLM, dict[str, int]]:
    """Agent whose first tool asks for a pause of its own run."""
    calls = {"step": 0}
    holder: dict[str, Agent] = {}

    @tool()
    async def step() -> str:
        calls["step"] += 1
        if calls["step"] == 1:
            await holder["agent"].request_pause(_current_run_id(), reason="operator hold")
        return "stepped"

    llm = ScriptedLLM([tool_call("step", pid="t1"), *(extra or [LLMResponse(text="done")])])
    agent = Agent(provider=llm, prompt="Test.", tools=[step], state_store=store)
    holder["agent"] = agent
    return agent, llm, calls


class TestRequestedPause:
    async def test_pauses_at_next_checkpoint_with_full_state(self, store) -> None:
        agent, llm, calls = _pausing_agent(store)
        result = await agent.run("go")

        assert result.status == RunStatus.PAUSED
        assert result.iteration_count == 1
        assert calls["step"] == 1
        assert llm.calls == 1  # the second LLM call never started

        record = await store.get_run(result.run_id)
        assert record.status == "paused"
        assert record.pause_requested is False
        pause = PauseState.from_dict(await store.get_pause_state(result.run_id))
        assert pause.pending_tool_calls == []
        assert pause.pause_reason == "requested"
        assert pause.pause_message == "operator hold"
        assert pause.iteration == 1
        from dendrux.runtime.replay import rebuild_history

        assert pause.schema_version == 2 and pause.history == []
        messages = await rebuild_history(store, result.run_id)
        assert [m.role.value for m in messages.messages] == ["system", "user", "assistant", "tool"]

        paused = await _events(store, result.run_id, "run.paused")
        assert len(paused) == 1
        assert paused[0].data["status"] == "paused"
        assert paused[0].data["reason"] == "requested"
        assert paused[0].data["message"] == "operator hold"

    @pytest.mark.parametrize("how", ["resume", "submit_resume", "resume_stream"])
    async def test_resume_continues_with_full_history(self, store, how) -> None:
        agent, llm, calls = _pausing_agent(store)
        paused = await agent.run("go")
        assert paused.status == RunStatus.PAUSED

        if how == "resume":
            result = await agent.resume(paused.run_id, expected_pause_id=paused.pause_id)
        elif how == "submit_resume":
            result = await agent.submit_resume(paused.run_id)
        else:
            events = []
            async with agent.resume_stream(
                paused.run_id, expected_pause_id=paused.pause_id
            ) as stream:
                async for event in stream:
                    events.append(event)
            assert events[0].type == RunEventType.RUN_RESUMED
            assert events[-1].type == RunEventType.RUN_COMPLETED
            result = events[-1].run_result

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "done"
        assert result.iteration_count == 2
        assert calls["step"] == 1  # nothing re-executed
        # The continuation saw the whole history: user, assistant(tool call), tool.
        assert [m.role.value for m in llm.messages[1]] == ["system", "user", "assistant", "tool"]
        assert llm.contents(1)[-1] == '"stepped"'
        record = await store.get_run(paused.run_id)
        assert record.status == "success"
        resumed = await _events(store, paused.run_id, "run.resumed")
        assert resumed[-1].data["resumed_from"] == "paused"

    async def test_in_flight_call_is_not_preempted(self, store) -> None:
        @tool()
        async def step() -> str:
            return "stepped"

        llm = ScriptedLLM([tool_call("step", pid="t1"), LLMResponse(text="done")], gate_at=0)
        agent = Agent(provider=llm, prompt="Test.", tools=[step], state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = (await store.list_runs())[0].id

        outcome = await agent.request_pause(run_id)
        assert outcome.status_before == RunStatus.RUNNING
        assert outcome.affected == [(run_id, "requested")]
        llm.gate.set()
        result = await asyncio.wait_for(task, 5)

        # The blocked LLM call and its tool completed; the pause took effect
        # at the next checkpoint.
        assert result.status == RunStatus.PAUSED
        assert llm.cancelled is False
        tools = await store.get_tool_calls(run_id)
        assert [t.tool_name for t in tools] == ["step"]

    async def test_submit_resume_rejects_non_paused(self, store) -> None:
        @tool(target="client")
        async def client_read() -> str:
            return ""

        llm = ScriptedLLM([tool_call("client_read", pid="t1")])
        agent = Agent(provider=llm, prompt="Test.", tools=[client_read], state_store=store)
        waiting = await agent.run("go")
        with pytest.raises(PauseStatusMismatchError):
            await agent.submit_resume(waiting.run_id)
        with pytest.raises(RunNotFoundError):
            await agent.submit_resume("missing")

    async def test_cancel_run_on_paused_finalizes_cancelled(self, store) -> None:
        agent, _, _ = _pausing_agent(store)
        paused = await agent.run("go")
        result = await agent.cancel_run(paused.run_id)
        assert result.status == RunStatus.CANCELLED
        record = await store.get_run(paused.run_id)
        assert record.status == "cancelled"
        assert await store.get_pause_state(paused.run_id) is None
        assert (await _event_types(store, paused.run_id))[-1] == "run.cancelled"

    @_INTERRUPTIBLE
    async def test_cancel_wins_over_pause(self, store) -> None:
        holder: dict[str, Agent] = {}

        @tool()
        async def step() -> str:
            run_id = _current_run_id()
            await holder["agent"].request_pause(run_id)
            await holder["agent"].cancel_run(run_id)
            return "stepped"

        llm = ScriptedLLM([tool_call("step", pid="t1"), LLMResponse(text="unreached")])
        agent = Agent(provider=llm, prompt="Test.", tools=[step], state_store=store)
        holder["agent"] = agent
        result = await agent.run("go")

        assert result.status == RunStatus.CANCELLED
        record = await store.get_run(result.run_id)
        assert record.status == "cancelled"
        assert record.pause_requested is False and record.cancel_requested is False
        assert "run.paused" not in await _event_types(store, result.run_id)

    async def test_cancel_between_checkpoint_and_pause_cas_wins(self, store) -> None:
        """A cancel that lands after the loop decided to pause, before the
        pause CAS: the CAS loses and the run ends CANCELLED, not PAUSED."""
        agent, _, _ = _pausing_agent(store)
        original = store.pause_with_event

        async def cancel_first(run_id: str, **kwargs: Any) -> Any:
            await store.request_cancel(run_id)
            return await original(run_id, **kwargs)

        store.pause_with_event = cancel_first  # type: ignore[method-assign]
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        events = await _event_types(store, result.run_id)
        assert "run.paused" not in events
        assert events.count("run.cancelled") == 1

    async def test_waiting_pause_satisfies_request(self, store) -> None:
        """Pre-pause point: a client/approval/input pause persists and clears
        the request; after the submit the run continues without pausing."""
        holder: dict[str, Agent] = {}

        @tool()
        async def step() -> str:
            await holder["agent"].request_pause(_current_run_id())
            return "stepped"

        @tool(target="client")
        async def client_read() -> str:
            return ""

        llm = ScriptedLLM(
            [
                tool_calls(("step", {}, "t1"), ("client_read", {}, "t2")),
                LLMResponse(text="done"),
            ]
        )
        agent = Agent(provider=llm, prompt="Test.", tools=[step, client_read], state_store=store)
        holder["agent"] = agent
        waiting = await agent.run("go")
        assert waiting.status == RunStatus.WAITING_CLIENT_TOOL
        assert (await store.get_run(waiting.run_id)).pause_requested is False

        call_id = waiting.pending.tool_calls[0].id
        result = await agent.submit_tool_results(
            waiting.run_id, [ToolResult(name="client_read", call_id=call_id, payload='"x"')]
        )
        assert result.status == RunStatus.SUCCESS

    async def test_request_on_waiting_run_is_non_latching(self, store) -> None:
        @tool(target="client")
        async def client_read() -> str:
            return ""

        llm = ScriptedLLM([tool_call("client_read", pid="t1"), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="Test.", tools=[client_read], state_store=store)
        waiting = await agent.run("go")

        outcome = await agent.request_pause(waiting.run_id)
        assert outcome.status_before == outcome.status_after == RunStatus.WAITING_CLIENT_TOOL
        assert outcome.affected == [(waiting.run_id, "already_waiting")]
        assert (await store.get_run(waiting.run_id)).pause_requested is False

        call_id = waiting.pending.tool_calls[0].id
        result = await agent.submit_tool_results(
            waiting.run_id, [ToolResult(name="client_read", call_id=call_id, payload='"x"')]
        )
        assert result.status == RunStatus.SUCCESS

    async def test_request_on_paused_and_terminal_is_noop(self, store) -> None:
        agent, _, _ = _pausing_agent(store)
        paused = await agent.run("go")
        outcome = await agent.request_pause(paused.run_id)
        assert outcome.affected == [(paused.run_id, "already_paused")]
        done = await agent.submit_resume(paused.run_id)
        outcome = await agent.request_pause(done.run_id)
        assert outcome.status_after == RunStatus.SUCCESS
        assert outcome.affected == [(done.run_id, "terminal")]
        with pytest.raises(RunNotFoundError):
            await agent.request_pause("missing")
        with pytest.raises(ValueError):
            await agent.request_pause(done.run_id, reason="x" * 1001)

    async def test_resume_clears_stale_pause_flag(self, store) -> None:
        agent, _, calls = _pausing_agent(store)
        paused = await agent.run("go")
        await _set_row(store, paused.run_id, pause_requested=True)
        result = await agent.submit_resume(paused.run_id)
        assert result.status == RunStatus.SUCCESS
        assert calls["step"] == 1

    async def test_stream_pauses_and_resumes(self, store) -> None:
        agent, _, calls = _pausing_agent(store)
        events = []
        async with agent.stream("go") as stream:
            async for event in stream:
                events.append(event)
        assert events[-1].type == RunEventType.RUN_PAUSED
        assert events[-1].run_result.status == RunStatus.PAUSED
        run_id = stream.run_id
        assert (await store.get_run(run_id)).status == "paused"

        result = await agent.submit_resume(run_id)
        assert result.status == RunStatus.SUCCESS
        assert calls["step"] == 1


# ---------------------------------------------------------------------------
# Approval-claim barriers
# ---------------------------------------------------------------------------


def _approval_agent(
    store: Any, *, on_a: str | None = None
) -> tuple[Agent, ScriptedLLM, dict[str, int]]:
    """Two approval-gated tools; ``a`` can pause or cancel its own run."""
    counts = {"a": 0, "b": 0}
    holder: dict[str, Agent] = {}

    @tool()
    async def a() -> str:
        counts["a"] += 1
        if on_a == "pause":
            await holder["agent"].request_pause(_current_run_id())
        elif on_a == "cancel":
            # Another Agent instance (as from another process): durable flag only.
            await holder["other"].cancel_run(_current_run_id())
        return "a-done"

    @tool()
    async def b() -> str:
        counts["b"] += 1
        return "b-done"

    llm = ScriptedLLM([tool_calls(("a", {}, "t1"), ("b", {}, "t2")), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="Test.",
        tools=[a, b],
        require_approval=["a", "b"],
        state_store=store,
    )
    holder["agent"] = agent
    holder["other"] = Agent(provider=ScriptedLLM([]), prompt="Other.", state_store=store)
    return agent, llm, counts


class TestApprovalBarrier:
    async def test_pause_between_approved_calls_persists_remainder(self, store) -> None:
        agent, llm, counts = _approval_agent(store, on_a="pause")
        waiting = await agent.run("go")
        assert waiting.status == RunStatus.WAITING_APPROVAL

        paused = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert paused.status == RunStatus.PAUSED
        assert counts == {"a": 1, "b": 0}

        pause = PauseState.from_dict(await store.get_pause_state(paused.run_id))
        assert [tc.name for tc in pause.pending_tool_calls] == ["b"]
        b_id = pause.pending_tool_calls[0].id
        assert pause.pending_decisions == {b_id: {"approved": True, "reason": None}}
        assert set(pause.approval_gates or {}) == {b_id}
        assert pause.pause_reason == "requested"

        result = await agent.submit_resume(paused.run_id)
        assert result.status == RunStatus.SUCCESS
        assert counts == {"a": 1, "b": 1}
        # One TOOL message per call; the model saw both results.
        tool_msgs = [m for m in llm.messages[1] if m.role.value == "tool"]
        assert sorted(m.content for m in tool_msgs) == ['"a-done"', '"b-done"']
        # approval.decided is emitted once (for the human decision).
        assert (await _event_types(store, paused.run_id)).count("approval.decided") == 1

    async def test_remainder_keeps_per_call_denial(self, store) -> None:
        agent, llm, counts = _approval_agent(store, on_a="pause")
        waiting = await agent.run("go")
        ids = {tc.name: tc.id for tc in waiting.pending.tool_calls}
        paused = await agent.submit_approval(
            waiting.run_id,
            decisions={
                ids["a"]: ApprovalDecision.approve(),
                ids["b"]: ApprovalDecision.deny("not today"),
            },
            expected_pause_id=waiting.pause_id,
        )
        assert paused.status == RunStatus.PAUSED
        result = await agent.resume(paused.run_id, expected_pause_id=paused.pause_id)
        assert result.status == RunStatus.SUCCESS
        assert counts == {"a": 1, "b": 0}
        b_msg = next(m for m in llm.messages[1] if m.role.value == "tool" and m.name == "b")
        assert b_msg.content == '{"denied": "not today"}'

    async def test_cancel_between_approved_calls_stops_side_effects(self, store) -> None:
        agent, _, counts = _approval_agent(store, on_a="cancel")
        waiting = await agent.run("go")
        result = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert result.status == RunStatus.CANCELLED
        assert counts == {"a": 1, "b": 0}
        assert (await store.get_run(waiting.run_id)).status == "cancelled"

    async def test_pause_landing_right_after_claim_runs_nothing(self, store) -> None:
        # This fault injection / pause blob exercises the format-1 write path.
        from tests._fixtures.storage import pin_storage_format

        await pin_storage_format(store._engine, 1)
        agent, _, counts = _approval_agent(store)
        waiting = await agent.run("go")
        original = store.claim_paused_run

        async def claim_then_pause(run_id: str, *, expected_status: str, **kwargs) -> int:
            won = await original(run_id, expected_status=expected_status, **kwargs)
            if won:
                await store.request_pause(run_id)
            return won

        store.claim_paused_run = claim_then_pause  # type: ignore[method-assign]
        paused = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        store.claim_paused_run = original  # type: ignore[method-assign]

        assert paused.status == RunStatus.PAUSED
        assert counts == {"a": 0, "b": 0}
        pause = PauseState.from_dict(await store.get_pause_state(paused.run_id))
        assert [tc.name for tc in pause.pending_tool_calls] == ["a", "b"]

        result = await agent.submit_resume(paused.run_id)
        assert result.status == RunStatus.SUCCESS
        assert counts == {"a": 1, "b": 1}

    async def test_cancel_landing_right_after_claim_runs_nothing(self, store) -> None:
        # This fault injection / pause blob exercises the format-1 write path.
        from tests._fixtures.storage import pin_storage_format

        await pin_storage_format(store._engine, 1)
        agent, _, counts = _approval_agent(store)
        waiting = await agent.run("go")
        original = store.claim_paused_run

        async def claim_then_cancel(run_id: str, *, expected_status: str, **kwargs) -> int:
            won = await original(run_id, expected_status=expected_status, **kwargs)
            if won:
                await store.request_cancel(run_id)
            return won

        store.claim_paused_run = claim_then_cancel  # type: ignore[method-assign]
        result = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert result.status == RunStatus.CANCELLED
        assert counts == {"a": 0, "b": 0}

    async def test_pause_request_while_waiting_approval_does_not_latch(self, store) -> None:
        agent, _, counts = _approval_agent(store)
        waiting = await agent.run("go")
        outcome = await agent.request_pause(waiting.run_id)
        assert outcome.affected == [(waiting.run_id, "already_waiting")]
        result = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert result.status == RunStatus.SUCCESS
        assert counts == {"a": 1, "b": 1}

    async def test_stream_resume_of_paused_remainder(self, store) -> None:
        agent, _, counts = _approval_agent(store, on_a="pause")
        waiting = await agent.run("go")
        paused = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        events = []
        async with agent.resume_stream(paused.run_id, expected_pause_id=paused.pause_id) as stream:
            async for event in stream:
                events.append(event)
        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert counts == {"a": 1, "b": 1}
        results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert [e.tool_call.name for e in results] == ["b"]
