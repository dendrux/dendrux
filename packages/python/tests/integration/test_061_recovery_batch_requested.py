"""A requested pause cannot approve governance that recovery has not evaluated."""

import pytest
from sqlalchemy import event

from dendrux.types import ToolCall
from tests._helpers.recovery_checkpoint import checkpoint, expire
from tests.integration.test_061_recovery_batch import Crash, batch_agent
from tests.integration.test_061_recovery_pause import _continue


async def _pause_before_preparation(agent, store, engine, monkeypatch, calls, phases, *, item=None):
    if item is None:
        item = await checkpoint(agent, store, engine, calls=calls, phases=phases)
    claim = store.reclaim_run

    async def request_pause(*args, **kwargs):
        result = await claim(*args, **kwargs)
        assert await store.request_pause(item.run_id)
        return result

    with monkeypatch.context() as changes:
        changes.setattr(store, "reclaim_run", request_pause)
        paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == "paused"
    raw = await store.get_pause_state(item.run_id)
    assert raw["pause_reason"] == "requested"
    assert [p["phase"] for p in (await store.get_loop_state(item.run_id))["pending"]] == phases
    return paused


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("hook", [False, True])
@pytest.mark.parametrize("approved", [False, True])
async def test_061_recovery_batch_requested_pause(
    store, engine, monkeypatch, entry, reverse, hook, approved
):
    effects = []
    agent = batch_agent(store, effects, hook=hook)
    write, notice = ToolCall("write_region"), ToolCall("post_review_notice")
    calls = [write, notice] if reverse else [notice, write]
    phases = ["gated" if c == write and not hook else "dispatchable" for c in calls]
    paused = await _pause_before_preparation(agent, store, engine, monkeypatch, calls, phases)
    assert effects == [] and await store.get_tool_calls(paused.run_id) == []
    await agent.close()
    agent = batch_agent(store, effects, hook=hook)
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        pending = await _continue(agent, paused.run_id, paused.pause_id, entry)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        print(
            "requested_continuation_statements", engine.dialect.name, entry, hook, len(statements)
        )
    assert pending.status.value == "waiting_approval" and effects == []
    assert pending.pause_id and pending.pause_id != paused.pause_id
    assert await store.get_tool_calls(paused.run_id) == []
    raw = await store.get_pause_state(paused.run_id)
    assert raw["pause_id"] == pending.pause_id
    assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
    assert raw["approval_gates"] == {
        write.id: {
            "source": "hook" if hook else "static",
            "reason": "Review region" if hook else "requires_approval",
        }
    }
    events = await store.get_run_events(paused.run_id)
    assert [e.event_type for e in events if e.event_type.startswith("approval.")] == [
        "approval.requested"
    ]
    final = await agent.submit_approval(
        paused.run_id,
        expected_pause_id=pending.pause_id,
        tenant_id="tenant",
        approved=approved,
    )
    assert final.status.value == "success"
    assert effects == ([c.name for c in calls] if approved else [])
    events = await store.get_run_events(paused.run_id)
    assert sum(e.event_type == "approval.decided" for e in events) == 1
    await agent.close()


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
async def test_061_recovery_batch_requested_ungated_runs(store, engine, monkeypatch, entry):
    effects = []
    agent = batch_agent(store, effects, hook=True)
    calls = [ToolCall("post_review_notice")]
    paused = await _pause_before_preparation(
        agent, store, engine, monkeypatch, calls, ["dispatchable"]
    )
    assert effects == []
    result = await _continue(agent, paused.run_id, paused.pause_id, entry)
    assert result.status.value == "success" and effects == ["post_review_notice"]
    events = await store.get_run_events(paused.run_id)
    assert not any(e.event_type.startswith("approval.") for e in events)
    await agent.close()


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
async def test_061_recovery_batch_requested_keeps_prior_approval(store, engine, monkeypatch, entry):
    effects = []
    agent = batch_agent(store, effects, hook=True)
    calls = [ToolCall("write_region"), ToolCall("post_review_notice")]
    item = await checkpoint(
        agent, store, engine, calls=calls, phases=["dispatchable", "dispatchable"]
    )
    pending = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    consume = store.consume_submission

    async def crash_after_decision(*args, **kwargs):
        await consume(*args, **kwargs)
        raise Crash()

    with monkeypatch.context() as changes:
        changes.setattr(store, "consume_submission", crash_after_decision)
        with pytest.raises(Crash):
            await agent.submit_approval(
                item.run_id, expected_pause_id=pending.pause_id, tenant_id="tenant", approved=True
            )
    assert effects == [] and await store.get_tool_calls(item.run_id) == []
    state = await store.get_loop_state(item.run_id)
    decision_id = state["pending"][0]["decision_id"]
    assert decision_id and state["submission"] is None
    item = await expire(store, engine, item.run_id)
    paused = await _pause_before_preparation(
        agent, store, engine, monkeypatch, calls, ["approved", "dispatchable"], item=item
    )
    state = await store.get_loop_state(item.run_id)
    assert state["pending"][0]["decision_id"] == decision_id
    result = await _continue(agent, paused.run_id, paused.pause_id, entry)
    assert result.status.value == "success" and effects == [c.name for c in calls]
    events = await store.get_run_events(item.run_id)
    assert sum(e.event_type == "approval.requested" for e in events) == 1
    assert len(await store.get_tool_calls(item.run_id)) == 2
    await agent.close()
