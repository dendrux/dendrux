"""Requested recovery pauses retain unresolved approval work across continuations."""

from functools import wraps

import pytest
from sqlalchemy import event

from dendrux import tool
from dendrux.types import ToolCall, ToolDecision
from tests._helpers.recovery_checkpoint import checkpoint
from tests.integration.test_060_reclaim_agent import make_agent


async def _continue(agent, run_id, pause_id, entry):
    kwargs = dict(tenant_id="tenant", expected_pause_id=pause_id)
    if entry == "resume_stream":
        async with agent.resume_stream(run_id, **kwargs) as stream:
            async for _ in stream:
                pass
        return stream.result
    return await getattr(agent, entry)(run_id, **kwargs)


def _agent(store, effects, gate="static", *, idempotent=False):
    @tool(idempotent=idempotent)
    async def write_region() -> str:
        """Record an approved region change."""
        effects.append("write_region")
        return "ok"

    async def before(context):
        return ToolDecision.require_approval("Review region change")

    return make_agent(
        store,
        tools=[write_region],
        **({"require_approval": ["write_region"]} if gate == "static" else {"before_tool": before}),
    )


async def _requested_pause(agent, store, engine, monkeypatch, *, started=False, cancel=False):
    calls = [ToolCall("write_region"), ToolCall("write_region")]
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=calls,
        phases=["gated"] * 2,
        started=started,
        idempotent=started,
    )
    original = store.reclaim_run

    @wraps(original)
    async def claim(*args, **kwargs):
        result = await original(*args, **kwargs)
        assert await store.request_pause(item.run_id)
        if cancel:
            assert await store.request_cancel(item.run_id)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(store, "reclaim_run", claim)
        paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == ("cancelled" if cancel else "paused")
    if not cancel:
        raw = await store.get_pause_state(item.run_id)
        assert raw["pause_reason"] == "requested"
        assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
        state = await store.get_loop_state(item.run_id)
        assert [p["call_id"] for p in state["pending"]] == [c.id for c in calls]
        assert all(p["phase"] == "gated" for p in state["pending"])
        if not started:
            assert all(p["decision_id"] is None for p in state["pending"])
        assert (await store.get_run(item.run_id)).status == "paused"
    return paused, calls


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
@pytest.mark.parametrize("gate", ["static", "hook"])
@pytest.mark.parametrize("approved", [True, False])
async def test_061_resume_requested_recovery_reaches_approval(
    store, engine, monkeypatch, entry, gate, approved
):
    effects = []
    agent = _agent(store, effects, gate)
    paused, calls = await _requested_pause(agent, store, engine, monkeypatch)
    assert effects == []
    # A fresh Agent must reconstruct everything from durable state.
    await agent.close()
    agent = _agent(store, effects, gate)
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await _continue(agent, paused.run_id, paused.pause_id, entry)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        print(
            "continuation_statements", engine.dialect.name, entry, gate, approved, len(statements)
        )
    assert result.status.value == "waiting_approval"
    assert result.pause_id and result.pause_id != paused.pause_id
    assert effects == []
    assert await store.get_tool_calls(paused.run_id) == []
    raw = await store.get_pause_state(paused.run_id)
    assert raw["pause_id"] == result.pause_id
    assert set(raw["approval_gates"]) == {c.id for c in calls}
    final = await agent.submit_approval(
        paused.run_id, tenant_id="tenant", expected_pause_id=result.pause_id, approved=approved
    )
    assert final.status.value == "success"
    assert effects == (["write_region"] * 2 if approved else [])
    rows = await store.get_tool_calls(paused.run_id)
    assert {r.tool_call_id for r in rows} == ({c.id for c in calls} if approved else set())
    assert len(rows) == (2 if approved else 0)
    events = await store.get_run_events(paused.run_id)
    sequence = [e.event_type for e in events]
    assert sequence.count("run.paused") == 2
    assert sequence.count("run.resumed") == 3
    assert sequence.count("approval.requested") == 2
    assert sequence.count("approval.decided") == 1
    print("event_sequence", entry, approved, sequence)
    await agent.close()


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
async def test_061_recovery_pause_can_repeat(store, engine, monkeypatch, entry):
    from dendrux.runtime import runner

    effects = []
    agent = _agent(store, effects)
    paused, calls = await _requested_pause(agent, store, engine, monkeypatch)
    original = runner._resume_barrier

    async def request_again(owned, run_id):
        assert await store.request_pause(run_id)
        return await original(owned, run_id)

    for _ in range(2):
        with monkeypatch.context() as patch:
            patch.setattr(runner, "_resume_barrier", request_again)
            again = await _continue(agent, paused.run_id, paused.pause_id, entry)
        assert again.status.value == "paused"
        assert again.pause_id != paused.pause_id
        assert effects == []
        assert [p["call_id"] for p in (await store.get_loop_state(paused.run_id))["pending"]] == [
            c.id for c in calls
        ]
        paused = again
    result = await _continue(agent, paused.run_id, paused.pause_id, entry)
    assert result.status.value == "waiting_approval" and effects == []
    final = await agent.submit_approval(
        result.run_id, tenant_id="tenant", expected_pause_id=result.pause_id, approved=False
    )
    assert final.status.value == "success" and effects == []
    await agent.close()


async def test_061_cancel_wins_over_recovery_pause(store, engine, monkeypatch):
    effects = []
    agent = _agent(store, effects)
    result, _ = await _requested_pause(agent, store, engine, monkeypatch, cancel=True)
    assert (await store.get_run(result.run_id)).status == "cancelled"
    assert await store.get_pause_state(result.run_id) is None
    assert await store.get_tool_calls(result.run_id) == []
    assert effects == []
    await agent.close()


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
async def test_061_started_idempotent_recovery_retains_approval(store, engine, monkeypatch, entry):
    effects = []
    agent = _agent(store, effects, idempotent=True)
    paused, _ = await _requested_pause(agent, store, engine, monkeypatch, started=True)
    result = await _continue(agent, paused.run_id, paused.pause_id, entry)
    assert result.status.value == "waiting_approval" and effects == []
    final = await agent.submit_approval(
        result.run_id, tenant_id="tenant", expected_pause_id=result.pause_id, approved=True
    )
    assert final.status.value == "success" and effects == ["write_region"] * 2
    rows = await store.get_tool_calls(result.run_id)
    assert len(rows) == 2 and all(r.replayed_at is not None for r in rows)
    await agent.close()


async def test_061_gated_intent_still_requires_submission(store, engine):
    from dendrux.errors import PersistenceError
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.state import ClaimResult, claimed_store

    effects = []
    agent = _agent(store, effects)
    call = ToolCall("write_region")
    item = await checkpoint(agent, store, engine, calls=[call], phases=["gated"])
    state = await store.get_loop_state(item.run_id)
    recorder = PersistenceRecorder(
        claimed_store(store, item.run_id, ClaimResult(item.claim_epoch, 2, state, None)),
        item.run_id,
    )
    with pytest.raises(PersistenceError) as caught:
        await recorder.on_tool_started(item.run_id, call, 1)
    assert caught.value.code == "submission_missing"
    assert await store.get_tool_calls(item.run_id) == [] and effects == []
    await agent.close()


async def test_061_concurrent_recovery_continuations_have_one_winner(store, engine, monkeypatch):
    import asyncio

    from dendrux.errors import PauseStatusMismatchError, RunAlreadyClaimedError, RunNotPausedError
    from dendrux.runtime.stores._factory import store_for_engine

    if engine.dialect.name != "postgresql":
        pytest.skip("Concurrent claims require independent PostgreSQL connections")
    effects = []
    agent = _agent(store, effects)
    paused, calls = await _requested_pause(agent, store, engine, monkeypatch)
    barrier = asyncio.Barrier(8)
    contenders = []
    for _ in range(8):
        contender_store = store_for_engine(engine)
        original = contender_store.submit_and_claim

        async def race(*args, _submit=original, **kwargs):
            await asyncio.wait_for(barrier.wait(), 10)
            return await _submit(*args, **kwargs)

        monkeypatch.setattr(contender_store, "submit_and_claim", race)
        contenders.append(_agent(contender_store, effects))
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    _continue(a, paused.run_id, paused.pause_id, "submit_resume")
                    for a in contenders
                ),
                return_exceptions=True,
            ),
            20,
        )
        winners = [r for r in results if not isinstance(r, BaseException)]
        losers = [r for r in results if isinstance(r, BaseException)]
        assert len(winners) == 1 and len(losers) == 7
        assert all(
            isinstance(r, (RunAlreadyClaimedError, RunNotPausedError, PauseStatusMismatchError))
            for r in losers
        )
        winner = winners[0]
        assert winner.status.value == "waiting_approval" and effects == []
        state = await store.get_loop_state(paused.run_id)
        assert state["submission"] is None
        assert [p["call_id"] for p in state["pending"]] == [c.id for c in calls]
        assert await store.get_tool_calls(paused.run_id) == []
        events = await store.get_run_events(paused.run_id)
        assert sum(e.event_type == "run.resumed" for e in events) == 2
        final = await agent.submit_approval(
            paused.run_id, tenant_id="tenant", expected_pause_id=winner.pause_id, approved=True
        )
        assert final.status.value == "success" and effects == ["write_region"] * 2
        assert (await store.get_loop_state(paused.run_id))["pending"] == []
        assert len(await store.get_tool_calls(paused.run_id)) == 2
    finally:
        await asyncio.gather(*(a.close() for a in contenders))
        await agent.close()
