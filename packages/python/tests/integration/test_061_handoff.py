"""Store-created Tasks retain committed ownership through cancellation."""

import asyncio
from functools import wraps

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db.models import AgentRun
from tests.integration import test_060_handoff_deadlines as deadlines
from tests.integration import test_060_ownership_cancellation as ownership


@pytest.fixture(params=[False, True], ids=["coroutine", "task"])
def task_store(store, monkeypatch, request):
    if request.param:
        for name in ("create_run", "submit_and_claim", "claim_paused_run", "reclaim_run"):
            original = getattr(store, name)

            def wrap(method):
                @wraps(method)
                def spawned(*args, **kwargs):
                    return asyncio.create_task(method(*args, **kwargs))

                return spawned

            monkeypatch.setattr(store, name, wrap(original))
    return store


@pytest.mark.parametrize("entry", ownership.ENTRIES)
async def test_061_task_store_handoff_inherits_commit_marker(
    task_store, engine, monkeypatch, entry
):
    await ownership.test_cancel_ownership_handoff(
        task_store, engine, monkeypatch, entry, "commit_after"
    )


@pytest.mark.parametrize("entry", ["run", "submit_tool_results", "submit_resume", "reclaim"])
async def test_task_store_precommit_cancellation(task_store, engine, monkeypatch, entry):
    await deadlines.test_precommit_cancellation_preserves_state(
        task_store, engine, monkeypatch, entry
    )


@pytest.mark.parametrize("slow", [False, True])
async def test_task_store_cleanup_deadline(task_store, engine, monkeypatch, caplog, slow):
    await deadlines.test_commit_cancel_deadline(task_store, engine, monkeypatch, caplog, slow)


@pytest.mark.parametrize("entry", ["run", "submit_tool_results", "submit_resume", "reclaim"])
async def test_task_store_repeated_cleanup_cancellation(task_store, engine, monkeypatch, entry):
    original_commit = AsyncSession.commit
    original_finalize = task_store.finalize_with_event
    committed, release = asyncio.Event(), asyncio.Event()
    cleaning, finish = asyncio.Event(), asyncio.Event()
    agent, call, run_id, epoch, calls = await ownership.invocation(entry, task_store, engine)
    before_calls = calls()
    armed = True
    finalizations = []

    async def commit(session):
        nonlocal armed
        gate = armed
        armed = False
        await original_commit(session)
        if gate:
            committed.set()
            await release.wait()

    async def finalize(rid, *, claim_epoch=None, **kwargs):
        finalizations.append((rid, claim_epoch))
        cleaning.set()
        await finish.wait()
        return await original_finalize(rid, claim_epoch=claim_epoch, **kwargs)

    monkeypatch.setattr(AsyncSession, "commit", commit)
    monkeypatch.setattr(task_store, "finalize_with_event", finalize)
    caller = asyncio.create_task(call)
    try:
        await asyncio.wait_for(committed.wait(), 5)
        caller.cancel()
        release.set()
        await asyncio.wait_for(cleaning.wait(), 5)
        for _ in range(3):
            caller.cancel()
            await asyncio.sleep(0)
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, 5)
        assert len(finalizations) == 1
        acquired_id, acquired_epoch = finalizations[0]
        assert run_id is None or acquired_id == run_id
        assert acquired_epoch == epoch
        row = await task_store.get_run(acquired_id)
        assert row.status == "cancelled" and row.claim_epoch == epoch
        assert calls() == before_calls
    finally:
        release.set()
        finish.set()
        await asyncio.gather(caller, return_exceptions=True)
        await agent.close()


async def test_postcommit_cancel_races_with_another_claim(task_store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent owners require Postgres")

    from dendrux.runtime.runner import _persist_task_cancellation
    from dendrux.runtime.state import claimed_store
    from dendrux.runtime.submit import governance_snapshot
    from tests._helpers.recovery_checkpoint import expire

    agent, call, run_id, epoch, calls = await ownership.invocation("reclaim", task_store, engine)
    original = AsyncSession.commit
    committed, release = asyncio.Event(), asyncio.Event()
    armed = True

    async def commit(session):
        nonlocal armed
        gate = armed
        armed = False
        await original(session)
        if gate:
            committed.set()
            await release.wait()

    monkeypatch.setattr(AsyncSession, "commit", commit)
    caller = asyncio.create_task(call)
    next_owner = None
    try:
        await asyncio.wait_for(committed.wait(), 5)
        caller.cancel()
        await asyncio.sleep(0)

        async def take_over():
            item = await expire(task_store, engine, run_id)
            claim = await task_store.reclaim_run(
                run_id,
                tenant_id="tenant",
                agent_name=agent.name,
                expected_claim_epoch=item.claim_epoch,
                governance=governance_snapshot(agent),
                governance_policy="raise",
                skill_pins={},
                loop_type="react",
            )
            assert claim and claim.claim_epoch == epoch + 2
            return claim

        next_owner = asyncio.create_task(take_over())
        claim = await asyncio.wait_for(next_owner, 5)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, 5)
        row = await task_store.get_run(run_id)
        assert row.status == "running" and row.claim_epoch == claim.claim_epoch
        assert calls() == 0
        assert not any(
            event.event_type == "run.cancelled" for event in await task_store.get_run_events(run_id)
        )

        await _persist_task_cancellation(claimed_store(task_store, run_id, claim), run_id)
        async with engine.connect() as conn:
            assert not (
                await conn.execute(select(AgentRun.id).where(AgentRun.status == "running"))
            ).all()
        cancellations = [
            event
            for event in await task_store.get_run_events(run_id)
            if event.event_type == "run.cancelled"
        ]
        assert len(cancellations) == 1
        assert (await task_store.get_run(run_id)).claim_epoch == claim.claim_epoch
    finally:
        release.set()
        await asyncio.gather(caller, *([next_owner] if next_owner else []), return_exceptions=True)
        await agent.close()
