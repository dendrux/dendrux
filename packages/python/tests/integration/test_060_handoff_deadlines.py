"""Cancellation before COMMIT is interruptible; committed ownership is bounded."""

import asyncio
import time

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db.models import AgentRun
from dendrux.runtime import durability
from dendrux.runtime.submit import _cancel_claim
from dendrux.runtime.tasks import _drain_cleanup
from tests._helpers.gates import wait_for_gate
from tests.integration.test_060_ownership_cancellation import (
    ENTRIES,
    GateBus,
    continuation,
    invocation,
)

# A hang tripwire, not a speed claim: these cancels take about 60 ms, but a busy machine can stop
# the event loop for over 0.5 s (a full garbage collection, or no CPU). It stays well under the
# 5 s deadlines a regression would wait for. Speed is measured in the benchmark at the end.
CANCEL_TRIPWIRE_S = 2.5


async def stop_red_children(task):
    for child in asyncio.all_tasks():
        if child is not task and "_without_live_flush" in child.get_coro().__qualname__:
            child.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_drain_failure_preserves_cancel(caplog):
    entered, release = asyncio.Event(), asyncio.Event()

    async def failing():
        entered.set()
        await release.wait()
        raise ValueError("private_failure_sentinel")

    task = asyncio.create_task(_drain_cleanup(failing()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "persistence_error" in caplog.text
    assert "private_failure_sentinel" not in caplog.text


@pytest.mark.parametrize("entry", ENTRIES)
async def test_precommit_cancellation_preserves_state(store, engine, monkeypatch, entry):
    assert CANCEL_TRIPWIRE_S * 2 <= durability._OWNERSHIP_DEADLINE_S
    agent, call, run_id, epoch, calls = await invocation(entry, store, engine)
    before = await store.get_run(run_id) if run_id else None
    async with engine.connect() as conn:
        before_ids = (await conn.execute(select(AgentRun.id))).all()
    count = calls()
    gate = asyncio.Event()
    original = store._write_tx

    async def blocked(*args, **kwargs):
        gate.set()
        await asyncio.Event().wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "_write_tx", blocked)
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(gate, task=task)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TRIPWIRE_S)
        assert done, "pre-COMMIT cancellation did not return promptly"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls() == count
        if run_id is None:
            async with engine.connect() as conn:
                assert (await conn.execute(select(AgentRun.id))).all() == before_ids
        else:
            assert await store.get_run(run_id) == before
            if entry == "reclaim":
                expired = await store.get_lease_expired(limit=100, cursor=None)
                assert any(item.run_id == run_id for item in expired)
        monkeypatch.setattr(store, "_write_tx", original)
        if run_id is not None:
            retry = (
                agent.reclaim(run_id, tenant_id="tenant")
                if entry == "reclaim"
                else continuation(entry, agent, run_id, await store.get_pause_state(run_id))
            )
        elif entry == "retry":
            retry = agent.retry(before_ids[0][0], tenant_id="tenant")
        elif entry == "runs":
            retry = agent.run("again", runs=[before_ids[0][0]], tenant_id="tenant")
        elif entry == "stream":
            from tests.integration.test_060_ownership_cancellation import consume

            retry = consume(agent.stream("go", tenant_id="tenant"))
        else:
            retry = agent.run("go", tenant_id="tenant")
        result = await retry
        if isinstance(result, list):
            result = result[-1].run_result
        assert result.status.value == "success"
    finally:
        await stop_red_children(task)
        await agent.close()


@pytest.mark.parametrize("slow", [False, True])
async def test_commit_cancel_deadline(store, engine, monkeypatch, caplog, slow):
    # The slow case proves a 0.1 s COMMIT is kept, so it runs under the real deadline.
    if not slow:
        monkeypatch.setattr(durability, "_OWNERSHIP_DEADLINE_S", 0.3)
    agent, call, _, _, calls = await invocation("run", store, engine)
    entered, released = asyncio.Event(), asyncio.Event()
    original = AsyncSession.commit
    armed = True

    async def commit(session):
        nonlocal armed
        await original(session)
        if armed:
            armed = False
            entered.set()
            try:
                await asyncio.sleep(0.1 if slow else 10)
            finally:
                released.set()

    monkeypatch.setattr(AsyncSession, "commit", commit)
    task = asyncio.create_task(call)
    await wait_for_gate(entered, task=task)
    started = time.monotonic()
    task.cancel()
    try:
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TRIPWIRE_S)
        assert done, "committed handoff exceeded private deadline"
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed = time.monotonic() - started
        assert released.is_set() and calls() == 0
        async with engine.connect() as conn:
            run_id = (await conn.execute(select(AgentRun.id))).scalar_one()
        row = await store.get_run(run_id)
        assert row.status == ("cancelled" if slow else "running")
        assert row.claim_epoch == 0
        if not slow:
            assert elapsed >= 0.25
            assert any(
                getattr(r, "code", None) == "cancel_persist_failed"
                and getattr(r, "run_id", None) == run_id
                for r in caplog.records
            )
            from tests._helpers.recovery_checkpoint import expire

            item = await expire(store, engine, run_id)
            assert item.claim_epoch == 1
        print(f"COMMIT_DEADLINE slow={slow} elapsed={elapsed:.3f}")
    finally:
        await stop_red_children(task)
        await agent.close()


async def test_handoff_cancel_flushes_hint(store, engine, monkeypatch):
    bus = GateBus()
    sent = []

    async def send(messages):
        sent.extend(messages)

    monkeypatch.setattr(bus, "send", send)
    monkeypatch.setattr(store, "_live_updates", bus)
    agent, call, _, _, _ = await invocation("run", store, engine)
    original = AsyncSession.commit
    entered, release = asyncio.Event(), asyncio.Event()
    armed = True

    async def commit(session):
        nonlocal armed
        await original(session)
        if armed:
            armed = False
            entered.set()
            await release.wait()

    monkeypatch.setattr(AsyncSession, "commit", commit)
    task = asyncio.create_task(call)
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sent and not bus._pending
    await agent.close()


@pytest.mark.parametrize("result", [False, True, 0])
async def test_boolean_claim_has_no_cleanup_authority(store, engine, result):
    agent, call, run_id, _, _ = await invocation("submit_tool_results", store, engine)
    call.close()
    await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    if type(result) is int:
        from sqlalchemy import update

        async with engine.begin() as conn:
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=0))
    before = await store.get_run(run_id)
    await _cancel_claim(store, run_id, result)
    assert await store.get_run(run_id) == before
    assert not any(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id))
    await agent.close()


@pytest.mark.parametrize("entry", ["run", "stream", "submit_tool_results", "timeout"])
async def test_store_failure_never_replaces_caller_cancel(store, engine, monkeypatch, entry):
    from dendrux.errors import PersistenceError

    agent, call, _, _, _ = await invocation(
        "stream" if entry == "timeout" else entry, store, engine
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = AsyncSession.commit
    armed = True

    async def fail(session):
        nonlocal armed
        await original(session)
        if armed:
            armed = False
            entered.set()
            await release.wait()
            raise PersistenceError()

    monkeypatch.setattr(AsyncSession, "commit", fail)
    continued = []
    deadline = asyncio.timeout(None)

    async def consume():
        if entry == "timeout":
            async with deadline:
                await call
        else:
            await call
        continued.append(True)

    task = asyncio.create_task(consume())
    await entered.wait()
    if entry == "timeout":
        # Expire the caller's own timeout only once COMMIT is in flight.
        deadline.reschedule(asyncio.get_running_loop().time())
        while not deadline.expired():
            await asyncio.sleep(0)
    else:
        task.cancel()
        await asyncio.sleep(0)
    release.set()
    with pytest.raises(TimeoutError if entry == "timeout" else asyncio.CancelledError):
        await task
    assert continued == []
    await agent.close()


async def test_cleanup_deadline_and_hint_deadline_are_joined(store, engine, monkeypatch, caplog):
    agent, call, run_id, epoch, calls = await invocation("submit_tool_results", store, engine)
    bus = GateBus()
    bus.flush_deadline_s = bus.send_timeout_s = 0.05
    sends = []
    release = asyncio.Event()  # Only set on failure, so a regression fails by name.

    async def blocked_send(messages):
        sends.append(messages)
        bus.entered.set()
        await release.wait()

    monkeypatch.setattr(bus, "send", blocked_send)
    monkeypatch.setattr(store, "_live_updates", bus)
    monkeypatch.setattr(durability, "_OWNERSHIP_DEADLINE_S", 0.1)
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def hung_finalize(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
        finally:
            stopped.set()

    monkeypatch.setattr(store, "finalize_with_event", hung_finalize)
    bus.enabled = True
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(bus.entered, task=task)
        started = time.monotonic()
        task.cancel()
        await wait_for_gate(entered, task=task)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TRIPWIRE_S)
        elapsed = time.monotonic() - started
        assert done, "caller was not released by the ownership deadline"
        with pytest.raises(asyncio.CancelledError):
            task.result()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
    assert elapsed >= 0.09  # The second cancel did not cut the 0.1 s deadline short.
    assert len(sends) == 1  # The interrupted hint was lossy; cleanup never committed another.
    assert any(
        getattr(r, "code", None) == "cancel_persist_failed" and getattr(r, "run_id", None) == run_id
        for r in caplog.records
    )
    assert stopped.is_set()
    row = await store.get_run(run_id)
    assert row.status == "running" and row.claim_epoch == epoch
    assert calls() == 1
    from tests._helpers.recovery_checkpoint import expire

    item = await expire(store, engine, run_id)
    assert item.claim_epoch == epoch + 1
    await agent.close()


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
@pytest.mark.parametrize("superseded", [False, True])
async def test_legacy_governance_cancel_has_no_epoch_authority(
    store, engine, monkeypatch, entry, superseded
):
    from sqlalchemy import update

    from dendrux.runtime import runner, submit

    agent, call, run_id, epoch, _ = await invocation(entry, store, engine)
    original = store.submit_and_claim

    async def legacy(*args, **kwargs):
        return bool(await original(*args, **kwargs))

    monkeypatch.setattr(store, "submit_and_claim", legacy)
    entered = asyncio.Event()

    async def governance(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        submit if entry == "submit_tool_results" else runner, "report_governance_change", governance
    )
    task = asyncio.create_task(call)
    await wait_for_gate(entered, task=task)
    if superseded:
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=epoch + 1)
            )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await store.get_run(run_id)
    assert row.status == "running" and row.claim_epoch == epoch + int(superseded)
    assert not any(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id))
    await agent.close()


async def test_final_cleanup_hint_has_one_flush_deadline(store, engine, monkeypatch):
    agent, call, run_id, epoch, calls = await invocation("submit_tool_results", store, engine)
    bus = GateBus()
    bus.flush_deadline_s = bus.send_timeout_s = 0.05
    sends = []
    release = asyncio.Event()  # Only set on failure, so a regression fails by name.

    async def send(messages):
        sends.append(messages)
        bus.entered.set()
        await release.wait()

    monkeypatch.setattr(bus, "send", send)
    monkeypatch.setattr(store, "_live_updates", bus)
    task = asyncio.create_task(call)
    await bus.entered.wait()
    started = time.monotonic()
    task.cancel()
    try:
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TRIPWIRE_S)
        elapsed = time.monotonic() - started
        assert done, "the final hint held the caller past its flush deadline"
        with pytest.raises(asyncio.CancelledError):
            task.result()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
    assert elapsed >= 0.04  # The final hint really had its 0.05 s to send.
    assert len(sends) == 2 and not bus._pending
    row = await store.get_run(run_id)
    assert row.status == "cancelled" and row.claim_epoch == epoch
    assert calls() == 1
    await agent.close()


async def test_precommit_cancel_does_not_wait_for_unrelated_hint(store, engine, monkeypatch):
    agent, call, run_id, _, _ = await invocation("submit_tool_results", store, engine)
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    entered = asyncio.Event()

    async def before_commit(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(store, "_write_tx", before_commit)
    task = asyncio.create_task(call)
    try:
        await entered.wait()
        bus.enabled = True
        bus.enqueue([run_id])
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TRIPWIRE_S)
        assert done, "an unrelated pending hint delayed pre-COMMIT cancellation"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not bus.entered.is_set()
    finally:
        bus.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.benchmark
async def test_precommit_cancel_is_fast_on_a_quiet_machine(store, engine, monkeypatch):
    agent, call, run_id, _, _ = await invocation("submit_tool_results", store, engine)
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    entered = asyncio.Event()

    async def before_commit(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(store, "_write_tx", before_commit)
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(entered, task=task)
        bus.enabled = True
        bus.enqueue([run_id])
        started = time.monotonic()
        task.cancel()
        await asyncio.wait({task}, timeout=CANCEL_TRIPWIRE_S)
        assert time.monotonic() - started < 0.1
    finally:
        bus.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()
