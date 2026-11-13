"""Each sweep entry shares one budget and publishes earlier committed runs on failure."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import update

from dendrux import sweep
from dendrux.db.models import AgentRun
from dendrux.types import generate_ulid
from tests.integration.test_live_updates_store_transactions import live_store as live_store
from tests.integration.test_live_updates_store_transactions import sent


async def stale(store, engine, count, *, waiting=False):
    ids = sorted(generate_ulid() for _ in range(count))
    for run_id in ids:
        await store.create_run(run_id, "sweep", strategy="react")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_(ids))
            .values(
                last_progress_at=datetime.now(UTC) - timedelta(hours=2),
                updated_at=datetime.now(UTC) - timedelta(hours=2),
                status_changed_at=datetime.now(UTC) - timedelta(hours=2),
                status="waiting_human_input" if waiting else "running",
                loop_state={"v": 1, "loop": "react", "phase": "model_next"},
                storage_format=2,
            )
        )
    return ids


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze the flush-budget clock so results do not depend on runner speed."""
    from dendrux.db import _live_updates
    from dendrux.runtime import durability

    monkeypatch.setattr(durability, "monotonic", lambda: 100.0)
    monkeypatch.setattr(_live_updates, "monotonic", lambda: 100.0)


async def invoke(store, entry):
    if entry == "sweep":
        return await sweep(
            state_store=store,
            stale_running=timedelta(hours=1),
            abandoned_waiting=timedelta(hours=1),
            batch_size=2,
            on_stale="error",
        )
    if entry in ("_scan_expired_runs", "mark_lease_expired"):
        return await getattr(store, entry)(
            cutoff=datetime.now(UTC) - timedelta(hours=1),
            batch_size=2,
            max_runs=100,
            max_returned=100,
            reclaim_limit=3,
            cursor=None,
        )
    if entry == "_sweep_runs":
        return await store._sweep_runs(
            timedelta(hours=1), waiting=False, batch_size=2, max_runs=100
        )
    return await getattr(store, entry)(timedelta(hours=1), batch_size=2, max_runs=100)


@pytest.mark.parametrize(
    "entry",
    [
        "sweep",
        "_scan_expired_runs",
        "_sweep_runs",
        "mark_lease_expired",
        "sweep_stale_runs",
        "sweep_abandoned_runs",
    ],
)
async def test_all_sweep_boundaries_share_deadline_and_flush_between_sessions(
    live_store, engine, monkeypatch, entry
):
    from dendrux.db import _live_updates
    from dendrux.runtime import durability

    store, bus = live_store
    ids = await stale(store, engine, 20, waiting=entry == "sweep_abandoned_runs")
    if entry == "sweep":
        ids += await stale(store, engine, 20, waiting=True)
    bus.messages.clear()
    now = [100.0]
    monkeypatch.setattr(durability, "monotonic", lambda: now[0])
    monkeypatch.setattr(_live_updates, "monotonic", lambda: now[0])
    send = bus.send

    async def exhausting(messages):
        async with store._session_factory():
            pass  # deadlocks if a sweep still holds the serialized-session lock
        await send(messages)
        now[0] += bus.flush_deadline_s

    monkeypatch.setattr(bus, "send", exhausting)
    flush = AsyncMock(wraps=bus._flush_before)
    monkeypatch.setattr(bus, "_flush_before", flush)
    await invoke(store, entry)
    assert sent(bus) == ids[:2]
    assert len(bus.messages) == 1
    assert bus.stats().pending == len(ids) - 2
    assert {call.args[0] for call in flush.await_args_list} == {101.0}
    assert len(flush.await_args_list) == len(ids) // 2 + 1


@pytest.mark.parametrize("failed_write", [2, 4])
@pytest.mark.parametrize("entry", ["sweep", "_scan_expired_runs", "_sweep_runs"])
async def test_partial_failure_flushes_committed_runs_preserves_error(
    live_store, engine, monkeypatch, frozen_clock, entry, failed_write
):
    store, bus = live_store
    ids = await stale(store, engine, 4)
    bus.messages.clear()
    write = store._write_tx
    calls = 0
    error = ValueError("original")

    async def fail_second(session, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_write:
            raise error
        await write(session, **kwargs)

    monkeypatch.setattr(store, "_write_tx", fail_second)
    with pytest.raises(ValueError) as raised:
        await invoke(store, entry)
    assert raised.value is error
    assert sent(bus) == ids[: failed_write - 1]
    assert bus.stats().pending == 0


async def test_nested_sweep_establishes_budget_inside_existing_store_call(
    live_store, engine, monkeypatch
):
    from dendrux.runtime.durability import _safe_database_call

    store, bus = live_store
    await stale(store, engine, 4)
    bus.messages.clear()
    deadlines = []
    original = bus._flush_before

    async def capture(deadline):
        deadlines.append(deadline)
        await original(deadline)

    monkeypatch.setattr(bus, "_flush_before", capture)

    @_safe_database_call
    async def outer(self):
        await invoke(self, "sweep")

    await outer(store)
    # The outer normal call may flush with infinity; every sweep page must have a budget.
    assert len(set(deadlines[:-1])) == 1
    assert deadlines[0] < float("inf")


@pytest.mark.parametrize("kind", ["policy", "cursor", "capability"])
async def test_refused_sweep_leaves_pending_hints_without_flushing(live_store, kind):
    from dataclasses import replace

    store, bus = live_store
    bus.enqueue([generate_ulid()])
    kwargs = dict(stale_running=timedelta(hours=1), on_stale="reclaim")
    if kind == "policy":
        kwargs["on_stale"] = "invalid"
    elif kind == "cursor":
        kwargs["cursor"] = 1
    else:
        store.capabilities = replace(store.capabilities, recovery=False)
    with pytest.raises(NotImplementedError if kind == "capability" else ValueError):
        await sweep(state_store=store, **kwargs)
    assert bus.stats().pending == 1
    assert sent(bus) == []


@pytest.mark.parametrize(
    "entry",
    [
        "sweep",
        "_scan_expired_runs",
        "_sweep_runs",
        "mark_lease_expired",
        "sweep_stale_runs",
        "sweep_abandoned_runs",
    ],
)
async def test_sweep_sends_one_batched_message_per_page(live_store, engine, monkeypatch, entry):
    from dendrux.db import _live_updates
    from dendrux.runtime import durability

    store, bus = live_store
    ids = await stale(store, engine, 5, waiting=entry == "sweep_abandoned_runs")
    bus.messages.clear()
    monkeypatch.setattr(durability, "monotonic", lambda: 100.0)
    monkeypatch.setattr(_live_updates, "monotonic", lambda: 100.0)
    await invoke(store, entry)
    assert [json.loads(message) for message in bus.messages] == [ids[:2], ids[2:4], ids[4:]]
    assert bus.stats().pending == 0


@pytest.mark.parametrize("entry", ["_scan_expired_runs", "_sweep_runs"])
async def test_failed_commit_in_page_stays_pending_after_earlier_success_flush(
    live_store, engine, monkeypatch, frozen_clock, entry
):
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.errors import StoreOutcomeUnknown

    store, bus = live_store
    ids = await stale(store, engine, 4)
    bus.messages.clear()
    commit = AsyncSession.commit
    calls = 0
    error = StoreOutcomeUnknown()

    async def lost_ack(session):
        nonlocal calls
        calls += 1
        await commit(session)
        if calls == 2:
            raise error

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        with pytest.raises(StoreOutcomeUnknown) as raised:
            await invoke(store, entry)
    assert raised.value is error
    assert sent(bus) == ids[:1]
    assert bus.stats().pending == 1
    await bus.flush()
    assert sent(bus) == ids[:2]


async def test_successful_retry_in_page_does_not_leave_a_duplicate_hint(
    live_store, engine, monkeypatch, frozen_clock
):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    class DeadlockError(Exception):
        sqlstate = "40P01"

    store, bus = live_store
    ids = await stale(store, engine, 2)
    bus.messages.clear()
    commit = AsyncSession.commit
    calls = 0

    async def retry_once(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OperationalError("commit-fault", {}, DeadlockError())
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", retry_once)
    await invoke(store, "_scan_expired_runs")
    assert sent(bus) == ids
    assert len(bus.messages) == 1
    assert bus.stats().pending == 0
