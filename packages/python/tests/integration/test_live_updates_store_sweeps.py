"""Each sweep entry shares one budget and publishes earlier committed runs on failure."""

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
    assert len(sent(bus)) == 1
    assert sent(bus)[0] == ids[0]
    assert bus.stats().pending == len(ids) - 1
    assert {call.args[0] for call in flush.await_args_list} == {101.0}
    assert len(flush.await_args_list) >= len(ids)


@pytest.mark.parametrize("entry", ["sweep", "_scan_expired_runs", "_sweep_runs"])
async def test_partial_failure_flushes_committed_runs_preserves_error(
    live_store, engine, monkeypatch, entry
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
        if calls == 2:
            raise error
        await write(session, **kwargs)

    monkeypatch.setattr(store, "_write_tx", fail_second)
    with pytest.raises(ValueError) as raised:
        await invoke(store, entry)
    assert raised.value is error
    assert sent(bus) == ids[:1]
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
