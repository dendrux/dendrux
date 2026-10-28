"""Recovery marking, keyset scans, tenancy and competing reclaimers."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux.db.models import AgentRun
from dendrux.errors import RunNotFoundError
from dendrux.types import generate_ulid


async def stale(store, engine, *, tenant="tenant", count=1):
    ids = []
    for _ in range(count):
        run_id = generate_ulid()
        await store.create_run(run_id, "recovery", strategy="react", tenant_id=tenant)
        ids.append(run_id)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_(ids))
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    return ids


async def mark(store, **kwargs):
    assert hasattr(store, "mark_lease_expired"), "recovery needs an atomic lease-expiry mark"
    return await store.mark_lease_expired(
        cutoff=datetime.now(UTC) - timedelta(minutes=1),
        **{
            **dict(batch_size=10, max_runs=100, max_returned=100, reclaim_limit=3, cursor=None),
            **kwargs,
        },
    )


async def reclaim(store, item, tenant="tenant"):
    return await store.reclaim_run(
        item.run_id,
        tenant_id=tenant,
        agent_name="recovery",
        expected_claim_epoch=item.claim_epoch,
        governance={},
        governance_policy="raise",
        skill_pins={},
        loop_type="react",
    )


async def test_a2_twenty_reclaimers_one_winner(store, engine):
    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    results = await asyncio.gather(
        *(reclaim(store, item) for _ in range(20)), return_exceptions=True
    )
    winners = [r for r in results if not isinstance(r, BaseException) and r is not False]
    assert len(winners) == 1
    losers = [r for r in results if isinstance(r, BaseException)]
    assert len(losers) == 19 and all(type(r).__name__ == "RunNotReclaimableError" for r in losers)
    row = await store.get_run(item.run_id)
    assert row.claim_epoch == 2 and row.reclaim_attempts == 1
    assert row.lease_expired_at is None and row.last_reclaimed_at is not None
    assert [e.event_type for e in await store.get_run_events(item.run_id)] == [
        "run.lease_expired",
        "run.reclaimed",
    ]


async def test_a13_a14_two_markers_rediscover_without_duplicate_events(store, engine):
    ids = await stale(store, engine)
    first, second = await asyncio.gather(mark(store), mark(store))
    assert first.reclaimable[0].claim_epoch == second.reclaimable[0].claim_epoch == 1
    again = await store.get_lease_expired(limit=10, cursor=None)
    assert [r.run_id for r in again] == ids
    assert len(await store.get_run_events(ids[0])) == 1


async def test_a15_cursor_covers_backlog_never_wraps(store, engine):
    ids = await stale(store, engine, count=7)
    seen, cursor = [], None
    for _ in range(5):
        page = await mark(store, max_runs=3, max_returned=2, cursor=cursor)
        seen.extend(r.run_id for r in page.reclaimable)
        cursor = page.cursor
        if cursor is None:
            break
    assert seen == ids and len(set(seen)) == 7 and cursor is None
    assert not (await mark(store, cursor=ids[-1])).reclaimable


async def test_a12_foreign_and_missing_refused_without_mutation(store, engine):
    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    before = await store.get_run(item.run_id)
    with pytest.raises(RunNotFoundError):
        await reclaim(store, item, tenant="foreign")
    after = await store.get_run(item.run_id)
    assert before == after
    from dataclasses import replace

    with pytest.raises(RunNotFoundError):
        await reclaim(store, replace(item, run_id=generate_ulid()))


async def test_a6_limit_mark_finalizes_exhausted_run(store, engine):
    await stale(store, engine)
    item = (await mark(store, reclaim_limit=1)).reclaimable[0]
    await reclaim(store, item)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == item.run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    result = await mark(store, reclaim_limit=1)
    assert not result.reclaimable
    assert result.finalised[0].failure_reason == "reclaim_limit"
    assert (await store.get_run(item.run_id)).status == "error"


async def test_fresh_scan_has_no_per_run_read(store, engine):
    from sqlalchemy import event

    for _ in range(20):
        await store.create_run(generate_ulid(), "fresh", strategy="react")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await mark(store, batch_size=100)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert result.inspected == 20 and not result.reclaimable
    assert len(statements) <= 2, "fresh runs are inspected in a page, without N+1 row reads"


async def test_recovery_reads_exclude_unneeded_payload_columns(store, engine):
    from sqlalchemy import event

    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await store.get_lease_expired(limit=10, cursor=None)
        await reclaim(store, item)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    for sql in statements:
        if sql.lstrip().upper().startswith("SELECT") or " RETURNING " in sql:
            assert all(
                name not in sql
                for name in ("input_data", "output_data", "pii_mapping", "pause_data")
            ), "recovery must project execution metadata without loading conversation payloads"


async def test_new_mark_persists_the_current_reclaim_limit(store, engine):
    await stale(store, engine)
    first = (await mark(store, reclaim_limit=1)).reclaimable[0]
    await reclaim(store, first)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == first.run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    second = await mark(store, reclaim_limit=2)
    assert len(second.reclaimable) == 1 and not second.finalised
    assert second.reclaimable[0].reclaim_limit == 2
