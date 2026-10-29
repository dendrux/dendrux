"""Public sweep policy, pagination and fencing on both databases."""

import asyncio
import warnings
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux import sweep
from dendrux.db.models import AgentRun
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import generate_ulid


async def candidate(store, engine, *, fmt=2, marker="react", started=True, **values):
    run_id = generate_ulid()
    await store.create_run(run_id, "recovery", tenant_id="tenant")
    if started:
        await store.save_run_event(run_id, event_type="run.started")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                storage_format=fmt,
                loop_state={"loop": marker} if marker else {},
                last_progress_at=datetime.now(UTC) - timedelta(minutes=10),
                **values,
            )
        )
    return run_id


async def scan(store, **kwargs):
    return await sweep(state_store=store, stale_running=timedelta(minutes=1), **kwargs)


@pytest.mark.parametrize("policy", [None, "error", "reclaim"])
@pytest.mark.parametrize(
    "fmt,marker", [(1, None), (2, "react"), (2, "single_call"), (2, None), (2, "invalid")]
)
async def test_outcome_table(store, engine, caplog, policy, fmt, marker):
    run_id = await candidate(store, engine, fmt=fmt, marker=marker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await scan(store, **({"on_stale": policy} if policy else {}))
    if policy == "reclaim" and fmt == 2 and marker == "react":
        assert [r.run_id for r in result.reclaimable] == [run_id]
        assert (await store.get_run(run_id)).status == "running"
    else:
        assert [r.run_id for r in result.stale_running] == [run_id]
        expected = "stale_running_legacy" if policy == "reclaim" and fmt == 1 else "stale_running"
        assert result.stale_running[0].failure_reason == expected
        assert (await store.get_run(run_id)).status == "error"
        assert any(
            getattr(r, "run_id", None) == run_id and getattr(r, "code", None)
            for r in caplog.records
        )
    assert len([w for w in caught if w.category is FutureWarning]) == (policy is None)
    if policy is None:
        assert caught[0].filename == __file__


async def test_warning_only_with_stale_rows_and_no_once_flag(store, engine):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await scan(store)
        assert not caught
        for _ in range(2):
            await candidate(store, engine)
            await scan(store)
        assert len(caught) == 2


@pytest.mark.parametrize("policy", ["error", "reclaim"])
async def test_public_sweep_preserves_candidate_cap_warning(store, engine, caplog, policy):
    await candidate(store, engine)
    await scan(store, on_stale=policy, max_runs=1)
    warning = next(
        (r for r in caplog.records if getattr(r, "code", None) == "sweep_cap_reached"), None
    )
    assert warning is not None
    assert warning.category == "running" and warning.cap == warning.inspected == 1


async def test_poisoned_row_does_not_hide_good_row(store, engine):
    bad = await candidate(store, engine, marker=None)
    good = await candidate(store, engine)
    result = await scan(store, on_stale="reclaim")
    assert [r.run_id for r in result.stale_running] == [bad]
    assert [r.run_id for r in result.reclaimable] == [good]


async def test_limit_and_rediscovery(store, engine):
    run_id = await candidate(store, engine)
    first = await scan(store, on_stale="reclaim", max_reclaims=2)
    again = await scan(store, on_stale="reclaim", max_reclaims=10)
    assert first.reclaimable == again.reclaimable
    assert again.reclaimable[0].reclaim_limit == 2
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(reclaim_attempts=2))
    last = await scan(store, on_stale="reclaim")
    assert last.stale_running[0].failure_reason == "reclaim_limit"
    assert not last.reclaimable


@pytest.mark.parametrize("policy", ["error", "reclaim"])
async def test_cursor_counts_fresh_rows_and_never_wraps(store, engine, policy):
    ids = [await candidate(store, engine) for _ in range(7)]
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_(ids[:2]))
            .values(last_progress_at=datetime.now(UTC))
        )
    seen, cursor = [], None
    for _ in range(10):
        page = await scan(store, on_stale=policy, max_runs=3, max_returned=2, cursor=cursor)
        seen.extend(r.run_id for r in page.reclaimable + page.stale_running)
        cursor = page.cursor
        if cursor is None:
            break
    assert seen == ids[2:]
    assert cursor is None
    end = await scan(store, on_stale=policy, cursor=ids[-1])
    assert not end.reclaimable and not end.stale_running and end.cursor is None


async def test_two_public_sweepers_mark_once(engine, store):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent connections require PostgreSQL")
    run_id = await candidate(store, engine)
    barrier = asyncio.Barrier(2)

    async def actor():
        await barrier.wait()
        return await scan(SQLAlchemyStateStore(engine), on_stale="reclaim")

    results = await asyncio.gather(actor(), actor())
    assert all(r.reclaimable[0].run_id == run_id for r in results)
    assert (
        len([e for e in await store.get_run_events(run_id) if e.event_type == "run.lease_expired"])
        == 1
    )


@pytest.mark.parametrize("policy", ["error", "reclaim"])
async def test_renewal_between_scan_and_finalise_wins(store, engine, policy, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    run_id = await candidate(store, engine, marker=None)
    original = AsyncSession.execute
    renewed = False

    async def execute(session, statement, *args, **kwargs):
        nonlocal renewed
        if statement.is_update and not renewed:
            renewed = True
            async with engine.begin() as conn:
                await conn.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .values(last_progress_at=datetime.now(UTC))
                )
        return await original(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", execute)
    result = await scan(store, on_stale=policy)
    assert renewed and not result.stale_running and not result.reclaimable
    assert (await store.get_run(run_id)).status == "running"


async def test_native_scan_database_failure_is_safe(store, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.errors import PersistenceError

    async def fail(*args, **kwargs):
        raise OperationalError("private-sentinel", {}, Exception("private-sentinel"))

    monkeypatch.setattr(AsyncSession, "execute", fail)
    with pytest.raises(PersistenceError) as caught:
        await scan(store, on_stale="reclaim")
    assert "private-sentinel" not in str(caught.value)
