"""Twenty independent callers exercise native InnoDB serialization."""

# ruff: noqa: F401, F811
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select, text, update

from dendrux.db.models import AgentRun, RunEvent, RunInbox
from dendrux.types import generate_ulid
from tests.integration._mysql import mysql_engine, store
from tests.integration.test_060_core_recovery import mark, reclaim, stale


async def warm_connections(mysql_engine, count):
    checked_out = await asyncio.gather(*(mysql_engine.connect() for _ in range(count)))
    await asyncio.gather(*(connection.close() for connection in checked_out))


@pytest_asyncio.fixture(autouse=True)
async def prewarm_race_pool(store, mysql_engine):
    await store.get_run("absent")
    await warm_connections(mysql_engine, 20)


async def contenders(operation):
    barrier = asyncio.Barrier(20)

    async def call(index):
        await asyncio.wait_for(barrier.wait(), 10)
        return await operation(index)

    return await asyncio.wait_for(
        asyncio.gather(*(call(i) for i in range(20)), return_exceptions=True), 45
    )


async def test_claim_twenty(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    await store.pause_run(
        run_id, status="waiting_client_tool", pause_data={"pause_id": "generation"}
    )
    results = await contenders(
        lambda _: store.claim_paused_run(
            run_id, expected_status="waiting_client_tool", expected_pause_id="generation"
        )
    )
    assert not any(isinstance(r, BaseException) for r in results), [
        type(r).__name__ for r in results
    ]
    winners = [r for r in results if r is not False]
    assert len(winners) == 1
    row = await store.get_run(run_id)
    assert row.claim_epoch == winners[0].claim_epoch == 1
    assert row.status == "running"


async def test_reclaim_twenty(store, mysql_engine):
    from dendrux.errors import RunNotReclaimableError

    await stale(store, mysql_engine)
    item = (await mark(store)).reclaimable[0]
    results = await contenders(lambda _: reclaim(store, item))
    winners = [r for r in results if not isinstance(r, BaseException)]
    assert len(winners) == 1
    assert sum(isinstance(r, RunNotReclaimableError) for r in results) == 19
    row = await store.get_run(item.run_id)
    assert row.claim_epoch == winners[0].claim_epoch == item.claim_epoch + 1
    assert row.reclaim_attempts == 1
    assert [e.event_type for e in await store.get_run_events(item.run_id)] == [
        "run.lease_expired",
        "run.reclaimed",
    ]


@pytest.mark.parametrize("status", ["success", "cancelled"])
async def test_finalize_cancel_twenty(store, status):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    results = await contenders(
        lambda _: store.finalize_with_event(
            run_id, status=status, allowed_current_statuses=["running"], event_type=f"run.{status}"
        )
    )
    assert not any(isinstance(r, BaseException) for r in results), [
        type(r).__name__ for r in results
    ]
    assert sum(r is not None for r in results) == 1
    assert (await store.get_run(run_id)).status == status
    assert len(await store.get_run_events(run_id)) == 1


async def test_note_cap_twenty(store):
    from dendrux.errors import NoteLimitExceededError

    run_id = generate_ulid()
    await store.create_run(run_id, "agent", tenant_id="tenant")
    results = await contenders(
        lambda _: store.add_note(
            run_id, note_id=generate_ulid(), text="bounded", tenant_id="tenant", max_notes=1
        )
    )
    assert sum(r is True for r in results) == 1
    assert sum(isinstance(r, NoteLimitExceededError) for r in results) == 19
    async with store._session_factory() as session:
        assert (
            await session.scalar(
                select(func.count()).select_from(RunInbox).where(RunInbox.run_id == run_id)
            )
            == 1
        )


@pytest.mark.parametrize("same_identity", [False, True])
async def test_event_append_twenty(store, same_identity):
    from dendrux.errors import PersistenceError

    run_id, row_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    results = await contenders(
        lambda i: store.save_run_event(
            run_id,
            event_type="append",
            data={"caller": i},
            event_id=row_id if same_identity else generate_ulid(),
        )
    )
    events = await store.get_run_events(run_id)
    if same_identity:
        assert sum(isinstance(r, int) for r in results) == 1
        assert (
            sum(isinstance(r, PersistenceError) and r.code == "write_conflict" for r in results)
            == 19
        )
        assert len(events) == 1
    else:
        assert sorted(results) == list(range(20))
        assert [e.sequence_index for e in events] == list(range(20))
        assert {row.data["caller"]: row.sequence_index for row in events} == dict(
            enumerate(results)
        )
    async with store._session_factory() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .outerjoin(AgentRun, AgentRun.id == RunEvent.agent_run_id)
                .where(AgentRun.id.is_(None))
            )
            == 0
        )


@pytest.mark.parametrize("operation", ["mark", "sweep"])
async def test_order_cancel_requested_stale_run(store, mysql_engine, operation):
    ids = await stale(store, mysql_engine)
    async with mysql_engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_(ids))
            .values(
                cancel_requested=True,
                status_changed_at=datetime.now(UTC) - timedelta(minutes=10),
            )
        )
    if operation == "mark":
        await mark(store)
    else:
        await store.sweep_stale_runs(timedelta(minutes=1))
    row = await store.get_run(ids[0])
    assert row.status == "cancelled"
    assert row.cancel_requested is False
    assert row.status_changed_at > datetime.now(UTC) - timedelta(minutes=1)


async def test_duplicate_insert_keeps_transaction_without_savepoint(store, mysql_engine):
    run_id, event_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    await store.save_run_event(run_id, event_type="one", event_id=event_id)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await store.save_run_event(run_id, event_type="one", event_id=event_id) == 0
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    assert not any("SAVEPOINT" in s.upper() for s in statements)
    assert len(await store.get_run_events(run_id)) == 1


@pytest.mark.parametrize("operation,target", [("claim", 2), ("reclaim", 3)])
async def test_named_increment_statement_budget(store, mysql_engine, operation, target):
    if operation == "claim":
        run_id = generate_ulid()
        await store.create_run(run_id, "agent")
        await store.pause_run(
            run_id, status="waiting_client_tool", pause_data={"pause_id": "generation"}
        )

        async def action():
            return await store.claim_paused_run(
                run_id, expected_status="waiting_client_tool", expected_pause_id="generation"
            )
    else:
        await stale(store, mysql_engine)
        item = (await mark(store)).reclaimable[0]
        run_id = item.run_id

        async def action():
            return await reclaim(store, item)

    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await action()
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == target
    assert result.claim_epoch == (await store.get_run(run_id)).claim_epoch


async def test_lock_timeout_rolls_back_whole_attempt(store, mysql_engine, monkeypatch):
    from dendrux.runtime import durability
    from dendrux.runtime.stores._errors import classify_error

    run_id, other_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    await store.create_run(other_id, "agent")
    monkeypatch.setattr(durability, "_BASE_BACKOFF_S", 0)
    blocker = await mysql_engine.connect()
    transaction = await blocker.begin()
    await blocker.execute(select(AgentRun.id).where(AgentRun.id == run_id).with_for_update())
    attempts = 0
    try:

        async def attempt():
            nonlocal attempts
            attempts += 1
            async with store._session_factory() as session:
                await session.execute(text("SET SESSION innodb_lock_wait_timeout = 1"))
                await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == other_id)
                    .values(iteration_count=AgentRun.iteration_count + 1)
                )
                try:
                    await store._fence(session, run_id, 0)
                except Exception as exc:
                    assert classify_error(exc).code == 1205
                    await transaction.rollback()
                    raise
                await store._write_tx(session)

        await durability.retry_transient_db(
            attempt, label="timeout_test", run_id=run_id, classify=store.classify_db_error
        )
        assert attempts == 2
        assert (await store.get_run(other_id)).iteration_count == 1
    finally:
        if transaction.is_active:
            await transaction.rollback()
        await blocker.close()


async def test_native_deadlock_classification(store):
    from dendrux.runtime.stores._errors import classify_error

    ids = [generate_ulid(), generate_ulid()]
    for run_id in ids:
        await store.create_run(run_id, "agent")
    barrier = asyncio.Barrier(2)

    async def lock(order):
        async with store._session_factory() as session:
            await session.execute(
                select(AgentRun.id).where(AgentRun.id == ids[order]).with_for_update()
            )
            await barrier.wait()
            try:
                await session.execute(
                    select(AgentRun.id).where(AgentRun.id == ids[1 - order]).with_for_update()
                )
            except Exception as exc:
                result = classify_error(exc)
                assert result.code == 1213 and result.kind == "deadlock_or_serialization"
                return "deadlock"
            return "winner"

    assert sorted(await asyncio.wait_for(asyncio.gather(lock(0), lock(1)), 15)) == [
        "deadlock",
        "winner",
    ]


@pytest.mark.parametrize("operation", ["claim", "reclaim"])
async def test_commit_loss_never_authorizes_claim(store, mysql_engine, operation, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.errors import StoreOutcomeUnknown

    if operation == "reclaim":
        await stale(store, mysql_engine)
        item = (await mark(store)).reclaimable[0]
        run_id = item.run_id
    else:
        run_id = generate_ulid()
        await store.create_run(run_id, "agent")
        await store.pause_run(run_id, status="waiting_client_tool", pause_data={})
    original = AsyncSession.commit
    commits = 0

    async def lose_ack(session):
        nonlocal commits
        commits += 1
        await original(session)
        raise OperationalError(None, None, Exception(2013, "PRIVATE_SENTINEL"))

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    with pytest.raises(StoreOutcomeUnknown) as caught:
        if operation == "reclaim":
            await reclaim(store, item)
        else:
            await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    assert "PRIVATE_SENTINEL" not in str(caught.value)
    assert commits == 1
    assert (await store.get_run(run_id)).status == "running"


async def test_create_idempotency_twenty(store):
    results = await contenders(
        lambda _: store.create_run(
            generate_ulid(),
            "agent",
            tenant_id="t ",
            idempotency_key="same",
            idempotency_fingerprint="digest",
        )
    )
    assert not any(isinstance(r, BaseException) for r in results), [
        type(r).__name__ for r in results
    ]
    assert sum(r.outcome == "created" for r in results) == 1
    assert len({r.run_id for r in results}) == 1


async def test_metadata_merge_twenty_no_lost_keys(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent", meta={"initial": True})
    results = await contenders(lambda i: store.update_run_meta(run_id, {str(i): i}))
    assert not any(isinstance(r, BaseException) for r in results), [
        type(r).__name__ for r in results
    ]
    assert (await store.get_run(run_id)).meta == {"initial": True, **{str(i): i for i in range(20)}}


async def test_two_independent_stores_share_only_database_coordination(store, mysql_engine):
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore

    run_id = generate_ulid()
    await store.create_run(run_id, "agent", tenant_id="t ", meta={"initial": True})
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={"pause_id": "p"})
    instances = [
        MySQLStateStore.from_url(
            mysql_engine.url.render_as_string(hide_password=False), pool_size=10, max_overflow=0
        )
        for _ in range(2)
    ]
    try:
        await asyncio.gather(*(instance.get_run(run_id) for instance in instances))
        await asyncio.gather(*(warm_connections(instance._engine, 10) for instance in instances))
        results = await contenders(
            lambda i: instances[i % 2].claim_paused_run(
                run_id, expected_status="waiting_client_tool", expected_pause_id="p"
            )
        )
        assert not any(isinstance(r, BaseException) for r in results)
        winners = [r for r in results if r is not False]
        assert len(winners) == 1
        assert winners[0].claim_epoch == (await store.get_run(run_id)).claim_epoch == 1
        merged = await contenders(lambda i: instances[i % 2].update_run_meta(run_id, {str(i): i}))
        assert not any(isinstance(r, BaseException) for r in merged)
        assert (await store.get_run(run_id)).meta == {
            "initial": True,
            **{str(i): i for i in range(20)},
        }
        for instance in instances:
            assert await instance.get_run(run_id, tenant_id="t") is None
    finally:
        await asyncio.gather(*(instance._engine.dispose() for instance in instances))
