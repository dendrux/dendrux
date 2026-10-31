"""Exercise the portable helpers against the production backend matrix."""

import asyncio

import pytest
from sqlalchemy import and_, select

from dendrux.db.models import AgentRun


async def test_lost_claim_ack_never_proves_ownership(store, monkeypatch, caplog):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.errors import PersistenceError
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={})
    from dendrux.runtime.state import EventWrite, _write_event

    write = store._write_tx

    async def with_evidence(session, **kwargs):
        await _write_event(
            session, run_id, EventWrite(generate_ulid(), "ownership.proof", 0, {}, None, None)
        )
        assert session.info["dendrux.fresh_evidence"]
        await write(session, **kwargs)

    monkeypatch.setattr(store, "_write_tx", with_evidence)
    commit = AsyncSession.commit
    calls = 0

    class LostConnectionError(Exception):
        sqlstate = "08006"

    async def lose_ack(session):
        nonlocal calls
        calls += 1
        await commit(session)
        raise OperationalError(None, None, LostConnectionError("PRIVATE-DRIVER-SENTINEL"))

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    with pytest.raises(PersistenceError) as caught:
        await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    assert type(caught.value).__name__ == "StoreOutcomeUnknown"
    assert "PRIVATE-DRIVER-SENTINEL" not in str(caught.value)
    assert "PRIVATE-DRIVER-SENTINEL" not in caplog.text
    assert any(
        getattr(record, "code", None) == "store_outcome_unknown" for record in caplog.records
    )
    assert calls == 1


async def test_update_one_reads_epoch_by_immutable_key(store):
    assert callable(getattr(store, "update_one", None)), "portable conditional update is required"
    store.supports_update_returning = False
    await store.create_run("generic-one", "agent")
    async with store._session_factory() as session:
        row = await store.update_one(
            session,
            table=AgentRun.__table__,
            pk=("generic-one",),
            predicate=AgentRun.claim_epoch == 0,
            values=[(AgentRun.__table__.c.claim_epoch, AgentRun.claim_epoch + 1)],
            returning=[AgentRun.claim_epoch, AgentRun.storage_format],
        )
        assert row["claim_epoch"] == 1
        await session.commit()


async def test_update_many_exact_bounded_set(store):
    assert callable(getattr(store, "update_many", None)), "portable batch update is required"
    store.supports_update_returning = False
    for i in range(5):
        await store.create_run(f"batch-{i}", "agent")
    async with store._session_factory() as session:
        rows = await store.update_many(
            session,
            table=AgentRun.__table__,
            predicate=AgentRun.claim_epoch == 0,
            values=[(AgentRun.__table__.c.claim_epoch, AgentRun.claim_epoch + 1)],
            returning=[AgentRun.id, AgentRun.claim_epoch],
            page_size=3,
        )
        assert {row["id"] for row in rows} == {"batch-0", "batch-1", "batch-2"}
        assert all(row["claim_epoch"] == 1 for row in rows)
        untouched = await session.scalars(select(AgentRun.id).where(AgentRun.claim_epoch == 0))
        assert set(untouched) == {"batch-3", "batch-4"}
        await session.commit()


@pytest.mark.parametrize("many", [False, True])
async def test_generic_update_twenty_contenders(store, engine, many):
    assert callable(getattr(store, "update_one", None)), "portable conditional update is required"
    if engine.dialect.name != "postgresql":
        pytest.skip("row-lock contention requires Postgres")
    store.supports_update_returning = False
    await store.create_run("twenty", "agent")

    async def contend():
        async with store._session_factory() as session:
            options = dict(
                table=AgentRun.__table__,
                predicate=and_(AgentRun.claim_epoch == 0, AgentRun.lease_expired_at.is_(None)),
                values=[(AgentRun.__table__.c.claim_epoch, AgentRun.claim_epoch + 1)],
                returning=[AgentRun.id, AgentRun.claim_epoch],
            )
            if many:
                affected = await store.update_many(session, page_size=1, **options)
                row = affected[0] if affected else None
            else:
                row = await store.update_one(session, pk=("twenty",), **options)
            await session.commit()
            return row

    rows = await asyncio.wait_for(asyncio.gather(*(contend() for _ in range(20))), 30)
    winners = [row for row in rows if row is not None]
    assert len(winners) == 1
    assert winners[0]["claim_epoch"] == 1


async def test_generic_reclaim_reuses_validation_lock(store, engine):
    from sqlalchemy import event

    from tests.integration.test_060_core_recovery import mark, reclaim, stale

    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    store.supports_update_returning = False
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        claim = await reclaim(store, item)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert claim.claim_epoch == 2
    assert len(statements) == (4 if store.supports_row_locks else 5), (
        "validation lock/reservation + UPDATE + immutable-PK readback + audit INSERT"
    )


async def test_generic_reclaim_twenty_contenders(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("real row-lock contention requires Postgres")
    from dendrux.errors import RunNotReclaimableError
    from tests.integration.test_060_core_recovery import mark, reclaim, stale

    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    store.supports_update_returning = False
    results = await asyncio.wait_for(
        asyncio.gather(*(reclaim(store, item) for _ in range(20)), return_exceptions=True), 30
    )
    winners = [result for result in results if not isinstance(result, BaseException)]
    assert len(winners) == 1 and winners[0].claim_epoch == item.claim_epoch + 1
    assert sum(isinstance(result, RunNotReclaimableError) for result in results) == 19
    record = await store.get_run(item.run_id)
    assert record.claim_epoch == winners[0].claim_epoch and record.reclaim_attempts == 1


async def test_plain_insert_twenty_contenders(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("plain INSERT conflict handling is certified on Postgres")
    from dendrux.errors import PersistenceError
    from dendrux.runtime.stores.sql.base import SQLStateStore
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore

    class PortablePostgres(PostgresStateStore):
        supports_insert_returning = False
        supports_update_returning = False
        insert_if_absent_stmt = SQLStateStore.insert_if_absent_stmt

    store = PortablePostgres(engine)
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")

    async def contend(i):
        try:
            return await store.save_run_event(
                run_id, event_type="one", data={"caller": i}, event_id="same-event"
            )
        except PersistenceError as exc:
            assert exc.code == "write_conflict"
            return None

    rows = await asyncio.wait_for(asyncio.gather(*(contend(i) for i in range(20))), 30)
    assert [row for row in rows if row is not None] == [0]
    events = await store.get_run_events(run_id)
    assert len(events) == 1 and events[0].sequence_index == 0


@pytest.mark.parametrize(
    "path,value,kind,expected",
    [
        (("a.b",), True, "bool", True),
        (("a.b",), True, "str", None),
        (("a.b",), "true", "bool", None),
        (("a.b",), "true", "str", "true"),
        (("a.b",), None, "json", None),
        (("a.b",), 12, "int", 12),
        (("a.b",), "12", "int", None),
        (("missing",), 12, "str", None),
        (("arr", -1), "unused", "str", "last"),
    ],
)
async def test_json_hook_truth_table(store, path, value, kind, expected):
    await store.create_run("json-hooks", "agent", meta={"a.b": value, "arr": ["first", "last"]})
    async with store._session_factory() as session:
        actual = await session.scalar(
            select(store.json_get(AgentRun.meta, path, kind)).where(AgentRun.id == "json-hooks")
        )
    assert actual == expected


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("null", None, True),
        ("missing", None, False),
        ("bool", True, True),
        ("bool", "true", False),
        ("string", "true", True),
        ("string", True, False),
    ],
)
async def test_json_contains_distinguishes_types(store, field, value, expected):
    await store.create_run("contains", "agent", meta={"null": None, "bool": True, "string": "true"})
    async with store._session_factory() as session:
        actual = await session.scalar(
            select(store.json_contains(AgentRun.meta, (field,), value)).where(
                AgentRun.id == "contains"
            )
        )
    assert actual is expected


@pytest.mark.parametrize("field,expected", [("null", True), ("missing", False), ("bool", False)])
async def test_json_null_is_explicit(store, field, expected):
    await store.create_run("explicit-null", "agent", meta={"null": None, "bool": True})
    async with store._session_factory() as session:
        actual = await session.scalar(
            select(store.json_is_null(AgentRun.meta, (field,))).where(
                AgentRun.id == "explicit-null"
            )
        )
    assert actual is expected


async def test_update_validation_and_composite_key_bounds(store):
    from dendrux.db.models import Blob

    async with store._session_factory() as session:
        with pytest.raises(ValueError, match="complete"):
            await store.update_one(
                session,
                table=Blob.__table__,
                pk=("only-one-component",),
                predicate=AgentRun.id == "x",
                values=[],
                returning=[Blob.generation],
            )
        for page_size in (0, True, 1001):
            with pytest.raises(ValueError):
                await store.update_many(
                    session,
                    table=AgentRun.__table__,
                    predicate=AgentRun.id == "x",
                    values=[],
                    returning=[AgentRun.id],
                    page_size=page_size,
                )


async def test_first_connection_facts_resolve_once(store, monkeypatch):
    from dendrux.runtime.stores.sql._types import DialectFacts

    calls = 0

    async def checks(conn):
        nonlocal calls
        calls += 1
        return DialectFacts(conn.dialect.name, features={"supports_insert_returning": False})

    monkeypatch.setattr(store, "connection_checks", checks)
    await store.create_run("facts", "agent")
    await store.get_run("facts")
    assert calls == 1
    assert store.supports_insert_returning is False


async def test_generic_blob_touch_keeps_complete_projection(engine):
    from tests.integration.test_blob_lifecycle_faults import catalog, live_external

    _, services, sha, _ = await live_external(engine, 731)
    store = catalog(services)
    before = await store.get_row("-", sha)
    store.supports_update_returning = False
    result = await store.touch_live("-", [sha, "0" * 64])
    assert set(result) == {sha}
    after = result[sha]
    for name in (
        "media_type",
        "byte_length",
        "width",
        "height",
        "frame_count",
        "storage",
        "state",
        "generation",
        "created_at",
        "purged_at",
        "purge_reason",
        "ingest_transform",
        "state_changed_at",
    ):
        assert getattr(after, name) == getattr(before, name)
    assert after.last_used_at >= before.last_used_at


@pytest.mark.parametrize("atomic", [False, True])
async def test_create_and_pause_finalize_lost_ack_are_proven(store, monkeypatch, atomic):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.types import generate_ulid
    from tests._helpers.db_errors import LostConnectionError

    commit = AsyncSession.commit
    calls = 0

    async def lose_ack(session):
        nonlocal calls
        calls += 1
        await commit(session)
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    run_id = generate_ulid()
    created = await store.create_run(
        run_id, "agent", idempotency_key="lost-key", idempotency_fingerprint="digest"
    )
    assert created.run_id == run_id
    if atomic:
        assert (
            await store.pause_with_event(
                run_id, status="waiting_client_tool", pause_data={"pause_id": "pause"}
            )
            == 0
        )
        assert (
            await store.finalize_with_event(
                run_id,
                status="success",
                allowed_current_statuses=["waiting_client_tool"],
                event_type="run.completed",
                answer="done",
            )
            == 1
        )
    else:
        from dendrux.runtime.durability import StoreOutcomeUnknown

        with pytest.raises(StoreOutcomeUnknown):
            await store.pause_run(
                run_id, status="waiting_client_tool", pause_data={"pause_id": "pause"}
            )
        with pytest.raises(StoreOutcomeUnknown):
            await store.finalize_run(run_id, status="success", answer="done")
    assert calls == 3
    assert (await store.get_run(run_id)).status == "success"


@pytest.mark.parametrize("operation", ["finalize_run", "pause_with_event"])
async def test_uncommitted_transition_retries_without_returning_a_cas_miss(
    store, monkeypatch, operation
):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.types import generate_ulid
    from tests._helpers.db_errors import LostConnectionError

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")

    from dendrux.runtime import durability

    commit = AsyncSession.commit
    calls = 0

    async def lose_before_commit(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OperationalError(None, None, LostConnectionError())
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", lose_before_commit)
    monkeypatch.setattr(durability, "_BASE_BACKOFF_S", 0)
    if operation == "finalize_run":
        assert await store.finalize_run(run_id, status="success") is True
        expected = "success"
    else:
        assert (
            await store.pause_with_event(
                run_id, status="waiting_client_tool", pause_data={"pause_id": "this-call"}
            )
            == 0
        )
        expected = "waiting_client_tool"
    assert calls == 2
    assert (await store.get_run(run_id)).status == expected


async def test_partial_sweep_records_each_committed_run(store, engine, monkeypatch):
    from tests.integration.test_060_core_recovery import mark, stale

    run_ids = set(await stale(store, engine, count=2))
    committed = []
    write = store._write_tx

    async def observe(session, **kwargs):
        await write(session, **kwargs)
        committed.append(set(session.info["dendrux.touched_runs"]))

    monkeypatch.setattr(store, "_write_tx", observe)
    await mark(store)
    assert len(committed) == 2 and all(len(ids) == 1 for ids in committed)
    assert set.union(*committed) == run_ids


async def test_consumed_evidence_proves_ack_after_later_progress(store, monkeypatch):
    from sqlalchemy import update
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.runtime.state import LoopStatePatch, TraceWrite
    from dendrux.types import generate_ulid
    from tests._helpers.db_errors import LostConnectionError

    run_id, submission_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "submission": {"submission_id": submission_id}},
        pii_mapping=None,
        trace=None,
    )
    commit = AsyncSession.commit

    async def lose_after_later_progress(session):
        await commit(session)
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(loop_state={"v": 1, "phase": "later"})
        )
        await commit(session)
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", lose_after_later_progress)
    assert (
        await store.consume_submission(
            run_id,
            claim_epoch=0,
            submission_id=submission_id,
            traces=[TraceWrite(generate_ulid(), "tool", "done", 1)],
            patch=LoopStatePatch(set_phase="model_next"),
        )
        is True
    )
    assert (await store.get_loop_state(run_id))["phase"] == "later"


@pytest.mark.parametrize("operation", ["trace", "tool", "event"])
async def test_append_lost_ack_reads_its_retry_identity(store, monkeypatch, operation):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.types import generate_ulid
    from tests._helpers.db_errors import LostConnectionError

    run_id, row_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    commit = AsyncSession.commit
    calls = 0

    async def lose_ack(session):
        nonlocal calls
        calls += 1
        await commit(session)
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    if operation == "trace":
        assert (
            await store.save_trace(run_id, "assistant", "ok", order_index=0, trace_id=row_id)
            == row_id
        )
        assert (await store.get_trace(run_id, row_id)).content == "ok"
    elif operation == "tool":
        assert (
            await store.save_tool_call(
                run_id,
                tool_call_id="call",
                provider_tool_call_id=None,
                tool_name="tool",
                target="server",
                params={},
                result_payload='"ok"',
                success=True,
                duration_ms=1,
                iteration_index=1,
                row_id=row_id,
            )
            == row_id
        )
        assert len(await store.get_tool_calls(run_id)) == 1
    else:
        assert (
            await store.save_run_event(
                run_id, event_type="tool.completed", data={"ok": True}, event_id=row_id
            )
            == 0
        )
        assert len(await store.get_run_events(run_id)) == 1
    assert calls == 1


async def test_generic_claim_twenty_contenders(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("real row-lock contention requires Postgres")
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    await store.pause_run(
        run_id, status="waiting_client_tool", pause_data={"pause_id": "same-generation"}
    )
    store.supports_update_returning = False
    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                store.claim_paused_run(
                    run_id,
                    expected_status="waiting_client_tool",
                    expected_pause_id="same-generation",
                )
                for _ in range(20)
            )
        ),
        30,
    )
    winners = [result for result in results if result is not False]
    assert len(winners) == 1 and winners[0].claim_epoch == 1
    assert (await store.get_run(run_id)).status == "running"


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
@pytest.mark.parametrize("operation", ["one", "many", "reclaim"])
async def test_file_sqlite_reserves_writer_before_validation(engine, store, monkeypatch, operation):
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.sql import Select, Update

    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.types import generate_ulid
    from tests.integration.test_060_core_recovery import mark, reclaim, stale

    item = None
    if operation == "reclaim":
        await stale(store, engine)
        item = (await mark(store)).reclaimable[0]
        run_id = item.run_id
    else:
        run_id = generate_ulid()
        await store.create_run(run_id, "agent")
    pooled = create_async_engine(engine.url, pool_size=20, max_overflow=0)
    contender = store_for_engine(pooled)
    if operation == "one":
        contender.supports_update_returning = False
    original = AsyncSession.execute
    barrier = asyncio.Barrier(20)
    entered = set()
    writers = set()

    async def synchronize_unlocked_reads(session, statement, *args, **kwargs):
        actor = asyncio.current_task()
        if isinstance(statement, Update):
            writers.add(actor)
        result = await original(session, statement, *args, **kwargs)
        if (
            isinstance(statement, Select)
            and "agent_runs" in str(statement)
            and actor not in entered
            and actor not in writers
        ):
            entered.add(actor)
            await asyncio.wait_for(barrier.wait(), 15)
        return result

    monkeypatch.setattr(AsyncSession, "execute", synchronize_unlocked_reads)

    async def contend():
        if operation == "reclaim":
            return await reclaim(contender, item)
        async with contender._session_factory() as session:
            options = dict(
                table=AgentRun.__table__,
                predicate=(AgentRun.id == run_id) & (AgentRun.claim_epoch == 0),
                values=[(AgentRun.__table__.c.claim_epoch, AgentRun.claim_epoch + 1)],
                returning=[AgentRun.claim_epoch],
            )
            if operation == "one":
                result = await contender.update_one(session, pk=(run_id,), **options)
            else:
                rows = await contender.update_many(session, page_size=1, **options)
                result = rows[0] if rows else None
            await session.commit()
            return result

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(contend() for _ in range(20)), return_exceptions=True), 30
        )
        winners = [
            result
            for result in results
            if result is not None and result is not False and not isinstance(result, BaseException)
        ]
        assert len(winners) == 1
    finally:
        await pooled.dispose()
