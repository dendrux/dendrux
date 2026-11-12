"""Transaction failure regressions shared with the native MySQL-family matrix."""

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.runtime.durability import retry_transient_db
from dendrux.types import generate_ulid
from tests._helpers.db_errors import LostConnectionError
from tests._helpers.gates import wait_for_gate


@pytest.fixture
def backend_engine(engine):
    return engine


async def test_readback_lock_timeout_is_unknown(store, monkeypatch):
    from dendrux.errors import StoreOutcomeUnknown

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    execute = AsyncSession.execute
    failed = False
    calls = 0

    async def lose_ack(session):
        nonlocal failed, calls
        calls += 1
        failed = True
        raise OperationalError(None, None, LostConnectionError())

    async def timeout_readback(session, statement, *args, **kwargs):
        if failed:
            raise OperationalError(None, None, Exception(1205, "PRIVATE_SENTINEL"))
        return await execute(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    monkeypatch.setattr(AsyncSession, "execute", timeout_readback)
    with pytest.raises(StoreOutcomeUnknown):
        await store.finalize_run(run_id, status="success", claim_epoch=0)
    assert calls == 1


async def test_inflight_commit_readback_waits_for_writer(backend_engine, store, monkeypatch):
    import asyncio

    from sqlalchemy import update

    from dendrux.db.models import AgentRun

    if not store.supports_row_locks:
        pytest.skip("independent row-lock transactions require a server database")
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    commit = AsyncSession.commit
    held = AsyncSession(backend_engine)
    pending = asyncio.Event()
    calls = 0

    async def delayed(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            expected = session.info["dendrux.commit_proofs"][0][2]
            await session.rollback()
            await held.execute(update(AgentRun).where(AgentRun.id == run_id).values(**expected))
            pending.set()
            raise OperationalError(None, None, LostConnectionError())
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", delayed)
    task = asyncio.create_task(
        store.finalize_run(
            run_id, status="success", expected_current_status="running", claim_epoch=0
        )
    )
    try:
        await wait_for_gate(pending, task=task)
        await asyncio.sleep(0.15)
        assert not task.done()
        await commit(held)
        assert await asyncio.wait_for(task, 5) is True
        assert calls == 1
    finally:
        await held.close()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("operation", ["create", "keyed_create", "meta"])
async def test_flush_failure_retries_whole_transaction(
    backend_engine, store, monkeypatch, operation
):
    run_id = generate_ulid()
    if operation == "meta":
        await store.create_run(run_id, "agent", meta={"before": True})
    cuts = 0
    commits = 0

    def cut(conn, cursor, sql, params, context, many):
        nonlocal cuts
        target = "UPDATE agent_runs SET meta=" if operation == "meta" else "INSERT INTO agent_runs"
        if cuts == 0 and sql.startswith(target):
            cuts += 1
            raise OperationalError(sql, params, LostConnectionError(), connection_invalidated=True)

    def committed(conn):
        nonlocal commits
        commits += 1

    event.listen(backend_engine.sync_engine, "before_cursor_execute", cut)
    event.listen(backend_engine.sync_engine, "commit", committed)
    try:
        if operation == "meta":
            await store.update_run_meta(run_id, {"after": True})
        else:
            result = await store.create_run(
                run_id, "agent", idempotency_key="key" if operation == "keyed_create" else None
            )
            assert result.outcome == "created"
        assert cuts == 1
        assert commits == 1
        row = await store.get_run(run_id)
        assert row is not None
        if operation == "meta":
            assert row.meta == {"before": True, "after": True}
    finally:
        event.remove(backend_engine.sync_engine, "before_cursor_execute", cut)
        event.remove(backend_engine.sync_engine, "commit", committed)


@pytest.mark.parametrize("operation", ["create", "keyed_create", "trace"])
@pytest.mark.parametrize("landed", [False, True])
async def test_commit_identity_resolves_or_retries(store, monkeypatch, operation, landed):
    run_id, row_id = generate_ulid(), generate_ulid()
    if operation == "trace":
        await store.create_run(run_id, "agent")
    commit = AsyncSession.commit
    calls = 0

    async def lose_ack(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            if landed:
                await commit(session)
            raise OperationalError(None, None, LostConnectionError(), connection_invalidated=True)
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    if operation == "trace":
        result = await retry_transient_db(
            lambda: store.save_trace(run_id, "assistant", "ok", order_index=0, trace_id=row_id),
            label="trace",
            run_id=run_id,
        )
        assert result == row_id
        assert (await store.get_trace(run_id, row_id)).content == "ok"
    else:
        result = await store.create_run(
            run_id,
            "agent",
            input_data={"input": "same"},
            tenant_id="tenant",
            idempotency_key="key" if operation == "keyed_create" else None,
        )
        assert result.run_id == run_id and result.outcome == "created"
    assert calls == (1 if landed else 2)


async def test_keyed_create_normalizes_none_fingerprint(store):
    first = await store.create_run(generate_ulid(), "agent", idempotency_key="key")
    second = await store.create_run(generate_ulid(), "agent", idempotency_key="key")
    assert first.run_id == second.run_id


async def test_keyed_create_normalizes_legacy_none_fingerprint(store, backend_engine):
    from sqlalchemy import update

    from dendrux.db.models import AgentRun

    first = await store.create_run(generate_ulid(), "agent", idempotency_key="key")
    async with backend_engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == first.run_id).values(idempotency_fingerprint=None)
        )
    second = await store.create_run(generate_ulid(), "agent", idempotency_key="key")
    assert first.run_id == second.run_id


async def test_precommit_heartbeat_retries(store, monkeypatch):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    execute = AsyncSession.execute
    calls = 0

    async def cut(session, statement, *args, **kwargs):
        nonlocal calls
        if str(statement).startswith("UPDATE agent_runs"):
            calls += 1
            if calls == 1:
                raise OperationalError(None, None, LostConnectionError())
        return await execute(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", cut)
    assert await store.touch_progress(run_id, claim_epoch=0)
    assert calls == 2


@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize(
    "mismatch", [None, "agent_name", "tenant_id", "input_data", "idempotency_fingerprint"]
)
async def test_inflight_create_reproves_identity(
    backend_engine, store, monkeypatch, keyed, mismatch
):
    import asyncio

    from sqlalchemy import insert

    from dendrux.db.models import AgentRun
    from dendrux.errors import StoreOutcomeUnknown

    if not store.supports_row_locks:
        pytest.skip("server transaction visibility")
    run_id = generate_ulid()
    commit = AsyncSession.commit
    held = AsyncSession(backend_engine)
    pending = None
    calls = 0

    async def publish():
        await asyncio.sleep(0.3)
        await commit(held)

    async def delayed(session):
        nonlocal calls, pending
        calls += 1
        if calls == 1:
            conn = await session.connection()
            table = AgentRun.__table__
            row = dict(
                (await conn.execute(table.select().where(table.c.id == run_id))).mappings().one()
            )
            await session.rollback()
            if mismatch is not None:
                row[mismatch] = {"different": True} if mismatch == "input_data" else "different"
            await held.execute(insert(table).values(**row))
            pending = asyncio.create_task(publish())
            raise OperationalError(None, None, LostConnectionError(), connection_invalidated=True)
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", delayed)
    try:
        write = store.create_run(
            run_id,
            "agent",
            tenant_id="tenant",
            input_data={"i": 1},
            idempotency_key="key" if keyed else None,
        )
        if mismatch:
            with pytest.raises(StoreOutcomeUnknown):
                await write
        else:
            assert (await write).outcome == "created"
        assert await store.get_run(run_id) is not None
    finally:
        if pending is not None:
            await pending
        await held.close()


async def test_postgres_readback_lock_wait_is_bounded(backend_engine, store, monkeypatch):
    import asyncio

    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.errors import StoreOutcomeUnknown
    from dendrux.runtime.stores.sql import base

    if backend_engine.dialect.name != "postgresql":
        pytest.skip("Postgres local lock timeout")
    monkeypatch.setattr(base, "_COMMIT_READBACK_LOCK_TIMEOUT_MS", 50, raising=False)
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    held = AsyncSession(backend_engine)

    async def lose_ack(session):
        await session.rollback()
        await held.execute(update(AgentRun).where(AgentRun.id == run_id).values(status="success"))
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    try:
        with pytest.raises(StoreOutcomeUnknown):
            await asyncio.wait_for(store.finalize_run(run_id, status="success", claim_epoch=0), 10)
    finally:
        await held.rollback()
        await held.close()


async def test_locked_validation_callback_preserves_attribute_error(backend_engine, store):
    from dendrux.db.models import AgentRun

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    error = AttributeError("caller validation bug")

    def validate(row):
        raise error

    async with store._session_factory() as session:
        with pytest.raises(AttributeError) as caught:
            await store.update_one(
                session,
                table=AgentRun.__table__,
                pk=(run_id,),
                predicate=AgentRun.status == "running",
                values=[(AgentRun.status, "success")],
                returning=[AgentRun.id],
                validate_locked=validate,
            )
        assert caught.value is error
    assert (await store.get_run(run_id)).status == "running"
