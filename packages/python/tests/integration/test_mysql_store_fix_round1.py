"""Native regression proofs for the first independent review."""

# ruff: noqa: F401, F811
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, func, insert, select, text, update

from dendrux.db.models import AgentRun, RunEvent
from dendrux.types import generate_ulid
from tests.integration._mysql import mysql_engine, store


@pytest.mark.parametrize("initial_status", ["running", "waiting_client_tool"])
async def test_agent_cancels_1500_child_tree(store, mysql_engine, initial_status):
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM

    root = generate_ulid()
    await store.create_run(root, "agent", tenant_id="t")
    if initial_status != "running":
        await store.pause_run(root, status=initial_status, pause_data={"pause_id": "p"})
    # Running rows are cooperatively stopped; paused rows finalize immediately.
    old = datetime.now(UTC) - timedelta(minutes=10)
    children = [
        dict(
            id=generate_ulid(),
            agent_name="agent",
            tenant_id="t",
            parent_run_id=root,
            storage_format=2,
            status=initial_status,
            pause_data={"pause_id": "p"} if initial_status != "running" else None,
            last_progress_at=old,
        )
        for _ in range(1500)
    ]
    async with mysql_engine.begin() as conn:
        await conn.execute(insert(AgentRun), children)
        await conn.execute(update(AgentRun).where(AgentRun.id == root).values(last_progress_at=old))
    agent = Agent(prompt="p", provider=MockLLM([]), state_store=store)
    try:
        result = await agent.cancel_run(root, cascade=True, tenant_id="t")
        assert result.meta["cascade_count"] == 1501
        if initial_status == "running":
            async with store._session_factory() as session:
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(AgentRun)
                        .where(AgentRun.cancel_requested.is_(True))
                    )
                    == 1501
                )
            await store.sweep_stale_runs(timedelta(minutes=1))
        async with store._session_factory() as session:
            assert (
                await session.scalar(
                    select(func.count()).select_from(AgentRun).where(AgentRun.status == "cancelled")
                )
                == 1501
            )
    finally:
        await agent.close()


async def test_create_commit_proof_uses_generated_scope(store, mysql_engine, monkeypatch, caplog):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    caplog.set_level("ERROR", logger="asyncmy")
    commit = AsyncSession.commit
    calls = 0
    captured = []

    async def lose_ack(session):
        nonlocal calls
        calls += 1
        await commit(session)
        raise OperationalError(None, None, Exception(2013, "PRIVATE_SENTINEL"))

    def capture(conn, cursor, sql, params, context, many):
        if (
            sql.lstrip().startswith("SELECT")
            and "WHERE" in sql
            and "agent_runs.idempotency_key" in sql.split("WHERE", 1)[1]
        ):
            captured.append((sql, params))

    monkeypatch.setattr(AsyncSession, "commit", lose_ack)
    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        run_id = generate_ulid()
        assert (
            await store.create_run(
                run_id,
                "agent",
                tenant_id="t ",
                idempotency_key="proof",
                idempotency_fingerprint="digest",
            )
        ).run_id == run_id
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    assert calls == 1 and len(captured) >= 2
    assert all("agent_runs.tenant_scope" in sql.split("WHERE", 1)[1] for sql, _ in captured)
    async with mysql_engine.connect() as conn:
        for sql, params in captured:
            plan = [
                dict(row)
                for row in (await conn.exec_driver_sql("EXPLAIN " + sql, params)).mappings()
            ]
            print("PROOF_EXPLAIN", mysql_engine.dialect.is_mariadb, plan)
            assert any(row.get("key") == "uq_agent_runs_tenant_idempotency_key" for row in plan)


async def test_first_use_is_one_select_and_ignores_default_engine(store, mysql_engine):
    statements = []

    def capture(conn, cursor, sql, params, context, many):
        statements.append(sql)

    async with mysql_engine.connect() as conn:
        old = await conn.scalar(text("SELECT @@default_storage_engine"))
        await conn.execute(text("SET SESSION default_storage_engine = 'MyISAM'"))
        event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
        try:
            facts = await store.connection_checks(conn)
        finally:
            event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
            await conn.execute(text("SET SESSION default_storage_engine = :value"), {"value": old})
    assert facts.max_allowed_packet > 0
    assert len(statements) == 1


async def test_native_error_diagnostics(store, caplog):
    from sqlalchemy.exc import IntegrityError

    run_id = generate_ulid()
    await store.create_run(
        run_id,
        "agent",
        tenant_id="t",
        idempotency_key="PRIVATE_SENTINEL",
        idempotency_fingerprint="digest",
    )
    async with store._session_factory() as session:
        with pytest.raises(IntegrityError) as caught:
            await session.execute(
                insert(AgentRun).values(
                    id=generate_ulid(),
                    agent_name="agent",
                    tenant_id="t",
                    idempotency_key="PRIVATE_SENTINEL",
                )
            )
        safe = store.classify_db_error(caught.value)
        assert safe.kind == "unique_violation" and safe.code == 1062
        assert safe.constraint == "uq_agent_runs_tenant_idempotency_key"
        assert "PRIVATE_SENTINEL" not in repr(safe)
        assert "PRIVATE_SENTINEL" not in caplog.text


async def test_prior_write_survives_targeted_1062_and_commits(store):
    run_id, event_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    await store.save_run_event(run_id, event_type="one", event_id=event_id)
    conflict = store._conflict(RunEvent.__table__, ["id"])
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(iteration_count=7)
        )
        duplicate = store.insert_if_absent_stmt(
            RunEvent.__table__,
            dict(id=event_id, agent_run_id=run_id, event_type="dup", sequence_index=5),
            conflict=conflict,
        )
        assert await store._execute_insert_plan(session, duplicate, conflict) is None
        fresh = store.insert_if_absent_stmt(
            RunEvent.__table__,
            dict(id=generate_ulid(), agent_run_id=run_id, event_type="new", sequence_index=6),
            conflict=conflict,
        )
        assert await store._execute_insert_plan(session, fresh, conflict) is not None
        await store._write_tx(session)
    assert (await store.get_run(run_id)).iteration_count == 7
    assert len(await store.get_run_events(run_id)) == 2


async def test_first_use_facts_need_one_application_select(store, mysql_engine):
    statements = []

    def capture(conn, cursor, sql, params, context, many):
        statements.append(sql)

    async with mysql_engine.connect() as conn:
        event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
        try:
            await store.connection_checks(conn)
        finally:
            event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 1


async def test_event_sequence_zero_one_two_without_readback(store, mysql_engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    statements = []

    def capture(conn, cursor, sql, params, context, many):
        statements.append(sql)

    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        returned = [
            await store.save_run_event(run_id, event_type="one", data={"index": i})
            for i in range(3)
        ]
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    assert returned == [0, 1, 2]
    assert len(statements) == 6  # Three owning-run fences and three INSERTs.
    events = await store.get_run_events(run_id)
    assert {row.data["index"]: row.sequence_index for row in events} == dict(enumerate(returned))
