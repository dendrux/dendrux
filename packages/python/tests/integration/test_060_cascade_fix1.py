"""Admission and frontier regressions exercised on real database transactions."""

import asyncio
import sqlite3
import sys
import time
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent
from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from dendrux.runtime import durability
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import generate_ulid


@asynccontextmanager
async def sqlite_file(tmp_path, mode):
    path = tmp_path / f"{mode}.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    options = {"timeout": 0.02}
    if mode == "autocommit_false":
        if sys.version_info < (3, 12):
            pytest.skip("sqlite3 autocommit=False requires Python 3.12")
        options["autocommit"] = False
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", connect_args=options)
    if mode == "explicit_begin":

        @event.listens_for(engine.sync_engine, "connect")
        def connection_mode(connection, record):
            connection.isolation_level = None

        @event.listens_for(engine.sync_engine, "begin")
        def begin(connection):
            connection.exec_driver_sql("BEGIN")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.run_sync(lambda c: seed_settings(c, active=True))
        yield engine
    finally:
        await engine.dispose()


@pytest.mark.parametrize("mode", ["legacy", "explicit_begin", "autocommit_false"])
async def test_sqlite_file_creator_lock_closes_admission(
    tmp_path, monkeypatch, mode, record_property
):
    async with sqlite_file(tmp_path, mode) as engine:
        store, creator = SQLAlchemyStateStore(engine), SQLAlchemyStateStore(engine)
        root, child = generate_ulid(), generate_ulid()
        await store.create_run(root, "parent", tenant_id="t")
        checked, release, retried = asyncio.Event(), asyncio.Event(), asyncio.Event()
        factory = creator._session_factory

        @asynccontextmanager
        async def held_commit():
            async with factory() as session:
                commit = session.commit

                async def delayed():
                    checked.set()
                    await release.wait()
                    await commit()

                session.commit = delayed
                yield session

        async def backoff(*args):
            retried.set()
            await release.wait()

        monkeypatch.setattr(creator, "_session_factory", held_commit)
        monkeypatch.setattr(durability, "_backoff", backoff)
        agent = Agent(prompt="p", state_store=store, strict_tenancy=False)
        creation = asyncio.create_task(
            creator.create_run(child, "child", parent_run_id=root, tenant_id="t")
        )
        cancellation = retry_seen = None
        try:
            await asyncio.wait_for(checked.wait(), 5)
            started = time.perf_counter()
            cancellation = asyncio.create_task(agent.cancel_run(root, cascade=True, tenant_id="t"))
            retry_seen = asyncio.create_task(retried.wait())
            done, _ = await asyncio.wait(
                {cancellation, retry_seen}, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            assert retry_seen in done, "cascade escaped the creator lock without a retry"
            assert not cancellation.done()
            waited = time.perf_counter() - started
            record_property("lock_wait_seconds", waited)
            print(f"SQLite {mode} creator lock/busy wait: {waited:.6f}s")
            release.set()
            await asyncio.wait_for(creation, 5)
            result = await asyncio.wait_for(cancellation, 5)
            assert {row["run_id"] for row in result.meta["cascade"]} == {root, child}
            assert (await store.get_run(child)).cancel_requested
        finally:
            release.set()
            if retry_seen is not None:
                retry_seen.cancel()
            await asyncio.gather(
                *[task for task in (creation, cancellation, retry_seen) if task is not None],
                return_exceptions=True,
            )
            await agent.close()


@pytest.mark.parametrize("mode", ["legacy", "explicit_begin", "autocommit_false"])
async def test_sqlite_parent_reservation_is_first_and_does_not_touch_timestamp(tmp_path, mode):
    async with sqlite_file(tmp_path, mode) as engine:
        store = SQLAlchemyStateStore(engine)
        parent = generate_ulid()
        await store.create_run(parent, "parent")
        before = await store.get_run(parent)
        statements = []

        def capture(conn, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith(("SELECT", "UPDATE", "INSERT")):
                statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await store.create_run(generate_ulid(), "child", parent_run_id=parent)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert statements[0].startswith(
            "UPDATE agent_runs SET cancel_requested = cancel_requested WHERE id ="
        ), statements
        assert "updated_at" not in statements[0]
        assert (await store.get_run(parent)).updated_at == before.updated_at


@pytest.mark.parametrize("operation", ["mark_cancel_requested", "get_children_page"])
async def test_frontier_operations_retry_fresh_transactions(store, monkeypatch, operation):
    parent, child = generate_ulid(), generate_ulid()
    await store.create_run(parent, "parent")
    await store.create_run(child, "child", parent_run_id=parent)
    original = AsyncSession.execute
    attempts, retries = [], []

    async def execute(session, statement, *args, **kwargs):
        attempts.append(session)
        if len(attempts) == 1:
            raise OperationalError("frontier", {}, sqlite3.OperationalError("database is locked"))
        return await original(session, statement, *args, **kwargs)

    async def backoff(number, label, run_id, error):
        retries.append((number, label, run_id))

    monkeypatch.setattr(AsyncSession, "execute", execute)
    monkeypatch.setattr(durability, "_backoff", backoff)
    if operation == "mark_cancel_requested":
        assert await store.mark_cancel_requested([parent], tenant_id=None) == 1
    else:
        assert await store.get_children_page([parent], after=None, limit=500, tenant_id=None) == [
            child
        ]
    assert attempts[0] is not attempts[1]
    assert retries == [(0, operation, parent)]


@pytest.mark.parametrize("tenant", [None, "t"])
async def test_sqlite_frontier_plans_ignore_unrelated_tenant_history(
    tmp_path, tenant, record_property
):
    from sqlalchemy import insert

    from dendrux.db.models import AgentRun

    async with sqlite_file(tmp_path, "legacy") as engine:
        store = SQLAlchemyStateStore(engine)
        parent, child = "7R" + "0" * 24, "7Z" + "0" * 24
        async with engine.begin() as connection:
            await connection.execute(
                insert(AgentRun),
                [
                    dict(id=f"{i:026d}", agent_name="history", status="running", tenant_id=tenant)
                    for i in range(5000)
                ]
                + [dict(id=parent, agent_name="parent", status="running", tenant_id=tenant)],
            )
            await connection.execute(
                insert(AgentRun),
                [
                    dict(
                        id=child,
                        agent_name="child",
                        status="running",
                        tenant_id=tenant,
                        parent_run_id=parent,
                    )
                ],
            )
        queries = []

        def capture(conn, cursor, statement, parameters, context, many):
            queries.append((statement, parameters))

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            assert await store.mark_cancel_requested([parent], tenant_id=tenant) == 1
            assert await store.get_children_page(
                [parent], after=None, limit=500, tenant_id=tenant
            ) == [child]
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(queries) == 2
        for name, (statement, parameters) in zip(("mark", "children"), queries, strict=True):
            async with engine.connect() as connection:
                plan = (
                    await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters)
                ).all()
            details = [row[3] for row in plan]
            record_property(f"{name}_plan", str(details))
            print(f"SQLite tenant={tenant!r} {name}: {details}")
            assert not any("tenant_id" in row and "INDEX" in row for row in details), details
            expected = (
                "sqlite_autoindex_agent_runs_1" if name == "mark" else "ix_agent_runs_parent_run_id"
            )
            assert any(expected in row for row in details), details


async def test_postgres_child_plan_is_bounded_by_frontier_not_history(
    store, engine, record_property
):
    import json

    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres LATERAL/parent-index plan on one million unrelated runs")
    columns = (
        "id,agent_name,status,iteration_count,delegation_level,total_input_tokens,"
        "total_output_tokens,total_cache_read_tokens,total_cache_creation_tokens,"
        "created_at,updated_at,tenant_id,parent_run_id"
    )
    insert_sql = (
        "INSERT INTO agent_runs (" + columns + ") SELECT {identity},'plan', 'running',"
        "0,0,0,0,0,0,now(),now(),'t',{parent} FROM generate_series(1,{size}) g"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                insert_sql.format(
                    identity="'0'||lpad(g::text,25,'0')",
                    parent=(
                        "CASE WHEN g%10=0 OR g<10 THEN NULL "
                        "ELSE '0'||lpad((g-g%10)::text,25,'0') END"
                    ),
                    size=1000000,
                )
            )
        )
        await connection.execute(
            text(
                insert_sql.format(
                    identity="'7P'||lpad(g::text,24,'0')",
                    parent="NULL",
                    size=2000,
                )
            )
        )
        await connection.execute(
            text(
                insert_sql.format(
                    identity="'7Z'||lpad(g::text,24,'0')",
                    parent="'7P'||lpad(((g-1)/3+1)::text,24,'0')",
                    size=6000,
                )
            )
        )
        await connection.execute(text("ANALYZE agent_runs"))
    parents = ["7P" + f"{i:024d}" for i in range(1, 2001)]
    statements = []

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        first = await store.get_children_page(parents, after=None, limit=500, tenant_id="t")
        second = await store.get_children_page(
            [*reversed(parents), *parents[:20]], after=first[-1], limit=500, tenant_id="t"
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert first + second == ["7Z" + f"{i:024d}" for i in range(1, 1001)]
    assert len(statements) == 2

    def nodes(plan):
        yield plan
        for child in plan.get("Plans", []):
            yield from nodes(child)

    for number, (statement, parameters) in enumerate(statements):
        async with engine.connect() as connection:
            result = await connection.exec_driver_sql(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement, parameters
            )
            plan = result.scalar_one()
            if isinstance(plan, str):
                plan = json.loads(plan)
        record_property(f"page_{number}_plan", json.dumps(plan))
        scans = list(nodes(plan[0]["Plan"]))
        print(
            f"Postgres page={number}: ms={plan[0]['Execution Time']} scans="
            + json.dumps(
                [
                    {
                        key: node[key]
                        for key in (
                            "Node Type",
                            "Index Name",
                            "Rows Removed by Filter",
                            "Actual Loops",
                            "Actual Rows",
                        )
                        if key in node
                    }
                    for node in scans
                ]
            )
        )
        assert not any(
            node.get("Index Name") == "agent_runs_pkey" and "Filter" in node for node in scans
        ), plan
        assert any(node.get("Index Name") == "ix_agent_runs_parent_run_id" for node in scans), plan


@pytest.mark.parametrize("status", ["success", "error", "cancelled", "max_iterations"])
@pytest.mark.parametrize("old_flag", [False, True])
async def test_terminal_frontier_preserves_row_but_scans_children(store, status, old_flag):
    from sqlalchemy import select, update

    from dendrux.db.models import AgentRun

    root, child = generate_ulid(), generate_ulid()
    await store.create_run(root, "root", tenant_id="t")
    await store.create_run(child, "child", parent_run_id=root, tenant_id="t")
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == root)
            .values(status=status, cancel_requested=old_flag)
        )
        await session.commit()
        before = (
            await session.execute(
                select(AgentRun.cancel_requested, AgentRun.updated_at).where(AgentRun.id == root)
            )
        ).one()
    assert await store.mark_cancel_requested([root], tenant_id="t") == 0
    agent = Agent(prompt="p", state_store=store, strict_tenancy=False)
    try:
        result = await agent.cancel_run(root, cascade=True)
        async with store._session_factory() as session:
            after = (
                await session.execute(
                    select(AgentRun.cancel_requested, AgentRun.updated_at).where(
                        AgentRun.id == root
                    )
                )
            ).one()
        assert after == before
        assert result.status.value == status
        assert result.meta["cancel_requested"] is old_flag
        assert (await store.get_run(child)).cancel_requested
        assert result.meta["cascade_count"] == 2
    finally:
        await agent.close()
