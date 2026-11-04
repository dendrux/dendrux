"""Batched reads and index plans on all supported database families."""

from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import AgentRun, Base, RunEvent
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import generate_ulid
from tests.db.conftest import mysql_database
from tests.integration.conftest import pg_database as pg_database


@pytest.fixture(params=["sqlite", "postgres", "mysql", "mariadb"])
async def read_store(request, pg_database):
    backend = request.param
    with (
        contextmanager(mysql_database)(backend)
        if backend in ("mysql", "mariadb")
        else _url(backend, pg_database) as url
    ):
        from dendrux.db._engine_options import engine_options

        engine = create_async_engine(url, **engine_options(url))
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        store = store_for_engine(engine)
        yield store
        await store.close()
        # Other integration fixtures share this worker's Postgres schema. Leave it
        # present until their session teardown (the private database is disposable).
        if backend != "postgres":
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@contextmanager
def _url(backend, pg):
    if backend == "postgres" and pg is None:
        pytest.skip("DENDRUX_TEST_PG_URL not set")
    yield pg if backend == "postgres" else "sqlite+aiosqlite://"


async def seed(store, count=200, events=10):
    ids = [generate_ulid() for _ in range(count)]
    async with store._engine.begin() as conn:
        await conn.execute(
            insert(AgentRun),
            [{"id": rid, "agent_name": "batch", "status": "running"} for rid in ids],
        )
        rows = [
            {
                "id": generate_ulid(),
                "agent_run_id": rid,
                "event_type": "test",
                "sequence_index": seq,
                "iteration_index": seq,
                "data": {"seq": seq},
                "created_at": datetime.now(UTC),
            }
            for rid in ids
            for seq in reversed(range(events))
        ]
        await conn.execute(insert(RunEvent), rows)
    return ids


async def test_batch_bounds_order_heads_missing_and_one_query(read_store):
    store = read_store
    ids = await seed(store)  # 10 x a realistic chunk of 20 runs and 200 rows
    await store.get_run_status(ids[0])  # warm connection checks
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(store._engine.sync_engine, "before_cursor_execute", capture)
    result = await store.get_events_for_runs(dict.fromkeys(ids, -1), limit_per_run=10)
    assert len(statements) == 1
    assert set(result) == set(ids)
    assert sum(map(len, result.values())) == 2000
    for rows in result.values():
        assert [r.sequence_index for r in rows] == list(range(10))
        assert [r.data["seq"] for r in rows] == list(range(10))
    statements.clear()
    assert await store.get_run_statuses(ids) == dict.fromkeys(ids, "running")
    assert len(statements) == 1
    assert await store.get_events_for_runs({ids[0]: 7, generate_ulid(): -1}, limit_per_run=100) == {
        ids[0]: result[ids[0]][8:]
    }
    assert await store.get_run_statuses([ids[0], ids[0], generate_ulid()]) == {ids[0]: "running"}
    statements.clear()
    assert await store.get_events_for_runs({}, limit_per_run=1) == {}
    assert await store.get_run_statuses([]) == {}
    assert statements == []


@pytest.mark.parametrize(
    "kind", ["runs", "statuses", "limit_low", "limit_high", "limit_bool", "head"]
)
async def test_hostile_read_limits_refused_without_query(read_store, kind):
    store = read_store
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(store._engine.sync_engine, "before_cursor_execute", capture)
    heads = {"sentinel": -1}
    limit = {"limit_low": 0, "limit_high": 101, "limit_bool": True}.get(kind, 1)
    if kind == "runs":
        heads = {str(i): -1 for i in range(201)}
    if kind == "head":
        heads = {"sentinel": -2}
    with pytest.raises(ValueError) as raised:
        if kind == "statuses":
            await store.get_run_statuses([str(i) for i in range(201)])
        else:
            await store.get_events_for_runs(heads, limit_per_run=limit)
    assert "sentinel" not in str(raised.value)
    assert statements == []


async def test_limit_100_passes_and_explain_200_runs_uses_index(read_store, record_property):
    store = read_store
    ids = await seed(store, events=100)  # ten times 10 events per run
    heads = dict.fromkeys(ids, -1)
    rows = await store.get_events_for_runs({ids[0]: -1}, limit_per_run=100)
    assert len(rows[ids[0]]) == 100
    query = store.events_for_runs_query(heads, limit_per_run=10)
    compiled = query.compile(dialect=store._engine.dialect, compile_kwargs={"literal_binds": True})
    dialect = store._engine.dialect.name
    prefix = "EXPLAIN QUERY PLAN " if dialect == "sqlite" else "EXPLAIN "
    async with store._engine.connect() as conn:
        plan = (await conn.execute(text(prefix + str(compiled)))).all()
    output = "\n".join(map(str, plan))
    record_property("batched_events_explain", output)
    print("\nBATCHED EXPLAIN " + dialect + "\n" + output)
    indexes = {"ix_run_events_agent_run_id_seq", "uq_run_events_run_seq"}
    if dialect in {"mysql", "mariadb"}:
        scans = [row._mapping for row in plan if row._mapping.get("table") == "run_events"]
        assert len(scans) == 200
        assert all(row["key"] in indexes for row in scans), output
    else:
        assert any(index in output for index in indexes)


async def test_python_grouping_sorts_an_adapter_query_with_reversed_rows(read_store, monkeypatch):
    store = read_store
    ids = await seed(store, count=2, events=4)
    original = store.events_for_runs_query

    def reversed_query(heads, *, limit_per_run):
        rows = original(heads, limit_per_run=limit_per_run).subquery()
        return select(rows).order_by(rows.c.sequence_index.desc(), rows.c.agent_run_id.desc())

    monkeypatch.setattr(store, "events_for_runs_query", reversed_query)
    result = await store.get_events_for_runs(dict.fromkeys(ids, -1), limit_per_run=4)
    assert set(result) == set(ids)
    assert [[row.sequence_index for row in result[run_id]] for run_id in ids] == [
        list(range(4))
    ] * 2
