"""Sweep saturation and online status-clock migration regressions."""

import asyncio
import datetime as dt
import json
import sys
import threading
import time

import pytest
from sqlalchemy import event, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Select

from dendrux import sweep
from dendrux.db.models import AgentRun, RunEvent
from dendrux.runtime.state import SQLAlchemyStateStore
from tests.integration._migration_schema import create_pre_activation_schema, migration_schema
from tests.integration._migration_schema import test_database_url as _pg_url
from tests.integration.test_migration_run_control import _alembic


async def test_default_running_bound_reaches_orphan_after_target_concurrency(store, engine):
    now = dt.datetime.now(dt.UTC)
    old = now - dt.timedelta(days=2)
    async with engine.begin() as conn:
        for base in range(0, 1500, 500):
            await conn.execute(
                AgentRun.__table__.insert(),
                [
                    {
                        "id": f"{i:026d}",
                        "agent_name": "live",
                        "status": "running",
                        "last_progress_at": now,
                    }
                    for i in range(base, base + 500)
                ],
            )
        await conn.execute(
            AgentRun.__table__.insert(),
            {
                "id": "00000000000000000000002000",
                "agent_name": "lost",
                "status": "running",
                "last_progress_at": old,
            },
        )
    result = await sweep(state_store=store, stale_running=dt.timedelta(days=1))
    assert [r.run_id for r in result.stale_running] == ["00000000000000000000002000"], (
        "default running bound must reach newer orphans behind >1000 live runs"
    )


@pytest.mark.parametrize("waiting", [False, True])
async def test_sweep_candidate_cap_warns_with_safe_structured_fields(
    store, engine, caplog, waiting
):
    for i in range(4):
        await store.create_run(f"cap-{i}", "agent")
        if waiting:
            await store.pause_run(f"cap-{i}", status="paused", pause_data={})
    if waiting:
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).values(
                    status_changed_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=2),
                    updated_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=2),
                )
            )
        await store.sweep_abandoned_runs(dt.timedelta(days=1), max_runs=2)
    else:
        assert await store.sweep_stale_runs(dt.timedelta(days=1), max_runs=2) == []
    records = [r for r in caplog.records if getattr(r, "code", None) == "sweep_cap_reached"]
    assert len(records) == 1, "candidate saturation must never be silent"
    record = records[0]
    assert record.category == ("abandoned_waiting" if waiting else "stale_running")
    assert record.cap == record.inspected == 2
    assert record.levelname == "WARNING"
    assert "cap-" not in record.getMessage()


async def test_sweep_categories_share_the_same_maximum(store):
    assert await store.sweep_stale_runs(dt.timedelta(days=1), max_runs=100000) == []
    assert await store.sweep_abandoned_runs(dt.timedelta(days=1), max_runs=100000) == []
    result = await sweep(
        state_store=store,
        stale_running=dt.timedelta(days=1),
        abandoned_waiting=dt.timedelta(days=1),
        max_runs=100000,
    )
    assert result.stale_running == result.abandoned_waiting == []
    for method in (store.sweep_stale_runs, store.sweep_abandoned_runs):
        with pytest.raises(ValueError, match="max_runs"):
            await method(dt.timedelta(days=1), max_runs=100001)
    with pytest.raises(ValueError, match="max_runs"):
        await sweep(
            state_store=store,
            stale_running=dt.timedelta(days=1),
            abandoned_waiting=dt.timedelta(days=1),
            max_runs=100001,
        )


async def test_ten_thousand_running_inspections_are_bounded(
    store, engine, monkeypatch, record_property, caplog
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres timing and buffer evidence")
    now = dt.datetime.now(dt.UTC)
    async with engine.begin() as conn:
        for base in range(0, 10000, 500):
            await conn.execute(
                AgentRun.__table__.insert(),
                [
                    {
                        "id": f"scan-{i:05}",
                        "agent_name": "live",
                        "status": "running",
                        "last_progress_at": now,
                    }
                    for i in range(base, base + 500)
                ],
            )
            await conn.execute(
                RunEvent.__table__.insert(),
                [
                    {
                        "id": f"scan-event-{i}-{j}",
                        "agent_run_id": f"scan-{i:05}",
                        "event_type": "run.started" if j == 0 else "tool.completed",
                        "sequence_index": j,
                    }
                    for i in range(base, base + 500)
                    for j in range(5)
                ],
            )
        await conn.execute(text("ANALYZE agent_runs"))
        await conn.execute(text("ANALYZE run_events"))
    statements = []
    original = AsyncSession.execute

    async def capture(session, statement, *args, **kwargs):
        if isinstance(statement, Select) and "started" in statement.selected_columns:
            statements.append(statement)
        return await original(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", capture)
    started = time.perf_counter()
    assert await store.sweep_stale_runs(dt.timedelta(days=1)) == []
    elapsed = time.perf_counter() - started
    assert len(statements) == 100, "default must inspect all 10k candidates in 100-row pages"
    plans = []
    async with engine.connect() as conn:
        for statement in statements:
            sql = str(
                statement.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
            )
            plans.append(
                (
                    await conn.execute(text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql))
                ).scalar_one()[0]
            )
    assert sum(p["Plan"]["Actual Rows"] for p in plans) == 10000
    metrics = {
        "seconds": elapsed,
        "pages": len(plans),
        "execution_ms": sum(p["Execution Time"] for p in plans),
        "shared_hit_blocks": sum(p["Plan"]["Shared Hit Blocks"] for p in plans),
        "shared_read_blocks": sum(p["Plan"]["Shared Read Blocks"] for p in plans),
    }
    record_property("running_10k", json.dumps(metrics))
    print("running_10k", json.dumps(metrics))


async def _prior_schema(engine, count):
    async with engine.begin() as conn:
        await conn.run_sync(create_pre_activation_schema)
        for name in (
            "ix_agent_runs_sweep_running",
            "ix_agent_runs_sweep_waiting",
            "ix_run_events_started",
        ):
            await conn.execute(text(f"DROP INDEX {name}"))
        await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN status_changed_at"))
        await conn.execute(
            text("""
            INSERT INTO agent_runs (id, agent_name, status, iteration_count,
                delegation_level, total_input_tokens, total_output_tokens,
                total_cache_read_tokens, total_cache_creation_tokens, created_at, updated_at)
            SELECT 'run-' || lpad(i::text, 6, '0'), 'migration', 'running', 0, 0, 0, 0,
                   0, 0, now() - interval '2 days', now() - interval '1 day'
            FROM generate_series(1, :count) i
        """),
            {"count": count},
        )
        await conn.execute(text("ANALYZE agent_runs"))
    await asyncio.to_thread(
        _alembic, engine.url.render_as_string(hide_password=False), "stamp", "d7e8f9a0b1c2"
    )


# 11,000 rows (three backfill batches) check correctness in every run. 300,000 rows also
# time a concurrent writer; that latency depends on the machine, so it is a benchmark
# (`make bench`), not a gate.
@pytest.mark.parametrize(
    "rows",
    [11_000, pytest.param(300_000, marks=pytest.mark.benchmark)],
    ids=["11k", "300k"],
)
async def test_online_clock_migration_writer_block(engine, record_property, rows):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres concurrent writer and autocommit evidence")
    async with migration_schema(_pg_url(), "t7_fix3_online") as target:
        await _prior_schema(target, rows)
        stop = asyncio.Event()
        ready = asyncio.Event()
        durations = []
        statements = []
        backfill_plans = []
        migrator_thread = None

        def capture(conn, cursor, statement, parameters, context, executemany):
            if threading.get_ident() == migrator_thread:
                statements.append(statement)
                if (
                    "UPDATE agent_runs" in statement
                    and not statement.startswith("EXPLAIN")
                    and len(backfill_plans) < 2
                ):
                    backfill_plans.append(
                        conn.exec_driver_sql(
                            "EXPLAIN (FORMAT JSON) " + statement, parameters
                        ).scalar_one()[0]
                    )

        async def writer():
            async with target.connect() as conn:
                conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                while not stop.is_set():
                    started = time.perf_counter()
                    await conn.execute(
                        text(
                            "UPDATE agent_runs SET iteration_count=iteration_count+1 "
                            f"WHERE id='run-{rows:06d}'"
                        )
                    )
                    durations.append(time.perf_counter() - started)
                    if len(durations) >= 10:
                        ready.set()
                    await asyncio.sleep(0)

        def migrate():
            nonlocal migrator_thread
            migrator_thread = threading.get_ident()
            _alembic(target.url.render_as_string(hide_password=False), "upgrade", "head")

        event.listen(target.sync_engine.__class__, "before_cursor_execute", capture)
        worker = asyncio.create_task(writer())
        try:
            await asyncio.wait_for(ready.wait(), 20)
            started = time.perf_counter()
            await asyncio.wait_for(asyncio.to_thread(migrate), 120)
            elapsed = time.perf_counter() - started
        finally:
            stop.set()
            await worker
            event.remove(target.sync_engine.__class__, "before_cursor_execute", capture)
        metrics = {
            "rows": rows,
            "seconds": elapsed,
            "writer_samples": len(durations),
            "max_writer_ms": max(durations) * 1000,
        }
        record_property("migration_writer", json.dumps(metrics))
        record_property("backfill_plans", json.dumps(backfill_plans))
        print("migration_writer", json.dumps(metrics))
        async with target.connect() as conn:
            assert (
                await conn.execute(
                    text("SELECT count(*) FROM agent_runs WHERE status_changed_at IS NULL")
                )
            ).scalar_one() == 0
        if rows >= 300_000:  # latency is only meaningful (and only asserted) in the benchmark
            assert max(durations) < 0.2, "migration statements must not block writers for 200ms"

        def nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from nodes(child)

        assert len(backfill_plans) == 2
        assert not any(
            n["Node Type"] == "Seq Scan" and n.get("Relation Name") == "agent_runs"
            for plan in backfill_plans
            for n in nodes(plan["Plan"])
        ), "a 5k backfill batch must not scan the full growing run table"
        sql = "\n".join(statements).upper()
        assert "LIMIT 5000" in sql, "backfill must use bounded id-keyed batches"
        assert "NOT VALID" in sql and "VALIDATE CONSTRAINT" in sql


async def test_online_backfill_commits_batches_and_preserves_concurrent_runtime_clocks(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres backfill visibility")
    async with migration_schema(_pg_url(), "t7_fix3_batches") as target:
        await _prior_schema(target, 12000)
        loop = asyncio.get_running_loop()
        batch_done = asyncio.Event()
        release = threading.Event()
        thread_id = None
        held = False

        def hold(conn, cursor, statement, parameters, context, executemany):
            nonlocal held
            if threading.get_ident() == thread_id and "UPDATE agent_runs" in statement and not held:
                held = True
                loop.call_soon_threadsafe(batch_done.set)
                assert release.wait(20), "test must release first committed batch"

        def migrate():
            nonlocal thread_id
            thread_id = threading.get_ident()
            _alembic(target.url.render_as_string(hide_password=False), "upgrade", "head")

        event.listen(target.sync_engine.__class__, "after_cursor_execute", hold)
        task = asyncio.create_task(asyncio.to_thread(migrate))
        clocks = {}
        try:
            await asyncio.wait_for(batch_done.wait(), 20)
            async with target.connect() as conn:
                count = (
                    await conn.execute(
                        text("SELECT count(*) FROM agent_runs WHERE status_changed_at IS NOT NULL")
                    )
                ).scalar_one()
                assert count == 5000, "each bounded backfill must commit independently"
            runtime = SQLAlchemyStateStore(target)
            # An insert and a transition ahead of the cursor both set the clock.
            await runtime.create_run("run-new", "new")
            # A raw writer omitting the clock also works behind the cursor:
            # ADD COLUMN and SET DEFAULT must commit in the same short DDL.
            async with target.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO agent_runs (id, agent_name, status, iteration_count,
                        delegation_level, total_input_tokens, total_output_tokens,
                        total_cache_read_tokens, total_cache_creation_tokens,
                        created_at, updated_at)
                    SELECT 'aaa-raw', agent_name, status, iteration_count, delegation_level,
                        total_input_tokens, total_output_tokens, total_cache_read_tokens,
                        total_cache_creation_tokens, created_at, updated_at
                    FROM agent_runs WHERE id='run-000001'
                """)
                )
            await runtime.pause_run("run-012000", status="paused", pause_data={})
            async with target.connect() as conn:
                clocks = dict(
                    (
                        await conn.execute(
                            text(
                                "SELECT id, status_changed_at FROM agent_runs "
                                "WHERE id IN ('aaa-raw', 'run-new', 'run-012000')"
                            )
                        )
                    ).all()
                )
            assert all(clocks.values())
        finally:
            release.set()
            await task
            event.remove(target.sync_engine.__class__, "after_cursor_execute", hold)
        async with target.connect() as conn:
            actual = dict(
                (
                    await conn.execute(
                        text(
                            "SELECT id, status_changed_at FROM agent_runs "
                            "WHERE id IN ('aaa-raw', 'run-new', 'run-012000')"
                        )
                    )
                ).all()
            )
            assert actual == clocks, (
                "backfill must not overwrite clocks from concurrent runtime writes"
            )
            assert (
                await conn.execute(
                    text("SELECT count(*) FROM agent_runs WHERE status_changed_at IS NULL")
                )
            ).scalar_one() == 0


async def test_two_migration_runners_serialize_before_reading_revision(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres advisory lock")
    # Alembic's context proxy is process-global, so use real runner processes.
    script = r"""
import sys
from sqlalchemy import Engine, event
from tests.integration.test_migration_run_control import _alembic
url, schema, hold = sys.argv[1:]
def connected(dbapi, record):
    previous = dbapi.autocommit
    dbapi.autocommit = True
    cursor = dbapi.cursor()
    cursor.execute('SET search_path TO "' + schema + '"')
    cursor.close()
    dbapi.autocommit = previous
def before(conn, cursor, statement, parameters, context, executemany):
    if 'alembic_version' in statement:
        print('REVISION_READ', flush=True)
def after(conn, cursor, statement, parameters, context, executemany):
    if hold == 'yes' and 'pg_try_advisory_lock(' in statement:
        print('LOCKED', flush=True)
        assert sys.stdin.readline().strip() == 'release'
event.listen(Engine, 'connect', connected)
event.listen(Engine, 'before_cursor_execute', before)
event.listen(Engine, 'after_cursor_execute', after)
try:
    _alembic(url, 'upgrade', 'head')
except RuntimeError as error:
    print(str(error), file=sys.stderr, flush=True)
    sys.exit(2)
"""
    async with migration_schema(_pg_url(), "t7_fix3_runners") as target:
        await _prior_schema(target, 10)
        args = [
            sys.executable,
            "-c",
            script,
            target.url.render_as_string(hide_password=False),
            "t7_fix3_runners",
        ]
        first = await asyncio.create_subprocess_exec(
            *args,
            "yes",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            line = await asyncio.wait_for(first.stdout.readline(), 20)
            assert line == b"LOCKED\n", "runner must lock before reading the revision"
            second = await asyncio.create_subprocess_exec(
                *args, "no", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await asyncio.wait_for(second.communicate(), 20)
            assert second.returncode == 2
            assert b"Another Dendrux database migration is running" in err
            assert b"REVISION_READ" not in out, "refused runner must not read stale revision"
        finally:
            if first.returncode is None:
                first.stdin.write(b"release\n")
                await first.stdin.drain()
            out, err = await asyncio.wait_for(first.communicate(), 60)
        assert first.returncode == 0, err.decode()
        # A later runner succeeds: the session lock was explicitly released.
        await asyncio.to_thread(
            _alembic, target.url.render_as_string(hide_password=False), "upgrade", "head"
        )
        async with target.connect() as conn:
            assert (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one() == "f9a0b1c2d3e4"
