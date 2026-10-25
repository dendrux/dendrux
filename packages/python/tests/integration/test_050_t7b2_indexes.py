"""Concurrent index DDL must fail clearly behind idle writer transactions."""

import asyncio
import threading
import time

import pytest
from sqlalchemy import event, text

from tests.integration._migration_schema import migration_schema
from tests.integration._migration_schema import test_database_url as pg_url
from tests.integration.test_050_t7b_fix3 import _prior_schema
from tests.integration.test_migration_run_control import _alembic


@pytest.mark.parametrize("action", ["upgrade", "downgrade"])
async def test_concurrent_index_wait_is_bounded_and_retry_repairs(engine, action, record_property):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres concurrent index waits")
    async with migration_schema(pg_url(), f"t7b2_index_{action}") as target:
        await _prior_schema(target, 20)
        url = target.url.render_as_string(hide_password=False)
        if action == "downgrade":
            await asyncio.to_thread(_alembic, url, "upgrade", "head")
        revision = "head" if action == "upgrade" else "d7e8f9a0b1c2"
        marker = (
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_agent_runs_sweep_running"
            if action == "upgrade"
            else "DROP INDEX CONCURRENTLY IF EXISTS ix_agent_runs_sweep_running"
        )
        reached = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        thread_id = None
        timeouts = []

        def before(conn, cursor, statement, parameters, context, executemany):
            if threading.get_ident() == thread_id and "CONCURRENTLY" in statement:
                value = conn.exec_driver_sql("SHOW lock_timeout").scalar_one()
                timeouts.append(value)
                if statement.startswith(marker):
                    loop.call_soon_threadsafe(reached.set)
                    assert release.wait(20)
                    assert value == "5s", "CONCURRENTLY steps must have a bounded lock wait"

        def migrate():
            nonlocal thread_id
            thread_id = threading.get_ident()
            _alembic(url, action, revision)

        async with target.connect() as blocker:
            transaction = await blocker.begin()
            event.listen(target.sync_engine.__class__, "before_cursor_execute", before)
            task = asyncio.create_task(asyncio.to_thread(migrate))
            try:
                await asyncio.wait_for(reached.wait(), 20)
                await blocker.execute(
                    text("UPDATE agent_runs SET iteration_count=1 WHERE id='run-000001'")
                )
                started = time.perf_counter()
                release.set()
                with pytest.raises(RuntimeError, match="[Rr]etry"):
                    await asyncio.wait_for(asyncio.shield(task), 12)
                seconds = time.perf_counter() - started
                assert seconds < 10
                assert all(v == "5s" for v in timeouts)
                record_property(f"concurrent_index_{action}_seconds", seconds)
            finally:
                release.set()
                await transaction.rollback()
                await asyncio.gather(task, return_exceptions=True)
                event.remove(target.sync_engine.__class__, "before_cursor_execute", before)
        if action == "upgrade":
            async with target.connect() as conn:
                valid = (
                    await conn.execute(
                        text(
                            "SELECT indisvalid FROM pg_index "
                            "WHERE indexrelid='ix_agent_runs_sweep_running'::regclass"
                        )
                    )
                ).scalar_one()
                assert not valid, (
                    "timed-out concurrent build should leave a repairable invalid index"
                )
        await asyncio.to_thread(_alembic, url, action, revision)
        if action == "downgrade":
            await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with target.connect() as conn:
            assert (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one() == "e8f9a0b1c2d3"
            assert (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                        "WHERE i.indrelid IN ('agent_runs'::regclass,'run_events'::regclass) "
                        "AND i.indisvalid AND i.indisready AND c.relname IN "
                        "('ix_agent_runs_sweep_running','ix_agent_runs_sweep_waiting',"
                        "'ix_run_events_started')"
                    )
                )
            ).scalar_one() == 3
