"""Every additive revision DDL/seed step retains the five-second lock bound."""

import asyncio
import threading
import time

import pytest
from sqlalchemy import event, text

from dendrux.db._schema_check import ensure_schema_current
from dendrux.db._storage_schema import ADDITIONS
from tests.integration._migration_schema import migration_schema
from tests.migrations.test_060_activate import HEAD, PRIOR, alembic

_STEPS = [
    (action, table, column)
    for action in ("upgrade", "downgrade")
    for table, columns in ADDITIONS.items()
    for column in columns
] + [
    ("upgrade", "dendrux_settings", "create"),
    ("upgrade", "dendrux_settings", "seed"),
    ("downgrade", "dendrux_settings", "drop"),
]


@pytest.mark.parametrize(("action", "table", "column"), _STEPS)
async def test_060_each_step_is_lock_bounded(pg_test_url, action, table, column):
    schema = f"bounds_060_{action}_{table}_{column}"
    async with migration_schema(pg_test_url, schema) as engine:
        await asyncio.to_thread(alembic, pg_test_url, "upgrade", PRIOR)
        if action == "downgrade":
            await asyncio.to_thread(alembic, pg_test_url, "upgrade", HEAD)
        if column == "create":
            marker = "create table if not exists dendrux_settings"
        elif column == "seed":
            marker = "insert into dendrux_settings"
        else:
            operation = (
                "add column if not exists" if action == "upgrade" else "drop column if exists"
            )
            marker = (
                "drop table if exists dendrux_settings"
                if column == "drop"
                else f"alter table {table} {operation} {column}"
            )
        ready = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        thread_id = None
        observed = []

        def capture(conn, cursor, statement, parameters, context, executemany):
            if threading.get_ident() == thread_id and marker in statement.lower():
                observed.append(conn.exec_driver_sql("SHOW lock_timeout").scalar_one())
                loop.call_soon_threadsafe(ready.set)
                assert release.wait(20)

        def migrate():
            nonlocal thread_id
            thread_id = threading.get_ident()
            alembic(pg_test_url, action, HEAD if action == "upgrade" else PRIOR)

        event.listen(engine.sync_engine.__class__, "before_cursor_execute", capture)
        task = asyncio.create_task(asyncio.to_thread(migrate))
        try:
            await asyncio.wait_for(ready.wait(), 20)
            async with engine.connect() as blocker:
                if column == "create":
                    await blocker.execute(
                        text(
                            "CREATE TABLE dendrux_settings (key VARCHAR(64) PRIMARY KEY, "
                            "value VARCHAR(255) NOT NULL, updated_at TIMESTAMPTZ NOT NULL)"
                        )
                    )
                elif column == "seed":
                    await blocker.execute(text("LOCK TABLE dendrux_settings IN SHARE MODE"))
                else:
                    await blocker.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                started = time.perf_counter()
                release.set()
                try:
                    with pytest.raises(RuntimeError, match="[Rr]etry"):
                        await asyncio.wait_for(asyncio.shield(task), 12)
                    assert 4.5 < time.perf_counter() - started < 10
                    assert observed == ["5s"]
                finally:
                    await blocker.rollback()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            event.remove(engine.sync_engine.__class__, "before_cursor_execute", capture)
        async with engine.connect() as conn:
            assert (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one() == (PRIOR if action == "upgrade" else HEAD)
        await asyncio.to_thread(
            alembic, pg_test_url, action, HEAD if action == "upgrade" else PRIOR
        )
        if action == "downgrade":
            await asyncio.to_thread(alembic, pg_test_url, "upgrade", HEAD)
        await ensure_schema_current(engine)
