"""Existing PostgreSQL upgrades retain main-thread cancellation and lock behavior."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import event, text

from tests.migrations.test_060_activate import PRIOR, prior_schema
from tests.migrations.test_fresh_postgres_migrate import fresh_pg as fresh_pg
from tests.migrations.test_fresh_postgres_migrate import migrate

_SLOW_PROCESS = """
import asyncio
import threading
import sys
from contextlib import contextmanager
from alembic import command
from sqlalchemy import text
from dendrux.db._dialects import postgres
from dendrux.cli.main import app
upgrade = command.upgrade
lock = postgres.migration_lock

def record_upgrade(*args, **kwargs):
    try:
        asyncio.get_running_loop()
        running_loop = True
    except RuntimeError:
        running_loop = False
    main_thread = threading.current_thread() is threading.main_thread()
    print(f"CONTEXT:{int(main_thread)}:{int(running_loop)}", flush=True)
    return upgrade(*args, **kwargs)

@contextmanager
def slow_lock(conn):
    with lock(conn):
        pid = conn.scalar(text("SELECT pg_backend_pid()"))
        conn.commit()
        print(f"SLOW:{pid}", flush=True)
        conn.execute(text("SELECT pg_sleep(:delay)"), {"delay": float(sys.argv[1])})
        conn.commit()
        yield

command.upgrade = record_upgrade
postgres.migration_lock = slow_lock
app(args=["db", "migrate"])
"""


async def _read_slow_marker(process):
    lines = []
    async with asyncio.timeout(20):
        while True:
            line = (await process.stdout.readline()).decode().strip()
            assert line, "migration child exited before its slow step"
            lines.append(line)
            if line.startswith("SLOW:"):
                return int(line.split(":")[1]), lines


async def _wait_for_sleep(engine, pid):
    async with asyncio.timeout(10):
        while True:
            async with engine.connect() as conn:
                waiting = await conn.scalar(
                    text("SELECT wait_event = 'PgSleep' FROM pg_stat_activity WHERE pid=:pid"),
                    {"pid": pid},
                )
            if waiting:
                return
            await asyncio.sleep(0.01)


@asynccontextmanager
async def _slow_migration(engine, *, delay=30):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _SLOW_PROCESS,
        str(delay),
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pid = None
    try:
        pid, lines = await _read_slow_marker(process)
        await _wait_for_sleep(engine, pid)
        yield process, lines
    finally:
        await _stop_migration(engine, process, pid)


async def _stop_migration(engine, process, pid):
    if process.returncode is None:
        process.kill()
    await process.communicate()
    if pid is not None:
        async with engine.connect() as conn:
            await conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE pid=:pid AND datname=current_database()"
                ),
                {"pid": pid},
            )


async def test_existing_migrate_sigint_stops_before_upgrade(fresh_pg):
    async with fresh_pg.begin() as conn:
        await conn.run_sync(prior_schema)
    async with _slow_migration(fresh_pg) as (process, _):
        started = time.monotonic()
        process.send_signal(signal.SIGINT)
        await asyncio.wait_for(process.communicate(), 15)
        elapsed = time.monotonic() - started
        async with fresh_pg.connect() as conn:
            revision = await conn.scalar(text("SELECT version_num FROM alembic_version"))
        print(f"SIGINT elapsed={elapsed:.3f}s returncode={process.returncode} revision={revision}")
        assert process.returncode != 0
        assert revision == PRIOR
        assert elapsed < 15


async def test_existing_migrate_releases_probe_and_refuses_competitor(fresh_pg):
    async with fresh_pg.begin() as conn:
        await conn.run_sync(prior_schema)
    async with _slow_migration(fresh_pg) as (process, lines):
        async with fresh_pg.connect() as conn:
            sessions = await conn.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname=current_database() AND pid<>pg_backend_pid() "
                    "AND backend_type='client backend'"
                )
            )
        started = time.monotonic()
        result = await migrate()
        elapsed = time.monotonic() - started
        print(f"competitor elapsed={elapsed:.3f}s exit={result.exit_code} sessions={sessions}")
        assert (sessions, result.exit_code) == (1, 1)
        assert "CONTEXT:1:0" in lines
        assert "Another Dendrux database migration is running" in " ".join(result.output.split())
        assert elapsed < 15
        assert process.returncode is None


@pytest.mark.parametrize("activation_fails", [False, True])
async def test_bootstrap_lock_is_explicitly_released(fresh_pg, monkeypatch, activation_fails):
    from dendrux.cli import db
    from dendrux.db import _activate

    unlocked = []
    make_engine = db._maintenance_engine

    def observe_unlock(conn, cursor, statement, parameters, context, executemany):
        if "pg_advisory_unlock" in statement:
            unlocked.append(parameters == (db._POSTGRES_BOOTSTRAP_LOCK,))

    def engine_with_observer(url):
        engine = make_engine(url)
        event.listen(engine.sync_engine, "before_cursor_execute", observe_unlock)
        return engine

    async def fail(*args, **kwargs):
        raise RuntimeError("injected activation failure")

    monkeypatch.setattr(db, "_maintenance_engine", engine_with_observer)
    if activation_fails:
        monkeypatch.setattr(_activate, "activate_database", fail)
    result = await migrate()
    assert result.exit_code == int(activation_fails), result.output
    assert True in unlocked, "the CLI lock must be explicitly unlocked before closing"


async def test_fresh_interrupt_finishes_upgrade_and_rerun_guides_activation(fresh_pg):
    from tests.migrations.test_fresh_postgres_migrate import settings

    async with _slow_migration(fresh_pg, delay=5) as (process, _):
        process.send_signal(signal.SIGINT)
        output, _ = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode != 0
        assert b"Migrations complete" not in output
        assert await settings(fresh_pg) == {"storage_format": "1"}
    result = await migrate()
    assert result.exit_code == 0, result.output
    assert await settings(fresh_pg) == {"storage_format": "1"}
    assert "dendrux db activate" in " ".join(result.output.split())
