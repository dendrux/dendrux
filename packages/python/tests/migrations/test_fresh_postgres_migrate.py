"""CLI bootstrap on private PostgreSQL databases, including concurrent processes."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db.models import AgentRun, Base
from tests._helpers.pg_url import maintenance_url, render, validate_test_pg_url
from tests.migrations.test_060_activate import alembic, prior_schema


@pytest.fixture
async def fresh_pg(monkeypatch):
    raw = os.environ.get("DENDRUX_TEST_PG_URL")
    if not raw:
        pytest.skip("DENDRUX_TEST_PG_URL not set")
    base = validate_test_pg_url(raw, append_suffix=False)
    name = f"{(base.database or '').removesuffix('_test')[:36]}_fresh_{uuid.uuid4().hex[:8]}_test"
    url = render(base.set(database=name))
    admin = create_async_engine(
        maintenance_url(base), poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
        try:
            yield engine
        finally:
            await engine.dispose()
            async with admin.connect() as conn:
                await conn.execute(text(f'DROP DATABASE "{name}"'))
    finally:
        await admin.dispose()


async def migrate():
    return await asyncio.to_thread(CliRunner().invoke, app, ["db", "migrate"])


async def settings(engine):
    async with engine.connect() as conn:
        return dict((await conn.execute(text("SELECT key, value FROM dendrux_settings"))).all())


async def assert_active(engine):
    values = await settings(engine)
    assert values["storage_format"] == "2"
    assert values["activate_step"] == "done"
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert set(tables) >= set(Base.metadata.tables) | {"alembic_version"}
        assert (await conn.execute(text("SELECT count(*) FROM alembic_version"))).scalar_one() == 1


@pytest.mark.parametrize("application_table", [False, True])
async def test_fresh_postgres_is_active(fresh_pg, application_table):
    if application_table:
        async with fresh_pg.begin() as conn:
            await conn.execute(text("CREATE TABLE application_data (value TEXT)"))
            await conn.execute(text("INSERT INTO application_data VALUES ('keep')"))
    result = await migrate()
    assert result.exit_code == 0, result.output
    await assert_active(fresh_pg)
    if application_table:
        async with fresh_pg.connect() as conn:
            assert (await conn.execute(text("SELECT value FROM application_data"))).all() == [
                ("keep",)
            ]


@pytest.mark.parametrize("at_head", [False, True])
async def test_existing_empty_postgres_stays_format_one(fresh_pg, at_head):
    async with fresh_pg.begin() as conn:
        await conn.run_sync(prior_schema)
    if at_head:
        await asyncio.to_thread(alembic, render(fresh_pg.url), "upgrade", "head")
    result = await migrate()
    assert result.exit_code == 0, result.output
    assert (await settings(fresh_pg)) == {"storage_format": "1"}
    output = " ".join(result.output.split())
    assert "dendrux db activate" in output
    assert "every worker and reader is upgraded" in output
    async with fresh_pg.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one() == 0


async def test_activated_migrate_is_noop(fresh_pg):
    assert (await migrate()).exit_code == 0
    before = await settings(fresh_pg)
    result = await migrate()
    assert result.exit_code == 0, result.output
    await assert_active(fresh_pg)
    assert await settings(fresh_pg) == before
    assert "dendrux db activate" not in result.output


async def test_activation_failure_is_masked_and_rerun_stays_format_one(fresh_pg, monkeypatch):
    from dendrux.db import _activate

    async def fail(*args, **kwargs):
        raise RuntimeError(
            "failed postgresql+asyncpg://sentinel_user:sentinel_password@localhost/db"
        )

    with monkeypatch.context() as patch:
        patch.setattr(_activate, "activate_database", fail)
        result = await migrate()
    assert result.exit_code == 1
    assert "Migration failed" in result.output
    assert "sentinel_user" not in result.output and "sentinel_password" not in result.output
    assert await settings(fresh_pg) == {"storage_format": "1"}
    result = await migrate()
    assert result.exit_code == 0, result.output
    assert await settings(fresh_pg) == {"storage_format": "1"}
    assert "dendrux db activate" in " ".join(result.output.split())


_PROCESS = """
import time
from contextlib import contextmanager
from dendrux.db._dialects import postgres
from dendrux.cli.main import app
original = postgres.migration_lock
@contextmanager
def slow_lock(conn):
    with original(conn):
        time.sleep(0.5)
        yield
postgres.migration_lock = slow_lock
print("ready", flush=True)
input()
app(args=["db", "migrate"])
"""


async def test_two_concurrent_migrate_processes(fresh_pg):
    processes = []
    try:
        for _ in range(2):
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _PROCESS,
                env=os.environ.copy(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            processes.append(process)
        for process in processes:
            assert await asyncio.wait_for(process.stdout.readline(), 20) == b"ready\n"
        results = await asyncio.wait_for(
            asyncio.gather(*(process.communicate(b"go\n") for process in processes)), 45
        )
        assert [p.returncode for p in processes] == [0, 0], results
        await assert_active(fresh_pg)
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def test_run_on_fresh_migrated_database_uses_format_two(fresh_pg):
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.types import LLMResponse, RunStatus

    assert (await migrate()).exit_code == 0
    store = store_for_engine(fresh_pg)
    try:
        agent = Agent(
            prompt="p",
            provider=MockLLM([LLMResponse(text="done")]),
            state_store=store,
            strict_tenancy=False,
        )
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        async with fresh_pg.connect() as conn:
            assert (
                await conn.execute(
                    select(AgentRun.storage_format).where(AgentRun.id == result.run_id)
                )
            ).scalar_one() == 2
    finally:
        await store.close()


async def test_empty_revision_table_prevents_auto_activation(fresh_pg):
    async with fresh_pg.begin() as conn:
        await conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
    result = await migrate()
    assert result.exit_code == 0, result.output
    assert await settings(fresh_pg) == {"storage_format": "1"}
    assert "dendrux db activate" in " ".join(result.output.split())


@pytest.mark.parametrize("table", list(Base.metadata.tables))
async def test_any_dendrux_table_prevents_fresh_classification(fresh_pg, table):
    from dendrux.cli.db import _fresh_postgres

    async with fresh_pg.begin() as conn:
        await conn.execute(text(f'CREATE TABLE "{table}" (placeholder INTEGER)'))
    async with fresh_pg.connect() as conn:
        assert not await _fresh_postgres(conn)


async def test_cli_coordination_wait_is_bounded_and_releases_on_failure(fresh_pg, monkeypatch):
    from dendrux.cli import db

    monkeypatch.setattr(db, "_POSTGRES_BOOTSTRAP_WAIT_SECONDS", 0.01)
    async with fresh_pg.connect() as blocker:
        await blocker.execute(
            text("SELECT pg_advisory_lock(:key)"), {"key": db._POSTGRES_BOOTSTRAP_LOCK}
        )
        await blocker.commit()
        result = await asyncio.wait_for(migrate(), 5)
        assert result.exit_code == 1
        assert "retry when it finishes" in " ".join(result.output.split())
    # The failed attempt must not retain a connection or lock; retry after the holder closes.
    result = await migrate()
    assert result.exit_code == 0, result.output
    await assert_active(fresh_pg)


@pytest.mark.parametrize("existing", [False, True])
async def test_status_read_failure_does_not_fail_migration(fresh_pg, monkeypatch, existing):
    from dendrux.cli import db

    if existing:
        async with fresh_pg.begin() as conn:
            await conn.run_sync(prior_schema)

    async def fail_status(url):
        raise RuntimeError(
            "status-detail-sentinel postgresql+asyncpg://sentinel_user:sentinel_password@host/db"
        )

    monkeypatch.setattr(db, "_read_activation_status", fail_status)
    result = await migrate()
    values = await settings(fresh_pg)
    assert values["storage_format"] == ("1" if existing else "2")
    assert result.exit_code == 0, result.output
    assert "Could not read database status. Run `dendrux db status`." in result.output
    assert "Migrations complete" in result.output and "Migration failed" not in result.output
    assert "sentinel" not in result.output


async def test_percent_encoded_password_reaches_fresh_bootstrap(fresh_pg, monkeypatch):
    password = fresh_pg.url.password
    if not password:
        pytest.skip("the selected PostgreSQL URL has no password to encode")
    encoded = "".join(f"%{byte:02X}" for byte in password.encode("utf-8"))
    url = render(fresh_pg.url.set(password="password_placeholder"))
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url.replace("password_placeholder", encoded))
    result = await migrate()
    assert result.exit_code == 0, result.output
    await assert_active(fresh_pg)


@pytest.mark.parametrize("outcome", ["success", "activation_failure", "upgrade_failure"])
async def test_lost_bootstrap_session_preserves_migration_outcome(fresh_pg, monkeypatch, outcome):
    from alembic import command

    from dendrux.cli import db
    from dendrux.db import _activate

    unlock = db._unlock_postgres_bootstrap
    upgrade = command.upgrade

    async def lose_session_then_unlock(conn):
        pid = await conn.scalar(text("SELECT pg_backend_pid()"))
        await conn.commit()
        driver = (await conn.get_raw_connection()).driver_connection
        async with fresh_pg.connect() as killer:
            terminated = await killer.scalar(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE pid=:pid AND datname=current_database()"
                ),
                {"pid": pid},
            )
        assert terminated is True
        async with asyncio.timeout(5):
            while not driver.is_closed():
                await asyncio.sleep(0.01)
        await unlock(conn)

    async def fail_activation(*args, **kwargs):
        raise RuntimeError("activation-detail-sentinel")

    def fail_after_upgrade(*args, **kwargs):
        upgrade(*args, **kwargs)
        raise RuntimeError("original-upgrade-failure")

    monkeypatch.setattr(db, "_unlock_postgres_bootstrap", lose_session_then_unlock)
    if outcome == "activation_failure":
        monkeypatch.setattr(_activate, "activate_database", fail_activation)
    if outcome == "upgrade_failure":
        monkeypatch.setattr(command, "upgrade", fail_after_upgrade)
    result = await migrate()
    if outcome == "success":
        await assert_active(fresh_pg)
        assert result.exit_code == 0, result.output
    else:
        assert result.exit_code == 1
        assert await settings(fresh_pg) == {"storage_format": "1"}
        expected = (
            "PostgreSQL activation failed."
            if outcome == "activation_failure"
            else "original-upgrade-failure"
        )
        assert expected in result.output
    assert "activation-detail-sentinel" not in result.output
    assert "pg_advisory_unlock" not in result.output


@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
async def test_unlock_preserves_base_exceptions(error):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.cli import db

    conn = SimpleNamespace(
        rollback=AsyncMock(), execute=AsyncMock(side_effect=error), commit=AsyncMock()
    )
    with pytest.raises(type(error)) as caught:
        await db._unlock_postgres_bootstrap(conn)
    assert caught.value is error
