"""Independent Alembic graphs and restartable migration history adoption."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from typer.testing import CliRunner

from dendrux.cli.main import app
from tests.integration._migration_schema import migration_schema
from tests.migrations.test_060_activate import HEAD, PRIOR, prior_schema
from tests.migrations.test_fresh_postgres_migrate import fresh_pg as fresh_pg
from tests.unit.test_061_migrations import DEDICATED, LEGACY, snapshot, stamp


@pytest.fixture(params=["sqlite", "postgres"])
async def migration_database(request, tmp_path, pg_database, monkeypatch):
    if request.param == "sqlite":
        url = f"sqlite+aiosqlite:///{tmp_path / 'migration.db'}"
        engine = create_async_engine(url)
        monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
        try:
            yield engine
        finally:
            await engine.dispose()
    else:
        if pg_database is None:
            pytest.skip("DENDRUX_TEST_PG_URL not set")
        monkeypatch.setenv("DENDRUX_DATABASE_URL", pg_database)
        async with migration_schema(pg_database, "migration_061") as engine:
            yield engine


def app_config(directory):
    directory.mkdir()
    (directory / "versions").mkdir()
    (directory / "env.py").write_text(
        "from alembic import context\n"
        "context.configure(connection=context.config.attributes['connection'])\n"
        "with context.begin_transaction():\n    context.run_migrations()\n"
    )
    (directory / "versions" / "first.py").write_text(
        "from alembic import op\nimport sqlalchemy as sa\n"
        "revision='app_one'\ndown_revision=None\n"
        "def upgrade():\n    op.create_table('sales_jobs', sa.Column('id', sa.Integer))\n"
    )
    (directory / "versions" / "second.py").write_text(
        "from alembic import op\nimport sqlalchemy as sa\n"
        "revision='app_two'\ndown_revision='app_one'\n"
        "def upgrade():\n    op.add_column('sales_jobs', sa.Column('name', sa.String(32)))\n"
    )
    cfg = Config()
    cfg.set_main_option("script_location", str(directory))
    return cfg


def app_upgrade(conn, cfg, revision):
    cfg.attributes["connection"] = conn
    command.upgrade(cfg, revision)
    return MigrationContext.configure(conn).get_current_heads()


async def migrate():
    return await asyncio.to_thread(CliRunner().invoke, app, ["db", "migrate"])


async def test_061_alembic_histories_coexist(migration_database, tmp_path):
    engine = migration_database
    cfg = app_config(tmp_path / "app_migrations")
    async with engine.begin() as conn:
        await conn.run_sync(prior_schema)
        await conn.execute(text(f"ALTER TABLE {LEGACY} RENAME TO {DEDICATED}"))
        assert await conn.run_sync(lambda c: app_upgrade(c, cfg, "app_one")) == ("app_one",)
        await conn.execute(text("INSERT INTO sales_jobs (id) VALUES (42)"))
    result = await migrate()
    assert result.exit_code == 0, result.output
    async with engine.begin() as conn:
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
        assert await conn.run_sync(lambda c: app_upgrade(c, cfg, "head")) == ("app_two",)
        assert await conn.scalar(text("SELECT id FROM sales_jobs")) == 42
        before = await conn.run_sync(snapshot)
    assert (await migrate()).exit_code == 0
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before
        assert (
            await conn.scalar(text("SELECT value FROM dendrux_settings WHERE key='storage_format'"))
            == "1"
        )


async def test_061_legacy_upgrade_preserves_activation(migration_database):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(prior_schema)
    result = await migrate()
    assert result.exit_code == 0, result.output
    async with engine.connect() as conn:
        assert not await conn.run_sync(lambda c: inspect(c).has_table(LEGACY))
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
        assert (
            await conn.scalar(text("SELECT value FROM dendrux_settings WHERE key='storage_format'"))
            == "1"
        )


@pytest.mark.parametrize(
    ("dedicated", "legacy", "schema"),
    [
        ([HEAD], [HEAD], True),
        (None, ["app_one", HEAD], True),
        (None, ["unknown"], True),
        (None, [HEAD], False),
        (None, None, True),
        (None, [], True),
        ([HEAD], None, False),
        ([], None, False),
        (["unknown"], None, True),
    ],
)
async def test_061_ownership_refusal_matrix(migration_database, dedicated, legacy, schema):
    engine = migration_database
    async with engine.begin() as conn:
        if schema:
            await conn.execute(text("CREATE TABLE agent_runs (id INTEGER)"))
            await conn.execute(text("INSERT INTO agent_runs VALUES (42)"))
        if dedicated is not None:
            await conn.run_sync(lambda c: stamp(c, DEDICATED, dedicated))
        if legacy is not None:
            await conn.run_sync(lambda c: stamp(c, LEGACY, legacy))
        before = await conn.run_sync(snapshot)
    result = await migrate()
    assert result.exit_code == 1, result.output
    assert "migration_version_table_conflict" in result.output
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before


@pytest.mark.parametrize("revisions", [None, [], ["app_one", "app_two"]])
async def test_061_fresh_postgres_ignores_app_stamp(fresh_pg, revisions):
    async with fresh_pg.begin() as conn:
        if revisions is not None:
            await conn.run_sync(lambda c: stamp(c, LEGACY, revisions))
    result = await migrate()
    assert result.exit_code == 0, result.output
    async with fresh_pg.connect() as conn:
        assert await conn.run_sync(lambda c: inspect(c).has_table(LEGACY)) == (
            revisions is not None
        )
        if revisions is not None:
            assert list(await conn.scalars(text(f"SELECT version_num FROM {LEGACY}"))) == revisions
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD

        assert (
            await conn.scalar(text("SELECT value FROM dendrux_settings WHERE key='storage_format'"))
            == "2"
        )


_PROCESS = """
import sys
from contextlib import contextmanager
from sqlalchemy import event, Engine, Connection
from dendrux.cli.main import app
from dendrux.db._dialects import postgres
mode = sys.argv[1]
renamed = False
commit = Connection.commit
lock = postgres.migration_lock
@contextmanager
def observed_lock(conn):
    try:
        with lock(conn):
            yield
    except RuntimeError:
        print('REFUSED', flush=True)
        raise
def observed_commit(conn):
    global renamed
    commit(conn)
    if renamed and mode == 'after_commit':
        print('CUT', flush=True)
        input()
def after_cursor(conn, cursor, statement, parameters, context, executemany):
    global renamed
    if statement.startswith('ALTER TABLE alembic_version RENAME TO dendrux_alembic_version'):
        renamed = True
        print('RENAMED', flush=True)
        if mode == 'before_commit':
            print('CUT', flush=True)
            input()
        elif mode == 'race':
            input()
Connection.commit = observed_commit
postgres.migration_lock = observed_lock
event.listen(Engine, 'after_cursor_execute', after_cursor)
print('READY', flush=True)
input()
app(args=['db', 'migrate'])
"""


async def process(mode):
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _PROCESS,
        mode,
        env=os.environ.copy(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert await asyncio.wait_for(child.stdout.readline(), 20) == b"READY\n"
    return child


async def stop(children):
    for child in children:
        if child.returncode is None:
            child.kill()
        await child.communicate()


async def race_marker(child):
    lines = []
    async with asyncio.timeout(20):
        while True:
            line = await child.stdout.readline()
            assert line, "migration exited before reaching the concurrency barrier"
            lines.append(line)
            if line in {b"RENAMED\n", b"REFUSED\n"}:
                return b"".join(lines), line


async def test_061_two_postgres_migrate_callers_rename_once(fresh_pg):
    async with fresh_pg.begin() as conn:
        await conn.run_sync(prior_schema)
    children = []
    try:
        children.extend([await process("race"), await process("race")])
        for child in children:
            child.stdin.write(b"go\n")
            await child.stdin.drain()
        observed = await asyncio.gather(*(race_marker(child) for child in children))
        assert {marker for _, marker in observed} == {b"RENAMED\n", b"REFUSED\n"}
        loser = next(i for i, (_, marker) in enumerate(observed) if marker == b"REFUSED\n")
        assert await asyncio.wait_for(children[loser].wait(), 3) == 1
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    child.communicate(None if i == loser else b"release\n")
                    for i, child in enumerate(children)
                )
            ),
            45,
        )
        assert sorted(child.returncode for child in children) == [0, 1], results
        assert b"retry when it finishes" in b" ".join(results[loser][0].split())
        output = b"".join(
            prefix + rest for (prefix, _), (rest, _) in zip(observed, results, strict=True)
        )
        assert output.count(b"RENAMED") == 1
        async with fresh_pg.connect() as conn:
            before = await conn.run_sync(snapshot)
        assert (await migrate()).exit_code == 0
        async with fresh_pg.connect() as conn:
            assert await conn.run_sync(snapshot) == before
            assert not await conn.run_sync(lambda c: inspect(c).has_table(LEGACY))
            assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
    finally:
        await stop(children)


@pytest.mark.parametrize("cut", ["before_commit", "after_commit"])
async def test_061_interrupted_postgres_rename_restarts(fresh_pg, cut):
    async with fresh_pg.begin() as conn:
        await conn.run_sync(prior_schema)
    child = await process(cut)
    try:
        child.stdin.write(b"go\n")
        await child.stdin.drain()
        async with asyncio.timeout(20):
            while True:
                line = await child.stdout.readline()
                assert line, "migration exited without reaching the rename cut"
                if line == b"CUT\n":
                    break
    finally:
        await stop([child])
    async with fresh_pg.connect() as conn:
        table = LEGACY if cut == "before_commit" else DEDICATED
        assert await conn.scalar(text(f"SELECT version_num FROM {table}")) == PRIOR
    result = await migrate()
    assert result.exit_code == 0, result.output
    async with fresh_pg.connect() as conn:
        assert not await conn.run_sync(lambda c: inspect(c).has_table(LEGACY))
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
