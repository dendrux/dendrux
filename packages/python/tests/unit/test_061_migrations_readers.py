"""Dedicated history readers and untouched automatic SQLite startup."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event, inspect
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db.models import Base
from tests.migrations.test_060_activate import HEAD, MIGRATIONS, PRIOR
from tests.unit.test_061_migrations import DEDICATED, LEGACY, snapshot, stamp
from tests.unit.test_061_migrations import database as database


@pytest.mark.parametrize("existing", [False, True])
async def test_061_auto_sqlite_needs_no_stamp(tmp_path, monkeypatch, existing):
    from dendrux.db import session

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DENDRUX_DATABASE_URL", raising=False)
    await session.reset_engine()
    if existing:
        engine = create_engine("sqlite:///dendrux.db")
        with engine.begin() as conn:
            Base.metadata.create_all(conn)
        engine.dispose()
    statements = []

    def record(conn, cursor, sql, parameters, context, executemany):
        statements.append(sql + repr(parameters))

    event.listen(Engine, "before_cursor_execute", record)
    try:
        engine = await session.get_engine()
        assert session.get_database_url() == "sqlite+aiosqlite:///./dendrux.db"
        assert Path(engine.url.database).resolve() == (tmp_path / "dendrux.db").resolve()
    finally:
        event.remove(Engine, "before_cursor_execute", record)
        await session.reset_engine()
    assert not any("alembic_version" in sql.lower() for sql in statements)
    engine = create_engine("sqlite:///dendrux.db")
    try:
        with engine.connect() as conn:
            names = inspect(conn).get_table_names()
            assert set(Base.metadata.tables) <= set(names)
            assert DEDICATED not in names and LEGACY not in names
    finally:
        engine.dispose()


@pytest.mark.parametrize("action", ["check", "status", "activate"])
def test_061_readers_use_dedicated_history(database, action):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, [HEAD])
        stamp(conn, LEGACY, ["app_one", "app_two"])
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", action])
    assert result.exit_code == 0, result.output
    if action == "status":
        assert f"revision={HEAD} storage_format=2 activate_step=done" in result.output
    with engine.connect() as conn:
        assert snapshot(conn) == before


@pytest.mark.parametrize(
    "action", [["check"], ["check", "--allow-unstamped"], ["status"], ["activate"]]
)
@pytest.mark.parametrize(
    "dedicated,legacy",
    [
        ([HEAD], [HEAD]),
        ([HEAD, PRIOR], None),
        (["private-revision-sentinel"], None),
        (None, [HEAD, "app_one"]),
    ],
)
def test_061_readers_refuse_conflicts_without_mutation(database, action, dedicated, legacy):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        if dedicated is not None:
            stamp(conn, DEDICATED, dedicated)
        if legacy is not None:
            stamp(conn, LEGACY, legacy)
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", *action])
    assert result.exit_code == (1 if action == ["activate"] else 2), result.output
    assert "migration_version_table_conflict" in result.output
    assert "private-revision-sentinel" not in result.output
    with engine.connect() as conn:
        assert snapshot(conn) == before


@pytest.mark.parametrize("app_revisions", [None, [], ["app_one", "app_two"]])
def test_061_allow_unstamped_is_read_only_with_app_history(database, app_revisions):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        if app_revisions is not None:
            stamp(conn, LEGACY, app_revisions)
        before = snapshot(conn)
    runner = CliRunner()
    assert runner.invoke(app, ["db", "check"]).exit_code == 2
    result = runner.invoke(app, ["db", "check", "--allow-unstamped"])
    assert result.exit_code == 0, result.output
    status = runner.invoke(app, ["db", "status"])
    assert status.exit_code == 0, status.output
    assert "revision=unstamped" in status.output
    assert runner.invoke(app, ["db", "migrate"]).exit_code == 1
    with engine.connect() as conn:
        assert snapshot(conn) == before


def test_061_dedicated_old_revision_is_behind(database):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, [PRIOR])
    assert CliRunner().invoke(app, ["db", "check"]).exit_code == 1
    assert CliRunner().invoke(app, ["db", "activate"]).exit_code == 1


def test_061_offline_sql_only_targets_dedicated_history():
    output = StringIO()
    cfg = Config(output_buffer=output)
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", "postgresql+asyncpg://unused/unused")
    command.upgrade(cfg, ScriptDirectory(str(MIGRATIONS)).get_base(), sql=True)
    sql = output.getvalue()
    assert f"CREATE TABLE {DEDICATED}" in sql
    assert f"INSERT INTO {DEDICATED}" in sql
    assert "alembic_version" not in sql.replace(DEDICATED, "")
    assert "RENAME" not in sql


@pytest.mark.parametrize("dialect", ["mysql+asyncmy", "mariadb+asyncmy"])
def test_061_mysql_offline_still_refuses(dialect):
    cfg = Config(output_buffer=StringIO())
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", f"{dialect}://unused/unused")
    with pytest.raises(RuntimeError, match="live connection"):
        command.upgrade(cfg, "head", sql=True)


def test_061_stamp_read_returns_only_packaged_revision(database):
    from dendrux.db._version_table import _read_stamp

    engine, _ = database
    with engine.begin() as conn:
        stamp(conn, LEGACY, ["unrelated-app-revision-" * 1000])
        found = _read_stamp(conn, LEGACY, {HEAD})
        assert found.rows == 1 and found.known == 0
        assert found.revision is None
