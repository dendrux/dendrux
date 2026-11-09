"""Migration ownership is explicit; application history is never adopted."""

from __future__ import annotations

import logging

import pytest
from sqlalchemy import create_engine, inspect, text
from typer.testing import CliRunner

from dendrux import _codes
from dendrux.cli.main import app
from dendrux.db.models import Base
from tests.migrations.test_060_activate import HEAD, PRIOR, alembic, prior_schema

DEDICATED = "dendrux_alembic_version"
LEGACY = "alembic_version"


def test_061_migration_conflict_code():
    assert "migration_version_table_conflict" in _codes.REGISTRY
    code = _codes.REGISTRY["migration_version_table_conflict"]
    assert code.level == logging.WARNING
    assert code.kind == "migration"


def stamp(conn, table, revisions):
    conn.execute(text(f"CREATE TABLE {table} (version_num VARCHAR(32) NOT NULL)"))
    for revision in revisions:
        conn.execute(text(f"INSERT INTO {table} VALUES (:revision)"), {"revision": revision})


def snapshot(conn):
    names = inspect(conn).get_table_names()
    return {
        name: sorted(tuple(row) for row in conn.execute(text(f'SELECT * FROM "{name}"')))
        for name in names
    }


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "migration.db"
    url = f"sqlite+aiosqlite:///{path}"
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
    monkeypatch.setattr("dendrux.cli.db._find_alembic_ini", lambda: None)
    engine = create_engine(f"sqlite:///{path}")
    try:
        yield engine, url
    finally:
        engine.dispose()


@pytest.mark.parametrize("legacy", [None, [], ["app_one", "app_two"]])
def test_061_dedicated_valid(database, legacy):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, [HEAD])
        if legacy is not None:
            stamp(conn, LEGACY, legacy)
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 0, result.output
    with engine.connect() as conn:
        assert snapshot(conn) == before


def test_061_verified_legacy_renamed_once(database):
    engine, url = database
    with engine.begin() as conn:
        prior_schema(conn)
    # Keep the target at the old revision to prove the rename preserves it.
    alembic(url, "upgrade", PRIOR)
    with engine.connect() as conn:
        assert not inspect(conn).has_table(LEGACY)
        assert conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == PRIOR
        before = snapshot(conn)
    alembic(url, "upgrade", PRIOR)
    with engine.connect() as conn:
        assert snapshot(conn) == before


@pytest.mark.parametrize(
    ("dedicated", "legacy", "schema"),
    [
        ([HEAD], [HEAD], "agent_runs"),
        (None, [HEAD, "app_one"], "agent_runs"),
        (None, [HEAD, PRIOR], "agent_runs"),
        (None, [HEAD, HEAD], "agent_runs"),
        (None, ["unknown"], "agent_runs"),
        (None, [HEAD], None),
        (None, None, "agent_runs"),
        (None, [], "agent_runs"),
        (None, None, "tool_calls"),
        ([HEAD], None, None),
        (["unknown"], None, None),
        ([HEAD, PRIOR], None, "agent_runs"),
        ([HEAD, "app_one"], None, "agent_runs"),
    ],
)
def test_061_conflict_refuses_without_mutation(database, dedicated, legacy, schema, caplog):
    engine, _ = database
    with engine.begin() as conn:
        if schema:
            conn.execute(text(f"CREATE TABLE {schema} (id INTEGER)"))
            conn.execute(text(f"INSERT INTO {schema} VALUES (7)"))
        if dedicated is not None:
            stamp(conn, DEDICATED, dedicated)
        if legacy is not None:
            stamp(conn, LEGACY, legacy)
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 1, result.output
    assert "migration_version_table_conflict" in result.output
    assert any(
        getattr(record, "code", None) == "migration_version_table_conflict"
        for record in caplog.records
    )
    with engine.connect() as conn:
        assert snapshot(conn) == before


@pytest.mark.parametrize("legacy", [None, [], ["app_one", "app_two"]])
def test_061_bootstrap_leaves_app_history(database, legacy):
    from alembic.script import ScriptDirectory

    from tests.migrations.test_060_activate import MIGRATIONS

    engine, url = database
    with engine.begin() as conn:
        if legacy is not None:
            stamp(conn, LEGACY, legacy)
    first = ScriptDirectory(str(MIGRATIONS)).get_base()
    # The first historical revision supports SQLite; the entire old chain does not.
    alembic(url, "upgrade", first)
    with engine.connect() as conn:
        assert conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == first
        assert inspect(conn).has_table(LEGACY) == (legacy is not None)
        if legacy is not None:
            assert sorted(conn.scalars(text(f"SELECT version_num FROM {LEGACY}"))) == legacy


def test_061_version_table_not_in_model_metadata():
    assert DEDICATED not in Base.metadata.tables
    assert LEGACY not in Base.metadata.tables


def test_061_malformed_stamp_reports_database_error(database):
    engine, _ = database
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {DEDICATED} (wrong_column VARCHAR(32))"))
        conn.execute(text(f"INSERT INTO {DEDICATED} VALUES ('private-revision-sentinel')"))
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 1
    assert "migration_version_table_conflict" not in result.output
    assert "Migration failed:" in result.output
    assert "private-revision-sentinel" not in result.output
    with engine.connect() as conn:
        assert snapshot(conn) == before


def test_061_revision_membership_is_case_sensitive(database):
    engine, _ = database
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE agent_runs (id INTEGER)"))
        conn.execute(text(f"CREATE TABLE {LEGACY} (version_num VARCHAR(32) COLLATE NOCASE)"))
        conn.execute(text(f"INSERT INTO {LEGACY} VALUES (:revision)"), {"revision": HEAD.upper()})
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 1
    assert "migration_version_table_conflict" in result.output
    with engine.connect() as conn:
        assert snapshot(conn) == before


def test_061_all_stamp_rows_are_inspected(database):
    engine, _ = database
    with engine.begin() as conn:
        stamp(conn, LEGACY, ["app_one"] * 10000 + [HEAD])
        before = snapshot(conn)
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 1
    assert "migration_version_table_conflict" in result.output
    with engine.connect() as conn:
        assert snapshot(conn) == before


def test_061_legacy_verification_is_not_a_schema_diff(database):
    engine, _ = database
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE agent_runs (id INTEGER)"))
        stamp(conn, LEGACY, [HEAD])
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 0, result.output
    with engine.connect() as conn:
        assert conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
        assert not inspect(conn).has_table(LEGACY)


def test_061_postgres_competitor_refuses_without_retry(monkeypatch):
    from dendrux.cli import db

    monkeypatch.setattr(db, "_POSTGRES_BOOTSTRAP_WAIT_SECONDS", 0.01)

    async def existing(*args):
        return False

    monkeypatch.setattr(db, "_migrate_postgres", existing)
    calls = []

    def busy(*args):
        calls.append(1)
        raise RuntimeError("Another Dendrux database migration is running; retry when it finishes.")

    with pytest.raises(RuntimeError, match="retry when it finishes"):
        db._upgrade_postgres("unused", None, busy)
    assert len(calls) == 1


@pytest.mark.parametrize("cut", ["before_commit", "after_commit"])
def test_061_sqlite_interrupted_rename_restarts(database, monkeypatch, cut):
    from sqlalchemy import Connection, Engine, event

    engine, url = database
    with engine.begin() as conn:
        prior_schema(conn)
    renamed = False
    commit = Connection.commit

    def after_cursor(conn, cursor, statement, parameters, context, executemany):
        nonlocal renamed
        if statement.startswith(f"ALTER TABLE {LEGACY} RENAME TO {DEDICATED}"):
            renamed = True
            if cut == "before_commit":
                raise RuntimeError("interrupted rename")

    def after_commit(conn):
        commit(conn)
        if renamed:
            raise RuntimeError("interrupted rename")

    event.listen(Engine, "after_cursor_execute", after_cursor)
    try:
        with monkeypatch.context() as patch:
            if cut == "after_commit":
                patch.setattr(Connection, "commit", after_commit)
            with pytest.raises(RuntimeError, match="interrupted rename"):
                alembic(url, "upgrade", PRIOR)
    finally:
        event.remove(Engine, "after_cursor_execute", after_cursor)
    with engine.connect() as conn:
        table = LEGACY if cut == "before_commit" else DEDICATED
        assert conn.scalar(text(f"SELECT version_num FROM {table}")) == PRIOR
    alembic(url, "upgrade", PRIOR)
    with engine.connect() as conn:
        assert not inspect(conn).has_table(LEGACY)
        assert conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == PRIOR
