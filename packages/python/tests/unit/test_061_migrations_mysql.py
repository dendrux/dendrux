"""MySQL-family migration SQL, without requiring or connecting to a server."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects.mysql import dialect
from sqlalchemy.dialects.mysql.mariadb import MariaDBDialect

from dendrux.db._dialects import mysql
from tests.unit.test_061_migrations import DEDICATED, HEAD, LEGACY


@pytest.fixture(params=[dialect, MariaDBDialect], ids=["mysql", "mariadb"])
def connection(request):
    conn = MagicMock()
    conn.dialect = request.param()
    return conn


def test_061_mysql_bootstrap_sql_owns_dedicated_table(connection, monkeypatch):
    from dendrux.db import _migration_ddl

    statements = []
    monkeypatch.setattr(
        _migration_ddl,
        "bounded_step",
        lambda conn, statement, **kw: statements.append(statement),
    )
    monkeypatch.setattr(mysql, "read_catalog", lambda *args: mysql.Catalog({}, {}))
    monkeypatch.setattr(mysql, "base_tables", lambda *args: set())
    monkeypatch.setattr(mysql, "check", lambda *args, **kw: mysql.SchemaReport(set(), [], [], []))
    monkeypatch.setattr(
        mysql,
        "read_settings",
        lambda *args: {
            "storage_format": "2",
            "activate_step": "done",
            "bootstrap_in_progress": HEAD,
        },
    )
    mysql._bootstrap(connection, HEAD)
    statements.extend(str(call.args[0]) for call in connection.execute.call_args_list)
    assert any(statement.startswith(f"CREATE TABLE {DEDICATED} ") for statement in statements)
    assert f"SELECT version_num FROM {DEDICATED}" in statements
    assert f"INSERT INTO {DEDICATED} (version_num) VALUES (:head)" in statements
    assert not any(
        f"TABLE {LEGACY} " in statement or f"INTO {LEGACY} " in statement
        for statement in statements
    )


def test_061_mysql_rename_sql(connection, monkeypatch):
    from dendrux.db import _version_table

    statements = []
    monkeypatch.setattr(
        mysql, "bounded_execute", lambda conn, statement, **kw: statements.append(statement)
    )
    _version_table._rename(connection)
    statements.extend(str(call.args[0]) for call in connection.execute.call_args_list)
    assert f"RENAME TABLE {LEGACY} TO {DEDICATED}" in statements


def test_061_mysql_inspects_all_rows_with_portable_sql(connection):
    from dendrux.db import _version_table

    connection.execute.return_value.one.return_value = (10001, 1, HEAD)
    found = _version_table._read_stamp(connection, LEGACY, {HEAD})
    assert found.rows == 10001 and found.known == 1 and not found.valid
    statement = connection.execute.call_args.args[0]
    sql = str(statement.compile(dialect=connection.dialect))
    assert "count(*)" in sql and "CASE WHEN" in sql
    assert "FILTER" not in sql and "LIMIT" not in sql
    assert "CAST(alembic_version.version_num AS BINARY)" in sql


@pytest.mark.parametrize("empty_stamp", [False, True])
def test_061_mysql_interrupted_bootstrap_still_resumes(connection, monkeypatch, empty_stamp):
    from dendrux.db import _version_table

    locked = False
    called = []

    @contextmanager
    def lock(conn):
        nonlocal locked
        locked = True
        try:
            yield
        finally:
            locked = False

    def prepare(conn, known, *, resumable):
        assert locked
        dedicated = _version_table._Stamp(0, 0) if empty_stamp else None
        assert (
            _version_table._classify(
                dedicated,
                None,
                has_schema=True,
                has_agent_runs=False,
                resumable=resumable,
            )
            == "bootstrap"
        )

    monkeypatch.setattr(mysql, "maintenance_lock", lock)
    monkeypatch.setattr(mysql, "base_tables", lambda *args: {"dendrux_settings"})
    monkeypatch.setattr(mysql, "read_settings", lambda *args: {"bootstrap_in_progress": HEAD})
    monkeypatch.setattr(mysql, "prepare_version_table", prepare)
    monkeypatch.setattr(mysql, "_bootstrap", lambda *args: called.append("bootstrap"))
    assert (
        mysql.migrate(connection, head=HEAD, upgradeable=set(), upgrade=lambda: None) == "bootstrap"
    )
    assert called == ["bootstrap"]
    assert not locked


@pytest.mark.parametrize("malformed", [False, True])
def test_061_mysql_invalid_bootstrap_claim_refuses_safely(connection, monkeypatch, malformed):
    from sqlalchemy.exc import DBAPIError

    def settings(conn):
        if malformed:
            raise DBAPIError("SELECT", {}, Exception("private-marker-sentinel"))
        return {"bootstrap_in_progress": "private-marker-sentinel"}

    monkeypatch.setattr(mysql, "maintenance_lock", lambda conn: nullcontext())
    monkeypatch.setattr(mysql, "base_tables", lambda *args: {"dendrux_settings"})
    monkeypatch.setattr(mysql, "read_settings", settings)
    with pytest.raises(RuntimeError, match="migration_version_table_conflict") as caught:
        mysql.migrate(connection, head=HEAD, upgradeable=set(), upgrade=lambda: None)
    assert "private-marker-sentinel" not in str(caught.value)
    connection.execute.assert_not_called()


def test_061_mysql_readers_refuse_unknown_bootstrap_marker(connection, monkeypatch):
    from dendrux.db import _version_table

    monkeypatch.setattr(
        _version_table, "_inspect_history", lambda *args: ({"dendrux_settings"}, None, None)
    )
    monkeypatch.setattr(
        mysql, "read_settings", lambda conn: {"bootstrap_in_progress": "private-marker-sentinel"}
    )
    with pytest.raises(RuntimeError, match="migration_version_table_conflict") as caught:
        _version_table.read_revision(connection)
    assert "private-marker-sentinel" not in str(caught.value)
    connection.commit.assert_not_called()


@pytest.mark.parametrize("stamped", [False, True])
def test_061_mysql_readers_preserve_interrupted_bootstrap(connection, monkeypatch, stamped):
    from dendrux.db import _version_table

    names = {"dendrux_settings", "agent_runs", DEDICATED}
    revision = _version_table._Stamp(1, 1, HEAD) if stamped else _version_table._Stamp(0, 0)
    monkeypatch.setattr(_version_table, "_inspect_history", lambda *args: (names, revision, None))
    monkeypatch.setattr(mysql, "read_settings", lambda conn: {"bootstrap_in_progress": HEAD})
    assert _version_table.read_revision(connection) == (HEAD if stamped else None)
    connection.commit.assert_not_called()


@pytest.mark.parametrize("legacy_claim", [False, True])
def test_061_mysql_explicit_stamp_uses_locked_repair(connection, monkeypatch, legacy_claim):
    from unittest.mock import AsyncMock

    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext

    from dendrux.db import _version_table
    from tests.migrations.test_060_activate import MIGRATIONS

    locked = False
    ran = []

    @contextmanager
    def lock(conn):
        nonlocal locked
        assert conn is connection
        locked = True
        try:
            yield
        finally:
            locked = False

    def history(conn, known):
        assert locked
        assert HEAD in known
        legacy = _version_table._Stamp(1, 1, HEAD) if legacy_claim else None
        return {"agent_runs"}, None, legacy

    def run(migration_context, **kw):
        assert locked
        assert migration_context.opts["version_table"] == DEDICATED
        ran.append(True)

    engine = MagicMock()
    async_connection = AsyncMock()
    async_connection.run_sync.side_effect = lambda fn: fn(connection)
    engine.connect.return_value.__aenter__.return_value = async_connection
    engine.dispose = AsyncMock()
    monkeypatch.setattr("sqlalchemy.ext.asyncio.async_engine_from_config", lambda *a, **kw: engine)
    monkeypatch.setattr(mysql, "maintenance_lock", lock)
    monkeypatch.setattr(mysql, "migrate", lambda *a, **kw: pytest.fail("stamp must not bootstrap"))
    monkeypatch.setattr(_version_table, "_inspect_history", history)
    monkeypatch.setattr(MigrationContext, "run_migrations", run)
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", "mysql+asyncmy://unused/unused")
    if legacy_claim:
        with pytest.raises(_version_table.MigrationVersionTableConflictError):
            command.stamp(cfg, HEAD)
        assert not ran
    else:
        command.stamp(cfg, HEAD)
        assert ran == [True]
    assert not locked
    engine.dispose.assert_awaited_once()
