"""Reader diagnostics and SQLite lock boundaries from the independent review."""

import logging

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import OperationalError
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db import _version_table
from dendrux.db.models import Base
from tests.migrations.test_060_activate import HEAD, MIGRATIONS
from tests.unit.test_061_migrations import DEDICATED, LEGACY, snapshot, stamp
from tests.unit.test_061_migrations import database as database


def test_061_history_read_error_is_not_ownership_conflict(database, monkeypatch, caplog):
    engine, _ = database
    with engine.begin() as conn:
        stamp(conn, DEDICATED, [HEAD])
    failure = OperationalError("SELECT version_num", {}, RuntimeError("connection lost"))

    def fail(*args):
        raise failure

    monkeypatch.setattr(_version_table, "_read_stamp", fail)
    with (
        caplog.at_level(logging.WARNING),
        engine.connect() as conn,
        pytest.raises(OperationalError) as raised,
    ):
        _version_table.read_revision(conn)
    assert raised.value is failure
    assert not any(
        getattr(r, "code", None) == "migration_version_table_conflict" for r in caplog.records
    )


@pytest.mark.parametrize("operation", ["current", "migrate"])
def test_061_sqlite_read_only_has_no_maintenance_lock(database, operation):
    engine, url = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, [HEAD])
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    with engine.connect() as holder:
        holder.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            if operation == "current":
                command.current(cfg)
            else:
                result = CliRunner().invoke(app, ["db", "migrate"])
                assert result.exit_code == 1
                assert "Cannot migrate SQLite" in " ".join(result.output.split())
                assert "Cannot activate" not in result.output
        finally:
            holder.rollback()


def test_061_newer_stamp_advises_upgrade_not_manual_repair(database):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, ["newer-private-revision"])
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "Upgrade Dendrux" in output
    assert "repair the stamps" not in output
    assert "newer-private-revision" not in output


def test_061_empty_dedicated_cannot_hide_legacy_claim(database):
    engine, _ = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, [])
        stamp(conn, LEGACY, [HEAD])
        before = snapshot(conn)
    assert CliRunner().invoke(app, ["db", "migrate"]).exit_code == 1
    with engine.connect() as conn:
        assert snapshot(conn) == before


def test_061_empty_stamp_schema_needs_explicit_repair(database):
    engine, url = database
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
        stamp(conn, DEDICATED, [])
        before = snapshot(conn)
    assert CliRunner().invoke(app, ["db", "check", "--allow-unstamped"]).exit_code == 0
    assert CliRunner().invoke(app, ["db", "migrate"]).exit_code == 1
    with engine.connect() as conn:
        assert snapshot(conn) == before
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    command.stamp(cfg, HEAD)
    assert CliRunner().invoke(app, ["db", "check"]).exit_code == 0
