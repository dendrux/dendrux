"""0.5.0 T1: schema — the R1 model indexes, SQLite sync of expression/partial
indexes and the legacy UNIQUE, the Postgres index check and ``dendrux db check``.

Postgres-only behaviour (``pg_indexes``, the migration itself) is covered in
``tests/integration/test_050_t1_r1_migration.py``.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import create_engine
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db import _sqlite_sync
from dendrux.db._schema_check import compare_indexes, required_indexes
from dendrux.db._sqlite_sync import create_sqlite_schema_sync, sync_sqlite_schema
from dendrux.db.models import TENANT_IDEMPOTENCY_INDEX, AgentRun, Base
from tests.unit.test_sqlite_schema_sync import _build_old_db

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _sync(path: Path) -> list[str]:
    eng = create_engine(f"sqlite:///{path}")
    try:
        with eng.connect() as conn:
            return sync_sqlite_schema(conn)
    finally:
        eng.dispose()


def _index_sql(path: Path) -> dict[str, str]:
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT name, sql FROM sqlite_master WHERE type = 'index'").fetchall()
    finally:
        con.close()
    return {name: sql for name, sql in rows}


class TestModels:
    def test_idempotency_key_is_not_globally_unique(self) -> None:
        column = AgentRun.__table__.c.idempotency_key
        assert not column.unique
        from sqlalchemy import UniqueConstraint

        assert not any(
            isinstance(c, UniqueConstraint) and "idempotency_key" in c.columns
            for c in AgentRun.__table__.constraints
        )

    def test_tenant_scoped_unique_index(self) -> None:
        index = next(i for i in AgentRun.__table__.indexes if i.name == TENANT_IDEMPOTENCY_INDEX)
        assert index.unique
        from sqlalchemy.dialects import postgresql, sqlite
        from sqlalchemy.schema import CreateIndex

        for dialect in (postgresql.dialect(), sqlite.dialect()):
            ddl = str(CreateIndex(index).compile(dialect=dialect))
            assert "coalesce(tenant_id, '')" in ddl
            assert "WHERE idempotency_key IS NOT NULL" in ddl

    def test_tenant_created_and_retry_indexes(self) -> None:
        names = {i.name for i in AgentRun.__table__.indexes}
        assert {"ix_agent_runs_tenant_id_created_at", "ix_agent_runs_retry_of_run_id"} <= names


class TestSqliteSync:
    def test_new_file_gets_every_index_and_is_stable(self, tmp_path: Path) -> None:
        db = tmp_path / "new.db"
        eng = create_engine(f"sqlite:///{db}")
        with eng.connect() as conn:
            create_sqlite_schema_sync(conn)
        eng.dispose()
        assert _sync(db) == []
        indexes = _index_sql(db)
        assert "coalesce" in indexes[TENANT_IDEMPOTENCY_INDEX]
        assert "DESC" in indexes["ix_agent_runs_tenant_id_created_at"]

    def test_legacy_file_gains_the_new_indexes_and_keeps_its_unique(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_sqlite_sync, "_LEGACY_WARNED", set())
        db = tmp_path / "old.db"
        _build_old_db(db)
        with caplog.at_level(logging.WARNING):
            changes = _sync(db)
        assert f"created index {TENANT_IDEMPOTENCY_INDEX}" in changes
        assert "created index ix_agent_runs_tenant_id_created_at" in changes
        assert caplog.text.count("global UNIQUE on agent_runs.idempotency_key") == 1
        con = sqlite3.connect(db)
        table_sql = con.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'agent_runs'"
        ).fetchone()[0]
        con.close()
        assert "UNIQUE (idempotency_key)" in table_sql  # no table rebuild
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert _sync(db) == []
        assert "global UNIQUE" not in caplog.text  # one warning per file

    def test_existing_index_with_matching_sql_is_accepted(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        con = sqlite3.connect(db)
        con.execute(
            'CREATE UNIQUE INDEX "uq_agent_runs_tenant_idempotency_key" ON agent_runs '
            "(COALESCE(tenant_id,''),idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
        con.commit()
        con.close()
        assert f"created index {TENANT_IDEMPOTENCY_INDEX}" not in _sync(db)

    def test_normalize_sql(self) -> None:
        a = _sqlite_sync._normalize_sql(
            'CREATE INDEX "ix" ON t ( a ,  b DESC )   WHERE  x IS NOT NULL'
        )
        b = _sqlite_sync._normalize_sql("create index ix on t(a,b desc) where x is not null")
        assert a == b


class TestIndexCheck:
    def test_required_indexes_cover_models(self) -> None:
        wanted = required_indexes()
        assert wanted[TENANT_IDEMPOTENCY_INDEX] == "agent_runs"
        assert wanted["uq_dendrux_blob_refs_owner"] == "dendrux_blob_refs"
        assert wanted["ix_run_events_agent_run_id_seq"] == "run_events"

    def test_compare_indexes_reports_only_existing_tables(self) -> None:
        present = set(required_indexes()) - {TENANT_IDEMPOTENCY_INDEX, "ix_dendrux_blobs_gc"}
        missing = compare_indexes(present, {"agent_runs"})
        assert missing == [f"agent_runs.{TENANT_IDEMPOTENCY_INDEX} (index)"]
        assert compare_indexes(set(required_indexes()), Base.metadata.tables) == []


class TestDbCheckCli:
    @pytest.fixture
    def db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        path = tmp_path / "check.db"
        monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{path}")
        return path

    def _create(self, path: Path) -> None:
        eng = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(eng)
        eng.dispose()

    def _stamp(self, path: Path, revision: str) -> None:
        eng = create_engine(f"sqlite:///{path}")
        with eng.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)"
            )
            conn.exec_driver_sql("DELETE FROM alembic_version")
            conn.exec_driver_sql(f"INSERT INTO alembic_version VALUES ('{revision}')")
        eng.dispose()

    def _invoke(self, *args: str) -> Any:
        return runner.invoke(app, ["db", "check", *args])

    def test_unstamped_is_unknown(self, db: Path) -> None:
        self._create(db)
        result = self._invoke()
        assert result.exit_code == 2, result.output
        assert "--allow-unstamped" in result.output

    def test_allow_unstamped_checks_the_schema(self, db: Path) -> None:
        self._create(db)
        assert self._invoke("--allow-unstamped").exit_code == 0

    def test_allow_unstamped_reports_missing_objects(self, db: Path) -> None:
        _build_old_db(db)
        result = self._invoke("--allow-unstamped")
        assert result.exit_code == 1
        assert "agent_runs.claim_epoch" in result.output

    def test_head_is_current(self, db: Path) -> None:
        self._create(db)
        self._stamp(db, "d7e8f9a0b1c2")
        result = self._invoke()
        assert result.exit_code == 0, result.output

    def test_older_revision_is_behind(self, db: Path) -> None:
        self._create(db)
        self._stamp(db, "c6d7e8f9a0b1")
        result = self._invoke()
        assert result.exit_code == 1
        assert "dendrux db migrate" in result.output

    def test_unknown_revision(self, db: Path) -> None:
        self._create(db)
        self._stamp(db, "ffffffffffff")
        assert self._invoke().exit_code == 2

    def test_output_masks_the_password(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv(
            "DENDRUX_DATABASE_URL", "postgresql+asyncpg://u:hunter2@127.0.0.1:1/none"
        )
        result = self._invoke()
        assert result.exit_code == 2
        assert "hunter2" not in result.output
