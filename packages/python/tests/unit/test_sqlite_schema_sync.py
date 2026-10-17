"""In-place upgrade of SQLite files written by older Dendrux versions."""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from dendrux import Agent
from dendrux.db._sqlite_sync import (
    SqliteSchemaSyncError,
    _add_column,
    create_sqlite_schema,
    sync_sqlite_schema,
)
from dendrux.db.models import AgentRun, Base
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus

if TYPE_CHECKING:
    from pathlib import Path

# agent_runs as written by 0.2.0a23: no pause_requested, no total_reasoning_tokens.
_AGENT_RUNS_COLUMNS = {
    "id": "VARCHAR(26) NOT NULL",
    "tenant_id": "VARCHAR(255)",
    "agent_name": "VARCHAR(255) NOT NULL",
    "status": "VARCHAR(50) NOT NULL",
    "input_data": "JSON",
    "output_data": "JSON",
    "iteration_count": "INTEGER NOT NULL",
    "model": "VARCHAR(255)",
    "strategy": "VARCHAR(100)",
    "parent_run_id": "VARCHAR(26)",
    "delegation_level": "INTEGER NOT NULL",
    "retry_of_run_id": "VARCHAR(26)",
    "total_input_tokens": "INTEGER NOT NULL",
    "total_output_tokens": "INTEGER NOT NULL",
    "total_cost_usd": "NUMERIC(10, 6)",
    "total_cache_read_tokens": "INTEGER NOT NULL DEFAULT '0'",
    "total_cache_creation_tokens": "INTEGER NOT NULL DEFAULT '0'",
    "total_reasoning_tokens": "INTEGER NOT NULL DEFAULT '0'",
    "meta": "JSON",
    "pause_data": "JSON",
    "pii_mapping": "JSON",
    "error": "TEXT",
    "last_progress_at": "DATETIME",
    "failure_reason": "VARCHAR(100)",
    "idempotency_key": "VARCHAR(255)",
    "idempotency_fingerprint": "VARCHAR(64)",
    "cancel_requested": "BOOLEAN NOT NULL DEFAULT '0'",
    "created_at": "DATETIME NOT NULL",
    "updated_at": "DATETIME NOT NULL",
}
_LEGACY_ROWS = [
    ("01OLDRUN0000000000000000A1", "success", 2, 10, 5, "2026-09-01 10:00:00.000000"),
    ("01OLDRUN0000000000000000A2", "running", 0, 0, 0, "2026-09-02 10:00:00.000000"),
]


def _agent_runs_ddl(without: set[str] = frozenset()) -> str:  # type: ignore[assignment]
    cols = [f"{n} {t}" for n, t in _AGENT_RUNS_COLUMNS.items() if n not in without]
    cols.append("PRIMARY KEY (id)")
    if "idempotency_key" not in without:
        cols.append("UNIQUE (idempotency_key)")
    return f"CREATE TABLE agent_runs ({', '.join(cols)})"


def _build_old_db(path: Path, *, without: set[str] = frozenset()) -> None:  # type: ignore[assignment]
    """Write a SQLite file in the 0.2.0a23 shape: no run_inbox, no pause_requested.

    Reasoning columns on the per-call tables are absent too.
    """
    without = set(without) | {"total_reasoning_tokens"}
    con = sqlite3.connect(path)
    con.executescript(
        _agent_runs_ddl(without)
        + ";"
        + """
        CREATE TABLE token_usage (
            id VARCHAR(26) NOT NULL, agent_run_id VARCHAR(26) NOT NULL,
            iteration_index INTEGER NOT NULL, model VARCHAR(255), provider VARCHAR(100),
            input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
            cost_usd NUMERIC(10, 6), duration_ms INTEGER,
            cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER,
            meta JSON, created_at DATETIME NOT NULL, PRIMARY KEY (id),
            FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
        );
        CREATE TABLE llm_interactions (
            id VARCHAR(26) NOT NULL, agent_run_id VARCHAR(26) NOT NULL,
            iteration_index INTEGER NOT NULL, model VARCHAR(255), provider VARCHAR(100),
            semantic_request JSON, semantic_response JSON,
            provider_request JSON, provider_response JSON,
            input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
            cost_usd NUMERIC(10, 6), duration_ms INTEGER,
            cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER,
            guardrail_findings JSON, created_at DATETIME NOT NULL, PRIMARY KEY (id),
            FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
        );
        CREATE INDEX ix_agent_runs_status ON agent_runs (status);
        """
    )
    for run_id, status, iters, tin, tout, created in _LEGACY_ROWS:
        row = {
            "id": run_id, "agent_name": "legacy", "status": status, "iteration_count": iters,
            "delegation_level": 0, "total_input_tokens": tin, "total_output_tokens": tout,
            "created_at": created, "updated_at": created,
        }  # fmt: skip
        row = {k: v for k, v in row.items() if k not in without}
        marks = ", ".join("?" for _ in row)
        con.execute(
            f"INSERT INTO agent_runs ({', '.join(row)}) VALUES ({marks})", list(row.values())
        )
    con.commit()
    con.close()
    eng = create_engine(f"sqlite:///{path}")
    keep = {"react_traces", "tool_calls", "run_events"}
    Base.metadata.create_all(eng, tables=[t for t in Base.metadata.sorted_tables if t.name in keep])
    eng.dispose()


def _columns(path: Path, table: str) -> dict[str, tuple[str, int, str | None]]:
    con = sqlite3.connect(path)
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    con.close()
    return {r[1]: (r[2], r[3], r[4]) for r in rows}


def _tables(path: Path) -> set[str]:
    con = sqlite3.connect(path)
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    con.close()
    return {r[0] for r in rows}


def _indexes(path: Path) -> set[str]:
    con = sqlite3.connect(path)
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    con.close()
    return {r[0] for r in rows}


def _sync(path: Path) -> list[str]:
    eng = create_engine(f"sqlite:///{path}")
    try:
        with eng.begin() as conn:
            return sync_sqlite_schema(conn)
    finally:
        eng.dispose()


def _model_columns(table: str) -> set[str]:
    return {c.name for c in Base.metadata.tables[table].columns}


class TestUpgradeEndToEnd:
    async def test_old_file_runs_after_agent_startup(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        assert "pause_requested" not in _columns(db, "agent_runs")
        assert "run_inbox" not in _tables(db)

        agent = Agent(
            provider=MockLLM([LLMResponse(text="done")]),
            prompt="Be brief.",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        try:
            result = await agent.run("hello")
        finally:
            await agent.close()

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "done"
        for table in Base.metadata.tables:
            assert _model_columns(table) <= set(_columns(db, table)), table
        assert "run_inbox" in _tables(db)

    async def test_existing_rows_readable_with_defaults(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        agent = Agent(
            provider=MockLLM([LLMResponse(text="ok")]),
            prompt="Be brief.",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        try:
            await agent.run("hello")
        finally:
            await agent.close()

        eng = create_engine(f"sqlite:///{db}")
        from sqlalchemy.orm import Session

        with Session(eng) as s:
            legacy = s.get(AgentRun, "01OLDRUN0000000000000000A1")
            assert legacy is not None
            assert legacy.pause_requested is False
            assert legacy.cancel_requested is False
            assert legacy.total_reasoning_tokens == 0
            assert legacy.agent_name == "legacy"
            assert legacy.total_input_tokens == 10
        eng.dispose()

    async def test_get_engine_path_upgrades(self, tmp_path: Path) -> None:
        from dendrux.db.session import get_engine, reset_engine

        db = tmp_path / "old.db"
        _build_old_db(db)
        try:
            await get_engine(f"sqlite+aiosqlite:///{db}")
        finally:
            await reset_engine()
        assert "pause_requested" in _columns(db, "agent_runs")
        assert "run_inbox" in _tables(db)

    async def test_sweep_path_upgrades(self, tmp_path: Path) -> None:
        from dendrux.runtime.sweep import sweep

        db = tmp_path / "old.db"
        _build_old_db(db)
        result = await sweep(
            database_url=f"sqlite+aiosqlite:///{db}", stale_running=timedelta(days=1)
        )
        assert len(result.stale_running) >= 1
        assert "pause_requested" in _columns(db, "agent_runs")


class TestSyncBehaviour:
    def test_adds_missing_columns_with_defaults(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        changes = _sync(db)

        agent_runs = _columns(db, "agent_runs")
        assert agent_runs["pause_requested"][1] == 1  # NOT NULL
        assert agent_runs["pause_requested"][2] is not None  # has default
        assert "added column agent_runs.pause_requested" in changes
        assert "added column token_usage.reasoning_tokens" in changes
        assert "reasoning_tokens" in _columns(db, "token_usage")
        assert "reasoning_tokens" in _columns(db, "llm_interactions")
        con = sqlite3.connect(db)
        values = con.execute("SELECT pause_requested FROM agent_runs").fetchall()
        con.close()
        assert {v[0] for v in values} == {0}

    def test_creates_missing_indexes(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        assert "ix_agent_runs_created_at" not in _indexes(db)
        changes = _sync(db)
        assert "ix_agent_runs_created_at" in _indexes(db)
        assert "ix_agent_runs_tenant_id" in _indexes(db)
        assert "created index ix_agent_runs_created_at" in changes

    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        eng = create_engine(f"sqlite:///{db}")
        with eng.begin() as conn:
            create_sqlite_schema(conn)
        with eng.begin() as conn:
            assert sync_sqlite_schema(conn) == []
        with eng.begin() as conn:
            create_sqlite_schema(conn)
            assert sync_sqlite_schema(conn) == []
        eng.dispose()

    def test_fresh_database_needs_no_sync(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        eng = create_engine(f"sqlite:///{db}")
        with eng.begin() as conn:
            create_sqlite_schema(conn)
            assert sync_sqlite_schema(conn) == []
        eng.dispose()

    def test_logs_changes_at_info(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        with caplog.at_level(logging.INFO, logger="dendrux.db._sqlite_sync"):
            _sync(db)
        assert any("agent_runs.pause_requested" in r.getMessage() for r in caplog.records)

    def test_existing_data_and_unknown_objects_untouched(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE app_owned (id INTEGER PRIMARY KEY)")
        con.execute("ALTER TABLE agent_runs ADD COLUMN app_extra TEXT")
        con.commit()
        con.close()

        _sync(db)

        assert "app_extra" in _columns(db, "agent_runs")
        assert set(_columns(db, "app_owned")) == {"id"}
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 2
        con.close()

    def test_adds_unique_column_as_unique_index_and_fk(self, tmp_path: Path) -> None:
        db = tmp_path / "ancient.db"
        _build_old_db(db, without={"idempotency_key", "retry_of_run_id"})
        assert "idempotency_key" not in _columns(db, "agent_runs")

        _sync(db)

        con = sqlite3.connect(db)
        con.execute("PRAGMA foreign_keys = ON")
        fks = con.execute("PRAGMA foreign_key_list(agent_runs)").fetchall()
        assert any(fk[2] == "agent_runs" and fk[3] == "retry_of_run_id" for fk in fks)
        base = (
            "INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,"
            " total_input_tokens, total_output_tokens, created_at, updated_at, idempotency_key)"
            " VALUES (?, 'a', 'pending', 0, 0, 0, 0, 'x', 'x', 'dup')"
        )
        con.execute(base, ("R1",))
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(base, ("R2",))
        con.close()

    def test_unsafe_column_raises_clear_error(self, tmp_path: Path) -> None:
        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        with pytest.raises(SqliteSchemaSyncError) as exc:
            _sync(db)
        assert "agent_runs.agent_name" in str(exc.value)
        assert "recreate" in str(exc.value).lower()

    def test_unsafe_column_makes_no_changes(self, tmp_path: Path) -> None:
        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        before = {t: _columns(db, t) for t in ("agent_runs", "token_usage")}
        before_indexes = _indexes(db)
        with pytest.raises(SqliteSchemaSyncError):
            _sync(db)
        assert {t: _columns(db, t) for t in before} == before
        assert _indexes(db) == before_indexes

    async def test_agent_startup_surfaces_unsafe_error(self, tmp_path: Path) -> None:
        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        agent = Agent(
            provider=MockLLM([LLMResponse(text="x")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        try:
            with pytest.raises(SqliteSchemaSyncError, match="agent_runs.agent_name"):
                await agent.run("hi")
        finally:
            await agent.close()


class TestConcurrency:
    def test_duplicate_column_race_is_tolerated(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        column = Base.metadata.tables["agent_runs"].columns["pause_requested"]
        eng = create_engine(f"sqlite:///{db}")
        with eng.begin() as conn:
            assert _add_column(conn, Base.metadata.tables["agent_runs"], column) is True
        with eng.begin() as conn:
            # Another process won the race: ALTER fails with "duplicate column".
            assert _add_column(conn, Base.metadata.tables["agent_runs"], column) is False
        eng.dispose()

    def test_two_concurrent_startups(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker() -> None:
            eng = create_engine(f"sqlite:///{db}", connect_args={"timeout": 30})
            try:
                barrier.wait()
                with eng.begin() as conn:
                    create_sqlite_schema(conn)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                eng.dispose()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for table in Base.metadata.tables:
            assert _model_columns(table) <= set(_columns(db, table)), table
        assert _sync(db) == []

    def test_non_duplicate_alter_failure_is_wrapped(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        table = Base.metadata.tables["agent_runs"]
        column = table.columns["pause_requested"]
        conn = MagicMock()
        conn.dialect = create_engine("sqlite://").dialect
        conn.exec_driver_sql.side_effect = [
            OperationalError("ALTER", {}, Exception("disk I/O error")),
            MagicMock(fetchall=lambda: []),
        ]
        with pytest.raises(SqliteSchemaSyncError, match="agent_runs.pause_requested"):
            _add_column(conn, table, column)


class TestOtherBackends:
    def test_non_sqlite_dialect_is_untouched(self) -> None:
        conn = MagicMock()
        conn.dialect.name = "postgresql"
        assert sync_sqlite_schema(conn) == []
        conn.exec_driver_sql.assert_not_called()
        conn.execute.assert_not_called()

    def test_sync_after_create_all_is_noop_for_models(self, tmp_path: Path) -> None:
        eng = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
        with eng.begin() as conn:
            Base.metadata.create_all(conn)
            assert sync_sqlite_schema(conn) == []
            assert conn.execute(text("SELECT COUNT(*) FROM run_inbox")).scalar() == 0
        eng.dispose()


def test_unique_violation_still_enforced_on_fresh_db(tmp_path: Path) -> None:
    eng = create_engine(f"sqlite:///{tmp_path / 'f.db'}")
    with eng.begin() as conn:
        create_sqlite_schema(conn)
    with pytest.raises(IntegrityError), eng.begin() as conn:
        for rid in ("A", "B"):
            conn.execute(
                text(
                    "INSERT INTO agent_runs (id, agent_name, status, iteration_count,"
                    " delegation_level, total_input_tokens, total_output_tokens,"
                    " total_cache_read_tokens, total_cache_creation_tokens,"
                    " total_reasoning_tokens, cancel_requested, pause_requested,"
                    " created_at, updated_at, idempotency_key) VALUES (:i, 'a', 'p', 0, 0,"
                    " 0, 0, 0, 0, 0, 0, 0, 'x', 'x', 'k')"
                ),
                {"i": rid},
            )
    eng.dispose()
