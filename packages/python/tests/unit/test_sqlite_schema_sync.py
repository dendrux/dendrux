"""In-place upgrade of SQLite files written by older Dendrux versions."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, func, text
from sqlalchemy import false as sa_false
from sqlalchemy.exc import IntegrityError, OperationalError

from dendrux import Agent
from dendrux.db._sqlite_sync import (
    _add_column,
    _unsafe_reason,
    create_sqlite_schema,
    create_sqlite_schema_sync,
    sync_sqlite_schema,
)
from dendrux.db.models import AgentRun, Base
from dendrux.errors import SqliteSchemaSyncError
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
        with eng.connect() as conn:
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
        with eng.connect() as conn:
            create_sqlite_schema_sync(conn)
        with eng.connect() as conn:
            assert sync_sqlite_schema(conn) == []
        with eng.connect() as conn:
            create_sqlite_schema_sync(conn)
            assert sync_sqlite_schema(conn) == []
        eng.dispose()

    def test_fresh_database_needs_no_sync(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        eng = create_engine(f"sqlite:///{db}")
        with eng.connect() as conn:
            create_sqlite_schema_sync(conn)
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
    def test_two_concurrent_startups(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker() -> None:
            eng = create_engine(f"sqlite:///{db}", connect_args={"timeout": 30})
            try:
                barrier.wait()
                with eng.connect() as conn:
                    create_sqlite_schema_sync(conn)
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


class TestWrapperOnOtherDialects:
    async def test_create_sqlite_schema_is_a_noop_on_postgres(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        monkeypatch.setattr(
            Base.metadata, "create_all", lambda *a, **k: pytest.fail("create_all must not run")
        )
        await create_sqlite_schema(engine)
        engine.connect.assert_not_called()

    def test_sync_wrapper_is_a_noop_on_postgres(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = MagicMock()
        conn.dialect.name = "postgresql"
        monkeypatch.setattr(
            Base.metadata, "create_all", lambda *a, **k: pytest.fail("create_all must not run")
        )
        assert create_sqlite_schema_sync(conn) == []
        conn.exec_driver_sql.assert_not_called()

    def test_sync_after_create_all_is_noop_for_models(self, tmp_path: Path) -> None:
        eng = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
        with eng.connect() as conn:
            Base.metadata.create_all(conn)
            assert sync_sqlite_schema(conn) == []
            assert conn.execute(text("SELECT COUNT(*) FROM run_inbox")).scalar() == 0
        eng.dispose()


def test_unique_violation_still_enforced_on_fresh_db(tmp_path: Path) -> None:
    eng = create_engine(f"sqlite:///{tmp_path / 'f.db'}")
    with eng.connect() as conn:
        create_sqlite_schema_sync(conn)
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


# ---------------------------------------------------------------------------
# Review round: RunStore, global engine, index reconciliation, atomicity
# ---------------------------------------------------------------------------


def _dup_key_insert(path: Path) -> None:
    """Insert two runs sharing a non-NULL idempotency_key; raises if not unique."""
    con = sqlite3.connect(path)
    sql = (
        "INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,"
        " total_input_tokens, total_output_tokens, created_at, updated_at, idempotency_key)"
        " VALUES (?, 'a', 'pending', 0, 0, 0, 0, 'x', 'x', 'dup')"
    )
    try:
        con.execute(sql, ("D1",))
        con.execute(sql, ("D2",))
    finally:
        con.close()


class TestRunStoreUpgrade:
    async def test_run_store_from_database_url_upgrades_legacy_file(self, tmp_path: Path) -> None:
        from dendrux.store import RunStore

        db = tmp_path / "old.db"
        _build_old_db(db)
        async with RunStore.from_database_url(f"sqlite+aiosqlite:///{db}") as store:
            runs = await store.list_runs()
            assert {r.run_id for r in runs} == {r for r, *_ in _LEGACY_ROWS}
            assert await store.get_notes("01OLDRUN0000000000000000A1") == []
        assert "pause_requested" in _columns(db, "agent_runs")
        assert "run_inbox" in _tables(db)

    async def test_run_store_unsafe_file_raises_clear_error_each_time(self, tmp_path: Path) -> None:
        from dendrux.store import RunStore

        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        async with RunStore.from_database_url(f"sqlite+aiosqlite:///{db}") as store:
            for _ in range(2):
                with pytest.raises(SqliteSchemaSyncError, match="agent_runs.agent_name"):
                    await store.list_runs()

    async def test_run_store_initialises_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.store as store_mod
        from dendrux.store import RunStore

        calls = []
        real = store_mod.create_sqlite_schema

        async def counting(engine):  # type: ignore[no-untyped-def]
            calls.append(1)
            await real(engine)

        monkeypatch.setattr(store_mod, "create_sqlite_schema", counting)
        db = tmp_path / "old.db"
        _build_old_db(db)
        async with RunStore.from_database_url(f"sqlite+aiosqlite:///{db}") as store:
            await store.list_runs()
            await store.list_runs()
        assert len(calls) == 1

    async def test_run_store_non_sqlite_url_does_no_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.store as store_mod
        from dendrux.store import RunStore

        async def forbidden(engine):  # type: ignore[no-untyped-def]
            pytest.fail("must not run")

        monkeypatch.setattr(store_mod, "create_sqlite_schema", forbidden)
        store = RunStore.from_database_url("postgresql+asyncpg://u:p@localhost:1/none")
        await store.close()


class TestGlobalEngineInit:
    async def test_failed_init_publishes_nothing_and_retries(self, tmp_path: Path) -> None:
        import dendrux.db.session as session
        from dendrux.db.session import get_engine, reset_engine

        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        url = f"sqlite+aiosqlite:///{db}"
        await reset_engine()
        try:
            for _ in range(2):
                with pytest.raises(SqliteSchemaSyncError):
                    await get_engine(url)
                assert session._engine is None
                assert session._session_factory is None
            con = sqlite3.connect(db)
            con.execute("ALTER TABLE agent_runs ADD COLUMN agent_name VARCHAR(255) DEFAULT 'x'")
            con.commit()
            con.close()
            engine = await get_engine(url)
            assert engine is session._engine
        finally:
            await reset_engine()

    async def test_concurrent_callers_share_one_published_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.db.session as session
        from dendrux.db.session import get_engine, reset_engine

        created = []
        real = session.create_async_engine
        monkeypatch.setattr(
            session, "create_async_engine", lambda *a, **k: (created.append(1), real(*a, **k))[1]
        )
        db = tmp_path / "old.db"
        _build_old_db(db)
        url = f"sqlite+aiosqlite:///{db}"
        await reset_engine()
        try:
            engines = await asyncio.gather(get_engine(url), get_engine(url), get_engine(url))
            assert len({id(e) for e in engines}) == 1
            assert engines[0] is session._engine
            assert "pause_requested" in _columns(db, "agent_runs")
        finally:
            await reset_engine()

    async def test_agent_private_engine_failed_init_is_disposed_and_retried(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        agent = Agent(
            provider=MockLLM([LLMResponse(text="x")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        for _ in range(2):
            with pytest.raises(SqliteSchemaSyncError):
                await agent.run("hi")
            assert agent._private_engine is None
            assert agent._lazy_store is None
        await agent.close()

    async def test_agent_concurrent_first_use_shares_one_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = []
        real = Agent._create_private_engine

        async def counting(self: Agent, url: str):  # type: ignore[no-untyped-def]
            created.append(1)
            return await real(self, url)

        monkeypatch.setattr(Agent, "_create_private_engine", counting)
        db = tmp_path / "old.db"
        _build_old_db(db)
        agent = Agent(
            provider=MockLLM([LLMResponse(text="x")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        try:
            stores = await asyncio.gather(
                agent._resolve_state_store(), agent._resolve_state_store()
            )
            assert stores[0] is stores[1]
        finally:
            await agent.close()


class TestIndexReconciliation:
    def test_interrupted_upgrade_column_added_index_missing(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db, without={"idempotency_key"})
        con = sqlite3.connect(db)
        con.execute("ALTER TABLE agent_runs ADD COLUMN idempotency_key VARCHAR(255)")
        con.commit()
        con.close()
        _sync(db)
        with pytest.raises(sqlite3.IntegrityError):
            _dup_key_insert(db)

    def test_unique_index_name_owned_by_other_table_is_an_error(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db, without={"idempotency_key"})
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE app_owned (k TEXT)")
        con.execute("CREATE UNIQUE INDEX uq_agent_runs_idempotency_key ON app_owned (k)")
        con.commit()
        con.close()
        with pytest.raises(SqliteSchemaSyncError, match="uq_agent_runs_idempotency_key"):
            _sync(db)

    def test_plain_index_name_owned_by_other_table_is_an_error(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE app_owned (k TEXT)")
        con.execute("CREATE INDEX ix_agent_runs_created_at ON app_owned (k)")
        con.commit()
        con.close()
        with pytest.raises(SqliteSchemaSyncError, match="ix_agent_runs_created_at"):
            _sync(db)

    def test_same_name_different_columns_is_an_error(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        con = sqlite3.connect(db)
        con.execute("CREATE INDEX ix_agent_runs_created_at ON agent_runs (agent_name)")
        con.commit()
        con.close()
        with pytest.raises(SqliteSchemaSyncError, match="ix_agent_runs_created_at"):
            _sync(db)

    def test_existing_duplicates_block_unique_index_with_clear_error(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db, without={"idempotency_key"})
        con = sqlite3.connect(db)
        con.execute("ALTER TABLE agent_runs ADD COLUMN idempotency_key VARCHAR(255)")
        con.execute("UPDATE agent_runs SET idempotency_key = 'same'")
        con.commit()
        con.close()
        with pytest.raises(SqliteSchemaSyncError, match="idempotency_key"):
            _sync(db)

    def test_partial_unique_index_does_not_satisfy_uniqueness(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db, without={"idempotency_key"})
        con = sqlite3.connect(db)
        con.execute("ALTER TABLE agent_runs ADD COLUMN idempotency_key VARCHAR(255)")
        con.execute(
            "CREATE UNIQUE INDEX uq_agent_runs_idempotency_key ON agent_runs (idempotency_key)"
            " WHERE status = 'never'"
        )
        con.commit()
        con.close()
        with pytest.raises(SqliteSchemaSyncError, match="uq_agent_runs_idempotency_key"):
            _sync(db)


class TestAtomicityAndPreflight:
    def test_unsafe_column_blocks_before_create_all(self, tmp_path: Path) -> None:
        db = tmp_path / "bad.db"
        _build_old_db(db, without={"agent_name"})
        con = sqlite3.connect(db)
        con.execute("DROP TABLE run_events")
        con.commit()
        con.close()
        before = _tables(db)
        eng = create_engine(f"sqlite:///{db}")
        with (
            pytest.raises(SqliteSchemaSyncError, match="agent_runs.agent_name"),
            eng.connect() as c,
        ):
            create_sqlite_schema_sync(c)
        eng.dispose()
        assert _tables(db) == before

    def test_failure_mid_sync_rolls_everything_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.db._sqlite_sync as mod

        db = tmp_path / "old.db"
        _build_old_db(db)
        before_cols = _columns(db, "agent_runs")
        before_tables = _tables(db)
        before_idx = _indexes(db)

        def boom(*a: object, **k: object) -> None:
            raise SqliteSchemaSyncError("synthetic failure")

        monkeypatch.setattr(mod, "_reconcile_indexes", boom)
        eng = create_engine(f"sqlite:///{db}")
        with pytest.raises(SqliteSchemaSyncError) as exc, eng.connect() as c:
            create_sqlite_schema_sync(c)
        eng.dispose()
        assert _columns(db, "agent_runs") == before_cols
        assert _tables(db) == before_tables
        assert _indexes(db) == before_idx
        assert "pause_requested" in str(exc.value)  # reports what had been applied
        assert "rolled back" in str(exc.value).lower()

    async def test_failure_mid_sync_rolls_back_on_async_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine

        import dendrux.db._sqlite_sync as mod

        db = tmp_path / "old.db"
        _build_old_db(db)
        before_cols = _columns(db, "agent_runs")

        def boom(*a: object, **k: object) -> None:
            raise SqliteSchemaSyncError("synthetic failure")

        monkeypatch.setattr(mod, "_reconcile_indexes", boom)
        eng = create_async_engine(f"sqlite+aiosqlite:///{db}")
        with pytest.raises(SqliteSchemaSyncError):
            await create_sqlite_schema(eng)
        await eng.dispose()
        assert _columns(db, "agent_runs") == before_cols


class TestCreateAllRace:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, side_effect):  # type: ignore[no-untyped-def]
        real = Base.metadata.create_all
        calls: list[int] = []

        def fake(bind, *a, **k):  # type: ignore[no-untyped-def]
            calls.append(1)
            return side_effect(len(calls), real, bind, *a, **k)

        monkeypatch.setattr(Base.metadata, "create_all", fake)
        return calls

    def test_transient_non_race_error_is_not_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def effect(n, real, bind, *a, **k):  # type: ignore[no-untyped-def]
            if n == 1:
                raise OperationalError("CREATE", {}, Exception("disk I/O error"))
            return real(bind, *a, **k)

        calls = self._patch(monkeypatch, effect)
        eng = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
        with pytest.raises(OperationalError, match="disk I/O"), eng.connect() as c:
            create_sqlite_schema_sync(c)
        eng.dispose()
        assert len(calls) == 1

    def test_persistent_error_surfaces_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def effect(n, real, bind, *a, **k):  # type: ignore[no-untyped-def]
            raise OperationalError("CREATE", {}, Exception(f"failure {n}"))

        calls = self._patch(monkeypatch, effect)
        eng = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
        with pytest.raises(OperationalError, match="failure 1"), eng.connect() as c:
            create_sqlite_schema_sync(c)
        eng.dispose()
        assert len(calls) == 1


_NOW = text("CURRENT_TIMESTAMP")


class TestServerDefaultClassification:
    CASES = [
        ("nullable_plain", {"nullable": True}, True),
        ("string_default", {"nullable": False, "server_default": "0"}, True),
        ("text_literal", {"nullable": False, "server_default": text("0")}, True),
        ("text_quoted", {"nullable": False, "server_default": text("'x'")}, True),
        ("false", {"nullable": False, "server_default": sa_false()}, True),
        ("nullable_with_literal", {"nullable": True, "server_default": "7"}, True),
        ("not_null_no_default", {"nullable": False}, False),
        ("current_timestamp", {"nullable": False, "server_default": _NOW}, False),
        ("nullable_current_timestamp", {"nullable": True, "server_default": _NOW}, False),
        ("func_now", {"nullable": False, "server_default": func.now()}, False),
        ("paren_expr", {"nullable": False, "server_default": text("(1 + 1)")}, False),
    ]  # fmt: skip

    @pytest.mark.parametrize(("name", "kwargs", "safe"), CASES, ids=[c[0] for c in CASES])
    def test_classification_matches_sqlite_on_populated_table(
        self, tmp_path: Path, name: str, kwargs: dict, safe: bool
    ) -> None:
        from sqlalchemy import Column, Integer, MetaData, Table
        from sqlalchemy.schema import CreateColumn

        table = Table("t", MetaData(), Column("id", Integer, primary_key=True))
        column = Column("added", Integer, **kwargs)
        table.append_column(column)
        assert (_unsafe_reason(column) is None) is safe

        con = sqlite3.connect(tmp_path / "m.db")
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO t VALUES (1)")
        spec = str(CreateColumn(column).compile(dialect=create_engine("sqlite://").dialect))
        if safe:
            con.execute(f"ALTER TABLE t ADD COLUMN {spec}")
        else:
            with pytest.raises(sqlite3.OperationalError):
                con.execute(f"ALTER TABLE t ADD COLUMN {spec}")
        con.close()


def test_error_is_public_and_aliased() -> None:
    import dendrux.db._sqlite_sync as mod
    import dendrux.errors as errors

    assert "SqliteSchemaSyncError" in errors.__all__
    assert mod.SqliteSchemaSyncError is errors.SqliteSchemaSyncError


# ---------------------------------------------------------------------------
# Re-check round: lock only when needed, lifecycle fences
# ---------------------------------------------------------------------------


def _hold_writer(path: Path) -> sqlite3.Connection:
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    return holder


class TestWriteLockOnlyWhenNeeded:
    def test_current_schema_needs_no_write_lock(self, tmp_path: Path) -> None:
        db = tmp_path / "cur.db"
        eng = create_engine(f"sqlite:///{db}", connect_args={"timeout": 0.05})
        with eng.connect() as conn:
            create_sqlite_schema_sync(conn)
        holder = _hold_writer(db)
        try:
            with eng.connect() as conn:
                assert create_sqlite_schema_sync(conn) == []
                assert sync_sqlite_schema(conn) == []
        finally:
            holder.close()
            eng.dispose()

    def test_busy_writer_gives_actionable_error_when_work_is_needed(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        eng = create_engine(f"sqlite:///{db}", connect_args={"timeout": 0.05})
        holder = _hold_writer(db)
        try:
            with eng.connect() as conn, pytest.raises(SqliteSchemaSyncError, match="busy timeout"):
                create_sqlite_schema_sync(conn)
        finally:
            holder.close()
            eng.dispose()
        assert "pause_requested" not in _columns(db, "agent_runs")

    def test_work_finished_by_another_process_while_waiting_is_not_redone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.db._sqlite_sync as mod

        db = tmp_path / "old.db"
        _build_old_db(db)
        real = mod._begin_immediate

        def upgrade_elsewhere_then_lock(conn):  # type: ignore[no-untyped-def]
            monkeypatch.setattr(mod, "_begin_immediate", real)
            other = create_engine(f"sqlite:///{db}")
            with other.connect() as c2:
                create_sqlite_schema_sync(c2)
            other.dispose()
            real(conn)

        monkeypatch.setattr(mod, "_begin_immediate", upgrade_elsewhere_then_lock)
        eng = create_engine(f"sqlite:///{db}")
        with eng.connect() as conn:
            assert create_sqlite_schema_sync(conn) == []
        eng.dispose()
        assert "pause_requested" in _columns(db, "agent_runs")

    async def test_async_helper_owns_its_transaction(self, tmp_path: Path) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine

        db = tmp_path / "old.db"
        _build_old_db(db)
        eng = create_async_engine(f"sqlite+aiosqlite:///{db}")
        await create_sqlite_schema(eng)
        await eng.dispose()
        assert "pause_requested" in _columns(db, "agent_runs")


class TestLifecycleFences:
    @staticmethod
    def _gate(
        monkeypatch: pytest.MonkeyPatch, module: object, real_name: str = "create_sqlite_schema"
    ):  # type: ignore[no-untyped-def]
        """Make the first schema init block until released."""
        entered, release = asyncio.Event(), asyncio.Event()
        real = getattr(module, real_name)
        first = []

        async def gated(engine):  # type: ignore[no-untyped-def]
            if not first:
                first.append(1)
                entered.set()
                await release.wait()
            await real(engine)

        monkeypatch.setattr(module, real_name, gated)
        return entered, release

    async def test_reset_during_init_discards_the_stale_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.db.session as session
        from dendrux.db.session import get_engine, reset_engine

        created = []
        real_engine = session.create_async_engine
        monkeypatch.setattr(
            session,
            "create_async_engine",
            lambda *a, **k: (created.append(real_engine(*a, **k)), created[-1])[1],
        )
        entered, release = self._gate(monkeypatch, session)
        db = tmp_path / "old.db"
        _build_old_db(db)
        url = f"sqlite+aiosqlite:///{db}"
        await reset_engine()
        try:
            task = asyncio.create_task(get_engine(url))
            await entered.wait()
            await reset_engine()
            release.set()
            engine = await task
            assert len(created) == 2
            assert engine is created[1] is session._engine
        finally:
            await reset_engine()

    async def test_agent_close_during_init_disposes_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.db._sqlite_sync as sync_mod

        entered, release = self._gate(monkeypatch, sync_mod)
        db = tmp_path / "old.db"
        _build_old_db(db)
        agent = Agent(
            provider=MockLLM([LLMResponse(text="x")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        task = asyncio.create_task(agent._resolve_state_store())
        await entered.wait()
        await agent.close()
        release.set()
        with pytest.raises(RuntimeError, match="closed"):
            await task
        assert agent._private_engine is None
        assert agent._lazy_store is None

    async def test_run_store_close_during_init_raises_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        import dendrux.store as store_mod
        from dendrux.store import RunStore

        entered, release = self._gate(monkeypatch, store_mod)
        db = tmp_path / "old.db"
        _build_old_db(db)
        store = RunStore.from_database_url(f"sqlite+aiosqlite:///{db}")
        task = asyncio.create_task(store.list_runs())
        await entered.wait()
        await store.close()
        release.set()
        with pytest.raises(SQLAlchemyError, match="closed"):
            await task


class TestPartialIndexAndNoDemotion:
    def test_partial_ordinary_index_does_not_satisfy_full_model_index(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        _build_old_db(db)
        con = sqlite3.connect(db)
        con.execute(
            "CREATE INDEX ix_agent_runs_created_at ON agent_runs (created_at)"
            " WHERE status = 'never'"
        )
        con.commit()
        con.close()
        with pytest.raises(SqliteSchemaSyncError, match="ix_agent_runs_created_at"):
            _sync(db)
