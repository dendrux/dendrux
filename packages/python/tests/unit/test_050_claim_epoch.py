"""0.5.0 C1: ``agent_runs.claim_epoch`` (INTEGER NOT NULL, server default 0).

The column must reach existing SQLite files through the in-place sync, and
rows written by 0.4.1 workers (which never name the column) must still insert.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import Integer, create_engine

from dendrux import Agent
from dendrux.db._sqlite_sync import _unsafe_reason, sync_sqlite_schema
from dendrux.db.models import AgentRun, Base
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus

_MIGRATIONS = Path(__file__).parents[2] / "src" / "dendrux" / "db" / "migrations" / "versions"


def _columns(path: Path) -> dict[str, tuple[str, int, str | None]]:
    con = sqlite3.connect(path)
    try:
        rows = con.execute("PRAGMA table_info(agent_runs)").fetchall()
    finally:
        con.close()
    return {r[1]: (r[2], r[3], r[4]) for r in rows}


def _insert_041_row(path: Path, run_id: str) -> None:
    """Insert the way a 0.4.1 worker does: every column it knows, none of claim_epoch."""
    con = sqlite3.connect(path)
    try:
        con.execute(
            "INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,"
            " total_input_tokens, total_output_tokens, total_cache_read_tokens,"
            " total_cache_creation_tokens, total_reasoning_tokens, cancel_requested,"
            " pause_requested, created_at, updated_at)"
            " VALUES (?, 'legacy', 'success', 1, 0, 1, 1, 0, 0, 0, 0, 0,"
            " '2026-10-18 10:00:00.000000', '2026-10-18 10:00:00.000000')",
            (run_id,),
        )
        con.commit()
    finally:
        con.close()


def _build_041_file(path: Path) -> None:
    """A SQLite file as 0.4.1 creates it: today's schema minus ``claim_epoch``."""
    eng = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(eng)
    eng.dispose()
    con = sqlite3.connect(path)
    try:
        con.execute("ALTER TABLE agent_runs DROP COLUMN claim_epoch")
        con.commit()
    finally:
        con.close()
    assert "claim_epoch" not in _columns(path)


def _epochs(path: Path) -> dict[str, int]:
    con = sqlite3.connect(path)
    try:
        return dict(con.execute("SELECT id, claim_epoch FROM agent_runs").fetchall())
    finally:
        con.close()


class TestModel:
    def test_column_shape(self) -> None:
        col = AgentRun.__table__.c.claim_epoch
        assert isinstance(col.type, Integer)
        assert col.nullable is False
        assert col.server_default is not None
        assert str(col.server_default.arg.text) == "0"  # type: ignore[attr-defined]
        assert _unsafe_reason(col) is None

    def test_create_all_gives_not_null_default_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        eng = create_engine(f"sqlite:///{db}")
        Base.metadata.create_all(eng)
        eng.dispose()
        col_type, notnull, default = _columns(db)["claim_epoch"]
        assert (col_type, notnull, default) == ("INTEGER", 1, "0")
        _insert_041_row(db, "01FRESH000000000000000000A")
        assert _epochs(db) == {"01FRESH000000000000000000A": 0}


class TestSqliteSyncFrom041:
    def test_sync_adds_column_and_backfills_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "v041.db"
        _build_041_file(db)
        _insert_041_row(db, "01LEGACY00000000000000000A")

        eng = create_engine(f"sqlite:///{db}")
        try:
            with eng.connect() as conn:
                changes = sync_sqlite_schema(conn)
            with eng.connect() as conn:
                assert sync_sqlite_schema(conn) == []  # idempotent
        finally:
            eng.dispose()

        assert any("claim_epoch" in c for c in changes)
        assert _columns(db)["claim_epoch"] == ("INTEGER", 1, "0")
        assert _epochs(db) == {"01LEGACY00000000000000000A": 0}

        # A 0.4.1 worker still writing during a rolling deploy.
        _insert_041_row(db, "01LEGACY00000000000000000B")
        assert _epochs(db)["01LEGACY00000000000000000B"] == 0

    async def test_agent_runs_on_a_041_file(self, tmp_path: Path) -> None:
        db = tmp_path / "v041.db"
        _build_041_file(db)
        agent = Agent(
            strict_tenancy=False,
            provider=MockLLM([LLMResponse(text="done")]),
            prompt="p",
            database_url=f"sqlite+aiosqlite:///{db}",
        )
        try:
            result = await agent.run("hi")
        finally:
            await agent.close()
        assert result.status == RunStatus.SUCCESS
        assert _epochs(db) == {result.run_id: 0}


class TestMigration:
    def test_sweep_revision_is_single_head_above_r1_and_041(self) -> None:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config()
        cfg.set_main_option("script_location", str(_MIGRATIONS.parent))
        script = ScriptDirectory.from_config(cfg)
        assert script.get_heads() == ["f9a0b1c2d3e4"]
        assert script.get_revision("f9a0b1c2d3e4").down_revision == "e8f9a0b1c2d3"
        assert script.get_revision("e8f9a0b1c2d3").down_revision == "d7e8f9a0b1c2"
        r1 = script.get_revision("d7e8f9a0b1c2")
        assert r1 is not None
        assert r1.down_revision == "c6d7e8f9a0b1"
        source = Path(r1.path).read_text()
        assert '"claim_epoch"' in source
        assert "uq_agent_runs_tenant_idempotency_key" in source
        assert "IF NOT EXISTS" in source
