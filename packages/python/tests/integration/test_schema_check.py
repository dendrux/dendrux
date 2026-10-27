"""Schema-currency check on Postgres: a database behind the models fails fast.

Uses a temporary schema inside ``DENDRUX_TEST_PG_URL`` for deliberately
out-of-date schemas; it never connects to a second database.
Skipped when that variable is unset. SQLite is covered at the end: it upgrades
itself and must never raise ``SchemaOutOfDateError``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from dendrux import Agent
from dendrux.db import session as db_session
from dendrux.db.models import Base
from dendrux.errors import PersistenceError, SchemaNotInitializedError, SchemaOutOfDateError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.sweep import sweep
from dendrux.store import RunStore
from dendrux.types import LLMResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from tests.integration._migration_schema import (
    migration_schema,
)

pytestmark = pytest.mark.asyncio

PRIOR = "a4b5c6d7e8f9"
_MIGRATIONS = Path(__file__).resolve().parents[2] / "src" / "dendrux" / "db" / "migrations"
# 0.2.0a23 shape: no run control (0.3.0) and no image blob tables (0.4.0).
_MISSING = {"agent_runs.pause_requested", "run_inbox", "dendrux_blobs", "dendrux_blob_refs"}
_MISSING_TABLES = {"run_inbox", "dendrux_blobs", "dendrux_blob_refs"}
# A database migrated to an earlier revision also lacks 0.5.0's R1: claim_epoch
# and the two agent_runs indexes (the index check runs on Postgres only).
_R1_INDEXES = {
    "agent_runs.uq_agent_runs_tenant_idempotency_key (index)",
    "agent_runs.ix_agent_runs_tenant_id_created_at (index)",
}
_SWEEP_INDEXES = {
    "agent_runs.ix_agent_runs_sweep_running (index)",
    "agent_runs.ix_agent_runs_sweep_waiting (index)",
    "run_events.ix_run_events_started (index)",
}
_MISSING_AT_PRIOR = (
    _MISSING
    | {"agent_runs.claim_epoch", "agent_runs.status_changed_at"}
    | _R1_INDEXES
    | _SWEEP_INDEXES
)
RUN_CONTROL = "b5c6d7e8f9a0"  # 0.3.0 head: everything but the 0.4.0 image blob tables


def _alembic(url: str, rev: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, rev)


@pytest_asyncio.fixture
async def pg_url(monkeypatch: pytest.MonkeyPatch, pg_test_url: str) -> AsyncIterator[str]:
    url = pg_test_url
    monkeypatch.delenv("DENDRUX_SKIP_SCHEMA_CHECK", raising=False)
    async with migration_schema(url, "schema_check"):
        try:
            yield url
        finally:
            await db_session.reset_engine()


def _admin(url: str) -> AsyncEngine:
    return create_async_engine(url, poolclass=NullPool)


async def _build_previous_schema(url: str) -> None:
    """create_all, then drop what 0.3.0 added: the 0.2.0a23 shape."""
    engine = _admin(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("DROP TABLE dendrux_blob_refs"))
        await conn.execute(text("DROP TABLE dendrux_blobs"))
        await conn.execute(text("DROP TABLE run_inbox"))
        await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN pause_requested"))
    await engine.dispose()


async def _build_current_schema(url: str) -> None:
    engine = _admin(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def _count_runs(url: str) -> int:
    engine = _admin(url)
    try:
        async with engine.connect() as conn:
            return (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one()
    finally:
        await engine.dispose()


def _agent(url: str | None = None) -> Agent:
    kwargs: dict[str, Any] = {"database_url": url} if url else {}
    return Agent(
        name="schema-check",
        prompt="Test.",
        provider=MockLLM([LLMResponse(text="ok")] * 3),
        **kwargs,
    )


async def test_agent_run_on_previous_schema_raises_clear_error(pg_url: str) -> None:
    await _build_previous_schema(pg_url)
    agent = _agent(pg_url)
    try:
        with pytest.raises(SchemaOutOfDateError) as info:
            await agent.run("hi")
    finally:
        await agent.close()
    err = info.value
    assert set(err.missing) == _MISSING
    assert err.missing_columns == ("agent_runs.pause_requested",)
    assert set(err.missing_tables) == _MISSING_TABLES
    assert "dendrux db migrate" in str(err)
    assert await _count_runs(pg_url) == 0


async def test_agent_recovers_after_alembic_upgrade(pg_url: str) -> None:
    await asyncio.to_thread(_alembic, pg_url, PRIOR)
    agent = _agent(pg_url)
    try:
        with pytest.raises(SchemaOutOfDateError) as info:
            await agent.run("hi")
        assert set(info.value.missing) == _MISSING_AT_PRIOR

        await asyncio.to_thread(_alembic, pg_url, "head")
        # A failed init is not cached: the same agent works after migrating.
        result = await agent.run("hi")
        assert result.answer == "ok"
    finally:
        await agent.close()
    assert await _count_runs(pg_url) == 1


async def test_current_schema_passes(pg_url: str) -> None:
    await _build_current_schema(pg_url)
    agent = _agent(pg_url)
    try:
        assert (await agent.run("hi")).answer == "ok"
    finally:
        await agent.close()


async def test_extra_table_column_and_index_are_fine(pg_url: str) -> None:
    await _build_current_schema(pg_url)
    engine = _admin(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE agent_runs ADD COLUMN app_owned text"))
        await conn.execute(text("CREATE TABLE app_owned (id int primary key)"))
        await conn.execute(text("CREATE INDEX app_owned_ix ON agent_runs (agent_name)"))
    await engine.dispose()
    agent = _agent(pg_url)
    try:
        assert (await agent.run("hi")).answer == "ok"
    finally:
        await agent.close()


async def test_missing_model_index_is_reported(pg_url: str) -> None:
    """0.5.0: named model indexes are required on Postgres (an un-migrated DB fails)."""
    await _build_current_schema(pg_url)
    engine = _admin(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("DROP INDEX ix_run_inbox_run_pending"))
    await engine.dispose()
    agent = _agent(pg_url)
    try:
        with pytest.raises(SchemaOutOfDateError) as info:
            await agent.run("hi")
    finally:
        await agent.close()
    assert info.value.missing == ("run_inbox.ix_run_inbox_run_pending (index)",)
    assert await _count_runs(pg_url) == 0


async def test_skip_env_var_bypasses_the_check(
    pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _build_previous_schema(pg_url)
    monkeypatch.setenv("DENDRUX_SKIP_SCHEMA_CHECK", "1")
    agent = _agent(pg_url)
    try:
        with pytest.raises(PersistenceError) as info:
            await agent.run("hi")
    finally:
        await agent.close()
    assert not isinstance(info.value, SchemaOutOfDateError)


async def test_empty_database_still_says_not_initialized(pg_url: str) -> None:
    agent = _agent(pg_url)
    try:
        with pytest.raises(SchemaNotInitializedError):
            await agent.run("hi")
    finally:
        await agent.close()


async def test_env_var_path_does_not_cache_a_failed_engine(
    pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _build_previous_schema(pg_url)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", pg_url)
    await db_session.reset_engine()
    agent = _agent()
    try:
        with pytest.raises(SchemaOutOfDateError):
            await agent.run("hi")
        assert db_session._engine is None

        engine = _admin(pg_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
        await _build_current_schema(pg_url)
        assert (await agent.run("hi")).answer == "ok"
    finally:
        await agent.close()
        await db_session.reset_engine()


async def test_run_store_is_not_checked_and_reads_view_backed_tables(pg_url: str) -> None:
    await _build_current_schema(pg_url)
    engine = _admin(pg_url)
    async with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            await conn.execute(text(f"ALTER TABLE {table.name} RENAME TO {table.name}_base"))
        for table in Base.metadata.sorted_tables:
            await conn.execute(text(f"CREATE VIEW {table.name} AS SELECT * FROM {table.name}_base"))
    await engine.dispose()
    async with RunStore.from_database_url(pg_url) as store:
        assert await store.list_runs() == []


async def test_wrong_column_type_is_not_detected(pg_url: str) -> None:
    await _build_current_schema(pg_url)
    engine = _admin(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN pause_requested"))
        await conn.execute(text("ALTER TABLE agent_runs ADD COLUMN pause_requested text"))
    await engine.dispose()
    from dendrux.db._schema_check import ensure_schema_current

    check = create_async_engine(pg_url, poolclass=NullPool)
    try:
        await ensure_schema_current(check)  # presence only: types are not compared
    finally:
        await check.dispose()


async def test_quoted_uppercase_tables_are_reported_missing(pg_url: str) -> None:
    await _build_current_schema(pg_url)
    engine = _admin(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text('ALTER TABLE agent_runs RENAME TO "AGENT_RUNS"'))
    await engine.dispose()
    check = create_async_engine(pg_url, poolclass=NullPool)
    from dendrux.db._schema_check import ensure_schema_current

    try:
        with pytest.raises(SchemaOutOfDateError) as info:
            await ensure_schema_current(check)
    finally:
        await check.dispose()
    assert "agent_runs" in info.value.missing_tables


async def test_sweep_checks_the_schema(pg_url: str) -> None:
    from datetime import timedelta

    await _build_previous_schema(pg_url)
    with pytest.raises(SchemaOutOfDateError):
        await sweep(database_url=pg_url, stale_running=timedelta(minutes=5))
    engine = _admin(pg_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    await _build_current_schema(pg_url)
    result = await sweep(database_url=pg_url, stale_running=timedelta(minutes=5))
    assert result.stale_running == []


def _record_statements(engine: Any, sink: list[str]) -> None:
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record(_c: Any, _cur: Any, statement: str, *_a: Any) -> None:
        sink.append(statement)


def _reflection(statements: list[str]) -> list[str]:
    return [s for s in statements if "pg_class" in s or "pg_catalog" in s]


async def test_check_is_read_only(pg_url: str) -> None:
    from dendrux.db._schema_check import ensure_schema_current

    await _build_current_schema(pg_url)
    statements: list[str] = []
    engine = create_async_engine(pg_url, poolclass=NullPool)
    _record_statements(engine, statements)
    await ensure_schema_current(engine)
    await engine.dispose()
    assert statements
    assert {s.lstrip().split(None, 1)[0].upper() for s in statements} <= {"SELECT", "WITH"}
    assert len(statements) <= 5  # catalog reads plus one pg_indexes read


async def test_agent_reflects_once_per_engine(pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlalchemy.ext.asyncio as sa_asyncio

    import dendrux.agent as agent_module

    await _build_current_schema(pg_url)
    statements: list[str] = []
    real = sa_asyncio.create_async_engine

    def recording(*args: Any, **kwargs: Any) -> Any:
        engine = real(*args, **kwargs)
        _record_statements(engine, statements)
        return engine

    monkeypatch.setattr(sa_asyncio, "create_async_engine", recording)
    verify_calls: list[str] = []

    async def no_verify(*_a: Any) -> None:
        verify_calls.append("called")

    monkeypatch.setattr(agent_module, "_verify_schema_initialized", no_verify)

    agent = _agent(pg_url)
    try:
        await agent.run("one")
        first = len(_reflection(statements))
        assert first > 0  # the single check ran
        await agent.run("two")
        await agent.run("three")
    finally:
        await agent.close()
    assert len(_reflection(statements)) == first
    assert verify_calls == []  # the redundant has_table query is gone


async def test_sqlite_is_unaffected(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'old.db'}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("DROP TABLE dendrux_blob_refs"))
        await conn.execute(text("DROP TABLE dendrux_blobs"))
        await conn.execute(text("DROP TABLE run_inbox"))
        await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN pause_requested"))
    await engine.dispose()

    agent = _agent(url)
    try:
        assert (await agent.run("hi")).answer == "ok"
    finally:
        await agent.close()


async def test_v030_schema_names_both_blob_tables(pg_url: str) -> None:
    """A Postgres DB at 0.3.0 (b5c6d7e8f9a0) must be migrated even for apps without images."""
    await asyncio.to_thread(_alembic, pg_url, RUN_CONTROL)
    agent = _agent(pg_url)
    try:
        with pytest.raises(SchemaOutOfDateError) as info:
            await agent.run("hi")
        err = info.value
        assert set(err.missing_tables) == {"dendrux_blobs", "dendrux_blob_refs"}
        assert err.missing_columns == ("agent_runs.claim_epoch", "agent_runs.status_changed_at")
        assert set(err.missing_indexes) == _R1_INDEXES | _SWEEP_INDEXES
        assert "dendrux_blobs" in str(err) and "dendrux_blob_refs" in str(err)
        assert "dendrux db migrate" in str(err)

        await asyncio.to_thread(_alembic, pg_url, "head")
        assert (await agent.run("hi")).answer == "ok"
    finally:
        await agent.close()
