"""R1 (d7e8f9a0b1c2) on a populated Postgres database, plus the Postgres-only
startup checks: the ``pg_indexes`` schema check, ``dendrux db check`` and the
shared engine's event-loop guard.

Uses a temporary schema inside ``DENDRUX_TEST_PG_URL``; skipped when unset.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from dendrux.db._schema_check import ensure_schema_current
from dendrux.errors import SchemaOutOfDateError
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import IdempotencyConflictError
from tests._helpers.pg_url import validate_test_pg_url
from tests.integration._migration_schema import migration_schema

pytestmark = pytest.mark.asyncio

PRIOR = "c6d7e8f9a0b1"  # 0.4.1 head
HEAD = "d7e8f9a0b1c2"
_MIGRATIONS = Path(__file__).resolve().parents[2] / "src" / "dendrux" / "db" / "migrations"


def _alembic(url: str, op: str, rev: str) -> None:
    from alembic import command
    from alembic.config import Config

    validate_test_pg_url(url, append_suffix=False)
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    getattr(command, op)(cfg, rev)


@pytest.fixture
async def pg(pg_test_url: str) -> Any:
    async with migration_schema(pg_test_url, "t1_r1_migration") as engine:
        yield pg_test_url, engine


async def _indexes(engine: Any) -> dict[str, str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'agent_runs' AND "
                "schemaname = current_schema()"
            )
        )
        return {r[0]: r[1] for r in rows}


async def _seed(engine: Any) -> None:
    now = dt.datetime.now(dt.UTC)
    async with engine.begin() as conn:
        for run_id, tenant, key in (
            ("r1", "A", "k1"),
            ("r2", "B", "k2"),
            ("r3", None, "k3"),
            ("r4", "A", None),
        ):
            await conn.execute(
                text(
                    "INSERT INTO agent_runs (id, tenant_id, agent_name, status, iteration_count,"
                    " delegation_level, total_input_tokens, total_output_tokens,"
                    " total_cache_read_tokens, total_cache_creation_tokens, cancel_requested,"
                    " idempotency_key, idempotency_fingerprint, created_at, updated_at)"
                    " VALUES (:id, :t, 'a', 'success', 1, 0, 0, 0, 0, 0, false, :k, 'fp',"
                    " :now, :now)"
                ),
                {"id": run_id, "t": tenant, "k": key, "now": now},
            )


async def test_upgrade_downgrade_on_a_populated_database(pg: Any) -> None:
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    before = await _indexes(engine)
    assert "ix_agent_runs_idempotency_key" in before

    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    after = await _indexes(engine)
    assert "ix_agent_runs_idempotency_key" not in after
    assert "COALESCE(tenant_id" in after["uq_agent_runs_tenant_idempotency_key"]
    assert "WHERE (idempotency_key IS NOT NULL)" in after["uq_agent_runs_tenant_idempotency_key"]
    assert "created_at DESC" in after["ix_agent_runs_tenant_id_created_at"]
    await ensure_schema_current(engine)  # the migrated schema passes the startup check

    store = SQLAlchemyStateStore(engine)
    other = await store.create_run(
        "n1", "a", tenant_id="B", idempotency_key="k1", idempotency_fingerprint="fp"
    )
    assert other.outcome == "created"  # k1 is tenant A's; tenant B may use it now
    same = await store.create_run(
        "n2", "a", tenant_id="A", idempotency_key="k1", idempotency_fingerprint="fp"
    )
    assert (same.run_id, same.outcome) == ("r1", "existing_terminal")
    async with engine.connect() as conn:
        claim = (
            await conn.execute(text("SELECT claim_epoch FROM agent_runs WHERE id='r1'"))
        ).scalar()
    assert claim == 0

    # Two tenants now share k1: the 0.4.1 global index cannot be rebuilt.
    with pytest.raises(RuntimeError, match="more than one tenant"):
        await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
    assert "uq_agent_runs_tenant_idempotency_key" in await _indexes(engine)

    async with engine.begin() as conn:
        await conn.execute(text("UPDATE agent_runs SET idempotency_key = NULL WHERE id = 'n1'"))
    await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
    restored = await _indexes(engine)
    assert "ix_agent_runs_idempotency_key" in restored
    assert "uq_agent_runs_tenant_idempotency_key" not in restored
    async with engine.connect() as conn:
        count = (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar()
    assert count == 5

    await asyncio.to_thread(_alembic, url, "upgrade", "head")  # upgrade again
    assert "uq_agent_runs_tenant_idempotency_key" in await _indexes(engine)


async def test_database_stamped_early_is_detected_and_recovers(pg: Any) -> None:
    """A DB stamped at R1 by the claim_epoch-only build: startup fails, then it heals."""
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    async with engine.begin() as conn:  # rebuild the claim_epoch-only R1 shape
        await conn.execute(text("DROP INDEX uq_agent_runs_tenant_idempotency_key"))
        await conn.execute(text("DROP INDEX ix_agent_runs_tenant_id_created_at"))
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX ix_agent_runs_idempotency_key ON agent_runs (idempotency_key)"
            )
        )
    with pytest.raises(SchemaOutOfDateError) as caught:
        await ensure_schema_current(engine)
    assert "agent_runs.uq_agent_runs_tenant_idempotency_key (index)" in str(caught.value)

    from typer.testing import CliRunner

    from dendrux.cli.main import app

    os.environ["DENDRUX_DATABASE_URL"] = url
    try:
        # A thread: the command runs its own event loop.
        result = await asyncio.to_thread(CliRunner().invoke, app, ["db", "check"])
    finally:
        del os.environ["DENDRUX_DATABASE_URL"]
    assert result.exit_code == 1
    assert "is missing" in result.output
    assert "root" not in result.output  # the password is masked

    # Even un-migrated, a cross-tenant key reuse fails closed instead of leaking.
    store = SQLAlchemyStateStore(engine)
    await store.create_run(
        "a1", "a", tenant_id="A", idempotency_key="k", idempotency_fingerprint="f"
    )
    with pytest.raises(IdempotencyConflictError) as conflict:
        await store.create_run(
            "b1", "a", tenant_id="B", idempotency_key="k", idempotency_fingerprint="f"
        )
    assert conflict.value.run_id is None

    await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    await ensure_schema_current(engine)


async def test_shared_engine_refuses_another_event_loop(pg: Any) -> None:
    from dendrux.db import session

    url, _engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    await session.reset_engine()
    try:
        engine = await session.get_engine(url)
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar() == 1

        def other_loop() -> None:
            async def use() -> None:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))

            asyncio.run(use())

        with pytest.raises(RuntimeError, match="different event loop"):
            await asyncio.to_thread(other_loop)
    finally:
        await session.reset_engine()


@pytest.mark.parametrize(
    "parts",
    [
        "INDEX {name} ON agent_runs (coalesce(tenant_id, ''), idempotency_key) WHERE "
        "idempotency_key IS NOT NULL",
        "UNIQUE INDEX {name} ON agent_runs (tenant_id, idempotency_key) WHERE idempotency_key "
        "IS NOT NULL",
        "UNIQUE INDEX {name} ON agent_runs (coalesce(tenant_id, 'x'), idempotency_key) WHERE "
        "idempotency_key IS NOT NULL",
        "UNIQUE INDEX {name} ON agent_runs (coalesce(tenant_id, ''), idempotency_key) WHERE "
        "tenant_id IS NOT NULL",
    ],
)
async def test_upgrade_rejects_wrong_prebuilt_index(pg: Any, parts: str) -> None:
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    async with engine.begin() as conn:
        await conn.execute(
            text("CREATE " + parts.format(name="uq_agent_runs_tenant_idempotency_key"))
        )
    with pytest.raises(RuntimeError, match="unexpected definition"):
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
    assert "ix_agent_runs_idempotency_key" in await _indexes(engine)


async def test_startup_rejects_invalid_concurrent_index(pg: Any) -> None:
    from sqlalchemy.exc import IntegrityError

    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    await _seed(engine)
    async with engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text("DROP INDEX uq_agent_runs_tenant_idempotency_key"))
        await conn.execute(text("UPDATE agent_runs SET idempotency_key='same', tenant_id='A'"))
        with pytest.raises(IntegrityError):
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX CONCURRENTLY uq_agent_runs_tenant_idempotency_key "
                    "ON agent_runs (coalesce(tenant_id, ''), idempotency_key) WHERE "
                    "idempotency_key IS NOT NULL"
                )
            )
        valid = (
            await conn.execute(
                text(
                    "SELECT indisvalid FROM pg_index WHERE indexrelid = "
                    "'uq_agent_runs_tenant_idempotency_key'::regclass"
                )
            )
        ).scalar_one()
        assert not valid
    with pytest.raises(SchemaOutOfDateError):
        await ensure_schema_current(engine)
