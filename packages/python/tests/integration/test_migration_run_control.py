"""Run-control migration (b5c6d7e8f9a0) applied to a populated database.

Postgres runs the real Alembic chain: upgrade to the prior revision, insert
rows, upgrade to head, check, downgrade, upgrade again. It uses its own
database (``<name>_migrate_test``, derived from ``DENDRUX_TEST_PG_URL``) and
is skipped when that variable is unset.

SQLite cannot run the full chain (earlier revisions ALTER constraints), so it
checks this revision alone: the prior schema is built from the models minus
this revision's objects, stamped at the prior revision, then migrated.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base
from dendrux.runtime.state import SQLAlchemyStateStore

pytestmark = pytest.mark.asyncio

PRIOR = "a4b5c6d7e8f9"
HEAD = "d7e8f9a0b1c2"  # 0.4.0 image blobs and 0.5.0 R1 sit on top of run control
_MIGRATIONS = Path(__file__).resolve().parents[2] / "src" / "dendrux" / "db" / "migrations"


def _alembic(url: str, op: str, rev: str) -> None:
    from alembic import command
    from alembic.config import Config

    # No ini file: fileConfig() would reconfigure (and disable) test loggers.
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    getattr(command, op)(cfg, rev)


def _migrate_pg_url() -> str | None:
    raw = os.environ.get("DENDRUX_TEST_PG_URL")
    if not raw:
        return None
    parsed = urlparse(raw)
    name = parsed.path.lstrip("/").removesuffix("_test") + "_migrate_test"
    return urlunparse(parsed._replace(path=f"/{name}"))


async def _seed_prior_rows(conn: Any) -> None:
    now: Any = dt.datetime.now(dt.UTC)
    if conn.dialect.name == "sqlite":
        now = now.isoformat(sep=" ")  # no implicit datetime adapter on 3.12+
    for run_id, status in (
        ("r-run", "running"),
        ("r-wait", "waiting_approval"),
        ("r-done", "success"),
    ):
        await conn.execute(
            text(
                "INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,"
                " total_input_tokens, total_output_tokens, total_cache_read_tokens,"
                " total_cache_creation_tokens, cancel_requested, pause_data,"
                " created_at, updated_at)"
                " VALUES (:id, 'a', :status, 1, 0, 0, 0, 0, 0, :cancel, :pd, :now, :now)"
            ),
            {
                "id": run_id,
                "status": status,
                "cancel": False,
                "pd": '{"agent_name": "a"}' if status == "waiting_approval" else None,
                "now": now,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO run_events (id, agent_run_id, event_type, sequence_index,"
                " iteration_index, created_at) VALUES (:id, :rid, 'run.started', 0, 0, :now)"
            ),
            {"id": f"e-{run_id}", "rid": run_id, "now": now},
        )


def _columns(sync_conn: Any, table: str) -> set[str]:
    return {c["name"] for c in inspect(sync_conn).get_columns(table)}


async def _check_head(engine: Any) -> None:
    async with engine.connect() as conn:
        cols = await conn.run_sync(_columns, "agent_runs")
        assert "pause_requested" in cols
        assert await conn.run_sync(_columns, "run_inbox") == {
            "id",
            "run_id",
            "text",
            "metadata",
            "created_at",
            "delivered_at",
            "delivered_iteration",
        }
        indexes = await conn.run_sync(lambda c: inspect(c).get_indexes("run_inbox"))
        pending = next(i for i in indexes if i["name"] == "ix_run_inbox_run_pending")
        assert pending["column_names"] == ["run_id", "delivered_at", "created_at", "id"]
        fks = await conn.run_sync(lambda c: inspect(c).get_foreign_keys("run_inbox"))
        assert fks[0]["referred_table"] == "agent_runs"
        assert (fks[0].get("options") or {}).get("ondelete", "").upper() == "CASCADE"
        flags = (
            await conn.execute(text("SELECT id, pause_requested FROM agent_runs ORDER BY id"))
        ).all()
        assert [(r[0], bool(r[1])) for r in flags] == [
            ("r-done", False),
            ("r-run", False),
            ("r-wait", False),
        ]

    # The migrated schema serves the new write paths.
    store = SQLAlchemyStateStore(engine)
    assert await store.request_pause("r-run", reason="m") is True
    assert await store.add_note("r-run", note_id="n1", text="hi", metadata={"k": 1}) is True
    assert await store.add_note("r-done", note_id="n2", text="hi") is False
    control = await store.get_run_control("r-run")
    assert control.pause_requested and control.has_pending_notes
    assert await store.try_pause_run("r-run", status="paused", pause_data={"p": 1}) is True
    assert (await store.get_notes("r-run"))[0].metadata == {"k": 1}
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM agent_runs WHERE id = 'r-run'"))
    assert await store.get_notes("r-run") == []  # ON DELETE CASCADE


async def _check_prior(engine: Any) -> None:
    async with engine.connect() as conn:
        assert "pause_requested" not in await conn.run_sync(_columns, "agent_runs")
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert "run_inbox" not in tables
        count = (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one()
        assert count == 2  # r-run was deleted at head; the rest survived the downgrade


async def test_postgres_upgrade_downgrade_on_populated_db() -> None:
    url = _migrate_pg_url()
    if url is None:
        pytest.skip("DENDRUX_TEST_PG_URL not set")
    name = urlparse(url).path.lstrip("/")
    assert name.endswith("_migrate_test")

    maintenance = create_async_engine(
        urlunparse(urlparse(url)._replace(path="/postgres")),
        isolation_level="AUTOCOMMIT",
        poolclass=NullPool,
    )
    async with maintenance.connect() as conn:
        exists = (
            await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name})
        ).first()
        if exists is None:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await maintenance.dispose()

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
        async with engine.begin() as conn:
            await _seed_prior_rows(conn)

        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        await _check_head(engine)

        await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
        await _check_prior(engine)

        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with engine.connect() as conn:
            assert "pause_requested" in await conn.run_sync(_columns, "agent_runs")
            version = (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
            assert version == HEAD
    finally:
        await engine.dispose()


async def test_sqlite_revision_on_populated_db(tmp_path: Path) -> None:
    from sqlalchemy import event

    url = f"sqlite+aiosqlite:///{tmp_path / 'migrate.db'}"
    engine = create_async_engine(url, poolclass=NullPool)

    # SQLite enforces ON DELETE CASCADE only with foreign_keys on per connection.
    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_conn: Any, _rec: Any) -> None:
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Back to the prior revision's shape (no run control, no image blobs).
            await conn.execute(text("DROP TABLE dendrux_blob_refs"))
            await conn.execute(text("DROP TABLE dendrux_blobs"))
            await conn.execute(text("DROP TABLE run_inbox"))
            await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN pause_requested"))
            await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN claim_epoch"))  # 0.5.0 R1
            await _seed_prior_rows(conn)
        await asyncio.to_thread(_alembic, url, "stamp", PRIOR)

        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        await _check_head(engine)

        await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
        await _check_prior(engine)
    finally:
        await engine.dispose()
