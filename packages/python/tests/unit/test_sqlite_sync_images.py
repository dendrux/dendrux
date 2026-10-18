"""SQLite self-upgrade for the 0.4.0 image blob tables (§3.3).

A v0.3.0 SQLite file gains ``dendrux_blobs`` and ``dendrux_blob_refs`` (with
their plain indexes and the owner-key uniqueness) and nothing else changes;
starting the same file repeatedly is a no-op. ``db/_sqlite_sync.py`` itself
is unchanged by 0.4.0: the new tables are created whole by ``create_all``.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._sqlite_sync import create_sqlite_schema_sync, sync_sqlite_schema
from dendrux.db.models import Base
from dendrux.runtime._blob_catalog import BlobServices
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import RunStore
from tests.unit.test_blob_catalog import make_png, store_png

if TYPE_CHECKING:
    from pathlib import Path

_BLOB_TABLES = {"dendrux_blobs", "dendrux_blob_refs"}


async def _master(engine: Any) -> dict[str, str]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name")
        )
        return {r[0]: r[1] for r in result.all()}


async def _make_v030_file(path: Path) -> Any:
    url = f"sqlite+aiosqlite:///{path}"
    engine = create_async_engine(url, poolclass=NullPool)
    now = dt.datetime.now(dt.UTC).isoformat(sep=" ")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("DROP TABLE dendrux_blob_refs"))
        await conn.execute(text("DROP TABLE dendrux_blobs"))
        await conn.execute(
            text(
                "INSERT INTO agent_runs (id, agent_name, status, iteration_count,"
                " delegation_level, total_input_tokens, total_output_tokens,"
                " total_cache_read_tokens, total_cache_creation_tokens, total_reasoning_tokens,"
                " cancel_requested, pause_requested, input_data, created_at, updated_at)"
                " VALUES ('R1', 'a', 'success', 1, 0, 0, 0, 0, 0, 0, 0, 0,"
                ' \'{"input": "hi"}\', :now, :now)'
            ),
            {"now": now},
        )
    return engine


async def test_v030_file_gains_blob_tables_and_nothing_else(tmp_path: Path) -> None:
    engine = await _make_v030_file(tmp_path / "v030.db")
    try:
        before = await _master(engine)
        assert not _BLOB_TABLES & set(before)
        async with engine.connect() as conn:
            changes = await conn.run_sync(create_sqlite_schema_sync)
        assert changes == []  # tables created whole by create_all: no column/index patching
        after = await _master(engine)
        for name, sql in before.items():
            assert after[name] == sql  # existing objects untouched
        new = set(after) - set(before)
        assert new >= _BLOB_TABLES
        assert {
            "ix_dendrux_blobs_gc",
            "ix_dendrux_blobs_work",
            "ix_dendrux_blob_refs_blob",
            "ix_dendrux_blob_refs_run",
        } <= new
        assert (
            "UNIQUE" in after["dendrux_blob_refs"]
            and "ON DELETE RESTRICT" in after["dendrux_blob_refs"]
        )
        async with engine.connect() as conn:
            row = (await conn.execute(text("SELECT input_data FROM agent_runs"))).scalar_one()
            assert row == '{"input": "hi"}'  # no rewrite-on-read / on-upgrade
            assert await conn.run_sync(sync_sqlite_schema) == []

        # The upgraded file serves the blob catalog.
        store = SQLAlchemyStateStore(engine)
        services = BlobServices.for_state_store(store)
        data = make_png(1)
        part = await store_png(services, "-", data)
        assert await services.accessor.read(part.ref) == data
    finally:
        await engine.dispose()


async def test_same_file_started_three_times(tmp_path: Path) -> None:
    path = tmp_path / "three.db"
    url = f"sqlite+aiosqlite:///{path}"
    results: list[list[str]] = []
    snapshots: list[dict[str, str]] = []
    for _ in range(3):
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                results.append(await conn.run_sync(create_sqlite_schema_sync))
            snapshots.append(await _master(engine))
        finally:
            await engine.dispose()
    assert results[1] == [] and results[2] == []
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert set(snapshots[0]) >= _BLOB_TABLES
    # The public entry point on the same file: also a no-op, and reads work.
    for _ in range(3):
        async with RunStore.from_database_url(url) as store:
            assert await store.list_runs() == []
            assert await store.blobs.info("a" * 64) is None
