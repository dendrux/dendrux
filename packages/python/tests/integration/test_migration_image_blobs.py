"""Image-blob migration (c6d7e8f9a0b1) on a populated database: up, down, up.

Postgres runs the real Alembic chain in a temporary schema inside
``DENDRUX_TEST_PG_URL``; skipped when that variable is unset. SQLite checks this revision alone
(it cannot run the full chain): the prior shape is ``create_all`` minus the
blob tables, stamped at ``b5c6d7e8f9a0``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._storage_schema import ADDITIONS
from dendrux.db.models import Base
from dendrux.media import BlobEdge
from dendrux.runtime._blob_catalog import BlobServices
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from tests.integration._migration_schema import (
    create_pre_activation_schema,
    migration_schema,
)
from tests.unit.test_blob_catalog import make_png, store_png

PRIOR = "b5c6d7e8f9a0"
HEAD = "f9a0b1c2d3e4"  # additive recovery schema sits above image blobs
_MIGRATIONS = Path(__file__).resolve().parents[2] / "src" / "dendrux" / "db" / "migrations"


def _alembic(url: str, op: str, rev: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    getattr(command, op)(cfg, rev)


async def _seed(conn: Any) -> None:
    now: Any = dt.datetime.now(dt.UTC)
    if conn.dialect.name == "sqlite":
        now = now.isoformat(sep=" ")
    for run_id, status in (("r-done", "success"), ("r-wait", "waiting_client_tool")):
        await conn.execute(
            text(
                "INSERT INTO agent_runs (id, agent_name, status, iteration_count,"
                " delegation_level, total_input_tokens, total_output_tokens,"
                " total_cache_read_tokens, total_cache_creation_tokens, cancel_requested,"
                " pause_requested, input_data, pause_data, created_at, updated_at)"
                " VALUES (:id, 'a', :status, 1, 0, 0, 0, 0, 0, :f, :f, :inp, :pd, :now, :now)"
            ),
            {
                "id": run_id,
                "status": status,
                "f": False,
                "inp": '{"input": "hi"}',
                "pd": '{"agent_name": "a"}' if status != "success" else None,
                "now": now,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO react_traces (id, agent_run_id, role, content, order_index,"
                " created_at, updated_at) VALUES (:id, :rid, 'user', 'hi', 0, :now, :now)"
            ),
            {"id": f"t-{run_id}", "rid": run_id, "now": now},
        )


async def _snapshot(conn: Any) -> list[Any]:
    # 0.5.0 R1 adds agent_runs.claim_epoch above this revision; leave it out.
    out: list[Any] = []
    for table in ("agent_runs", "react_traces"):
        result = await conn.execute(text(f"SELECT * FROM {table} ORDER BY id"))
        out.append(
            [
                tuple(
                    v
                    for k, v in r._mapping.items()
                    if k not in {"claim_epoch", "status_changed_at", *ADDITIONS[table]}
                )
                for r in result.all()
            ]
        )
    return out


async def _check_head(engine: Any, url: str) -> None:
    async with engine.connect() as conn:

        def _inspect(sync: Any) -> dict[str, Any]:
            i = inspect(sync)
            return {
                "blobs": {c["name"] for c in i.get_columns("dendrux_blobs")},
                "refs": {c["name"] for c in i.get_columns("dendrux_blob_refs")},
                "pk": i.get_pk_constraint("dendrux_blobs")["constrained_columns"],
                "blob_ix": {x["name"]: x["column_names"] for x in i.get_indexes("dendrux_blobs")},
                "ref_ix": {
                    x["name"]: x["column_names"] for x in i.get_indexes("dendrux_blob_refs")
                },
                "fks": i.get_foreign_keys("dendrux_blob_refs"),
                "uq": i.get_unique_constraints("dendrux_blob_refs"),
            }

        got = await conn.run_sync(_inspect)
    assert got["blobs"] == {c.name for c in Base.metadata.tables["dendrux_blobs"].columns}
    assert got["refs"] == {c.name for c in Base.metadata.tables["dendrux_blob_refs"].columns}
    assert got["pk"] == ["tenant_scope", "sha256"]
    assert got["blob_ix"]["ix_dendrux_blobs_gc"] == ["state", "last_used_at"]
    assert got["blob_ix"]["ix_dendrux_blobs_work"] == ["state", "state_changed_at"]
    assert got["ref_ix"]["ix_dendrux_blob_refs_blob"] == ["tenant_scope", "sha256"]
    assert got["ref_ix"]["ix_dendrux_blob_refs_run"] == ["run_id", "tenant_scope"]
    fks = {fk["referred_table"]: fk for fk in got["fks"]}
    assert (fks["agent_runs"].get("options") or {}).get("ondelete", "").upper() == "CASCADE"
    assert fks["dendrux_blobs"]["constrained_columns"] == ["tenant_scope", "sha256"]
    assert (fks["dendrux_blobs"].get("options") or {}).get("ondelete", "").upper() == "RESTRICT"
    assert [u["column_names"] for u in got["uq"]] == [
        ["run_id", "owner_kind", "owner_id", "ordinal"]
    ]

    # The migrated schema serves the catalog, edges and run-scoped reads.
    store = store_for_engine(engine)
    services = BlobServices.for_state_store(store)
    data = make_png(1)
    part = await store_png(services, "-", data)
    await services.catalog.add_blob_refs(  # type: ignore[union-attr]
        "r-wait", "-", [BlobEdge("-", part.ref.sha256, "seed", "seed", 0)]
    )
    run_store = RunStore(store)
    assert await run_store.read_image("r-wait", part.ref.sha256) == data


async def _check_prior(engine: Any) -> None:
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert "dendrux_blobs" not in tables and "dendrux_blob_refs" not in tables
        assert "run_inbox" in tables


async def test_postgres_up_down_up_on_populated_db(pg_test_url: str) -> None:
    url = pg_test_url
    async with migration_schema(url, "image_migration") as engine:
        await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
        async with engine.begin() as conn:
            await _seed(conn)
        async with engine.connect() as conn:
            before = await _snapshot(conn)

        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with engine.connect() as conn:
            assert await _snapshot(conn) == before  # purely additive
        await _check_head(engine, url)

        await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
        await _check_prior(engine)
        async with engine.connect() as conn:
            assert await _snapshot(conn) == before

        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with engine.connect() as conn:
            version = (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
            assert version == HEAD
            blobs = (await conn.execute(text("SELECT count(*) FROM dendrux_blobs"))).scalar_one()
            assert blobs == 0
        await _check_head(engine, url)


async def test_sqlite_revision_on_populated_db(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'imgmig.db'}"
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(create_pre_activation_schema)
            await conn.execute(text("DROP TABLE dendrux_blob_refs"))
            await conn.execute(text("DROP TABLE dendrux_blobs"))
            # 0.5.0 R1 sits above the image blobs.
            await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN claim_epoch"))
            await _seed(conn)
        async with engine.connect() as conn:
            before = await _snapshot(conn)
        await asyncio.to_thread(_alembic, url, "stamp", PRIOR)
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        await _check_head(engine, url)
        await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
        await _check_prior(engine)
        async with engine.connect() as conn:
            assert await _snapshot(conn) == before
    finally:
        await engine.dispose()
