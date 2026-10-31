"""T-EXPLAIN: the hot lookups use an index on MySQL 8.4 and MariaDB 10.11.

The tables are seeded with enough rows (and ``ANALYZE TABLE``) that a full scan would be a
real alternative. Each query is the shape the store uses: a run by id, a run by tenant
scope and idempotency key, a run's events after a sequence, the started event, and blob
references by run and by blob. ``-s`` prints the plans (the build report quotes them).
"""

from __future__ import annotations

from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import AgentRun, Base, Blob, BlobRef, RunEvent

RUNS = 400
EVENTS_PER_RUN = 8
SHA = "ab" * 32

QUERIES = {
    "run by id": (
        "SELECT * FROM agent_runs WHERE id = 'run00042'",
        {"PRIMARY"},
    ),
    "run by tenant scope and idempotency key": (
        "SELECT id, idempotency_fingerprint FROM agent_runs "
        "WHERE tenant_scope = 'tenant2' AND idempotency_key = 'key00042'",
        {"uq_agent_runs_tenant_idempotency_key"},
    ),
    "events of a run after a sequence": (
        "SELECT * FROM run_events WHERE agent_run_id = 'run00042' AND sequence_index > 3 "
        "ORDER BY sequence_index LIMIT 100",
        {"uq_run_events_run_seq"},
    ),
    "started event of a run": (
        "SELECT id FROM run_events WHERE agent_run_id = 'run00042' AND event_type = 'run.started'",
        {"ix_run_events_started", "uq_run_events_run_seq"},
    ),
    "blob refs of a run": (
        "SELECT * FROM dendrux_blob_refs WHERE run_id = 'run00042' AND tenant_scope = 'tenant2'",
        {"ix_dendrux_blob_refs_run", "uq_dendrux_blob_refs_owner"},
    ),
    "blob refs of a blob": (
        f"SELECT run_id FROM dendrux_blob_refs WHERE tenant_scope = 'tenant2' AND sha256 = '{SHA}'",
        {"ix_dendrux_blob_refs_blob"},
    ),
    "blob by key": (
        f"SELECT state FROM dendrux_blobs WHERE tenant_scope = 'tenant2' AND sha256 = '{SHA}'",
        {"PRIMARY"},
    ),
}


async def _seed(conn) -> None:  # noqa: ANN001
    await conn.execute(
        insert(AgentRun),
        [
            {
                "id": f"run{i:05d}",
                "tenant_id": f"tenant{i % 5}",
                "agent_name": "a",
                "status": "success",
                "idempotency_key": f"key{i:05d}",
            }
            for i in range(RUNS)
        ],
    )
    await conn.execute(
        insert(RunEvent),
        [
            {
                "id": f"ev{i:05d}{s}",
                "agent_run_id": f"run{i:05d}",
                "event_type": "run.started" if s == 0 else "llm.completed",
                "sequence_index": s,
            }
            for i in range(RUNS)
            for s in range(EVENTS_PER_RUN)
        ],
    )
    await conn.execute(
        insert(Blob),
        [
            {
                "tenant_scope": f"tenant{t}",
                "sha256": SHA if n == 0 else f"{n:064x}",
                "generation": "g",
                "state": "live",
                "media_type": "image/png",
                "byte_length": 1,
                "width": 1,
                "height": 1,
                "storage": "db",
            }
            for t in range(5)
            for n in range(40)
        ],
    )
    await conn.execute(
        insert(BlobRef),
        [
            {
                "id": f"ref{i:05d}{o}",
                "tenant_scope": f"tenant{i % 5}",
                "sha256": SHA if o == 0 else f"{(i % 39) + 1:064x}",
                "run_id": f"run{i:05d}",
                "owner_kind": "trace",
                "owner_id": f"t{i}",
                "ordinal": o,
            }
            for i in range(RUNS)
            for o in range(2)
        ],
    )


async def test_hot_lookups_use_an_index(mysql_url: str, mysql_backend: str) -> None:
    engine = create_async_engine(mysql_url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _seed(conn)
        async with engine.connect() as conn:
            for table in ("agent_runs", "run_events", "dendrux_blobs", "dendrux_blob_refs"):
                await conn.execute(text(f"ANALYZE TABLE {table}"))
            for label, (query, allowed) in QUERIES.items():
                rows = (await conn.execute(text(f"EXPLAIN {query}"))).mappings().all()
                plan = [
                    {k: row[k] for k in ("table", "type", "key", "rows", "Extra")} for row in rows
                ]
                print(f"[{mysql_backend}] {label}: {plan}")
                assert len(rows) == 1, plan
                assert rows[0]["type"] not in {"ALL", "index"}, plan
                assert rows[0]["key"] in allowed, plan
    finally:
        await engine.dispose()
