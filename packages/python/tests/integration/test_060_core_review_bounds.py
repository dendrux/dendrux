"""Bounded recovery scans/frontiers and validated rewrite metadata."""

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import event

from dendrux.runtime.state import TraceWrite
from dendrux.types import generate_ulid


@pytest.mark.parametrize("field", ["batch_size", "max_runs", "max_returned"])
async def test_recovery_scan_caps_and_realistic_maximum(store, field):
    kwargs = dict(
        cutoff=datetime.now(UTC),
        batch_size=100,
        max_runs=1000,
        max_returned=1000,
        reclaim_limit=3,
        cursor=None,
    )
    with pytest.raises(ValueError):
        await store.mark_lease_expired(**{**kwargs, field: 100001})
    result = await store.mark_lease_expired(**{**kwargs, field: 100000})
    assert result.inspected == 0


@pytest.mark.parametrize("size", [32, 10001])
async def test_children_frontier_queries_are_chunked(store, engine, size):
    await store.create_run("parent", "a")
    await store.create_run("child", "a", parent_run_id="parent")
    parents = ["parent", *(f"missing-{i}" for i in range(size - 1))]
    parameters_seen = []

    def capture(conn, cursor, statement, parameters, context, many):
        parameters_seen.append(len(parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        children = await store.get_children_page(parents, after=None, limit=10, tenant_id=None)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert children == ["child"]
    assert max(parameters_seen) <= 504


async def test_rewrite_rejects_nondictionary_entry_meta(store):
    await store.create_run("rewrite-meta", "a")
    trace = TraceWrite(
        generate_ulid(),
        "rewrite",
        json.dumps([{"role": "user", "content": "hello", "meta": ["bad"]}]),
        0,
        trace_kind="rewrite",
    )
    with pytest.raises(ValueError):
        await store.checkpoint_step(
            "rewrite-meta", claim_epoch=0, loop_state={}, pii_mapping=None, trace=trace
        )
    assert await store.get_traces("rewrite-meta") == []


async def test_claimed_note_acquires_advisory_lock_before_run_lock(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL advisory lock ordering")
    await store.create_run("note-lock", "a")
    statements = []

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await store.add_note(
            "note-lock", note_id=generate_ulid(), text="note", max_notes=10, claim_epoch=0
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    advisory = next(i for i, sql in enumerate(statements) if "pg_advisory_xact_lock" in sql)
    fence = next(i for i, sql in enumerate(statements) if sql.startswith("UPDATE agent_runs"))
    assert advisory < fence
