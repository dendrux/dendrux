"""Waiting-label contracts on the regular SQLite/Postgres integration matrix."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import event, null, update

from dendrux.db.models import AgentRun
from dendrux.errors import StoreCapabilityError
from dendrux.store import RunStore
from dendrux.testing.store_kit._declaration import Declaration
from dendrux.testing.store_kit._kit import Kit
from dendrux.testing.store_kit.contracts.test_waiting_label import (
    _LegacyStore,
    test_legacy_waiting_label_defaults_and_filter_refuses,
    test_waiting_count_filter_independently_excludes_waits,
    test_waiting_filter_preserves_status_scope_and_pagination,
    test_waiting_label_list_detail_subtree,
)

__all__ = [
    "test_legacy_waiting_label_defaults_and_filter_refuses",
    "test_waiting_count_filter_independently_excludes_waits",
    "test_waiting_filter_preserves_status_scope_and_pagination",
    "test_waiting_label_list_detail_subtree",
]


@pytest.fixture
def dendrux_kit(store: Any) -> Kit:
    return Kit({}, store, Declaration.of(store), needs_format_2=False)


@pytest.mark.parametrize("pause", [None, null(), [], "managed_wait", 4, True])
async def test_null_or_nonobject_pause_is_not_waiting(store, engine, pause) -> None:
    await store.create_run("null-pause", "Test")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == "null-pause")
            .values(status="paused", pause_data=pause)
        )
    reader = RunStore(store)
    rows = await reader.list_runs(include_waiting=False)
    assert len(rows) == 1 and rows[0].waiting_for is None
    assert await reader.count_runs(include_waiting=False) == 1


async def test_label_uses_one_query_and_keeps_pause_payload_deferred(store, engine) -> None:
    for i in range(10):
        await store.create_run(f"wait-{i}", "Test")
        await store.pause_run(
            f"wait-{i}",
            status="paused",
            pause_data={"v": 4, "pause_reason": "managed_wait", "history": ["large"] * 1000},
        )
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
        # Only the scalar expression may reference pause_data; the blob is deferred.
        assert "pause_data" not in [column[0] for column in cursor.description]
        assert "pii_mapping" not in [column[0] for column in cursor.description]

    event.listen(engine.sync_engine, "after_cursor_execute", capture)
    try:
        rows = await store.list_runs()
    finally:
        event.remove(engine.sync_engine, "after_cursor_execute", capture)
    assert len(statements) == 1
    assert "pause_reason" in statements[0] or "json" in statements[0].lower()
    assert len(rows) == 10 and all(row.waiting_for == "subagents" for row in rows)


async def test_result_projection_keeps_raw_status_and_no_label(store) -> None:
    await store.create_run("wait", "Test")
    await store.pause_run(
        "wait", status="paused", pause_data={"v": 3, "pause_reason": "managed_wait"}
    )
    reader = RunStore(store)
    assert (await reader.get_run("wait")).waiting_for == "subagents"
    record = await store._get_result_record("wait", tenant_id=None)
    assert record.status == "paused" and record.waiting_for is None


@pytest.mark.parametrize("surface", ["list", "detail", "subtree"])
@pytest.mark.parametrize("reason", ["managed_wait", "input"])
async def test_facade_waiting_labels_through_lazy_wrapper(store, engine, surface, reason) -> None:
    await store.create_run("parent", "Test", tenant_id="t")
    await store.create_run("child", "Test", tenant_id="t", parent_run_id="parent")
    for run_id in ("parent", "child"):
        await store.pause_run(run_id, status="paused", pause_data={"v": 4, "pause_reason": reason})
    async with RunStore.from_engine(engine) as reader:
        if surface == "list":
            records = await reader.list_runs(tenant_id="t")
        elif surface == "detail":
            records = [await reader.get_run("parent", tenant_id="t")]
        else:
            tree = await reader.get_subtree("parent", tenant_id="t")
            records = [node.run for node in tree.nodes]
        expected = "subagents" if reason == "managed_wait" else None
        assert all(
            record.status == "paused" and record.waiting_for == expected for record in records
        )
        assert await reader.count_runs(tenant_id="t", include_waiting=False) == (
            0 if expected else 2
        )
        assert len(await reader.list_runs(tenant_id="t", include_waiting=False)) == (
            0 if expected else 2
        )


@pytest.mark.parametrize("operation", ["list_runs", "count_runs"])
async def test_legacy_filter_refusal_is_independent_of_labels(operation) -> None:
    with pytest.raises(StoreCapabilityError):
        await getattr(RunStore(_LegacyStore()), operation)(include_waiting=False)


async def test_public_detail_label_query_keeps_runtime_read_unchanged(store, engine) -> None:
    await store.create_run("detail-wait", "Test", tenant_id="t")
    await store.pause_run(
        "detail-wait", status="paused", pause_data={"pause_reason": "managed_wait"}
    )
    await store.create_run("detail-running", "Test", tenant_id="t")
    statements = []

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        raw = await store.get_run("detail-wait", tenant_id="t")
        assert raw.waiting_for is None
        assert len(statements) == 1 and "CASE WHEN" not in statements[0]
        statements.clear()
        detail = await RunStore(store).get_run("detail-wait", tenant_id="t")
        assert detail.waiting_for == "subagents"
        assert len(statements) == 2
        assert "CASE WHEN" not in statements[0]
        assert "CASE WHEN" in statements[1] and "history" not in statements[1]
        statements.clear()
        assert await RunStore(store).get_run("detail-wait", tenant_id="other") is None
        assert len(statements) == 1
        statements.clear()
        running = await RunStore(store).get_run("detail-running", tenant_id="t")
        assert running.waiting_for is None
        assert len(statements) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
