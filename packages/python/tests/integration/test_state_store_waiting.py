"""Waiting-label contracts on the regular SQLite/Postgres integration matrix."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import event, null, update

from dendrux.db.models import AgentRun
from dendrux.errors import StoreCapabilityError
from dendrux.runtime.state import StoreCapabilities
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


class _ThirdPartyStore:
    """A minimal third-party store: delegates reads, declares the waiting capability."""

    capabilities = StoreCapabilities(deployment="single_process", waiting_filter=True)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def list_runs(self, *, include_waiting: bool = True, **kwargs: Any) -> list[Any]:
        return await self._inner.list_runs(include_waiting=include_waiting, **kwargs)

    async def count_runs(self, *, include_waiting: bool = True, **kwargs: Any) -> int:
        return await self._inner.count_runs(include_waiting=include_waiting, **kwargs)

    async def get_run(self, run_id: str, *, tenant_id: Any = None) -> Any:
        return await self._inner.get_run(run_id, tenant_id=tenant_id)

    async def get_subtree(self, run_id: str, **kwargs: Any) -> Any:
        return await self._inner.get_subtree(run_id, **kwargs)

    async def get_waiting_for(self, run_id: str, *, tenant_id: Any = None) -> str | None:
        return await self._inner.get_waiting_for(run_id, tenant_id=tenant_id)


@pytest.mark.parametrize("wrap", [False, True], ids=["sql", "third_party"])
@pytest.mark.parametrize("surface", ["list", "detail", "subtree"])
async def test_facade_reports_waiting_label_for_capable_stores(store, wrap, surface) -> None:
    await store.create_run("root", "Test", tenant_id="t")
    await store.create_run("kid", "Test", tenant_id="t", parent_run_id="root")
    for run_id in ("root", "kid"):
        await store.pause_run(
            run_id, status="paused", pause_data={"v": 4, "pause_reason": "managed_wait"}
        )
    facade = RunStore(_ThirdPartyStore(store) if wrap else store)
    if surface == "list":
        rows = await facade.list_runs(tenant_id="t")
    elif surface == "detail":
        rows = [await facade.get_run(run_id, tenant_id="t") for run_id in ("root", "kid")]
    else:
        rows = [node.run for node in (await facade.get_subtree("root", tenant_id="t")).nodes]
    assert len(rows) == 2 and {row.waiting_for for row in rows} == {"subagents"}
    assert await facade.list_runs(tenant_id="t", include_waiting=False) == []
    assert await facade.count_runs(tenant_id="t", include_waiting=False) == 0


async def test_facade_without_capability_has_no_label_and_refuses_filter() -> None:
    facade = RunStore(_LegacyStore())
    assert (await facade.list_runs())[0].waiting_for is None
    detail = await facade.get_run("legacy")
    assert detail is not None and detail.waiting_for is None
    with pytest.raises(StoreCapabilityError):
        await facade.list_runs(include_waiting=False)
    with pytest.raises(StoreCapabilityError):
        await facade.count_runs(include_waiting=False)


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


async def test_subtree_labels_cost_one_extra_read_not_one_per_run(store, engine) -> None:
    """Runtime BFS stays label-free; the public subtree adds one chunked label query."""
    await store.create_run("root", "Test")
    for i in range(12):
        await store.create_run(f"kid-{i}", "Test", parent_run_id="root")
    await store.create_run("grandkid", "Test", parent_run_id="kid-0")
    await store.pause_run(
        "grandkid", status="paused", pause_data={"v": 4, "pause_reason": "managed_wait"}
    )
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        tree = await RunStore(store).get_subtree("root")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert {n.run.run_id: n.run.waiting_for for n in tree.nodes}["grandkid"] == "subagents"
    # root + 3 BFS levels (kids, grandkid, empty) + usage + labels
    assert len(statements) == 6
    assert sum("pause_data" in s and "WHERE agent_runs.id IN" in s for s in statements) == 1


async def test_subtree_without_paused_runs_makes_no_label_query(store, engine) -> None:
    await store.create_run("root", "Test")
    await store.create_run("kid", "Test", parent_run_id="root")
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        tree = await RunStore(store).get_subtree("root")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert {n.run.waiting_for for n in tree.nodes} == {None}
    # root + 2 BFS levels (kid, empty) + usage; paused subtree above costs exactly one more
    assert len(statements) == 4
    assert not any("pause_data" in s and "WHERE agent_runs.id IN" in s for s in statements)
