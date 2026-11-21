"""Read-only waiting labels and optional filtering, including legacy stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import pytest

from dendrux.errors import StoreCapabilityError
from dendrux.runtime import _capabilities
from dendrux.runtime.state import RunRecord
from dendrux.testing.store_kit.contracts._marks import basic, contract
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

waiting = contract("basic", needs=("waiting_filter",))


@waiting
@pytest.mark.parametrize("surface", ["list", "detail", "subtree"])
@pytest.mark.parametrize(
    ("pause", "status", "label"),
    [
        ({"v": 3, "pause_reason": "managed_wait"}, "paused", "subagents"),
        ({"v": 4, "pause_reason": "managed_wait"}, "paused", "subagents"),
        ({"pause_reason": "managed_wait"}, "paused", "subagents"),
        ({"v": 4}, "paused", None),
        ({"v": 3, "pause_reason": None}, "paused", None),
        ({"pause_reason": "input"}, "paused", None),
        ({"pause_reason": True}, "paused", None),
        ({"pause_reason": 4}, "paused", None),
        ({"pause_reason": ["managed_wait"]}, "paused", None),
        ({"pause_reason": {"reason": "managed_wait"}}, "paused", None),
        ({"pause_reason": "MANAGED_WAIT"}, "paused", None),
        ({"pause_reason": "managed_wait "}, "paused", None),
        ({"pause_reason": "managed_wait"}, RunStatus.RUNNING.value, None),
        ({"pause_reason": "managed_wait"}, RunStatus.WAITING_CLIENT_TOOL.value, None),
        ({}, "paused", None),
    ],
)
async def test_waiting_label_list_detail_subtree(
    dendrux_kit: Kit, pause: dict[str, Any], status: str, label: str | None, surface: str
) -> None:
    store = dendrux_kit.store
    root = await dendrux_kit.new_run(tenant_id="t")
    child = await dendrux_kit.new_run(parent_run_id=root, tenant_id="t")
    for run_id in (root, child):
        await store.pause_run(run_id, status=status, pause_data=pause)
    reader = store
    if surface == "list":
        listed = await reader.list_runs(tenant_id="t")
        assert {row.id: row.waiting_for for row in listed} == {root: label, child: label}
    elif surface == "detail":
        for run_id in (root, child):
            detail = await reader.get_run(run_id, tenant_id="t")
            assert detail is not None and detail.status == status
            assert detail.waiting_for == label
    else:
        tree = await reader.get_subtree(root, tenant_id="t")
        assert tree is not None
        assert {node.run.id: node.run.waiting_for for node in tree.nodes} == {
            root: label,
            child: label,
        }
    for run_id in (root, child):
        raw = await store.get_run(run_id)
        assert raw is not None and raw.status == status
        assert raw.waiting_for == label
    expected = set() if label else {root, child}
    filtered = await reader.list_runs(tenant_id="t", include_waiting=False)
    assert {row.id for row in filtered} == expected
    assert await reader.count_runs(tenant_id="t", include_waiting=False) == len(expected)


class _WaitingOptions(TypedDict):
    tenant_id: str
    include_waiting: bool
    metadata_filter: dict[str, str]


@waiting
async def test_waiting_filter_preserves_status_scope_and_pagination(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    expected: set[str] = set()
    for tenant, reason in [("t", "input"), ("t", None), ("u", "input"), ("t", "managed_wait")]:
        run_id = await dendrux_kit.new_run(tenant_id=tenant, meta={"thread": "a"})
        await store.pause_run(run_id, status="paused", pause_data={"pause_reason": reason})
        if tenant == "t" and reason != "managed_wait":
            expected.add(run_id)
    running = await dendrux_kit.new_run(tenant_id="t", meta={"thread": "a"})
    reader = store
    assert len(await reader.list_runs(status="paused", tenant_id="t")) == 3
    assert await reader.count_runs(status="paused", tenant_id="t") == 3
    options: _WaitingOptions = {
        "tenant_id": "t",
        "include_waiting": False,
        "metadata_filter": {"thread": "a"},
    }
    rows = await reader.list_runs(status="paused", **options)
    assert {row.id for row in rows} == expected
    assert await reader.count_runs(status="paused", **options) == 2
    assert [
        row.id for row in await reader.list_runs(status="paused", limit=1, offset=1, **options)
    ] == [rows[1].id]
    mixed = await reader.list_runs(status=["paused", RunStatus.RUNNING.value], **options)
    assert {row.id for row in mixed} == expected | {running}
    assert await reader.count_runs(status=["paused", RunStatus.RUNNING.value], **options) == 3
    assert await reader.list_runs(status=[], **options) == []
    assert await reader.count_runs(status=[], **options) == 0
    assert await reader.get_run(running, tenant_id="u") is None


@waiting
async def test_waiting_count_filter_independently_excludes_waits(dendrux_kit: Kit) -> None:
    for reason in ("managed_wait", "input"):
        run_id = await dendrux_kit.new_run(tenant_id="t")
        await dendrux_kit.store.pause_run(
            run_id, status="paused", pause_data={"pause_reason": reason}
        )
    assert await dendrux_kit.store.count_runs(tenant_id="t", include_waiting=False) == 1
    assert await dendrux_kit.store.count_runs(tenant_id="t") == 2


class _LegacyStore:
    def __init__(self) -> None:
        self.record = RunRecord("legacy", "Legacy", "paused")

    async def list_runs(self, *, tenant_id: object = None, **kwargs: Any) -> list[RunRecord]:
        if "include_waiting" in kwargs:
            raise AssertionError("legacy signature must not receive the new keyword")
        return [self.record]

    async def count_runs(self, *, tenant_id: object = None, **kwargs: Any) -> int:
        if "include_waiting" in kwargs:
            raise AssertionError("legacy signature must not receive the new keyword")
        return 1

    async def get_run(self, run_id: str, *, tenant_id: object = None) -> RunRecord:
        return self.record


@basic()
async def test_legacy_waiting_label_defaults_and_filter_refuses() -> None:
    reader = _LegacyStore()
    assert (await reader.list_runs())[0].waiting_for is None
    detail = await reader.get_run("legacy")
    assert detail is not None and detail.waiting_for is None
    assert (
        await reader.count_runs(
            **_capabilities.waiting_filter_options(reader, include_waiting=True)
        )
        == 1
    )
    assert (
        len(
            await reader.list_runs(
                **_capabilities.waiting_filter_options(reader, include_waiting=True)
            )
        )
        == 1
    )
    with pytest.raises(StoreCapabilityError):
        await reader.list_runs(
            **_capabilities.waiting_filter_options(reader, include_waiting=False)
        )
    with pytest.raises(StoreCapabilityError):
        await reader.count_runs(
            **_capabilities.waiting_filter_options(reader, include_waiting=False)
        )
