"""Pure managed cancellation edge and lifecycle decisions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import ManagedStateError
from dendrux.runtime._managed import (
    LegacyCancelPages,
    ManagedCancelPages,
    cancel_child_id,
    legacy_cancel_levels,
    outcome_code,
)
from dendrux.runtime.state import ManagedHeader
from dendrux.types import ToolResult, generate_ulid

ROOT, PARENT, CHILD = (generate_ulid() for _ in range(3))


@pytest.mark.parametrize("membership", ["managed", "legacy", "other_root", "other_parent"])
def test_only_explicit_same_root_parent_edges_are_cancelled(membership: str) -> None:
    header = (
        None
        if membership == "legacy"
        else ManagedHeader(
            root_run_id=generate_ulid() if membership == "other_root" else ROOT,
            path=("worker",),
            parent_run_id=generate_ulid() if membership == "other_parent" else PARENT,
            tool_call_id="c1",
        ).to_dict()
    )
    assert cancel_child_id((CHILD, PARENT, header), ROOT) == (
        CHILD if membership == "managed" else None
    )


@pytest.mark.parametrize(("count", "has_more"), [(0, False), (1, False), (500, True), (501, True)])
async def test_legacy_traversal_does_not_require_a_managed_header(
    count: int, has_more: bool
) -> None:
    store = SimpleNamespace(get_children_page=AsyncMock(return_value=[CHILD] * count))
    page = await LegacyCancelPages(store).read([ROOT], after=None, limit=500, tenant_id=None)
    assert page.run_ids == (CHILD,) * count
    assert page.after == (CHILD if count else None)
    assert page.has_more is has_more
    store.get_children_page.assert_awaited_once_with([ROOT], after=None, limit=500, tenant_id=None)


async def test_filtered_page_keeps_raw_continuation_without_any_managed_ids() -> None:
    store = SimpleNamespace(_get_cancel_children_page=AsyncMock(return_value=[(CHILD, ROOT, None)]))
    page = await ManagedCancelPages(store, ROOT).read([ROOT], after=None, limit=1, tenant_id=None)
    assert page.run_ids == () and page.after == CHILD and page.has_more


@pytest.mark.parametrize("value", [{}, {"v": 2}, "SENTINEL-untrusted-header"])
def test_invalid_projected_header_fails_closed_without_text(value: Any) -> None:
    with pytest.raises(ManagedStateError) as info:
        cancel_child_id((CHILD, PARENT, value), ROOT)
    assert info.value.reason == "header_invalid"
    assert "SENTINEL" not in str(info.value)


@pytest.mark.parametrize(
    ("success", "code", "outcome"),
    [
        (True, None, "succeeded"),
        (False, "managed_child_failed", "failed"),
        (False, "managed_child_cancelled", "cancelled"),
        (False, "managed_report_too_large", "failed"),
    ],
)
def test_lifecycle_codes_distinguish_cancelled(
    success: bool, code: str | None, outcome: str
) -> None:
    report = ToolResult(
        name="worker",
        call_id="c1",
        payload='"report"',
        success=success,
        meta={"code": code} if code else {},
    )
    assert outcome_code(report) == outcome


@pytest.mark.parametrize("case", ["missing", "root_only", "ordered", "truncated"])
async def test_legacy_snapshot_cancel_preserves_depth_order_and_bound(case: str) -> None:
    sibling = generate_ulid()
    nodes = [
        SimpleNamespace(run=SimpleNamespace(id=CHILD), depth=2),
        SimpleNamespace(run=SimpleNamespace(id=ROOT), depth=0),
        SimpleNamespace(run=SimpleNamespace(id=PARENT), depth=1),
        SimpleNamespace(run=SimpleNamespace(id=sibling), depth=1),
    ]
    subtree = (
        None
        if case == "missing"
        else SimpleNamespace(nodes=[nodes[1]] if case == "root_only" else nodes)
    )
    store = SimpleNamespace(get_subtree=AsyncMock(return_value=subtree))
    levels, truncated = await legacy_cancel_levels(
        store, ROOT, None, 2 if case == "truncated" else 10
    )
    expected = [[ROOT]]
    if case == "truncated":
        expected.append([PARENT])
    elif case == "ordered":
        expected.extend([[PARENT, sibling], [CHILD]])
    assert levels == expected
    assert truncated is (case == "truncated")
    store.get_subtree.assert_awaited_once_with(ROOT, tenant_id=None)
