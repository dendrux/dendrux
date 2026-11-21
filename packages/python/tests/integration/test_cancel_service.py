"""The private cancel service: local signals, the traversal bound and managed-child refusal."""

from __future__ import annotations

from typing import Any

import pytest

from dendrux.errors import ManagedChildControlError, PersistenceError, RunNotFoundError
from dendrux.runtime import _cancel
from dendrux.runtime.state import UNSCOPED
from dendrux.types import RunStatus, generate_ulid
from tests.integration.test_managed_cancel import TENANT, chain


async def seed(store: Any, *, children: int = 0, paused: bool = True) -> tuple[str, list[str]]:
    root = generate_ulid()
    await store.create_run(root, "cancel", tenant_id=TENANT)
    kids = [generate_ulid() for _ in range(children)]
    for kid in kids:
        await store.create_run(kid, "cancel", parent_run_id=root, tenant_id=TENANT)
    for rid in (root, *kids) if paused else ():
        await store.pause_run(rid, status="paused", pause_data={})
    return root, kids


async def cancel(store: Any, run_id: str, signalled: list[str], **kwargs: Any) -> Any:
    options = {"tenant_id": TENANT, "cascade": False, "max_nodes": 100, **kwargs}
    return await _cancel.cancel_run(store, run_id, signal=signalled.append, **options)


@pytest.mark.parametrize("paused", [True, False])
async def test_requested_run_is_signalled_once_after_success(store: Any, paused: bool) -> None:
    root, _ = await seed(store, paused=paused)
    signalled: list[str] = []
    result = await cancel(store, root, signalled)
    assert result.status == (RunStatus.CANCELLED if paused else RunStatus.RUNNING)
    assert result.meta["cancel_requested"] is not paused
    assert signalled == [root]


async def test_terminal_run_is_not_signalled(store: Any) -> None:
    root, _ = await seed(store)
    await cancel(store, root, [])
    signalled: list[str] = []
    assert (await cancel(store, root, signalled)).status == RunStatus.CANCELLED
    assert signalled == []


@pytest.mark.parametrize("failure", ["finalize_with_event", "get_run"])
async def test_committed_request_is_signalled_once_when_cancel_fails(
    store: Any, monkeypatch: Any, failure: str
) -> None:
    root, _ = await seed(store, paused=failure == "finalize_with_event")
    original, calls = getattr(store, failure), 0

    async def fail_second_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if failure == "finalize_with_event" or calls == 2:
            raise PersistenceError()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, failure, fail_second_read)
    signalled: list[str] = []
    with pytest.raises(PersistenceError):
        await cancel(store, root, signalled)
    monkeypatch.undo()
    assert signalled == [root]
    assert (await store.get_run(root)).cancel_requested


async def test_traversal_bound_counts_the_root_and_truncates(store: Any) -> None:
    root, kids = await seed(store, children=3)
    result = await cancel(store, root, [], cascade=True, max_nodes=2)
    assert result.meta["cascade_count"] == 2
    assert result.meta["cascade_truncated"] is True
    assert result.meta["cascade"] == [
        {"run_id": kids[0], "outcome": "cancelled"},
        {"run_id": root, "outcome": "cancelled"},
    ]
    for kid in kids[1:]:
        row = await store.get_run(kid)
        assert (row.status, row.cancel_requested) == ("paused", False)


async def test_bound_large_enough_settles_every_descendant(store: Any) -> None:
    root, kids = await seed(store, children=3)
    result = await cancel(store, root, [], cascade=True, max_nodes=4)
    assert result.meta["cascade_count"] == 4
    assert result.meta["cascade_truncated"] is False
    assert [item["run_id"] for item in result.meta["cascade"]] == [*kids, root]


@pytest.mark.parametrize("cascade", [False, True])
async def test_managed_child_stays_refused_without_writes(store: Any, cascade: bool) -> None:
    root, child, leaf = await chain(store)
    before = {rid: await store.get_run(rid) for rid in (root, child, leaf)}
    signalled: list[str] = []
    with pytest.raises(ManagedChildControlError):
        await cancel(store, child, signalled, cascade=cascade, tenant_id=UNSCOPED)
    assert signalled == []
    for rid, row in before.items():
        after = await store.get_run(rid)
        assert (after.status, after.cancel_requested) == (row.status, row.cancel_requested)


async def test_foreign_tenant_is_not_found_and_unchanged(store: Any) -> None:
    root, _ = await seed(store)
    signalled: list[str] = []
    with pytest.raises(RunNotFoundError):
        await cancel(store, root, signalled, tenant_id="other")
    row = await store.get_run(root)
    assert (row.status, row.cancel_requested, signalled) == ("paused", False, [])
