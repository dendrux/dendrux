"""Postgres cancel/bind ordering with independent pools and explicit transaction gates."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import pytest

from dendrux.errors import RunCancelledError
from tests.integration.test_fanout_adoption import lead
from tests.integration.test_fanout_cancel import HANG_TIMEOUT
from tests.integration.test_fanout_store_races import (
    _pools,
    _postgres_only,
    batch,
    counts,
    fanout_parent,
)


@pytest.mark.parametrize("n", [2, 8, 32])
@pytest.mark.parametrize("round_", range(3))
async def test_r5_x4_mark_wins_against_eight_batch_binders(
    engine: Any,
    store: Any,
    monkeypatch: Any,
    n: int,
    round_: int,
) -> None:
    _postgres_only(engine)
    root, calls = await fanout_parent(store, n)
    request = batch(root, calls)
    async with AsyncExitStack() as stack:
        *binders, canceller = await _pools(stack, engine, [f"t8-mark-{i}" for i in range(9)])
        barrier, release = asyncio.Barrier(9), asyncio.Event()

        def hold(binder: Any) -> None:
            real = binder._lock_managed_root

            async def locked(session: Any, plan: Any) -> None:
                await asyncio.wait_for(barrier.wait(), HANG_TIMEOUT)
                await asyncio.wait_for(release.wait(), HANG_TIMEOUT)
                await real(session, plan)

            monkeypatch.setattr(binder, "_lock_managed_root", locked)

        for binder in binders:
            hold(binder)
        async with asyncio.timeout(HANG_TIMEOUT):
            tasks = [asyncio.create_task(b.bind_managed_children(request)) for b in binders]
            await asyncio.wait_for(barrier.wait(), HANG_TIMEOUT)
            try:
                cancelled = await lead(state_store=canceller).cancel_run(root, tenant_id="t1")
            finally:
                release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
    assert cancelled.status.value == "running"
    assert all(isinstance(r, RunCancelledError) for r in results)
    assert await counts(engine, root) == {
        "children": 0,
        "intents": 0,
        "events": 0,
        "seeds": 0,
        "epoch": 0,
    }
    assert (await store.get_run(root)).cancel_requested


@pytest.mark.parametrize("n", [2, 8, 32])
@pytest.mark.parametrize("round_", range(3))
async def test_r5_x4_bind_transaction_wins_then_cancel_sees_the_whole_batch(
    engine: Any,
    store: Any,
    monkeypatch: Any,
    n: int,
    round_: int,
) -> None:
    _postgres_only(engine)
    root, calls = await fanout_parent(store, n)
    request = batch(root, calls)
    async with AsyncExitStack() as stack:
        binder, canceller = await _pools(stack, engine, ["t8-bind", "t8-cancel"])
        partial, release, marking = asyncio.Event(), asyncio.Event(), asyncio.Event()
        child_write, mark = binder._bind_child, canceller.mark_cancel_requested

        async def held(*args: Any, **kwargs: Any) -> None:
            await child_write(*args, **kwargs)
            if not partial.is_set():
                partial.set()
                await asyncio.wait_for(release.wait(), HANG_TIMEOUT)

        async def marking_root(*args: Any, **kwargs: Any) -> Any:
            marking.set()
            return await mark(*args, **kwargs)

        monkeypatch.setattr(binder, "_bind_child", held)
        monkeypatch.setattr(canceller, "mark_cancel_requested", marking_root)
        async with asyncio.timeout(HANG_TIMEOUT):
            binding = asyncio.create_task(binder.bind_managed_children(request))
            await asyncio.wait_for(partial.wait(), HANG_TIMEOUT)
            cancellation = asyncio.create_task(
                lead(state_store=canceller).cancel_run(root, tenant_id="t1")
            )
            try:
                await asyncio.wait_for(marking.wait(), HANG_TIMEOUT)
                # The writer holds the root lock with only one child inserted.
                assert (await counts(engine, root))["children"] == 0
            finally:
                release.set()
            bound, cancelled = await binding, await cancellation
    assert bound.outcome == "created" and len(bound.children) == n
    assert sum(c.child_claim is not None for c in bound.children) == n
    assert cancelled.status.value == "cancelled" and cancelled.meta["cascade_count"] == n + 1
    assert (await counts(engine, root))["children"] == n
    assert (await counts(engine, root))["intents"] == n
    for child in request.children:
        record = await store.get_run(child.child_run_id)
        assert record.cancel_requested and record.status == "running"
        assert (await store.get_managed_run_control(child.child_run_id)).cancel_requested
    assert (await store.get_run(root)).status == "cancelled"
