"""Managed waits remain parked when a developer sweeps abandoned external pauses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts.test_managed_fanout import (
    _parent as fanout_parent,
)
from dendrux.testing.store_kit.contracts.test_managed_fanout import (
    _request,
)
from dendrux.testing.store_kit.contracts.test_managed_subagents import _parent as single_parent
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit


@contract(
    "managed_subagents",
    needs=("managed_subagents", "managed_fanout"),
    uses=("clock",),
    format2=True,
)
@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("terminal", [False, True])
async def test_managed_wait_is_exempt_from_abandonment(
    dendrux_kit: Kit, version: int, terminal: bool
) -> None:
    kit, store = dendrux_kit, dendrux_kit.store
    if version == 3:
        request = await single_parent(kit)
        await store.bind_managed_child(request)
        parent = request.parent_run_id
        children = [request.child_run_id]
    else:
        parent, calls = await fanout_parent(kit, 2)
        batch = _request(parent, calls)
        await store.bind_managed_children(batch)
        children = [child.child_run_id for child in batch.children]
    if terminal:
        for child in children:
            await store.finalize_run(
                child,
                status=RunStatus.SUCCESS.value,
                answer="ok",
                claim_epoch=0,
                expected_current_status=RunStatus.RUNNING.value,
            )
    before = await store.get_run(parent)
    pause = await store.get_pause_state(parent)
    checkpoint = await store.get_loop_state(parent)
    threshold = await kit.age()
    assert await store.sweep_abandoned_runs(threshold, batch_size=1, max_runs=1) == []
    after = await store.get_run(parent)
    assert after.status == before.status == RunStatus.PAUSED.value
    assert after.claim_epoch == before.claim_epoch
    assert await store.get_pause_state(parent) == pause
    assert await store.get_loop_state(parent) == checkpoint
