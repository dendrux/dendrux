"""The observation label stays out of runtime pending, wire and claim boundaries."""

from __future__ import annotations

import pytest

from dendrux._wire import result_to_dict
from dendrux.store import RunStore
from dendrux.types import RunStatus
from tests.integration.test_fanout_routing import (
    TENANT,
    approver,
    bind_one,
    fan_out,
    root_agent,
    start_root,
)


@pytest.mark.parametrize("version", [3, 4])
async def test_waiting_label_does_not_change_runtime_result_pending_or_events(
    store, version
) -> None:
    children = [approver("reviewer")]
    if version == 4:
        children.append(approver("editor"))
    root = root_agent(store, *children)
    parent = await start_root(store, root)
    if version == 3:
        await bind_one(store, root, parent, children[0])
    else:
        await fan_out(store, root, parent, children)
    before_pause = await store.get_pause_state(parent)
    before_events = await store.get_run_events(parent)
    reader = RunStore(store)
    detail = await reader.get_run(parent, tenant_id=TENANT)
    assert detail.waiting_for == "subagents" and detail.status == "paused"
    await reader.list_runs(tenant_id=TENANT, include_waiting=False)
    await reader.get_subtree(parent, tenant_id=TENANT)
    result = await root.get_result(parent, tenant_id=TENANT)
    wire = result_to_dict(result)
    assert not hasattr(result, "waiting_for")
    assert "waiting_for" not in wire
    for pending in result.pendings:
        assert not hasattr(pending, "waiting_for")
    assert RunStatus.PAUSED.value == "paused"
    assert "subagents" not in {status.value for status in RunStatus}
    assert await store.get_pause_state(parent) == before_pause
    assert await store.get_run_events(parent) == before_events
    # The same record still satisfies the raw persisted-status claim predicate.
    record = await store.get_run(parent)
    assert record.status == "paused" and record.waiting_for == "subagents"
    assert await store.claim_paused_run(parent, expected_status=record.status)
