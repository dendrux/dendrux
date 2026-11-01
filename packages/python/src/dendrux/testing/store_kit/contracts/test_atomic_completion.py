"""``atomic_completion``: a tool's intent, its result and a run's end publish all-or-nothing.

``start_tool_call`` writes the intent row and its event together. ``complete_tool_call`` writes
the result row, the observation trace, the event and the checkpoint patch together.
``finalize_with_event`` and ``pause_with_event`` move a run and publish their evidence in one
step. A reader never sees half of one.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from dendrux.errors import PersistenceError, RunFencedError
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    complete_call_kwargs,
    event_write,
    start_call_kwargs,
    trace_write,
)
from dendrux.types import InputValidationError

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

atomic = contract("atomic_completion", needs=("atomic_completion",), format2=True)


async def _run_with_pending(kit: Kit, *call_ids: str) -> str:
    run_id = await kit.new_run()
    await kit.store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "pending": [{"call_id": c} for c in call_ids]},
        pii_mapping=None,
        trace=None,
    )
    return run_id


@atomic
async def test_an_intent_publishes_its_row_and_event_together(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    [row] = await store.get_tool_calls(run_id)
    assert (row.tool_call_id, row.outcome) == ("a", "started")
    assert [e.event_type for e in await store.get_run_events(run_id)] == ["tool.started"]


@atomic
async def test_an_intent_with_unpersistable_values_writes_nothing(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    bad_event = replace(event_write("tool.started", "c"), data={"call_id": "\x00"})
    for kwargs in (
        start_call_kwargs("c", params={"q": "\x00"}),
        start_call_kwargs("c", event=bad_event),
    ):
        with pytest.raises(InputValidationError):
            await store.start_tool_call(run_id, claim_epoch=0, **kwargs)
    assert await store.get_tool_calls(run_id) == []
    assert await store.get_run_events(run_id) == []


@atomic
async def test_a_completion_publishes_result_trace_event_and_patch_together(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await _run_with_pending(dendrux_kit, "a", "b")
    for call_id in ("a", "b"):
        await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs(call_id))
    await store.complete_tool_call(run_id, claim_epoch=0, **complete_call_kwargs("a", 1))
    state = await store.get_loop_state(run_id)
    assert state is not None
    assert [p["call_id"] for p in state["pending"]] == ["b"]
    rows = {r.tool_call_id: r for r in await store.get_tool_calls(run_id)}
    assert (rows["a"].outcome, rows["a"].success) == ("completed", True)
    assert rows["b"].outcome == "started"
    assert [t.meta["call_id"] for t in await store.get_traces(run_id)] == ["a"]
    types = [e.event_type for e in await store.get_run_events(run_id)]
    assert types == ["tool.started", "tool.started", "tool.completed"]


@atomic
async def test_a_completion_repeated_after_a_lost_acknowledgement_is_a_no_op(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await _run_with_pending(dendrux_kit, "a")
    await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    kwargs = complete_call_kwargs("a", 1)
    await store.complete_tool_call(run_id, claim_epoch=0, **kwargs)
    await store.complete_tool_call(run_id, claim_epoch=0, **kwargs)
    assert len(await store.get_traces(run_id)) == 1
    assert len(await store.get_run_events(run_id)) == 2
    assert (await store.get_loop_state(run_id))["pending"] == []


@atomic
async def test_a_completion_with_a_different_result_conflicts(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await _run_with_pending(dendrux_kit, "a")
    await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    await store.complete_tool_call(run_id, claim_epoch=0, **complete_call_kwargs("a", 1))
    changed = complete_call_kwargs(
        "a",
        1,
        outcome="failed",
        result_payload='"changed"',
        success=False,
        error_message="safe",
        event=event_write("tool.completed", "a"),
    )
    with pytest.raises(PersistenceError) as caught:
        await store.complete_tool_call(run_id, claim_epoch=0, **changed)
    assert caught.value.code == "write_conflict"
    [row] = [r for r in await store.get_tool_calls(run_id) if r.tool_call_id == "a"]
    assert (row.outcome, row.success) == ("completed", True)


@atomic
async def test_a_completion_that_cannot_publish_everything_publishes_nothing(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await _run_with_pending(dendrux_kit, "a")
    await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    clash = trace_write("tool", '"first"', 1, meta={"call_id": "other"})
    await store.save_trace(run_id, "tool", '"first"', order_index=1, trace_id=clash.trace_id)
    with pytest.raises(PersistenceError):
        await store.complete_tool_call(run_id, claim_epoch=0, **complete_call_kwargs("a", 1))
    [row] = await store.get_tool_calls(run_id)
    assert row.outcome == "started", "the result row must roll back with the clashing trace"
    assert [p["call_id"] for p in (await store.get_loop_state(run_id))["pending"]] == ["a"]
    assert [e.event_type for e in await store.get_run_events(run_id)] == ["tool.started"]


@atomic
async def test_a_stale_epoch_cannot_start_or_complete_a_tool_call(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id, claim = await dendrux_kit.claimed_run()
    stale = claim.claim_epoch - 1
    with pytest.raises(RunFencedError):
        await store.start_tool_call(run_id, claim_epoch=stale, **start_call_kwargs("a"))
    with pytest.raises(RunFencedError):
        await store.complete_tool_call(run_id, claim_epoch=stale, **complete_call_kwargs("a", 1))
    assert await store.get_tool_calls(run_id) == []
    assert await store.get_run_events(run_id) == []
    assert await store.get_traces(run_id) == []


@atomic
async def test_finalize_with_event_ends_the_run_and_publishes_evidence_together(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.save_run_event(run_id, event_type="before")
    sequence = await store.finalize_with_event(
        run_id,
        status="success",
        allowed_current_statuses=["running"],
        event_type="run.completed",
        answer="done",
        claim_epoch=0,
        trace=trace_write("assistant", "done", 0),
        event=event_write("run.completed", run_id),
    )
    assert sequence == 1
    record = await store.get_run(run_id)
    assert record is not None
    assert (record.status, record.answer) == ("success", "done")
    assert [t.content for t in await store.get_traces(run_id)] == ["done"]
    assert [e.event_type for e in await store.get_run_events(run_id)] == ["before", "run.completed"]


@atomic
async def test_a_lost_finalize_publishes_no_evidence(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.finalize_run(run_id, status="cancelled", claim_epoch=0)
    sequence = await store.finalize_with_event(
        run_id,
        status="success",
        allowed_current_statuses=["running"],
        event_type="run.completed",
        answer="late",
        claim_epoch=0,
        trace=trace_write("assistant", "late", 0),
        event=event_write("run.completed", run_id),
    )
    assert sequence is None
    assert (await store.get_run(run_id)).status == "cancelled"
    assert await store.get_traces(run_id) == []
    assert await store.get_run_events(run_id) == []


@atomic
async def test_a_stale_finalize_with_event_publishes_nothing(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id, claim = await dendrux_kit.claimed_run()
    sequence = await store.finalize_with_event(
        run_id,
        status="success",
        allowed_current_statuses=["running"],
        event_type="run.completed",
        answer="stale",
        claim_epoch=claim.claim_epoch - 1,
        trace=trace_write("assistant", "stale", 0),
        event=event_write("run.completed", run_id),
    )
    assert sequence is None
    assert (await store.get_run(run_id)).status == "running"
    assert await store.get_traces(run_id) == []
    assert await store.get_run_events(run_id) == []


@atomic
async def test_pause_with_event_pauses_and_publishes_together_and_loses_to_a_cancel(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    cancelled = await dendrux_kit.new_run()
    await store.request_cancel(cancelled)
    assert (
        await store.pause_with_event(
            cancelled, status="waiting_client_tool", pause_data={"pause_id": "p"}, claim_epoch=0
        )
        is None
    )
    assert (await store.get_run(cancelled)).status == "running"
    assert await store.get_run_events(cancelled) == []
    run_id = await dendrux_kit.new_run()
    sequence = await store.pause_with_event(
        run_id,
        status="waiting_client_tool",
        pause_data={"pause_id": "p"},
        event_data={"why": "tools"},
        claim_epoch=0,
    )
    assert sequence == 0
    assert (await store.get_run(run_id)).status == "waiting_client_tool"
    [event] = await store.get_run_events(run_id)
    assert (event.event_type, event.sequence_index) == ("run.paused", 0)
