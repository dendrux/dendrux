"""Recovery-capable stores count only crashes since advancing durable progress."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from dendrux._lifecycle import StoredEventType
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    RECLAIM,
    lease_cutoff,
    mark,
    started_run,
    trace_write,
)
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

progress = contract("recovery", needs=("recovery", "checkpointing"), uses=("clock",), format2=True)


@progress
async def test_progress_resets_recovery_streak_in_its_checkpoint(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await started_run(dendrux_kit)
    for iteration in range(5):
        cutoff = await lease_cutoff(dendrux_kit)
        [item] = (await mark(dendrux_kit, cutoff)).reclaimable
        claim = await store.reclaim_run(
            run_id,
            tenant_id="t",
            agent_name="KitAgent",
            expected_claim_epoch=item.claim_epoch,
            **RECLAIM,
        )
        assert claim is not False
        record = await store.get_run(run_id)
        assert record.reclaim_attempts == 1
        stamp = record.last_reclaimed_at
        state = {**claim.loop_state, "iteration": iteration + 1}
        await store.checkpoint_step(
            run_id,
            claim_epoch=claim.claim_epoch,
            loop_state=state,
            pii_mapping=None,
            trace=trace_write("assistant", "advancing model step", iteration),
        )
        record = await store.get_run(run_id)
        assert record.reclaim_attempts == 0
        assert record.last_reclaimed_at == stamp


@progress
async def test_reused_observation_and_duplicate_checkpoint_preserve_streak(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await started_run(dendrux_kit)
    trace = trace_write("assistant", "original", 0)
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state={"v": 1, "loop": "react"}, pii_mapping=None, trace=trace
    )
    cutoff = await lease_cutoff(dendrux_kit)
    [item] = (await mark(dendrux_kit, cutoff)).reclaimable
    claim = await store.reclaim_run(
        run_id,
        tenant_id="t",
        agent_name="KitAgent",
        expected_claim_epoch=item.claim_epoch,
        **RECLAIM,
    )
    assert claim is not False
    for evidence in (
        None,
        trace,
        replace(trace_write("tool", "reused observation", 1), advances_progress=False),
    ):
        await store.checkpoint_step(
            run_id,
            claim_epoch=claim.claim_epoch,
            loop_state=claim.loop_state,
            pii_mapping=None,
            trace=evidence,
        )
        assert (await store.get_run(run_id)).reclaim_attempts == 1


@contract("recovery", needs=("recovery", "atomic_completion"), uses=("clock",), format2=True)
@pytest.mark.parametrize("path", ["server", "client", "final"])
async def test_committed_observations_and_answer_reset_streak(dendrux_kit: Kit, path: str) -> None:
    from dendrux.runtime.state import LoopStatePatch
    from dendrux.testing.store_kit.contracts._support import complete_call_kwargs, start_call_kwargs

    store = dendrux_kit.store
    run_id = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    [item] = (await mark(dendrux_kit, cutoff)).reclaimable
    claim = await store.reclaim_run(
        run_id,
        tenant_id="t",
        agent_name="KitAgent",
        expected_claim_epoch=item.claim_epoch,
        **RECLAIM,
    )
    assert claim is not False
    stamp = (await store.get_run(run_id)).last_reclaimed_at
    if path == "server":
        await store.start_tool_call(
            run_id, claim_epoch=claim.claim_epoch, **start_call_kwargs("call")
        )
        assert (await store.get_run(run_id)).reclaim_attempts == 1
        await store.complete_tool_call(
            run_id, claim_epoch=claim.claim_epoch, **complete_call_kwargs("call", 1)
        )
    elif path == "client":
        state = {**claim.loop_state, "submission": {"submission_id": "s", "kind": "tool_results"}}
        await store.checkpoint_step(
            run_id, claim_epoch=claim.claim_epoch, loop_state=state, pii_mapping=None, trace=None
        )
        assert (await store.get_run(run_id)).reclaim_attempts == 1
        assert await store.consume_submission(
            run_id,
            claim_epoch=claim.claim_epoch,
            submission_id="s",
            traces=[trace_write("tool", "client observation", 1)],
            patch=LoopStatePatch(),
        )
    else:
        await store.finalize_with_event(
            run_id,
            claim_epoch=claim.claim_epoch,
            status=RunStatus.SUCCESS.value,
            answer="done",
            allowed_current_statuses=[RunStatus.RUNNING.value],
            event_type=StoredEventType.RUN_COMPLETED.value,
        )
    record = await store.get_run(run_id)
    assert record.reclaim_attempts == 0 and record.last_reclaimed_at == stamp
