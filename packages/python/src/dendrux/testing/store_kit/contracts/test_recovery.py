"""``recovery``: leases, marking, reclaim and a store that survives being reopened.

A run whose owner stopped making progress is marked lease-expired, then reclaimed by a new owner
under a higher epoch; the old owner is fenced out. Marking is bounded and pages with a cursor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dendrux.errors import RunFencedError, RunNotFoundError, RunNotReclaimableError
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    EVENT,
    RECLAIM,
    lease_cutoff,
    mark,
    started_run,
    trace_write,
    ulid,
)

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

recovery = contract("recovery", needs=("recovery",), uses=("clock", "reopen"), format2=True)


@recovery
async def test_only_stale_runs_are_marked_lease_expired(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    stale = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    fresh = await started_run(dendrux_kit)
    marked = await mark(dendrux_kit, cutoff)
    assert [r.run_id for r in marked.reclaimable] == [stale]
    assert marked.reclaimable[0].reclaim_limit == 3
    assert [r.run_id for r in await store.get_lease_expired(limit=10, cursor=None)] == [stale]
    assert (await store.get_run(fresh)).status == "running"
    assert (await store.get_run(stale)).status == "running", "marking does not end the run"


@recovery
async def test_marking_is_repeatable_until_the_run_is_reclaimed(dendrux_kit: Kit) -> None:
    stale = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    first = await mark(dendrux_kit, cutoff)
    again = await mark(dendrux_kit, cutoff)
    assert [r.run_id for r in first.reclaimable] == [r.run_id for r in again.reclaimable] == [stale]
    assert first.reclaimable[0].claim_epoch == again.reclaimable[0].claim_epoch


@recovery
async def test_a_reclaim_gives_a_new_epoch_once_and_fences_the_old_owner(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    stale = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    [marked] = (await mark(dendrux_kit, cutoff)).reclaimable
    claim = await store.reclaim_run(
        stale,
        tenant_id="t",
        agent_name="KitAgent",
        expected_claim_epoch=marked.claim_epoch,
        **RECLAIM,
    )
    assert claim is not False
    assert claim.claim_epoch > marked.claim_epoch
    assert claim.loop_state == {"v": 1, "loop": "react", "phase": "dispatch"}
    record = await store.get_run(stale)
    assert record is not None
    assert (record.status, record.reclaim_attempts) == ("running", 1)
    with pytest.raises(RunNotReclaimableError):
        await store.reclaim_run(
            stale,
            tenant_id="t",
            agent_name="KitAgent",
            expected_claim_epoch=marked.claim_epoch,
            **RECLAIM,
        )
    with pytest.raises(RunFencedError):
        await store.save_run_event(stale, event_type=EVENT, claim_epoch=0)
    assert await store.save_run_event(stale, event_type=EVENT, claim_epoch=claim.claim_epoch) >= 1


@recovery
async def test_a_reclaim_is_refused_for_the_wrong_tenant_agent_or_epoch(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    stale = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    [marked] = (await mark(dendrux_kit, cutoff)).reclaimable
    epoch = marked.claim_epoch
    with pytest.raises(RunNotFoundError):
        await store.reclaim_run(
            stale, tenant_id="other", agent_name="KitAgent", expected_claim_epoch=epoch, **RECLAIM
        )
    with pytest.raises(RunNotReclaimableError):
        await store.reclaim_run(
            stale, tenant_id="t", agent_name="Other", expected_claim_epoch=epoch, **RECLAIM
        )
    with pytest.raises(RunNotReclaimableError):
        await store.reclaim_run(
            stale, tenant_id="t", agent_name="KitAgent", expected_claim_epoch=epoch - 1, **RECLAIM
        )
    assert (await store.get_run(stale)).claim_epoch == epoch, "refusals change nothing"


@recovery
async def test_a_fresh_run_cannot_be_reclaimed(dendrux_kit: Kit) -> None:
    run_id = await started_run(dendrux_kit)
    with pytest.raises(RunNotReclaimableError):
        await dendrux_kit.store.reclaim_run(
            run_id, tenant_id="t", agent_name="KitAgent", expected_claim_epoch=None, **RECLAIM
        )


@recovery
async def test_a_stale_run_with_a_pending_cancel_ends_cancelled(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await started_run(dendrux_kit)
    await store.request_cancel(run_id)
    cutoff = await lease_cutoff(dendrux_kit)
    marked = await mark(dendrux_kit, cutoff)
    assert marked.reclaimable == []
    assert [(s.run_id, s.status) for s in marked.finalised] == [(run_id, "cancelled")]
    assert (await store.get_run(run_id)).status == "cancelled"


@recovery
async def test_a_run_past_its_reclaim_limit_is_finalised(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    [marked] = (await mark(dendrux_kit, cutoff, reclaim_limit=1)).reclaimable
    claim = await store.reclaim_run(
        run_id,
        tenant_id="t",
        agent_name="KitAgent",
        expected_claim_epoch=marked.claim_epoch,
        **RECLAIM,
    )
    assert claim is not False
    cutoff = await lease_cutoff(dendrux_kit)
    ended = await mark(dendrux_kit, cutoff, reclaim_limit=1, finalise_unrecoverable=True)
    assert [s.failure_reason for s in ended.finalised] == ["reclaim_limit"]
    assert (await store.get_run(run_id)).status == "error"
    assert ended.reclaimable == []


@recovery
async def test_a_partial_sweep_pages_by_cursor_and_loses_no_run(dendrux_kit: Kit) -> None:
    runs = [await started_run(dendrux_kit) for _ in range(5)]
    cutoff = await lease_cutoff(dendrux_kit)
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page = await mark(dendrux_kit, cutoff, max_returned=2, cursor=cursor)
        assert len(page.reclaimable) <= 2
        seen.extend(r.run_id for r in page.reclaimable)
        cursor = page.cursor
        if cursor is None:
            break
    assert cursor is None, "the cursor must end"
    assert sorted(seen) == sorted(runs)
    assert len(seen) == len(set(seen)), "no run is returned twice"
    paged = await dendrux_kit.store.get_lease_expired(limit=2, cursor=None)
    assert len(paged) == 2


@recovery
async def test_the_marking_scan_is_bounded_by_max_runs(dendrux_kit: Kit) -> None:
    for _ in range(4):
        await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    page = await mark(dendrux_kit, cutoff, max_runs=2, max_returned=10)
    assert page.inspected <= 2
    assert page.cursor is not None, "a capped scan says there is more"


@recovery
async def test_everything_written_survives_reopening_the_store(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await started_run(dendrux_kit)
    await store.save_trace(run_id, "user", "hello ✓", order_index=0)
    await store.save_run_event(run_id, event_type=EVENT, data={"k": [1, None]})
    note = ulid()
    await store.add_note(run_id, note_id=note, text="pending note", tenant_id="t")
    paused = await dendrux_kit.paused_run()
    reopened = await dendrux_kit.reopen()
    record = await reopened.get_run(run_id)
    assert record is not None and record.status == "running"
    assert [t.content for t in await reopened.get_traces(run_id)] == ["hello ✓"]
    events = await reopened.get_run_events(run_id)
    assert [e.event_type for e in events] == ["run.started", EVENT]
    assert events[1].data == {"k": [1, None]}
    assert [n.text for n in await reopened.get_notes(run_id, undelivered_only=True)] == [
        "pending note"
    ]
    assert await reopened.get_loop_state(run_id) == {"v": 1, "loop": "react", "phase": "dispatch"}
    assert (
        await reopened.claim_paused_run(paused, expected_status="waiting_client_tool") is not False
    )


@recovery
async def test_a_reopened_store_reclaims_what_the_old_one_marked(dendrux_kit: Kit) -> None:
    stale = await started_run(dendrux_kit)
    cutoff = await lease_cutoff(dendrux_kit)
    [marked] = (await mark(dendrux_kit, cutoff)).reclaimable
    reopened = await dendrux_kit.reopen()
    claim = await reopened.reclaim_run(
        stale,
        tenant_id="t",
        agent_name="KitAgent",
        expected_claim_epoch=marked.claim_epoch,
        **RECLAIM,
    )
    assert claim is not False
    await reopened.save_trace(
        stale, "user", "after restart", order_index=0, claim_epoch=claim.claim_epoch
    )
    assert [t.content for t in await dendrux_kit.store.get_traces(stale)] == ["after restart"]


@recovery
async def test_a_trace_clash_after_reclaim_is_still_a_conflict(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    stale = await started_run(dendrux_kit)
    first = trace_write("user", "one", 0)
    await store.save_trace(stale, "user", "one", order_index=0, trace_id=first.trace_id)
    cutoff = await lease_cutoff(dendrux_kit)
    [marked] = (await mark(dendrux_kit, cutoff)).reclaimable
    claim = await store.reclaim_run(
        stale,
        tenant_id="t",
        agent_name="KitAgent",
        expected_claim_epoch=marked.claim_epoch,
        **RECLAIM,
    )
    assert claim is not False
    again = await store.save_trace(
        stale, "user", "one", order_index=0, trace_id=first.trace_id, claim_epoch=claim.claim_epoch
    )
    assert again == first.trace_id
