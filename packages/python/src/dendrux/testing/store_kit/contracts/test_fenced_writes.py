"""``fenced_writes``: a writer holding a stale claim epoch can change nothing.

A run's epoch is raised by every claim. The executor of an older epoch must be refused by every
write it can make, and a refused write must leave no trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dendrux.errors import RunFencedError
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import EVENT, tool_call_kwargs, ulid, usage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.testing.store_kit._kit import Kit

fenced = contract("fenced_writes", needs=("fenced_writes",))


def _raising_writers() -> dict[str, Callable[[Any, str, int], Awaitable[Any]]]:
    return {
        "save_trace": lambda s, r, e: s.save_trace(r, "user", "x", order_index=0, claim_epoch=e),
        "save_tool_call": lambda s, r, e: s.save_tool_call(r, **tool_call_kwargs(), claim_epoch=e),
        "save_usage": lambda s, r, e: s.save_usage(
            r, iteration_index=0, usage=usage(), claim_epoch=e
        ),
        "save_llm_interaction": lambda s, r, e: s.save_llm_interaction(
            r, iteration_index=0, usage=usage(), claim_epoch=e
        ),
        "save_run_event": lambda s, r, e: s.save_run_event(r, event_type=EVENT, claim_epoch=e),
        "add_note": lambda s, r, e: s.add_note(r, note_id=ulid(), text="n", claim_epoch=e),
        "mark_notes_delivered": lambda s, r, e: s.mark_notes_delivered(
            r, [ulid()], iteration=0, claim_epoch=e
        ),
        "update_run_meta": lambda s, r, e: s.update_run_meta(
            r, {"kit.fenced": True}, claim_epoch=e
        ),
    }


async def _nothing_was_written(kit: Kit, run_id: str) -> None:
    store = kit.store
    assert await store.get_traces(run_id) == []
    assert await store.get_tool_calls(run_id) == []
    assert await store.get_run_events(run_id) == []
    assert await store.get_notes(run_id) == []
    assert (await store.run_usage(run_id)).llm_calls == 0
    record = await store.get_run(run_id)
    assert record is not None
    assert "kit.fenced" not in (record.meta or {})


@fenced
@pytest.mark.parametrize("writer", sorted(_raising_writers()))
async def test_a_stale_epoch_is_refused_by_every_fenced_writer(
    dendrux_kit: Kit, writer: str
) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    stale = claim.claim_epoch - 1
    seeded = ulid()
    if writer == "mark_notes_delivered":
        # a real, undelivered note: a refusal must come from the fence, not from a missing note
        await dendrux_kit.store.add_note(
            run_id, note_id=seeded, text="n", claim_epoch=claim.claim_epoch
        )
    with pytest.raises(RunFencedError):
        if writer == "mark_notes_delivered":
            await dendrux_kit.store.mark_notes_delivered(
                run_id, [seeded], iteration=0, claim_epoch=stale
            )
        else:
            await _raising_writers()[writer](dendrux_kit.store, run_id, stale)
    if writer == "mark_notes_delivered":
        [note] = await dendrux_kit.store.get_notes(run_id)
        assert note.delivered_at is None, "a stale owner must not mark a note delivered"
        return
    await _nothing_was_written(dendrux_kit, run_id)


@fenced
@pytest.mark.parametrize("writer", sorted(_raising_writers()))
async def test_the_current_epoch_is_accepted_by_every_fenced_writer(
    dendrux_kit: Kit, writer: str
) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    if writer == "mark_notes_delivered":
        note = ulid()
        await dendrux_kit.store.add_note(
            run_id, note_id=note, text="n", claim_epoch=claim.claim_epoch
        )
        assert (
            await dendrux_kit.store.mark_notes_delivered(
                run_id, [note], iteration=0, claim_epoch=claim.claim_epoch
            )
            == 1
        )
        return
    await _raising_writers()[writer](dendrux_kit.store, run_id, claim.claim_epoch)


@fenced
async def test_touch_progress_reports_a_stale_epoch_as_false(dendrux_kit: Kit) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    store = dendrux_kit.store
    assert await store.touch_progress(run_id, claim_epoch=claim.claim_epoch - 1) is False
    assert await store.touch_progress(run_id, claim_epoch=claim.claim_epoch) is True


@fenced
async def test_a_stale_finalize_changes_nothing(dendrux_kit: Kit) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    store, stale = dendrux_kit.store, claim.claim_epoch - 1
    assert not await store.finalize_run(run_id, status="success", answer="x", claim_epoch=stale)
    assert not await store.finalize_run_if_status_in(
        run_id, status="success", allowed_current_statuses=["running"], claim_epoch=stale
    )
    record = await store.get_run(run_id)
    assert record is not None
    assert (record.status, record.answer) == ("running", None)
    assert await store.finalize_run(
        run_id, status="success", answer="x", claim_epoch=claim.claim_epoch
    )


@fenced
async def test_a_stale_pause_changes_nothing(dendrux_kit: Kit) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    store, stale = dendrux_kit.store, claim.claim_epoch - 1
    paused = await store.try_pause_run(
        run_id, status="waiting_client_tool", pause_data={"pause_id": "p"}, claim_epoch=stale
    )
    assert not paused
    assert (await store.get_run(run_id)).status == "running"
    state = await store.get_pause_state(run_id)
    assert state is None or state.get("pause_id") != "p", "the stale pause must not be stored"


@fenced
async def test_a_second_claim_fences_the_first_executor(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id, first = await dendrux_kit.claimed_run()
    await store.pause_run(
        run_id,
        status="waiting_client_tool",
        pause_data={"pause_id": "second"},
        claim_epoch=first.claim_epoch,
    )
    second = await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    assert second is not False
    assert second.claim_epoch > first.claim_epoch
    with pytest.raises(RunFencedError):
        await store.save_run_event(run_id, event_type=EVENT, claim_epoch=first.claim_epoch)
    assert await store.save_run_event(run_id, event_type=EVENT, claim_epoch=second.claim_epoch) == 0


async def _refused_or_ignored(call: Awaitable[Any]) -> Any:
    """A stale status write is refused (``RunFencedError``) or ignored; either way it is a loss."""
    try:
        return await call
    except RunFencedError:
        return None


@fenced
async def test_a_stale_pause_run_changes_nothing(dendrux_kit: Kit) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    store = dendrux_kit.store
    await _refused_or_ignored(
        store.pause_run(
            run_id,
            status="waiting_client_tool",
            pause_data={"pause_id": "stale-owner"},
            claim_epoch=claim.claim_epoch - 1,
        )
    )
    record = await store.get_run(run_id)
    assert record is not None and record.status == "running", "a stale owner cannot pause the run"
    assert (await store.get_pause_state(run_id) or {}).get("pause_id") != "stale-owner"
    await store.pause_run(
        run_id,
        status="waiting_client_tool",
        pause_data={"pause_id": "current-owner"},
        claim_epoch=claim.claim_epoch,
    )
    assert (await store.get_run(run_id)).status == "waiting_client_tool"
    assert (await store.get_pause_state(run_id))["pause_id"] == "current-owner"


@fenced
async def test_a_stale_pause_with_event_changes_nothing(dendrux_kit: Kit) -> None:
    run_id, claim = await dendrux_kit.claimed_run()
    store = dendrux_kit.store
    stale = await _refused_or_ignored(
        store.pause_with_event(
            run_id,
            status="waiting_client_tool",
            pause_data={"pause_id": "stale-owner"},
            claim_epoch=claim.claim_epoch - 1,
        )
    )
    assert stale is None, "a stale pause reports the loss"
    record = await store.get_run(run_id)
    assert record is not None and record.status == "running"
    assert await store.get_run_events(run_id) == [], "a lost pause publishes no event"
    sequence = await store.pause_with_event(
        run_id,
        status="waiting_client_tool",
        pause_data={"pause_id": "current-owner"},
        claim_epoch=claim.claim_epoch,
    )
    assert sequence == 0
    assert (await store.get_run(run_id)).status == "waiting_client_tool"
    assert [e.event_type for e in await store.get_run_events(run_id)] == ["run.paused"]


#: Every ``StateStore`` method that takes ``claim_epoch``, and the contract that refuses a stale
#: epoch for it. tests/conformance/test_kit_coverage.py keeps this equal to the protocol.
FENCED_WRITER_CONTRACTS: dict[str, str] = {
    "add_note": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "mark_notes_delivered": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "save_trace": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "save_tool_call": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "save_usage": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "save_llm_interaction": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "save_run_event": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "update_run_meta": "test_a_stale_epoch_is_refused_by_every_fenced_writer",
    "touch_progress": "test_touch_progress_reports_a_stale_epoch_as_false",
    "finalize_run": "test_a_stale_finalize_changes_nothing",
    "finalize_run_if_status_in": "test_a_stale_finalize_changes_nothing",
    "try_pause_run": "test_a_stale_pause_changes_nothing",
    "pause_run": "test_a_stale_pause_run_changes_nothing",
    "pause_with_event": "test_a_stale_pause_with_event_changes_nothing",
    "checkpoint_step": "test_a_stale_epoch_cannot_checkpoint_or_link",
    "set_trace_link": "test_a_stale_epoch_cannot_checkpoint_or_link",
    "consume_submission": "test_a_stale_epoch_cannot_consume_a_submission",
    "start_tool_call": "test_a_stale_epoch_cannot_start_or_complete_a_tool_call",
    "complete_tool_call": "test_a_stale_epoch_cannot_start_or_complete_a_tool_call",
    "finalize_with_event": "test_a_stale_finalize_with_event_publishes_nothing",
}
