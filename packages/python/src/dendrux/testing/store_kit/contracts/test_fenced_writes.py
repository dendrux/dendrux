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
    with pytest.raises(RunFencedError):
        await _raising_writers()[writer](dendrux_kit.store, run_id, stale)
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
