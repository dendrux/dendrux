"""Live updates, through the optional ``dendrux_live_store_factory``.

A store that announces changes hands the ids of the runs a write touched to a ``LiveUpdates``
once its COMMIT was issued, and flushes when a call that succeeded returns. The kit gives the
store a ``RecordingLiveUpdates`` whose probe reads the database over a second connection at the
moment each announcement is sent, and asserts:

- a committed write is announced, and what it changed is already visible to the second connection
  (an announcement sent before its commit shows up as a missing row, event or status);
- a write that is proved not to have applied is not announced;
- a write whose COMMIT was issued is announced even when the call then fails (a lost acknowledgement
  or a cancellation): at the next flush, never by delaying or changing the failing call.

Without the fixture these contracts skip and the report says "live updates: not tested".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import EVENT, ulid
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

live = contract("live_updates", uses=("live",))
live_not_applied = contract("live_updates", needs=("retry_ids",), uses=("live", "injector"))
live_issued = contract("live_updates", uses=("live", "injector"))


@live
async def test_live_a_created_run_is_announced_after_its_row_is_visible_elsewhere(
    dendrux_kit: Kit,
) -> None:
    async with dendrux_kit.live_store() as stored:
        run_id = await stored.new_run()
        assert run_id in stored.updates.ids, "create_run announced nothing before it returned"
        seen = stored.seen_when_announced(run_id)
        assert seen is not None and seen.exists, (
            "create_run announced the run before a second connection could see its row"
        )
        stored.visibility_checked()


@live
async def test_live_a_run_event_is_announced_after_the_event_is_visible_elsewhere(
    dendrux_kit: Kit,
) -> None:
    async with dendrux_kit.live_store() as stored:
        run_id = await stored.new_run()
        await stored.settle()
        await stored.store.save_run_event(run_id, event_type=EVENT, data={"n": 1})
        assert stored.updates.ids == [run_id], "save_run_event must announce exactly its run"
        seen = stored.seen_when_announced(run_id)
        assert seen is not None and seen.events == 1, (
            "save_run_event announced the run before a second connection could see its event"
        )
        assert stored.updates.batches[0].probe_error is None
        stored.visibility_checked()


@live
async def test_live_a_status_change_is_announced_after_the_new_status_is_visible_elsewhere(
    dendrux_kit: Kit,
) -> None:
    async with dendrux_kit.live_store() as stored:
        run_id = await stored.new_run()
        await stored.settle()
        await stored.store.finalize_run(run_id, status=RunStatus.SUCCESS.value, answer="done")
        assert stored.updates.ids == [run_id], "a status-only change must be announced"
        seen = stored.seen_when_announced(run_id)
        assert seen is not None and seen.status == RunStatus.SUCCESS.value, (
            "finalize_run announced the run before a second connection could see its new status"
        )
        stored.visibility_checked()


@live_not_applied
async def test_live_a_write_proved_not_applied_announces_nothing(dendrux_kit: Kit) -> None:
    async with dendrux_kit.live_store() as stored:
        run_id = await stored.new_run()
        await stored.settle()
        outcome = await dendrux_kit.faulted(
            "before_commit",
            # the event identity lets the store prove, after the failure, that nothing was written
            lambda: stored.store.save_run_event(run_id, event_type=EVENT, event_id=ulid()),
            sticky=True,
            store=stored.store,
        )
        outcome.expect_fired("before_commit")
        assert outcome.raised is not None, "a write that cannot commit must fail"
        await stored.updates.flush()  # send whatever the store left pending
        assert await stored.events(run_id) == 0, "nothing may persist"
        assert run_id not in stored.updates.ids, (
            "a write that was proved not to have applied was announced"
        )


@live_issued
@pytest.mark.parametrize(
    ("kind", "identity"), [("error", True), ("error", False), ("cancel", True)]
)
async def test_live_a_commit_that_was_issued_is_announced_even_when_the_call_failed(
    dendrux_kit: Kit, kind: str, identity: bool
) -> None:
    async with dendrux_kit.live_store() as stored:
        run_id = await stored.new_run()
        await stored.settle()
        extra = (
            {"event_id": ulid()} if identity and dendrux_kit.declaration.has("retry_ids") else {}
        )
        outcome = await dendrux_kit.faulted(
            "after_commit",
            lambda: stored.store.save_run_event(run_id, event_type=EVENT, **extra),
            kind=kind,  # type: ignore[arg-type]
            sticky=True,
            store=stored.store,
        )
        outcome.expect_fired("after_commit")
        assert await stored.events(run_id) == 1, "the commit happened: the write must persist"
        if outcome.raised is not None:
            # a failing call never waits for an announcement; its ids go out with the next flush
            assert run_id not in stored.updates.ids, "a call that raised flushed on its way out"
        await stored.updates.flush()
        assert run_id in stored.updates.ids, "a write whose COMMIT was issued was never announced"
        seen = stored.seen_when_announced(run_id)
        assert seen is not None and seen.events == 1
        stored.visibility_checked()
