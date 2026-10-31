"""Crash recovery, through the optional ``dendrux_fault_injector``.

A fault fails write transactions of a store at a named point. Each contract first asserts that
the cut it asked for really fired (an injector that does nothing fails certification), then
asserts the exact outcome:

- ``before_commit`` / ``mid_write`` with a fault that keeps firing: the call fails and nothing
  persisted, not even part of an operation;
- ``after_commit``: the write persisted although the caller saw a failure, and repeating it by its
  identity stores it exactly once.

Without the fault injector these contracts skip and the report says "crash recovery: not
certified".
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.errors import StoreOutcomeUnknown
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    EVENT,
    RECLAIM,
    complete_call_kwargs,
    event_write,
    lease_cutoff,
    mark,
    start_call_kwargs,
    started_run,
    tool_call_kwargs,
    trace_write,
    ulid,
    usage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.testing.store_kit._kit import Faulted, Kit
    from dendrux.testing.store_kit._protocols import FaultPoint

crash = contract("crash_recovery", uses=("injector",))
retry_crash = contract("crash_recovery", needs=("retry_ids",), uses=("injector",))
atomic_crash = contract(
    "crash_recovery", needs=("atomic_completion",), uses=("injector",), format2=True
)
POINTS: list[FaultPoint] = ["before_commit", "after_commit", "mid_write"]
ROLLED_BACK: list[FaultPoint] = ["before_commit", "mid_write"]


def writers(store: Any, run_id: str, ident: str) -> dict[str, Callable[[], Awaitable[Any]]]:
    return {
        "save_trace": lambda: store.save_trace(
            run_id, "user", "same", order_index=0, trace_id=ident
        ),
        "save_tool_call": lambda: store.save_tool_call(
            run_id, **tool_call_kwargs(tool_call_id="call-1"), row_id=ident
        ),
        "save_run_event": lambda: store.save_run_event(
            run_id, event_type=EVENT, data={"n": 1}, event_id=ident
        ),
        "save_llm_interaction": lambda: store.save_llm_interaction(
            run_id, iteration_index=0, usage=usage(), interaction_id=ident
        ),
    }


async def count(store: Any, run_id: str, writer: str) -> int:
    reads = {
        "save_trace": store.get_traces,
        "save_tool_call": store.get_tool_calls,
        "save_run_event": store.get_run_events,
        "save_llm_interaction": store.get_llm_interactions,
    }
    return len(await reads[writer](run_id))


WRITER_NAMES = ["save_trace", "save_tool_call", "save_run_event", "save_llm_interaction"]


@retry_crash
@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("writer", WRITER_NAMES)
async def test_a_failed_write_is_repeated_safely_with_the_same_identity(
    dendrux_kit: Kit, writer: str, point: FaultPoint
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    call = writers(store, run_id, ulid())[writer]
    outcome = await dendrux_kit.faulted(point, call, sticky=True)
    outcome.expect_fired(point)
    stored = await count(store, run_id, writer)
    if point in ROLLED_BACK:
        assert outcome.raised is not None, "a write that cannot commit must fail"
        assert stored == 0, f"{point}: nothing may persist"
    else:
        assert stored == 1, "after_commit: the write persisted although the caller saw a failure"
    first = await call()
    assert await count(store, run_id, writer) == 1, "repeating by identity stores it exactly once"
    assert await call() == first


@retry_crash
@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("writer", WRITER_NAMES)
async def test_a_one_shot_failure_is_absorbed_or_retried_into_exactly_one_write(
    dendrux_kit: Kit, writer: str, point: FaultPoint
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    call = writers(store, run_id, ulid())[writer]
    outcome = await dendrux_kit.faulted(point, call)
    outcome.expect_fired(point)
    await call()
    assert await count(store, run_id, writer) == 1


@crash
async def test_a_cancelled_write_is_not_half_applied_and_the_store_stays_usable(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    outcome = await dendrux_kit.faulted(
        "before_commit", lambda: store.save_run_event(run_id, event_type=EVENT), kind="cancel"
    )
    outcome.expect_fired("before_commit")
    assert isinstance(outcome.raised, asyncio.CancelledError)
    assert await store.get_run_events(run_id) == []
    assert await store.save_run_event(run_id, event_type=EVENT) == 0


@crash
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
async def test_finalize_that_fails_around_commit_leaves_one_final_state(
    dendrux_kit: Kit, point: FaultPoint
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()

    async def finalize() -> Any:
        return await store.finalize_run(
            run_id, status="success", answer="done", expected_current_status="running"
        )

    outcome = await dendrux_kit.faulted(point, finalize, sticky=True)
    outcome.expect_fired(point)
    record = await store.get_run(run_id)
    assert record is not None
    if point == "before_commit":
        assert outcome.raised is not None and record.status == "running"
        assert await finalize() is True
    else:
        assert (record.status, record.answer) == ("success", "done"), "the commit is visible"
    final = await store.get_run(run_id)
    assert final is not None and (final.status, final.answer) == ("success", "done")
    assert not await store.finalize_run(
        run_id, status="error", error="again", expected_current_status="running"
    ), "the second finalize loses"


CLAIM_OPERATIONS = ["claim_paused_run", "submit_and_claim", "reclaim_run", "release_run_claim"]


@crash
@pytest.mark.parametrize("form", ["invalidated", "native", "native_alt"])
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
@pytest.mark.parametrize("operation", CLAIM_OPERATIONS)
async def test_an_ambiguous_claim_commit_raises_outcome_unknown(
    dendrux_kit: Kit, operation: str, point: FaultPoint, form: str
) -> None:
    """After a commit that may or may not have landed, ownership is never reported.

    Two contenders expecting epoch ``e`` both see ``e + 1``, so a store cannot prove the claim
    was its own. It must raise ``StoreOutcomeUnknown`` and never return a claim.
    """
    store = dendrux_kit.store
    if form not in dendrux_kit.disconnect_forms():
        pytest.skip(f"the {form} disconnect form does not exist for this dialect")
    committed = point == "after_commit"
    if operation == "reclaim_run":
        if not dendrux_kit.declaration.has("recovery"):
            pytest.skip("recovery is not declared")
        dendrux_kit.require_format_2()
        run_id = await started_run(dendrux_kit)
        [marked] = (await mark(dendrux_kit, await lease_cutoff(dendrux_kit))).reclaimable

        async def call() -> Any:
            return await store.reclaim_run(
                run_id,
                tenant_id="t",
                agent_name="KitAgent",
                expected_claim_epoch=marked.claim_epoch,
                **RECLAIM,
            )

        async def landed() -> bool:
            record = await store.get_run(run_id)
            assert record is not None
            if committed:
                return bool(record.status == "running" and record.reclaim_attempts == 1)
            return bool(record.reclaim_attempts == 0 and record.lease_expired_at is not None)

    elif operation == "release_run_claim":
        if not hasattr(store, "release_run_claim"):
            pytest.skip("the optional release_run_claim is not implemented")
        run_id, claim = await dendrux_kit.claimed_run()

        async def call() -> Any:
            return await store.release_run_claim(
                run_id, status="waiting_client_tool", claim_epoch=claim.claim_epoch
            )

        async def landed() -> bool:
            record = await store.get_run(run_id)
            assert record is not None
            return bool(record.status == ("waiting_client_tool" if committed else "running"))

    else:
        run_id = await dendrux_kit.paused_run()
        envelope = {
            "submission_id": ulid(),
            "kind": "tool_results",
            "pause_id": "kit-pause",
            "call_ids": ["c1"],
        }

        async def call() -> Any:
            if operation == "claim_paused_run":
                return await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
            return await store.submit_and_claim(
                run_id,
                expected_status="waiting_client_tool",
                submitted_data={"submitted_tool_results": []},
                submission=envelope,
            )

        async def landed() -> bool:
            record = await store.get_run(run_id)
            assert record is not None
            return bool(record.status == ("running" if committed else "waiting_client_tool"))

    outcome = await dendrux_kit.faulted(point, call, sticky=True, form=form)
    outcome.expect_fired(point)
    assert isinstance(outcome.raised, StoreOutcomeUnknown), (
        f"{operation} after an ambiguous {point} ({form}) must raise StoreOutcomeUnknown, "
        f"got {type(outcome.raised).__name__ if outcome.raised else repr(outcome.result)[:60]}"
    )
    assert outcome.result is None, "no ownership may be reported"
    assert await landed(), f"{point}: the stored state must be exactly what the commit decided"
    if committed and operation in ("claim_paused_run", "submit_and_claim"):
        assert not await store.claim_paused_run(run_id, expected_status="waiting_client_tool"), (
            "nobody can claim a claimed run again"
        )


async def _all_or_nothing(
    point: FaultPoint,
    after: int,
    outcome: Faulted,
    done: Callable[[], Awaitable[bool]],
    nothing: Callable[[], Awaitable[bool]],
) -> None:
    """Exactly done, or exactly nothing: never half, and never a success that did not happen."""
    if after == 0 and point in ROLLED_BACK:
        assert outcome.raised is not None, (
            f"{point}: a rolled-back operation must fail, not succeed"
        )
        assert await nothing(), f"{point}: the failed operation left something behind"
    elif after == 0:
        assert await done(), "after_commit: the committed operation must be fully visible"
    else:
        assert await done() or await nothing(), "an operation cut between transactions is half done"
        if outcome.raised is None:
            assert await done(), "an operation that reported success must be complete"
        if outcome.fired == 0:
            assert await done(), "a single-transaction operation the cut missed must be complete"


@atomic_crash
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", POINTS)
async def test_finalize_with_event_is_all_or_nothing(
    dendrux_kit: Kit, point: FaultPoint, after: int
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    outcome = await dendrux_kit.faulted(
        point,
        lambda: store.finalize_with_event(
            run_id,
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
            answer="done",
            claim_epoch=0,
            trace=trace_write("assistant", "done", 0),
            event=event_write("run.completed", run_id),
        ),
        after=after,
        sticky=True,
    )
    if after == 0:
        outcome.expect_fired(point)

    async def done() -> bool:
        record = await store.get_run(run_id)
        events = [e.event_type for e in await store.get_run_events(run_id)]
        return (
            record is not None
            and record.status == "success"
            and len(await store.get_traces(run_id)) == 1
            and events == ["run.completed"]
        )

    async def nothing() -> bool:
        record = await store.get_run(run_id)
        return (
            record is not None
            and record.status == "running"
            and await store.get_traces(run_id) == []
            and await store.get_run_events(run_id) == []
        )

    await _all_or_nothing(point, after, outcome, done, nothing)


@atomic_crash
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", POINTS)
async def test_a_tool_completion_is_all_or_nothing_and_repeatable(
    dendrux_kit: Kit, point: FaultPoint, after: int
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "pending": [{"call_id": "a"}]},
        pii_mapping=None,
        trace=None,
    )
    await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    kwargs = complete_call_kwargs("a", 1)
    outcome = await dendrux_kit.faulted(
        point,
        lambda: store.complete_tool_call(run_id, claim_epoch=0, **kwargs),
        after=after,
        sticky=True,
    )
    if after == 0:
        outcome.expect_fired(point)

    async def done() -> bool:
        state = await store.get_loop_state(run_id)
        types = [e.event_type for e in await store.get_run_events(run_id)]
        return (
            state is not None
            and state["pending"] == []
            and len(await store.get_traces(run_id)) == 1
            and types == ["tool.started", "tool.completed"]
        )

    async def nothing() -> bool:
        state = await store.get_loop_state(run_id)
        types = [e.event_type for e in await store.get_run_events(run_id)]
        return (
            state is not None
            and [p["call_id"] for p in state["pending"]] == ["a"]
            and await store.get_traces(run_id) == []
            and types == ["tool.started"]
        )

    await _all_or_nothing(point, after, outcome, done, nothing)
    await store.complete_tool_call(run_id, claim_epoch=0, **kwargs)
    assert await done(), "repeating the completion finishes it exactly once"


@contract("crash_recovery", needs=("checkpointing",), uses=("injector",), format2=True)
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", ROLLED_BACK)
async def test_a_checkpoint_that_fails_leaves_the_previous_checkpoint(
    dendrux_kit: Kit, point: FaultPoint, after: int
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state={"v": 1, "phase": "one"}, pii_mapping=None, trace=None
    )
    outcome = await dendrux_kit.faulted(
        point,
        lambda: store.checkpoint_step(
            run_id,
            claim_epoch=0,
            loop_state={"v": 1, "phase": "two"},
            pii_mapping={"a": "b"},
            trace=trace_write("user", "t", 0),
        ),
        after=after,
        sticky=True,
    )
    if after == 0:
        outcome.expect_fired(point)

    async def done() -> bool:
        return (
            await store.get_loop_state(run_id) == {"v": 1, "phase": "two"}
            and len(await store.get_traces(run_id)) == 1
            and await store.get_pii_mapping(run_id) == {"a": "b"}
        )

    async def nothing() -> bool:
        return (
            await store.get_loop_state(run_id) == {"v": 1, "phase": "one"}
            and await store.get_traces(run_id) == []
            and await store.get_pii_mapping(run_id) is None
        )

    await _all_or_nothing(point, after, outcome, done, nothing)


@crash
@pytest.mark.parametrize("point", POINTS)
async def test_a_sweep_that_fails_midway_is_finished_by_the_next_sweep(
    dendrux_kit: Kit, point: FaultPoint
) -> None:
    store = dendrux_kit.store
    runs = [await dendrux_kit.new_run() for _ in range(4)]
    threshold = await dendrux_kit.age()
    swept: list[str] = []

    async def sweep() -> list[str]:
        return [s.run_id for s in await store.sweep_stale_runs(threshold, batch_size=2)]

    outcome = await dendrux_kit.faulted(point, sweep, sticky=True)
    outcome.expect_fired(point)
    if point in ROLLED_BACK:
        assert outcome.raised is not None, "a sweep that cannot commit must fail"
        for run_id in runs:
            record = await store.get_run(run_id)
            assert record is not None and record.status == "running", "nothing was swept"
    else:
        swept += [s for s in (outcome.result or [])]
    for _ in range(5):
        swept += await sweep()
    assert len(swept) == len(set(swept)), "no run is swept twice"
    for run_id in runs:
        record = await store.get_run(run_id)
        assert record is not None and record.status == "error", "no stale run is left behind"
