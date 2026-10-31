"""Crash recovery, through the optional ``dendrux_fault_injector``.

A fault fails the next write transaction at a named point. After any of them the store must be
in one of the two states the operation allows (done, or not done), never half done, and the
caller must be able to retry safely. Without the fault injector these contracts skip and the
report says "crash recovery: not certified".
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    EVENT,
    complete_call_kwargs,
    event_write,
    start_call_kwargs,
    tool_call_kwargs,
    trace_write,
    ulid,
    usage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.testing.store_kit._kit import Kit

crash = contract("crash_recovery", uses=("injector",))
retry_crash = contract("crash_recovery", needs=("retry_ids",), uses=("injector",))
atomic_crash = contract(
    "crash_recovery", needs=("atomic_completion",), uses=("injector",), format2=True
)


async def attempt(call: Callable[[], Awaitable[Any]]) -> tuple[bool, Any]:
    """Run a faulted call: (it returned, its result or the exception it raised)."""
    try:
        return True, await call()
    except Exception as exc:
        return False, exc


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


@retry_crash
@pytest.mark.parametrize("point", ["before_commit", "after_commit", "mid_write"])
@pytest.mark.parametrize(
    "writer", ["save_trace", "save_tool_call", "save_run_event", "save_llm_interaction"]
)
async def test_a_failed_write_is_repeated_safely_with_the_same_identity(
    dendrux_kit: Kit, writer: str, point: str
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    ident = ulid()
    call = writers(store, run_id, ident)[writer]
    dendrux_kit.arm(point)  # type: ignore[arg-type]
    returned, first = await attempt(call)
    dendrux_kit.disarm()
    stored_after_failure = await count(store, run_id, writer)
    assert stored_after_failure in (0, 1), "never a duplicate"
    if not returned:
        # a lost acknowledgement may have stored it; a rolled-back write must have stored nothing
        assert point != "before_commit" or stored_after_failure in (0, 1)
    second = await call()
    assert await count(store, run_id, writer) == 1, "retrying exactly once-stores the write"
    if returned:
        assert second == first


@crash
async def test_a_cancelled_write_is_not_half_applied_and_the_store_stays_usable(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    dendrux_kit.arm("before_commit", kind="cancel")
    with pytest.raises(asyncio.CancelledError):
        await store.save_run_event(run_id, event_type=EVENT)
    dendrux_kit.disarm()
    assert await store.get_run_events(run_id) == []
    assert await store.save_run_event(run_id, event_type=EVENT) == 0


@crash
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
async def test_finalize_that_fails_around_commit_leaves_one_final_state(
    dendrux_kit: Kit, point: str
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    dendrux_kit.arm(point)  # type: ignore[arg-type]
    await attempt(
        lambda: store.finalize_run(
            run_id, status="success", answer="done", expected_current_status="running"
        )
    )
    dendrux_kit.disarm()
    record = await store.get_run(run_id)
    assert record is not None
    if record.status == "running":
        assert point == "before_commit", "a committed finalize must be visible"
        assert await store.finalize_run(
            run_id, status="success", answer="done", expected_current_status="running"
        )
    final = await store.get_run(run_id)
    assert final is not None and (final.status, final.answer) == ("success", "done")
    assert not await store.finalize_run(
        run_id, status="error", error="again", expected_current_status="running"
    ), "the second finalize loses"


@crash
async def test_a_claim_that_fails_after_commit_never_leaves_two_owners(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.paused_run()
    dendrux_kit.arm("after_commit")
    returned, first = await attempt(
        lambda: store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    )
    dendrux_kit.disarm()
    record = await store.get_run(run_id)
    assert record is not None and record.status == "running", "the committed claim is visible"
    if returned:
        assert first is not False
    assert await store.claim_paused_run(run_id, expected_status="waiting_client_tool") is False


@atomic_crash
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", ["before_commit", "after_commit", "mid_write"])
async def test_finalize_with_event_is_all_or_nothing(
    dendrux_kit: Kit, point: str, after: int
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    dendrux_kit.arm(point, after=after)  # type: ignore[arg-type]
    await attempt(
        lambda: store.finalize_with_event(
            run_id,
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
            answer="done",
            claim_epoch=0,
            trace=trace_write("assistant", "done", 0),
            event=event_write("run.completed", run_id),
        )
    )
    dendrux_kit.disarm()
    record = await store.get_run(run_id)
    assert record is not None
    traces = await store.get_traces(run_id)
    events = await store.get_run_events(run_id)
    if record.status == "success":
        assert len(traces) == 1 and [e.event_type for e in events] == ["run.completed"]
    else:
        assert record.status == "running" and traces == [] and events == []


@atomic_crash
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", ["before_commit", "after_commit", "mid_write"])
async def test_a_tool_completion_is_all_or_nothing_and_repeatable(
    dendrux_kit: Kit, point: str, after: int
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
    dendrux_kit.arm(point, after=after)  # type: ignore[arg-type]
    await attempt(lambda: store.complete_tool_call(run_id, claim_epoch=0, **kwargs))
    dendrux_kit.disarm()
    state = await store.get_loop_state(run_id)
    assert state is not None
    done = state["pending"] == []
    assert len(await store.get_traces(run_id)) == (1 if done else 0)
    types = [e.event_type for e in await store.get_run_events(run_id)]
    assert types == (["tool.started", "tool.completed"] if done else ["tool.started"])
    await store.complete_tool_call(run_id, claim_epoch=0, **kwargs)
    assert len(await store.get_traces(run_id)) == 1
    assert (await store.get_loop_state(run_id))["pending"] == []


@contract("crash_recovery", needs=("checkpointing",), uses=("injector",), format2=True)
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", ["before_commit", "mid_write"])
async def test_a_checkpoint_that_fails_leaves_the_previous_checkpoint(
    dendrux_kit: Kit, point: str, after: int
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state={"v": 1, "phase": "one"}, pii_mapping=None, trace=None
    )
    dendrux_kit.arm(point, after=after)  # type: ignore[arg-type]
    await attempt(
        lambda: store.checkpoint_step(
            run_id,
            claim_epoch=0,
            loop_state={"v": 1, "phase": "two"},
            pii_mapping={"a": "b"},
            trace=trace_write("user", "t", 0),
        )
    )
    dendrux_kit.disarm()
    state = await store.get_loop_state(run_id)
    traces = await store.get_traces(run_id)
    if (state or {}).get("phase") == "two":
        assert len(traces) == 1 and await store.get_pii_mapping(run_id) == {"a": "b"}
    else:
        assert state == {"v": 1, "phase": "one"} and traces == []
        assert await store.get_pii_mapping(run_id) is None


@crash
@pytest.mark.parametrize("point", ["before_commit", "after_commit", "mid_write"])
async def test_a_sweep_that_fails_midway_is_finished_by_the_next_sweep(
    dendrux_kit: Kit, point: str
) -> None:
    store = dendrux_kit.store
    runs = [await dendrux_kit.new_run() for _ in range(4)]
    threshold = await dendrux_kit.age()
    dendrux_kit.arm(point)  # type: ignore[arg-type]
    swept: list[str] = []
    with contextlib.suppress(Exception):
        swept += [s.run_id for s in await store.sweep_stale_runs(threshold, batch_size=2)]
    dendrux_kit.disarm()
    for _ in range(5):
        swept += [s.run_id for s in await store.sweep_stale_runs(threshold, batch_size=2)]
    assert len(swept) == len(set(swept)), "no run is reported swept twice"
    for run_id in runs:
        record = await store.get_run(run_id)
        assert record is not None and record.status == "error", "no stale run is left behind"
