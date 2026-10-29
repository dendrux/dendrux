"""Tool intent, completion and submission transaction boundaries."""

import inspect

import pytest

from dendrux.errors import PersistenceError
from dendrux.runtime import state
from dendrux.types import generate_ulid


def event(kind, call_id):
    cls = getattr(state, "EventWrite", None)
    assert cls is not None, "atomic events need a stable write descriptor"
    return cls(generate_ulid(), kind, 0, {"call_id": call_id}, call_id, None)


def observation(call_id, order):
    return state.TraceWrite(generate_ulid(), "tool", '"ok"', order, meta={"call_id": call_id})


async def intent(store, run_id, call_id):
    assert hasattr(store, "start_tool_call"), "write intent before side effects"
    await store.start_tool_call(
        run_id,
        claim_epoch=0,
        row_id=generate_ulid(),
        tool_call_id=call_id,
        provider_tool_call_id=None,
        tool_name="search",
        target="server",
        params={},
        iteration_index=0,
        idempotent=True,
        event=event("tool.started", call_id),
    )


async def test_a20_parallel_completions_patch_current_pending(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "complete")
    assert hasattr(store, "checkpoint_step")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "pending": [{"call_id": "a"}, {"call_id": "b"}]},
        pii_mapping=None,
        trace=None,
    )
    await intent(store, run_id, "a")
    await intent(store, run_id, "b")
    patch_type = getattr(state, "LoopStatePatch", None)
    assert patch_type is not None, "parallel completions must patch under the run lock"

    async def finish(call_id, order, *, active_store=store):
        await active_store.complete_tool_call(
            run_id,
            claim_epoch=0,
            tool_call_id=call_id,
            outcome="completed",
            result_payload='"ok"',
            success=True,
            duration_ms=1,
            error_message=None,
            replayed=False,
            observation=observation(call_id, order),
            event=event("tool.completed", call_id),
            loop_state_patch=patch_type(remove_pending=(call_id,)),
        )

    from tests.integration.test_060_core_recovery import race

    outcomes = await race(
        store,
        engine,
        [
            lambda s: finish("b", 2, active_store=s),
            lambda s: finish("a", 1, active_store=s),
        ],
    )
    assert outcomes == [None, None]
    assert (await store.get_loop_state(run_id))["pending"] == []
    assert [r.meta["call_id"] for r in await store.get_traces(run_id)] == ["a", "b"]
    assert len(await store.get_run_events(run_id)) == 4
    # An acknowledgement lost after commit must not patch again or append evidence twice.
    await finish("a", 1)
    assert len(await store.get_run_events(run_id)) == 4
    with pytest.raises(PersistenceError) as caught:
        await store.complete_tool_call(
            run_id,
            claim_epoch=0,
            tool_call_id="a",
            outcome="failed",
            result_payload='"changed"',
            success=False,
            duration_ms=8,
            error_message="safe",
            replayed=False,
            observation=observation("a", 1),
            event=event("tool.failed", "a"),
            loop_state_patch=patch_type(remove_pending=("a",)),
        )
    assert caught.value.code == "write_conflict"


async def test_submission_consumed_once(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "submission")
    assert hasattr(store, "consume_submission"), "submission consumption must be atomic"
    descriptor = {
        "submission_id": generate_ulid(),
        "kind": "input",
        "pause_id": "p",
        "call_ids": [],
        "reserved_order_index": 1,
    }
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"phase": "awaiting_input", "submission": descriptor},
        pii_mapping=None,
        trace=None,
    )
    await store.pause_run(
        run_id,
        status="waiting_human_input",
        pause_data={
            "schema_version": 2,
            "pause_id": "p",
            "submission": {**descriptor, "payload": "hello"},
        },
    )
    claim = await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    trace = state.TraceWrite(generate_ulid(), "user", "hello", 1)
    patch = state.LoopStatePatch(set_phase="model_next")
    for expected in [True, False]:
        assert (
            await store.consume_submission(
                run_id,
                claim_epoch=claim.claim_epoch,
                submission_id=descriptor["submission_id"],
                traces=[trace],
                patch=patch,
            )
            is expected
        )
    assert (await store.get_loop_state(run_id)) == {"phase": "model_next", "submission": None}
    assert "submission" not in (await store.get_pause_state(run_id))
    assert len(await store.get_traces(run_id)) == 1


async def test_final_trace_and_finalize_atomic(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "final")
    assert "trace" in inspect.signature(store.finalize_with_event).parameters
    trace = state.TraceWrite(generate_ulid(), "assistant", "done", 0)
    seq = await store.finalize_with_event(
        run_id,
        status="success",
        allowed_current_statuses=["running"],
        event_type="run.completed",
        answer="done",
        claim_epoch=0,
        trace=trace,
        event=event("run.completed", run_id),
    )
    assert seq == 0
    assert (await store.get_run(run_id)).status == "success"
    assert (await store.get_traces(run_id))[0].content == "done"


@pytest.mark.parametrize("field", ["params", "event"])
async def test_intent_rejects_unpersistable_values_atomically(store, field):
    from dendrux.types import InputValidationError

    run_id = generate_ulid()
    await store.create_run(run_id, "invalid")
    item = event("tool.started", "c")
    if field == "event":
        from dataclasses import replace

        item = replace(item, data={"call_id": "\x00"})
    with pytest.raises(InputValidationError):
        await store.start_tool_call(
            run_id,
            claim_epoch=0,
            row_id=generate_ulid(),
            tool_call_id="c",
            provider_tool_call_id=None,
            tool_name="search",
            target="server",
            params={"q": "\x00"} if field == "params" else {},
            iteration_index=0,
            idempotent=True,
            event=item,
        )
    assert not await store.get_tool_calls(run_id)
    assert not await store.get_run_events(run_id)
