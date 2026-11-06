"""Exact announcement ids for the positive list and polling-only writers."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux.db.models import AgentRun
from dendrux.runtime.state import EventWrite, LoopStatePatch, TraceWrite
from dendrux.types import UsageStats, generate_ulid
from tests.integration.test_live_updates_store_transactions import live_store as live_store
from tests.integration.test_live_updates_store_transactions import sent

ANNOUNCING = [
    "create_run",
    "save_run_event",
    "start_tool_call",
    "complete_tool_call",
    "consume_submission",
    "finalize_run",
    "finalize_run_if_status_in",
    "finalize_with_event",
    "try_pause_run",
    "pause_run",
    "pause_with_event",
    "claim_paused_run",
    "submit_and_claim",
    "release_run_claim",
    "request_cancel",
    "request_pause",
    "mark_cancel_requested",
    "update_run_meta",
]
POLLING = [
    "save_trace",
    "save_tool_call",
    "save_usage",
    "save_llm_interaction",
    "checkpoint_step",
    "set_trace_link",
    "add_note",
    "mark_notes_delivered",
    "touch_progress",
]


def tool_args():
    return dict(
        tool_call_id="call",
        provider_tool_call_id=None,
        tool_name="tool",
        target="server",
        params={},
        iteration_index=1,
    )


def event():
    return EventWrite(generate_ulid(), "tool.started", 1, {}, None, None)


async def prepare(store, engine, run_id, method):
    await store.create_run(run_id, "live", strategy="react")
    if method in ("claim_paused_run", "submit_and_claim"):
        await store.pause_run(
            run_id, status="waiting_human_input", pause_data={"pause_id": generate_ulid()}
        )
    if method == "complete_tool_call":
        await store.checkpoint_step(
            run_id,
            claim_epoch=0,
            loop_state={"v": 1, "phase": "dispatch"},
            pii_mapping=None,
            trace=None,
        )
        await store.start_tool_call(
            run_id,
            claim_epoch=0,
            row_id=generate_ulid(),
            idempotent=True,
            event=event(),
            **tool_args(),
        )
    if method == "consume_submission":
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    loop_state={
                        "v": 1,
                        "phase": "model_next",
                        "submission": {"submission_id": "submission"},
                    }
                )
            )
    if method == "mark_notes_delivered":
        await store.add_note(run_id, note_id=run_id, text="note")


def arguments(method, run_id):
    pause = dict(status="waiting_human_input", pause_data={"pause_id": generate_ulid()})
    complete = dict(result_payload='"ok"', success=True, duration_ms=1)
    return {
        "create_run": ((run_id, "live"), {}),
        "save_run_event": ((run_id,), dict(event_type="run.started")),
        "start_tool_call": (
            (run_id,),
            dict(
                claim_epoch=0, row_id=generate_ulid(), idempotent=True, event=event(), **tool_args()
            ),
        ),
        "complete_tool_call": (
            (run_id,),
            dict(
                claim_epoch=0,
                tool_call_id="call",
                outcome="completed",
                error_message=None,
                replayed=False,
                observation=TraceWrite(generate_ulid(), "tool", "ok", 2),
                event=event(),
                loop_state_patch=LoopStatePatch(set_phase="model_next"),
                **complete,
            ),
        ),
        "consume_submission": (
            (run_id,),
            dict(
                claim_epoch=0,
                submission_id="submission",
                traces=[],
                patch=LoopStatePatch(set_phase="model_next"),
            ),
        ),
        "finalize_run": ((run_id,), dict(status="completed")),
        "finalize_run_if_status_in": (
            (run_id,),
            dict(status="completed", allowed_current_statuses=["running"]),
        ),
        "finalize_with_event": (
            (run_id,),
            dict(
                status="completed", allowed_current_statuses=["running"], event_type="run.completed"
            ),
        ),
        "try_pause_run": ((run_id,), pause),
        "pause_run": ((run_id,), pause),
        "pause_with_event": ((run_id,), pause),
        "claim_paused_run": ((run_id,), dict(expected_status="waiting_human_input")),
        "submit_and_claim": (
            (run_id,),
            dict(
                expected_status="waiting_human_input",
                submitted_data={"input": "ok"},
                submission={
                    "submission_id": generate_ulid(),
                    "kind": "input",
                    "pause_id": generate_ulid(),
                    "call_ids": [],
                },
            ),
        ),
        "release_run_claim": ((run_id,), dict(status="paused", claim_epoch=0)),
        "request_cancel": ((run_id,), {}),
        "request_pause": ((run_id,), {}),
        "mark_cancel_requested": (([run_id],), dict(tenant_id=None)),
        "update_run_meta": ((run_id, {"test": True}), {}),
        "save_trace": ((run_id, "user", "test"), dict(order_index=0)),
        "save_tool_call": ((run_id,), dict(**tool_args(), **complete)),
        "save_usage": ((run_id,), dict(iteration_index=0, usage=UsageStats())),
        "save_llm_interaction": ((run_id,), dict(iteration_index=0, usage=UsageStats())),
        "checkpoint_step": (
            (run_id,),
            dict(
                claim_epoch=0,
                loop_state={"v": 1, "phase": "model_next"},
                pii_mapping=None,
                trace=None,
            ),
        ),
        "set_trace_link": (
            (run_id,),
            dict(claim_epoch=0, trace_link="00-" + "a" * 32 + "-" + "b" * 16 + "-01"),
        ),
        "add_note": ((run_id,), dict(note_id=generate_ulid(), text="note")),
        "mark_notes_delivered": ((run_id, [run_id]), dict(iteration=0)),
        "touch_progress": ((run_id,), {}),
    }[method]


@pytest.mark.parametrize("method", ANNOUNCING + POLLING)
async def test_writer_exact_ids(live_store, engine, method):
    store, bus = live_store
    run_id = generate_ulid()
    if method != "create_run":
        await prepare(store, engine, run_id, method)
    bus.messages.clear()
    args, kwargs = arguments(method, run_id)
    await getattr(store, method)(*args, **kwargs)
    assert sent(bus) == ([run_id] if method in ANNOUNCING else [])
    assert bus.stats().pending == 0


@pytest.mark.parametrize("cancelled", [False, True])
async def test_reclaim_success_and_commit_then_raise(live_store, engine, cancelled):
    from dendrux.errors import RunNotReclaimableError

    store, bus = live_store
    run_id = generate_ulid()
    await store.create_run(run_id, "live", strategy="react")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                storage_format=2,
                loop_state={"v": 1, "loop": "react", "skills": {"pins": {}}},
                lease_expired_at=datetime.now(UTC) - timedelta(minutes=2),
                reclaim_limit=3,
                cancel_requested=cancelled,
            )
        )
    bus.messages.clear()
    kwargs = dict(
        tenant_id=None,
        agent_name="live",
        expected_claim_epoch=0,
        governance={},
        governance_policy="warn",
        skill_pins={},
        loop_type="react",
    )
    if cancelled:
        with pytest.raises(RunNotReclaimableError) as raised:
            await store.reclaim_run(run_id, **kwargs)
        assert raised.value.reason == "cancelled"
        assert sent(bus) == []
        assert bus.stats().pending == 1
        assert await store.get_run_status(run_id) == "cancelled"
    else:
        assert await store.reclaim_run(run_id, **kwargs)
    assert sent(bus) == [run_id]


async def test_frontier_announces_all_exact_requested_ids(live_store):
    store, bus = live_store
    ids = [generate_ulid() for _ in range(20)]
    for run_id in ids:
        await store.create_run(run_id, "frontier")
    bus.messages.clear()
    assert await store.mark_cancel_requested(ids, tenant_id=None) == 20
    assert set(sent(bus)) == set(ids)
    assert len(sent(bus)) == 20


async def test_blob_catalog_writes_never_announce(live_store):
    from dendrux.media._store import BlobInfo

    store, bus = live_store
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=1)
    sha, generation = "a" * 64, generate_ulid()
    info = BlobInfo(sha, "image/png", 1, 1, 1, 1, "db", "pending", generation, now, now)
    assert await store.insert_row("", sha, info, state="pending", generation=generation, data=b"x")
    next_generation = generate_ulid()
    assert await store.adopt_pending("", sha, generation, next_generation)
    generation = next_generation
    assert await store.promote("", sha, generation)
    await store.touch_live("", [sha])
    assert await store.mark_deleting("", sha, cutoff, stale_pending_cutoff=cutoff) == generation
    assert await store.revert_deleting("", sha, generation)
    assert await store.mark_deleting_row("", sha, cutoff, stale_pending_cutoff=cutoff)
    next_generation = generate_ulid()
    assert await store.revive("", sha, generation, next_generation, to_state="live", data=b"x")
    generation = next_generation
    assert await store.mark_purging("", sha, None) == generation
    await store.finish_purge("", sha, generation)
    other = "b" * 64
    assert await store.insert_row("", other, info, state="live", generation=generation, data=b"x")
    assert await store.mark_deleting("", other, cutoff, stale_pending_cutoff=cutoff)
    assert await store.delete_row("", other, generation)
    assert sent(bus) == []
    assert bus.stats().pending == 0


@pytest.mark.parametrize("outcome", ["success", "pause_mismatch", "validation_failure"])
async def test_nonreturning_claim_validates_before_announcing(live_store, monkeypatch, outcome):
    from dendrux.errors import PauseStatusMismatchError

    store, bus = live_store
    run_id, pause_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "fallback")
    await store.pause_run(run_id, status="waiting_human_input", pause_data={"pause_id": pause_id})
    monkeypatch.setattr(store, "supports_update_returning", False)
    bus.messages.clear()
    seen = []
    error = ValueError("validation")

    def validate(pause):
        seen.append(pause["pause_id"])
        if outcome == "validation_failure":
            raise error

    kwargs = dict(
        expected_status="waiting_human_input", expected_pause_id=pause_id, validate=validate
    )
    if outcome == "pause_mismatch":
        kwargs["expected_pause_id"] = generate_ulid()
        with pytest.raises(PauseStatusMismatchError):
            await store.claim_paused_run(run_id, **kwargs)
    elif outcome == "validation_failure":
        with pytest.raises(ValueError) as raised:
            await store.claim_paused_run(run_id, **kwargs)
        assert raised.value is error
    else:
        assert await store.claim_paused_run(run_id, **kwargs)
        assert seen == [pause_id]
        assert sent(bus) == [run_id]
        return
    assert bus.stats().pending == 0
    assert await store.get_run_status(run_id) == "waiting_human_input"
    assert sent(bus) == []


async def test_submission_validation_rollback_never_announces(live_store):
    store, bus = live_store
    run_id = generate_ulid()
    await store.create_run(run_id, "submission")
    await store.pause_run(run_id, status="waiting_human_input", pause_data={})
    bus.messages.clear()
    error = ValueError("validation")

    def reject(pause):
        raise error

    with pytest.raises(ValueError) as raised:
        await store.submit_and_claim(
            run_id, expected_status="waiting_human_input", submitted_data={}, validate=reject
        )
    assert raised.value is error
    assert bus.stats().pending == 0
    assert await store.get_run_status(run_id) == "waiting_human_input"
    assert sent(bus) == []
