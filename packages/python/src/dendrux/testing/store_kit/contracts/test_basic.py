"""Basic contracts: core reads and writes, on whatever storage format the store creates.

Nothing here inspects the store's own tables or reaches into its engine: every check goes
through the ``StateStore`` protocol, so any store can be held to it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dendrux.errors import NoteLimitExceededError
from dendrux.runtime.state import UNSCOPED
from dendrux.testing.store_kit.contracts._marks import basic
from dendrux.testing.store_kit.contracts._support import EVENT, tool_call_kwargs, ulid, usage
from dendrux.types import IdempotencyConflictError, compute_idempotency_fingerprint

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

# -- runs ---------------------------------------------------------------------------------


@basic()
async def test_create_run_round_trips_what_it_was_given(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run(
        tenant_id="acme",
        model="kit-model",
        strategy="kit-strategy",
        meta={"customer": "acme", "attempt": 2, "flag": True, "none": None},
        input_data={"input": "héllo ✓ 日本語", "nested": {"a": [1, 2.5, None, True]}},
    )
    record = await store.get_run(run_id)
    assert record is not None
    assert (record.id, record.agent_name, record.status) == (run_id, "KitAgent", "running")
    assert (record.tenant_id, record.model, record.strategy) == (
        "acme",
        "kit-model",
        "kit-strategy",
    )
    assert record.meta is not None
    assert record.meta["customer"] == "acme"
    assert record.meta["attempt"] == 2
    assert record.meta["flag"] is True
    assert record.meta["none"] is None
    assert record.input_data == {
        "input": "héllo ✓ 日本語",
        "nested": {"a": [1, 2.5, None, True]},
    }
    assert record.cancel_requested is False
    assert record.iteration_count == 0


@basic()
async def test_timestamps_are_timezone_aware_utc(dendrux_kit: Kit) -> None:
    run_id = await dendrux_kit.new_run()
    record = await dendrux_kit.store.get_run(run_id)
    assert record is not None
    for stamp in (record.created_at, record.updated_at, record.last_progress_at):
        assert stamp is not None
        assert stamp.utcoffset() == timedelta(0), "stored times must read back as aware UTC"
    assert abs(datetime.now(UTC) - record.created_at) < timedelta(minutes=5)


@basic()
async def test_reading_a_missing_run_is_empty_not_an_error(dendrux_kit: Kit) -> None:
    store, missing = dendrux_kit.store, ulid()
    assert await store.get_run(missing) is None
    assert await store.get_traces(missing) == []
    assert await store.get_tool_calls(missing) == []
    assert await store.get_run_events(missing) == []
    assert await store.get_notes(missing) == []
    assert await store.get_pause_state(missing) is None
    assert await store.is_cancel_requested(missing) is False
    assert await store.get_delegation_info(missing) is None
    assert await store.get_subtree(missing) is None


@basic()
async def test_an_idempotency_key_returns_the_run_it_created(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    fingerprint = compute_idempotency_fingerprint("KitAgent", "same input")
    first = await dendrux_kit.new_run(idempotency_key="key-1", idempotency_fingerprint=fingerprint)
    again = await store.create_run(
        ulid(),
        "KitAgent",
        idempotency_key="key-1",
        idempotency_fingerprint=fingerprint,
    )
    assert again.outcome == "existing_active"
    assert again.run_id == first
    assert await store.count_runs() == 1


@basic()
async def test_an_idempotency_key_with_different_input_conflicts(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    await dendrux_kit.new_run(
        idempotency_key="key-2",
        idempotency_fingerprint=compute_idempotency_fingerprint("KitAgent", "one"),
    )
    with pytest.raises(IdempotencyConflictError):
        await store.create_run(
            ulid(),
            "KitAgent",
            idempotency_key="key-2",
            idempotency_fingerprint=compute_idempotency_fingerprint("KitAgent", "two"),
        )
    assert await store.count_runs() == 1


@basic()
async def test_idempotency_keys_are_scoped_to_the_tenant(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    fingerprint = compute_idempotency_fingerprint("KitAgent", "x")
    one = await dendrux_kit.new_run(
        tenant_id="t1", idempotency_key="shared", idempotency_fingerprint=fingerprint
    )
    two = await dendrux_kit.new_run(
        tenant_id="t2", idempotency_key="shared", idempotency_fingerprint=fingerprint
    )
    assert one != two
    assert await store.count_runs() == 2


# -- events, traces, tool calls, usage ------------------------------------------------------


@basic()
async def test_event_sequence_starts_at_zero_without_gaps(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    other = await dendrux_kit.new_run()
    assert await store.get_next_event_sequence(run_id) == 0
    got = [await store.save_run_event(run_id, event_type=EVENT, data={"i": i}) for i in range(5)]
    assert got == [0, 1, 2, 3, 4]
    assert await store.save_run_event(other, event_type=EVENT) == 0, "sequences are per run"
    assert await store.get_next_event_sequence(run_id) == 5
    events = await store.get_run_events(run_id)
    assert [e.sequence_index for e in events] == [0, 1, 2, 3, 4]
    assert [e.data for e in events] == [{"i": i} for i in range(5)]


@basic()
async def test_event_reads_page_and_filter(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    for i in range(6):
        await store.save_run_event(run_id, event_type="a" if i % 2 else "b", iteration_index=i)
    after = await store.get_run_events(run_id, after_sequence_index=2)
    assert [e.sequence_index for e in after] == [3, 4, 5]
    limited = await store.get_run_events(run_id, limit=2)
    assert [e.sequence_index for e in limited] == [0, 1]
    only_a = await store.get_run_events(run_id, event_types=["a"])
    assert [e.sequence_index for e in only_a] == [1, 3, 5]
    assert {e.event_type for e in only_a} == {"a"}


@basic()
async def test_traces_come_back_in_order_and_page(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    for index, role in enumerate(["user", "assistant", "tool", "assistant"]):
        await store.save_trace(run_id, role, f"content {index} ✓", order_index=index)
    traces = await store.get_traces(run_id)
    assert [(t.order_index, t.role, t.content) for t in traces] == [
        (0, "user", "content 0 ✓"),
        (1, "assistant", "content 1 ✓"),
        (2, "tool", "content 2 ✓"),
        (3, "assistant", "content 3 ✓"),
    ]
    page = await store.get_traces(run_id, limit=2, offset=1)
    assert [t.order_index for t in page] == [1, 2]


@basic()
async def test_a_large_trace_round_trips_exactly(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    content = ("ünï😀cödé " * 20_000)[:150_000]
    await store.save_trace(run_id, "assistant", content, order_index=0)
    [trace] = await store.get_traces(run_id)
    assert trace.content == content


@basic()
async def test_tool_calls_round_trip_and_filter_by_iteration(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    first = tool_call_kwargs(iteration_index=0, params={"n": 1, "ok": True, "x": None})
    await store.save_tool_call(run_id, **first)
    await store.save_tool_call(
        run_id,
        **tool_call_kwargs(iteration_index=1, success=False, error_message="boom"),
    )
    rows = await store.get_tool_calls(run_id)
    assert [r.iteration_index for r in rows] == [0, 1]
    assert rows[0].tool_name == "kit_tool"
    assert rows[0].params == {"n": 1, "ok": True, "x": None}
    assert rows[0].success is True
    assert (rows[1].success, rows[1].error_message) == (False, "boom")
    only_second = await store.get_tool_calls(run_id, iteration_index=1)
    assert [r.iteration_index for r in only_second] == [1]


@basic()
async def test_usage_and_interactions_add_up(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.save_llm_interaction(
        run_id, iteration_index=0, usage=usage(10, 5), semantic_request={"messages": ["a"]}
    )
    await store.save_llm_interaction(run_id, iteration_index=1, usage=usage(3, 4))
    interactions = await store.get_llm_interactions(run_id)
    assert [i.iteration_index for i in interactions] == [0, 1]
    assert interactions[0].semantic_request == {"messages": ["a"]}
    rollup = await store.run_usage(run_id)
    assert (rollup.input_tokens, rollup.output_tokens, rollup.llm_calls) == (13, 9, 2)


# -- finishing, cancelling, pausing ---------------------------------------------------------


@basic()
async def test_finalize_run_writes_the_terminal_state_once(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    assert await store.finalize_run(
        run_id,
        status="success",
        answer="done ✓",
        iteration_count=2,
        total_usage=usage(7, 3),
        expected_current_status="running",
    )
    record = await store.get_run(run_id)
    assert record is not None
    assert (record.status, record.answer, record.iteration_count) == ("success", "done ✓", 2)
    assert (record.total_input_tokens, record.total_output_tokens) == (7, 3)
    lost = await store.finalize_run(
        run_id, status="error", error="late", expected_current_status="running"
    )
    assert lost is False
    after = await store.get_run(run_id)
    assert after is not None
    assert (after.status, after.answer) == ("success", "done ✓")


@basic()
async def test_finalize_run_if_status_in_is_a_compare_and_swap(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    miss = await store.finalize_run_if_status_in(
        run_id, status="success", allowed_current_statuses=["waiting_approval"], answer="x"
    )
    assert miss is False
    assert (await store.get_run(run_id)).status == "running"
    hit = await store.finalize_run_if_status_in(
        run_id, status="success", allowed_current_statuses=["running", "paused"], answer="y"
    )
    assert hit is True
    record = await store.get_run(run_id)
    assert record is not None
    assert (record.status, record.answer) == ("success", "y")


@basic()
async def test_cancel_is_a_flag_on_non_terminal_runs_only(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    assert await store.is_cancel_requested(run_id) is False
    assert await store.request_cancel(run_id) is True
    assert await store.request_cancel(run_id) is True, "idempotent while non-terminal"
    assert await store.is_cancel_requested(run_id) is True
    control = await store.get_run_control(run_id)
    assert control.cancel_requested is True
    await store.finalize_run(run_id, status="cancelled")
    assert await store.is_cancel_requested(run_id) is False, "a finished run holds no flag"
    assert await store.request_cancel(run_id) is False
    assert await store.request_cancel(ulid()) is False


@basic()
async def test_pause_state_is_stored_and_a_claim_has_one_winner(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.paused_run()
    record = await store.get_run(run_id)
    assert record is not None
    assert record.status == "waiting_client_tool"
    state = await store.get_pause_state(run_id)
    assert state is not None
    assert state["pause_id"] == "kit-pause"
    first = await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    second = await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    assert first is not False
    assert second is False
    assert (await store.get_run(run_id)).status == "running"


@basic()
async def test_a_claim_with_the_wrong_expected_status_loses(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.paused_run()
    assert await store.claim_paused_run(run_id, expected_status="waiting_approval") is False
    assert (await store.get_run(run_id)).status == "waiting_client_tool"
    assert await store.claim_paused_run(run_id, expected_status="waiting_client_tool") is not False


@basic()
async def test_submit_and_claim_merges_the_submission_and_claims_once(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.paused_run()
    envelope = {
        "submission_id": ulid(),
        "kind": "tool_results",
        "pause_id": "kit-pause",
        "call_ids": ["c1"],
    }
    won = await store.submit_and_claim(
        run_id,
        expected_status="waiting_client_tool",
        submitted_data={"submitted_tool_results": [{"call_id": "c1", "payload": '"v"'}]},
        submission=envelope,
    )
    assert won is not False
    assert (await store.get_run(run_id)).status == "running"
    lost = await store.submit_and_claim(
        run_id,
        expected_status="waiting_client_tool",
        submitted_data={"x": 1},
        submission=envelope,
    )
    assert lost is False


@basic()
async def test_a_pending_cancel_beats_a_pause(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.request_cancel(run_id)
    paused = await store.try_pause_run(
        run_id, status="waiting_client_tool", pause_data={"pause_id": "p"}
    )
    assert paused is False
    assert (await store.get_run(run_id)).status == "running"


@basic()
async def test_a_pause_request_only_latches_on_running_runs(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    running = await dendrux_kit.new_run()
    assert await store.request_pause(running, reason="please") is True
    assert (await store.get_run_control(running)).pause_requested is True
    waiting = await dendrux_kit.paused_run()
    assert await store.request_pause(waiting) is False
    assert await store.request_pause(ulid()) is False


@basic()
async def test_run_meta_updates_merge(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run(meta={"keep": 1})
    await store.update_run_meta(run_id, {"added": "yes"})
    await store.update_run_meta(run_id, {"keep": 2})
    record = await store.get_run(run_id)
    assert record is not None
    assert record.meta is not None
    assert (record.meta["keep"], record.meta["added"]) == (2, "yes")


@basic()
async def test_touch_progress_moves_the_progress_clock_forward(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    before = (await store.get_run(run_id)).last_progress_at
    assert await store.touch_progress(run_id) is True
    after = (await store.get_run(run_id)).last_progress_at
    assert before is not None and after is not None and after >= before
    assert await store.touch_progress(ulid()) is False


# -- notes ------------------------------------------------------------------------------------


@basic()
async def test_notes_are_kept_in_order_and_marked_delivered(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    ids = [ulid() for _ in range(3)]
    for index, note_id in enumerate(ids):
        assert await store.add_note(
            run_id, note_id=note_id, text=f"note {index}", metadata={"i": index}
        )
    notes = await store.get_notes(run_id)
    assert [n.text for n in notes] == ["note 0", "note 1", "note 2"]
    assert (await store.get_run_control(run_id)).has_pending_notes is True
    assert await store.mark_notes_delivered(run_id, ids[:2], iteration=3) == 2
    assert await store.mark_notes_delivered(run_id, ids[:2], iteration=4) == 0, "only once"
    pending = await store.get_notes(run_id, undelivered_only=True)
    assert [n.text for n in pending] == ["note 2"]
    delivered = [n for n in await store.get_notes(run_id) if n.delivered_at is not None]
    assert {n.delivered_iteration for n in delivered} == {3}


@basic()
async def test_a_note_is_refused_for_finished_missing_or_foreign_runs(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    done = await dendrux_kit.new_run()
    await store.finalize_run(done, status="success", answer="a")
    assert await store.add_note(done, note_id=ulid(), text="late") is False
    assert await store.add_note(ulid(), note_id=ulid(), text="nowhere") is False
    mine = await dendrux_kit.new_run(tenant_id="t1")
    assert await store.add_note(mine, note_id=ulid(), text="x", tenant_id="t2") is False
    assert await store.get_notes(mine, tenant_id="t2") == []
    assert await store.add_note(mine, note_id=ulid(), text="x", tenant_id="t1") is True


@basic()
async def test_the_note_cap_counts_exactly(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    for index in range(3):
        assert await store.add_note(run_id, note_id=ulid(), text=str(index), max_notes=3)
    with pytest.raises(NoteLimitExceededError) as caught:
        await store.add_note(run_id, note_id=ulid(), text="one too many", max_notes=3)
    assert caught.value.limit == 3
    assert len(await store.get_notes(run_id)) == 3


# -- tenants, listing, delegation ---------------------------------------------------------------


@basic()
async def test_tenant_scope_is_exact_not_padded_or_case_folded(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    ids = {name: await dendrux_kit.new_run(tenant_id=name) for name in ("t", "t ", "T", "é", "é")}
    ids["none"] = await dendrux_kit.new_run()
    for name in ("t", "t ", "T", "é", "é"):
        found = [r.id for r in await store.list_runs(tenant_id=name)]
        assert found == [ids[name]], f"tenant {name!r} must match only itself"
        assert await store.count_runs(tenant_id=name) == 1
    assert [r.id for r in await store.list_runs(tenant_id=None)] == [ids["none"]]
    assert len(await store.list_runs(tenant_id=UNSCOPED)) == 6
    with pytest.raises(ValueError):
        await store.list_runs(tenant_id="")


@basic()
async def test_list_runs_orders_newest_first_and_filters(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    first = await dendrux_kit.new_run(agent_name="Alpha", meta={"customer": "acme"})
    second = await dendrux_kit.new_run(agent_name="Beta", meta={"customer": "other"})
    third = await dendrux_kit.new_run(agent_name="Alpha")
    await store.finalize_run(second, status="success", answer="a")
    assert [r.id for r in await store.list_runs()] == [third, second, first]
    assert [r.id for r in await store.list_runs(limit=2)] == [third, second]
    assert [r.id for r in await store.list_runs(limit=1, offset=1)] == [second]
    assert [r.id for r in await store.list_runs(agent_name="Alpha")] == [third, first]
    assert [r.id for r in await store.list_runs(status="success")] == [second]
    assert {r.id for r in await store.list_runs(status=["success", "running"])} == {
        first,
        second,
        third,
    }
    assert [r.id for r in await store.list_runs(metadata_filter={"customer": "acme"})] == [first]
    assert await store.count_runs(agent_name="Alpha") == 2
    assert await store.count_runs(status="success") == 1


@basic()
async def test_time_filters_need_aware_datetimes(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    past = datetime.now(UTC) - timedelta(hours=1)
    future = datetime.now(UTC) + timedelta(hours=1)
    assert [r.id for r in await store.list_runs(started_after=past)] == [run_id]
    assert await store.list_runs(started_after=future) == []
    assert [r.id for r in await store.list_runs(started_before=future)] == [run_id]
    assert await store.list_runs(started_before=past) == []
    with pytest.raises(ValueError):
        await store.list_runs(started_after=datetime.now())  # noqa: DTZ005 - naive on purpose


@basic()
async def test_delegation_tree_reads(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    root = await dendrux_kit.new_run(agent_name="Root")
    child = await dendrux_kit.new_run(agent_name="Child", parent_run_id=root, delegation_level=1)
    grand = await dendrux_kit.new_run(agent_name="Grand", parent_run_id=child, delegation_level=2)
    await store.save_llm_interaction(root, iteration_index=0, usage=usage(1, 1))
    await store.save_llm_interaction(child, iteration_index=0, usage=usage(2, 2))
    await store.save_llm_interaction(grand, iteration_index=0, usage=usage(4, 4))
    info = await store.get_delegation_info(child)
    assert info is not None
    assert info.parent is not None and info.parent.run_id == root
    assert [c.run_id for c in info.children] == [grand]
    assert info.subtree_summary.descendant_count == 1
    tree = await store.get_subtree(root)
    assert tree is not None
    assert [(n.run.id, n.depth) for n in tree.nodes] == [(root, 0), (child, 1), (grand, 2)]
    assert (await store.subtree_usage(root)).input_tokens == 7
    assert (await store.run_usage(root)).input_tokens == 1
    assert [r.id for r in await store.list_runs(parent_run_id=root)] == [child]


@basic()
async def test_a_cancel_cascade_walks_children_in_pages(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    root = await dendrux_kit.new_run()
    kids = [await dendrux_kit.new_run(parent_run_id=root, delegation_level=1) for _ in range(5)]
    first = await store.get_children_page([root], after=None, limit=3, tenant_id=UNSCOPED)
    assert first == sorted(kids)[:3]
    rest = await store.get_children_page([root], after=first[-1], limit=3, tenant_id=UNSCOPED)
    assert rest == sorted(kids)[3:]
    assert await store.mark_cancel_requested([root, *kids], tenant_id=UNSCOPED) == 6
    assert all([await store.is_cancel_requested(r) for r in (root, *kids)])
    await store.finalize_run(kids[0], status="cancelled")
    assert await store.mark_cancel_requested([kids[0]], tenant_id=UNSCOPED) == 0


# -- sweeps -----------------------------------------------------------------------------------


@basic()
async def test_sweeping_stale_running_runs_spares_fresh_ones(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    stale = await dendrux_kit.new_run()
    threshold = await dendrux_kit.age()
    fresh = await dendrux_kit.new_run()
    swept = await store.sweep_stale_runs(threshold)
    assert [s.run_id for s in swept] == [stale]
    assert swept[0].previous_status == "running"
    assert (await store.get_run(stale)).status == "error"
    assert (await store.get_run(fresh)).status == "running"
    assert await store.sweep_stale_runs(threshold) == [], "a second sweep finds nothing new"


@basic()
async def test_sweeping_abandoned_waiting_runs_spares_fresh_ones(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    abandoned = await dendrux_kit.paused_run()
    running = await dendrux_kit.new_run()
    threshold = await dendrux_kit.age()
    waiting = await dendrux_kit.paused_run()
    swept = await store.sweep_abandoned_runs(threshold)
    assert [s.run_id for s in swept] == [abandoned]
    assert swept[0].previous_status == "waiting_client_tool"
    assert swept[0].failure_reason == "abandoned_waiting"
    assert (await store.get_run(abandoned)).status == "error"
    assert (await store.get_run(waiting)).status == "waiting_client_tool"
    assert (await store.get_run(running)).status == "running"


@basic()
async def test_a_sweep_of_a_cancelled_run_reports_cancelled(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.request_cancel(run_id)
    threshold = await dendrux_kit.age()
    [swept] = await store.sweep_stale_runs(threshold)
    assert (swept.run_id, swept.status) == (run_id, "cancelled")


@basic()
async def test_sweep_ages_must_not_be_negative(dendrux_kit: Kit) -> None:
    with pytest.raises(ValueError):
        await dendrux_kit.store.sweep_stale_runs(timedelta(seconds=-1))
    with pytest.raises(ValueError):
        await dendrux_kit.store.sweep_abandoned_runs(timedelta(seconds=-1))
