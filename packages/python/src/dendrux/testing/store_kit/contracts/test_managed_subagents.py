"""``managed_subagents``: atomic binding and creator-only execution authority.

All observations use the store protocol, never SQL or the repo's test helpers. Fault cuts
need the optional ``dendrux_fault_injector``; without it crash safety is not certified.
The injector can lose a commit acknowledgement but cannot interpose a competing creator
inside an implementation's private commit read-back. Independent bind contenders exercise
the separate ``second`` seam; backend-specific tests must cover that combined crash cut.
``ToolCallReadRecord`` omits private intent metadata: the kit observes the started intent
and validates the binding through the wait, child header and immutable read-back instead.
In 0.7, custom stores must honor the same run-specific root reclaim context as the driver;
the allowed-reclaim case deliberately enters that in-package context.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

from dendrux._lifecycle import CancelReason, StoredEventType
from dendrux.errors import (
    ManagedChildControlError,
    ManagedStateError,
    PersistenceError,
    RunCancelledError,
    RunFencedError,
    RunNotFoundError,
    StoreCapabilityError,
)
from dendrux.runtime._capabilities import (  # In-package on purpose; covered by kit tests.
    require_managed_subagents,
)
from dendrux.runtime.context import (  # In-package on purpose; covered by kit tests.
    managed_root_reclaim_scope,
)
from dendrux.runtime.state import (
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    BindManagedChildRequest,
    LoopStatePatch,
    ManagedHeader,
)
from dendrux.testing.store_kit._kit import store_skip
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    EVENT,
    RECLAIM,
    event_write,
    lease_cutoff,
    mark,
    started_run,
    trace_write,
    ulid,
    usage,
)
from dendrux.types import ManagedWait, PauseState, RunStatus, ToolCall

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit
    from dendrux.testing.store_kit._protocols import FaultPoint

managed = contract("managed_subagents", needs=("managed_subagents",), format2=True)
managed_crash = contract(
    "managed_subagents", needs=("managed_subagents",), uses=("injector",), format2=True
)


@managed
async def test_managed_root_cap_and_saturating_count(dendrux_kit: Kit) -> None:
    kit = dendrux_kit
    request = await _parent(kit)
    root = request.root_run_id
    assert await kit.store.get_managed_run_count(root, limit=128, tenant_id="t") == 1
    assert await kit.store.get_managed_run_count(root, limit=128, tenant_id=None) is None
    state = await kit.store.get_loop_state(root)
    state["managed_limits"] = {"max_managed_runs": 1}
    await kit.store.checkpoint_step(
        root, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    before = await _snapshot(kit.store, root)
    with pytest.raises(ManagedStateError) as error:
        await kit.store.bind_managed_child(request)
    assert error.value.code == "managed_run_limit"
    assert await _snapshot(kit.store, root) == before


@managed
async def test_managed_count_replay_and_legacy_exclusion(dendrux_kit: Kit) -> None:
    kit = dendrux_kit
    request = await _parent(kit)
    root = request.root_run_id
    await kit.store.create_run(ulid(), "legacy", tenant_id="t", parent_run_id=root)
    await kit.store.bind_managed_child(request)
    assert await kit.store.get_managed_run_count(root, limit=128, tenant_id="t") == 2
    assert await kit.store.get_managed_run_count(root, limit=1, tenant_id="t") == 1
    assert (await kit.store.bind_managed_child(request)).outcome == "existing"
    assert await kit.store.get_managed_run_count(root, limit=128, tenant_id="t") == 2


async def _parent(kit: Kit, tenant: str | None = "t") -> BindManagedChildRequest:
    parent = await started_run(kit, tenant)
    call_id = ulid()
    await kit.store.checkpoint_step(
        parent,
        claim_epoch=0,
        loop_state={
            "v": 1,
            "loop": "react",
            "phase": "dispatch",
            "pending": [{"call_id": call_id, "tool": "reviewer", "target": "agent"}],
            MANAGED_HEADER_KEY: ManagedHeader(parent, ()).to_dict(),
        },
        pii_mapping=None,
        trace=None,
    )
    return _request(parent, call_id, tenant)


def _request(parent: str, call_id: str, tenant: str | None) -> BindManagedChildRequest:
    child = ulid()
    brief = "kit managed task"
    wait = PauseState(
        agent_name="KitAgent",
        pending_tool_calls=[ToolCall(name="reviewer", params={"task": brief}, id=call_id)],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=1,
        usage=usage(),
        pending_targets={call_id: "agent"},
        pause_reason="managed_wait",
        pause_id=ulid(),
        schema_version=3,
        loop_state_ref=True,
        managed_wait=ManagedWait(call_id, child),
    )
    return BindManagedChildRequest(
        tenant_id=tenant,
        root_run_id=parent,
        parent_run_id=parent,
        parent_claim_epoch=0,
        tool_call_id=call_id,
        row_id=ulid(),
        provider_tool_call_id=None,
        tool_name="reviewer",
        params={"task": brief},
        iteration_index=1,
        child_run_id=child,
        child_agent_name="reviewer",
        child_path=("reviewer",),
        child_input_data={"input": brief},
        child_loop_state={"v": 1, "loop": "react", "phase": "dispatch"},
        child_seed_traces=(trace_write("user", brief, 0, kind="seed"),),
        parent_wait=wait,
        parent_patch=LoopStatePatch(set_fields={"iteration": 1}),
        intent_event=event_write(StoredEventType.TOOL_STARTED.value, call_id, 1),
    )


async def _snapshot(store: Any, run_id: str) -> dict[str, Any]:
    return deepcopy(
        {
            "run": await store.get_run(run_id),
            "state": await store.get_loop_state(run_id),
            "pause": await store.get_pause_state(run_id),
            "tools": await store.get_tool_calls(run_id, limit=10),
            "traces": await store.get_traces(run_id, limit=10),
            "events": await store.get_run_events(run_id, limit=10),
            "children": await store.list_runs(parent_run_id=run_id, limit=10),
        }
    )


async def _bound(store: Any, request: BindManagedChildRequest) -> None:
    parent, child_id = request.parent_run_id, request.child_run_id
    record = await store.get_run(parent)
    assert record is not None
    assert record.status == RunStatus.PAUSED.value
    assert record.claim_epoch == request.parent_claim_epoch + 1
    wait = PauseState.from_managed_dict(await store.get_pause_state(parent))
    assert wait.managed_wait == ManagedWait(request.tool_call_id, child_id)
    assert wait.pause_id == request.parent_wait.pause_id
    state = await store.get_loop_state(parent)
    assert state[MANAGED_BINDING_KEY] == {
        **wait.managed_wait.to_dict(),
        "pause_id": wait.pause_id,
    }
    assert [p["call_id"] for p in state["pending"]] == [request.tool_call_id]
    assert state["iteration"] == 1
    [intent] = await store.get_tool_calls(parent, limit=10)
    assert (intent.tool_call_id, intent.target, intent.outcome) == (
        request.tool_call_id,
        "agent",
        "started",
    )
    assert (intent.id, intent.params) == (request.row_id, request.params)
    events = await store.get_run_events(parent, limit=10)
    assert [e.event_type for e in events] == [
        StoredEventType.RUN_STARTED.value,
        request.intent_event.event_type,
    ]
    assert events[-1].id == request.intent_event.event_id
    assert events[-1].data == request.intent_event.data
    [child] = await store.list_runs(parent_run_id=parent, tenant_id=request.tenant_id, limit=10)
    assert child.id == child_id
    assert (child.status, child.claim_epoch, child.storage_format) == (
        RunStatus.RUNNING.value,
        0,
        2,
    )
    assert (child.parent_run_id, child.tenant_id) == (parent, request.tenant_id)
    assert child.input_data == request.child_input_data
    child_state = await store.get_loop_state(child_id)
    assert ManagedHeader.from_dict(child_state[MANAGED_HEADER_KEY], run_id=child_id) == (
        ManagedHeader(parent, ("reviewer",), parent, request.tool_call_id)
    )
    [seed] = await store.get_traces(child_id, limit=10)
    assert (seed.id, seed.role, seed.content, seed.trace_kind) == (
        request.child_seed_traces[0].trace_id,
        "user",
        "kit managed task",
        "seed",
    )


@contract("declaration")
async def test_managed_capability_is_required_before_managed_use(dendrux_kit: Kit) -> None:
    if dendrux_kit.declaration.has("managed_subagents"):
        assert require_managed_subagents(dendrux_kit.store).managed_subagents
    else:
        with pytest.raises(StoreCapabilityError) as refused:
            require_managed_subagents(dendrux_kit.store)
        assert refused.value.capability == "managed_subagents"


@managed
async def test_managed_bind_publishes_wait_intent_child_and_event(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    request = await _parent(dendrux_kit)
    created = await store.bind_managed_child(request)
    assert (created.outcome, created.child_run_id, created.parent_wait_id) == (
        "created",
        request.child_run_id,
        request.parent_wait.pause_id,
    )
    assert created.child_claim is not None
    assert (created.child_claim.claim_epoch, created.child_claim.storage_format) == (0, 2)
    assert created.child_claim.loop_state == await store.get_loop_state(request.child_run_id)
    await _bound(store, request)
    with pytest.raises(RunFencedError):
        await store.save_run_event(request.parent_run_id, event_type=EVENT, claim_epoch=0)


@managed
async def test_managed_bind_readback_never_grants_another_token(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    request = await _parent(dendrux_kit)
    await store.bind_managed_child(request)
    before = await _snapshot(store, request.parent_run_id)
    candidate = _request(request.parent_run_id, request.tool_call_id, request.tenant_id)
    for replay in (request, candidate):
        existing = await store.bind_managed_child(replay)
        assert (existing.outcome, existing.child_claim) == ("existing", None)
        assert existing.child_run_id == request.child_run_id
        assert existing.parent_wait_id == request.parent_wait.pause_id
    assert await store.get_run(candidate.child_run_id) is None
    assert await _snapshot(store, request.parent_run_id) == before
    await _bound(store, request)


@managed_crash
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", ["before_commit", "mid_write"])
async def test_managed_bind_is_all_or_nothing(
    dendrux_kit: Kit, point: FaultPoint, after: int
) -> None:
    store = dendrux_kit.store
    request = await _parent(dendrux_kit)
    before = await _snapshot(store, request.parent_run_id)
    outcome = await dendrux_kit.faulted(
        point, lambda: store.bind_managed_child(request), sticky=True, after=after
    )
    if after == 0:
        outcome.expect_fired(point)
    elif not outcome.fired:
        assert outcome.raised is None, "one atomic transaction must finish before the later cut"
        await _bound(store, request)
        return
    assert isinstance(outcome.raised, PersistenceError), "an uncommittable bind must fail"
    assert await _snapshot(store, request.parent_run_id) == before
    assert await store.get_run(request.child_run_id) is None
    assert await store.get_traces(request.child_run_id, limit=10) == []
    assert await store.get_run_events(request.child_run_id, limit=10) == []
    await store.bind_managed_child(request)
    await _bound(store, request)


@managed_crash
async def test_managed_bind_lost_ack_grants_only_creator_authority(dendrux_kit: Kit) -> None:
    forms = dendrux_kit.disconnect_forms()
    if not forms:
        store_skip(
            "no disconnect forms: managed bind lost acknowledgement untested", "managed_subagents"
        )
    store = dendrux_kit.store
    for form in forms:
        request = await _parent(dendrux_kit)
        outcome = await dendrux_kit.faulted(
            "after_commit", partial(store.bind_managed_child, request), form=form
        )
        outcome.expect_fired("after_commit")
        if outcome.raised is None:
            result = outcome.result
            assert result.child_run_id == request.child_run_id
            assert (result.outcome == "created") == (result.child_claim is not None)
            if result.child_claim is not None:
                assert (result.child_claim.claim_epoch, result.child_claim.storage_format) == (0, 2)
        else:
            assert isinstance(outcome.raised, PersistenceError)
        await _bound(store, request)
        # An identical public request carries no private proof of commit ownership.
        for replay in (request, _request(request.parent_run_id, request.tool_call_id, "t")):
            existing = await store.bind_managed_child(replay)
            assert (existing.outcome, existing.child_claim) == ("existing", None)
            assert existing.child_run_id == request.child_run_id
            if replay is not request:
                assert await store.get_run(replay.child_run_id) is None
        await _bound(store, request)


@managed
@pytest.mark.parametrize(
    "case",
    [
        "stale_epoch",
        CancelReason.CANCEL_REQUESTED.value,
        RunStatus.CANCELLED.value,
        "tenant",
        "unscoped",
        "null_tenant",
    ],
)
async def test_managed_bind_refuses_parent_without_writes(dendrux_kit: Kit, case: str) -> None:
    store = dendrux_kit.store
    request = await _parent(dendrux_kit, None if case == "null_tenant" else "t")
    expected: type[Exception]
    if case == "stale_epoch":
        await store.pause_run(
            request.parent_run_id,
            status=RunStatus.PAUSED.value,
            pause_data={"pause_id": ulid(), "pending_tool_calls": []},
            claim_epoch=0,
        )
        claim = await store.claim_paused_run(
            request.parent_run_id, expected_status=RunStatus.PAUSED.value
        )
        assert claim is not False and claim.claim_epoch > request.parent_claim_epoch
        expected = RunFencedError
    elif case in (CancelReason.CANCEL_REQUESTED.value, RunStatus.CANCELLED.value):
        if case == CancelReason.CANCEL_REQUESTED.value:
            await store.request_cancel(request.parent_run_id)
        else:
            assert await store.finalize_run(
                request.parent_run_id,
                status=RunStatus.CANCELLED.value,
                expected_current_status=RunStatus.RUNNING.value,
                claim_epoch=0,
            )
        expected = RunCancelledError
    else:
        request = replace(request, tenant_id=None if case == "unscoped" else "other")
        expected = RunNotFoundError
    before = await _snapshot(store, request.parent_run_id)
    with pytest.raises(expected):
        await store.bind_managed_child(request)
    assert await _snapshot(store, request.parent_run_id) == before
    assert await store.get_run(request.child_run_id) is None


@contract("managed_subagents", needs=("managed_subagents",), uses=("clock",), format2=True)
async def test_managed_child_reclaim_refuses_before_any_mutation(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    request = await _parent(dendrux_kit)
    await store.bind_managed_child(request)
    cutoff = await lease_cutoff(dendrux_kit)
    marked = await mark(dendrux_kit, cutoff, reclaim_limit=1)
    [child] = marked.reclaimable
    assert child.run_id == request.child_run_id
    before = await _snapshot(store, child.run_id)
    parent_before = await _snapshot(store, request.parent_run_id)
    kwargs = dict(
        tenant_id=request.tenant_id,
        agent_name="reviewer",
        expected_claim_epoch=child.claim_epoch,
        **RECLAIM,
    )
    # Repeated direct calls, even on the last attempt, may not spend the recovery allowance.
    for _ in range(2):
        with pytest.raises(ManagedChildControlError):
            await store.reclaim_run(child.run_id, **kwargs)
        assert await _snapshot(store, child.run_id) == before, "refusal must precede mutation"
        assert await _snapshot(store, request.parent_run_id) == parent_before
    with managed_root_reclaim_scope(ulid()), pytest.raises(ManagedChildControlError):
        await store.reclaim_run(child.run_id, **kwargs)
    assert await _snapshot(store, child.run_id) == before
    with managed_root_reclaim_scope(child.run_id):
        claim = await store.reclaim_run(child.run_id, **kwargs)
    assert claim is not False and claim.claim_epoch > child.claim_epoch
    reclaimed = await store.get_run(child.run_id)
    assert (reclaimed.status, reclaimed.reclaim_attempts) == (RunStatus.RUNNING.value, 1)
    assert reclaimed.lease_expired_at is None
    parent_after = await _snapshot(store, request.parent_run_id)
    assert {k: v for k, v in parent_after.items() if k != "children"} == {
        k: v for k, v in parent_before.items() if k != "children"
    }


@contract("managed_subagents", needs=("managed_subagents",), uses=("second",), format2=True)
async def test_managed_bind_contenders_have_one_creator(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    request = await _parent(dendrux_kit)
    candidate = _request(request.parent_run_id, request.tool_call_id, request.tenant_id)
    second = await dendrux_kit.second()
    results = await asyncio.gather(
        store.bind_managed_child(request), second.bind_managed_child(candidate)
    )
    assert sorted(r.outcome for r in results) == ["created", "existing"]
    assert sum(r.child_claim is not None for r in results) == 1
    assert len({r.child_run_id for r in results}) == 1
    winner = request if results[0].outcome == "created" else candidate
    loser = candidate if winner is request else request
    assert await store.get_run(loser.child_run_id) is None
    await _bound(store, winner)


@managed
async def test_managed_cancel_projection_is_scoped_ordered_and_bounded(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    getter = getattr(store, "get_managed_cancel_children_page", None)
    assert callable(getter), "managed stores must expose the public projected cancel page"
    request = await _parent(dendrux_kit)
    await store.bind_managed_child(request)
    legacy = ulid()
    await store.create_run(legacy, "legacy", parent_run_id=request.parent_run_id, tenant_id="t")
    page = await getter([request.parent_run_id] * 2, after=None, limit=1, tenant_id="t")
    assert len(page) == 1
    following = await getter([request.parent_run_id], after=page[-1][0], limit=1, tenant_id="t")
    complete = page + following
    assert [row[0] for row in complete] == sorted([request.child_run_id, legacy])
    assert all(row[1] == request.parent_run_id for row in complete)
    values = {row[0]: row[2] for row in complete}
    header = ManagedHeader.from_dict(values[request.child_run_id], run_id=request.child_run_id)
    assert header.root_run_id == request.root_run_id
    assert values[legacy] is None
    assert await getter([request.parent_run_id], after=None, limit=500, tenant_id="foreign") == []
    assert await getter([request.parent_run_id], after=None, limit=500, tenant_id=None) == []
    assert await getter([], after=None, limit=500, tenant_id="t") == []
    with pytest.raises(ValueError):
        await getter([request.parent_run_id], after=None, limit=0, tenant_id="t")


@managed
async def test_managed_cancel_controls_stop_on_root_and_missing_own(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    stopped = getattr(store, "is_managed_root_stopped", None)
    control = getattr(store, "get_managed_run_control", None)
    assert callable(stopped) and callable(control), (
        "managed stores must expose public control reads"
    )
    request = await _parent(dendrux_kit)
    await store.bind_managed_child(request)
    child, root = request.child_run_id, request.root_run_id
    assert not await stopped(child)
    own = await store.get_run_control(child)
    assert await control(child) == own
    assert await store.request_pause(child)
    await store.add_note(child, text="kit note", note_id=ulid(), tenant_id="t", claim_epoch=0)
    own = await store.get_run_control(child)
    combined = await control(child)
    assert combined == own and combined.pause_requested and combined.has_pending_notes
    await store.mark_cancel_requested([root], tenant_id=request.tenant_id)
    assert await stopped(child)
    assert (await control(child)).cancel_requested
    assert not (await store.get_run_control(child)).cancel_requested
    assert await store.finalize_run_if_status_in(
        root, status=RunStatus.CANCELLED.value, allowed_current_statuses=[RunStatus.PAUSED.value]
    )
    assert not (await store.get_run(root)).cancel_requested
    assert await stopped(child) and (await control(child)).cancel_requested
    missing = ulid()
    assert await stopped(missing), "missing own row must fail closed"
    assert (await control(missing)).cancel_requested
    foreign = await _parent(dendrux_kit, "foreign")
    state = await store.get_loop_state(child)
    for root_id in (foreign.root_run_id, ulid()):
        state[MANAGED_HEADER_KEY] = ManagedHeader(
            root_id, ("reviewer",), root, request.tool_call_id
        ).to_dict()
        await store.checkpoint_step(
            child, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
        )
        assert await stopped(child)
        assert await control(child) == replace(own, cancel_requested=True)
    live = await _parent(dendrux_kit)
    await store.bind_managed_child(live)
    assert await store.request_cancel(live.child_run_id)
    assert not await stopped(live.child_run_id)
    assert (await control(live.child_run_id)).cancel_requested


@managed
async def test_managed_cancel_page_caps_a_large_requested_limit(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    root = await started_run(dendrux_kit)
    children = []
    for _ in range(501):
        child = ulid()
        await store.create_run(child, "legacy", parent_run_id=root, tenant_id="t")
        children.append(child)
    page = await store.get_managed_cancel_children_page(
        [root], after=None, limit=5000, tenant_id="t"
    )
    assert [row[0] for row in page] == sorted(children)[:500]
    tail = await store.get_managed_cancel_children_page(
        [root], after=page[-1][0], limit=5000, tenant_id="t"
    )
    assert [row[0] for row in tail] == sorted(children)[500:]
