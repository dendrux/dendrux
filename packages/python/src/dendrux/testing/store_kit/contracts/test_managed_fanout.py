"""``managed_fanout`` (0.8): bind 2..32 managed children at once and read their status.

Observations use the store protocol, never SQL. The parent's wait is built in the stored JSON
shape the 0.8 spec freezes (pause ``schema_version`` 4 with a ``managed_fanout`` envelope of
ordered ``{tool_call_id, child_run_id}`` bindings); the store must persist exactly that value and
derive the checkpoint's ``managed_wait`` map from it. Fault cuts need the optional
``dendrux_fault_injector``; contenders need ``dendrux_store_second_instance``.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import fields, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

from dendrux._lifecycle import CancelReason, PauseReason, StoredEventType
from dendrux.errors import (
    ManagedStateError,
    PersistenceError,
    RunCancelledError,
    RunFencedError,
    RunNotFoundError,
    StoreCapabilityError,
)
from dendrux.runtime._capabilities import (  # In-package on purpose; covered by kit tests.
    fanout_support,
)
from dendrux.runtime._managed_fanout_store import (  # In-package on purpose; see above.
    BindManagedChildrenRequest,
    ManagedChildSpec,
)
from dendrux.runtime.state import (
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    UNSCOPED,
    LoopStatePatch,
    ManagedHeader,
    StoreCapabilities,
)
from dendrux.testing.store_kit._kit import store_skip
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import (
    complete_call_kwargs,
    event_write,
    started_run,
    trace_write,
    ulid,
    usage,
)
from dendrux.types import PauseState, RunStatus, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.testing.store_kit._kit import Kit
    from dendrux.testing.store_kit._protocols import FaultPoint

_NEEDS = ("managed_subagents", "managed_fanout")
fanout = contract("managed_subagents", needs=_NEEDS, format2=True)
fanout_crash = contract("managed_subagents", needs=_NEEDS, uses=("injector",), format2=True)
fanout_second = contract("managed_subagents", needs=_NEEDS, uses=("second",), format2=True)
_STARTED = StoredEventType.TOOL_STARTED.value


def _reason(name: str) -> str:
    """The reason a refusal carries: unregistered names read as ``wait_invalid``."""
    return name if name in ManagedStateError.REASONS else "wait_invalid"


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------


class _EncodedWait(PauseState):
    """A parent wait whose stored JSON is given verbatim (the frozen v4 shape)."""

    encoded: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.encoded)


def fanout_wait(
    bindings: Sequence[tuple[str, str]],
    *,
    pause_id: str | None = None,
    calls: Sequence[ToolCall] | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> PauseState:
    """A v4 fan-out wait over ``(tool_call_id, child_run_id)`` bindings, in call order."""
    base = PauseState(
        agent_name="KitAgent",
        pending_tool_calls=list(calls)
        if calls is not None
        else [ToolCall(name="reviewer", params={}, id=c) for c, _ in bindings],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=1,
        usage=usage(),
        pending_targets={c: "agent" for c, _ in bindings},
        pause_reason="managed_wait",
        pause_id=pause_id or ulid(),
        schema_version=2,
        loop_state_ref=True,
        provider_kwargs=dict(provider_kwargs or {}),
    )
    data = base.to_dict()
    data["schema_version"] = 4
    data["managed_fanout"] = {
        "v": 1,
        "n": len(bindings),
        "bindings": [{"tool_call_id": c, "child_run_id": ch} for c, ch in bindings],
    }
    wait = _EncodedWait(**{f.name: getattr(base, f.name) for f in fields(PauseState)})
    wait.encoded = data
    return wait


async def _parent(
    kit: Kit, n: int, tenant: str | None = "t", *, name: str = "reviewer"
) -> tuple[str, list[str]]:
    """A running managed root at epoch 0 with ``n`` pending managed calls."""
    parent = await started_run(kit, tenant)
    calls = [ulid() for _ in range(n)]
    await kit.store.checkpoint_step(
        parent,
        claim_epoch=0,
        loop_state={
            "v": 1,
            "loop": "react",
            "phase": "dispatch",
            "pending": [{"call_id": c, "tool": name, "target": "agent"} for c in calls],
            MANAGED_HEADER_KEY: ManagedHeader(parent, ()).to_dict(),
        },
        pii_mapping=None,
        trace=None,
    )
    return parent, calls


async def _nest(kit: Kit, child: str, n: int) -> list[str]:
    """Give a bound (running) child ``n`` pending managed calls of its own."""
    state = await kit.store.get_loop_state(child)
    calls = [ulid() for _ in range(n)]
    state["pending"] = [{"call_id": c, "tool": "reviewer", "target": "agent"} for c in calls]
    await kit.store.checkpoint_step(
        child, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    return calls


def _request(
    parent: str,
    calls: Sequence[str],
    tenant: str | None = "t",
    *,
    root: str | None = None,
    path: tuple[str, ...] = (),
    name: str = "reviewer",
    provider_id: str | None = None,
    child: Any = None,
    provider_kwargs: dict[str, Any] | None = None,
    **overrides: Any,
) -> BindManagedChildrenRequest:
    """A batch over ``calls``; ``child(i)`` returns per-child field overrides."""
    children = tuple(
        ManagedChildSpec(
            **{
                "tool_call_id": call,
                "row_id": ulid(),
                "provider_tool_call_id": provider_id,
                "tool_name": name,
                "params": {"task": f"kit brief {i}"},
                "child_run_id": ulid(),
                "child_agent_name": name,
                "child_path": (*path, name),
                "child_input_data": {"input": f"kit brief {i}"},
                "child_loop_state": {"v": 1, "loop": "react", "phase": "dispatch"},
                "intent_event": event_write(_STARTED, call, 1),
                "child_seed_traces": (trace_write("user", f"kit brief {i}", 0, kind="seed"),),
                **(child(i) if child is not None else {}),
            }
        )
        for i, call in enumerate(calls)
    )
    wait_calls = [
        ToolCall(name=c.tool_name, params={}, id=c.tool_call_id, provider_tool_call_id=provider_id)
        for c in children
    ]
    values: dict[str, Any] = {
        "tenant_id": tenant,
        "root_run_id": root or parent,
        "parent_run_id": parent,
        "parent_claim_epoch": 0,
        "iteration_index": 1,
        "parent_wait": fanout_wait(
            [(c.tool_call_id, c.child_run_id) for c in children],
            calls=wait_calls,
            provider_kwargs=provider_kwargs,
        ),
        "parent_patch": LoopStatePatch(set_fields={"iteration": 1}),
        "children": children,
        "max_parallel": max(len(children), 2),
    }
    values.update(overrides)
    return BindManagedChildrenRequest(**values)


async def _snapshot(store: Any, run_id: str) -> dict[str, Any]:
    return deepcopy(
        {
            "run": await store.get_run(run_id),
            "state": await store.get_loop_state(run_id),
            "pause": await store.get_pause_state(run_id),
            "tools": await store.get_tool_calls(run_id, limit=40),
            "traces": await store.get_traces(run_id, limit=40),
            "events": await store.get_run_events(run_id, limit=40),
            "children": await store.list_runs(parent_run_id=run_id, limit=40),
        }
    )


async def _absent(store: Any, request: BindManagedChildrenRequest) -> None:
    for child in request.children:
        assert await store.get_run(child.child_run_id) is None
        assert await store.get_traces(child.child_run_id, limit=5) == []


async def _published(store: Any, request: BindManagedChildrenRequest) -> None:
    """The parent parked at exactly this wait; N intents, N children, seeds and N events."""
    parent = request.parent_run_id
    record = await store.get_run(parent)
    assert record is not None
    assert (record.status, record.claim_epoch) == (RunStatus.PAUSED.value, 1)
    wait = request.parent_wait.to_dict()
    assert await store.get_pause_state(parent) == wait
    bindings = {c.tool_call_id: c.child_run_id for c in request.children}
    state = await store.get_loop_state(parent)
    assert state[MANAGED_BINDING_KEY] == {
        "v": 1,
        "pause_id": wait["pause_id"],
        "bindings": bindings,
    }
    assert {p["call_id"] for p in state["pending"]} == set(bindings)
    assert state["iteration"] == 1
    intents = {r.tool_call_id: r for r in await store.get_tool_calls(parent, limit=40)}
    assert set(intents) == set(bindings)
    for child in request.children:
        intent = intents[child.tool_call_id]
        assert (intent.id, intent.target, intent.outcome) == (child.row_id, "agent", "started")
        assert intent.params == child.params
    events = await store.get_run_events(parent, limit=40)
    assert [e.event_type for e in events] == [
        StoredEventType.RUN_STARTED.value,
        *[_STARTED] * len(request.children),
    ]
    assert [e.id for e in events[1:]] == [c.intent_event.event_id for c in request.children]
    children = await store.list_runs(parent_run_id=parent, tenant_id=request.tenant_id, limit=40)
    assert {c.id for c in children} == set(bindings.values())
    for child in request.children:
        await _child_published(store, request, child)


async def _child_published(
    store: Any, request: BindManagedChildrenRequest, child: ManagedChildSpec
) -> None:
    run = await store.get_run(child.child_run_id)
    assert (run.status, run.claim_epoch, run.storage_format) == (RunStatus.RUNNING.value, 0, 2)
    assert (run.parent_run_id, run.tenant_id) == (request.parent_run_id, request.tenant_id)
    assert run.input_data == child.child_input_data
    state = await store.get_loop_state(child.child_run_id)
    assert ManagedHeader.from_dict(state[MANAGED_HEADER_KEY], run_id=child.child_run_id) == (
        ManagedHeader(
            request.root_run_id, child.child_path, request.parent_run_id, child.tool_call_id
        )
    )
    assert {k: state[k] for k in child.child_loop_state} == dict(child.child_loop_state)
    [seed] = await store.get_traces(child.child_run_id, limit=5)
    assert (seed.id, seed.content, seed.trace_kind) == (
        child.child_seed_traces[0].trace_id,
        child.child_seed_traces[0].content,
        "seed",
    )


async def _claim_adoption(store: Any, request: BindManagedChildrenRequest) -> int:
    """Claim the parked parent as a v4 adoption (spec §7.3 descriptor and ids-only payload)."""
    calls = [c.tool_call_id for c in request.children]
    pause_id = request.parent_wait.to_dict()["pause_id"]
    claim = await store.submit_and_claim(
        request.parent_run_id,
        expected_status=RunStatus.PAUSED.value,
        submitted_data={"managed_fanout_adoption": {"v": 1, "calls": calls}},
        submission={
            "submission_id": ulid(),
            "kind": "tool_results",
            "pause_id": pause_id,
            "payload_ref": {"pause_data_key": "submission"},
            "call_ids": calls,
            "reserved_order_index": 1,
            "managed_fanout_adoption": {"v": 1},
        },
    )
    assert claim is not False
    return int(claim.claim_epoch)


async def _complete(
    store: Any, parent: str, epoch: int, call: str, order: int, *, failed: bool = False, **kw: Any
) -> None:
    outcome = {"outcome": "failed", "success": False, "error_message": "safe"} if failed else {}
    await store.complete_tool_call(
        parent, claim_epoch=epoch, **complete_call_kwargs(call, order, **outcome, **kw)
    )


# ---------------------------------------------------------------------------------------------
# K10, discovery reads the declaration
# ---------------------------------------------------------------------------------------------


class _UndeclaredMethods:
    """Has both fan-out methods but declares nothing: never fan-out capable."""

    async def bind_managed_children(self, request: Any) -> Any:
        raise AssertionError("discovery must not call or trust methods")

    async def get_managed_children(self, parent_run_id: Any, ids: Any, *, tenant_id: Any) -> Any:
        raise AssertionError("discovery must not call or trust methods")


class _FanoutOnly(_UndeclaredMethods):
    capabilities = StoreCapabilities(deployment="single_process", managed_fanout=True)


@contract("declaration")
async def test_fanout_support_reads_the_declaration_not_the_methods(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    declared = dendrux_kit.declaration.has("managed_subagents") and dendrux_kit.declaration.has(
        "managed_fanout"
    )
    assert fanout_support(store).capable is declared
    assert fanout_support(_UndeclaredMethods()).capable is False
    # Fan-out needs the managed declaration: narrowing managed_subagents turns it off too.
    assert fanout_support(_FanoutOnly()).capable is False
    if not declared and callable(getattr(store, "bind_managed_children", None)):
        parent, calls = await _parent(dendrux_kit, 2)
        before = await _snapshot(store, parent)
        with pytest.raises(StoreCapabilityError):
            await store.bind_managed_children(_request(parent, calls))
        assert await _snapshot(store, parent) == before


# ---------------------------------------------------------------------------------------------
# K1, one bind publishes everything
# ---------------------------------------------------------------------------------------------


@fanout
@pytest.mark.parametrize("n", [2, 8, 32])
async def test_fanout_bind_publishes_wait_intents_children_and_events(
    dendrux_kit: Kit, n: int
) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, n)
    request = _request(parent, calls)
    created = await store.bind_managed_children(request)
    assert created.outcome == "created"
    assert created.parent_wait_id == request.parent_wait.to_dict()["pause_id"]
    assert [(b.tool_call_id, b.child_run_id) for b in created.children] == [
        (c.tool_call_id, c.child_run_id) for c in request.children
    ]
    for bound in created.children:
        claim = bound.child_claim
        assert claim is not None and (claim.claim_epoch, claim.storage_format) == (0, 2)
        assert claim.loop_state == await store.get_loop_state(bound.child_run_id)
    await _published(store, request)
    with pytest.raises(RunFencedError):
        await store.save_run_event(parent, event_type="kit.event", claim_epoch=0)


@fanout
async def test_fanout_bind_snapshots_the_request_before_its_first_await(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    original = deepcopy(request)
    task = asyncio.ensure_future(store.bind_managed_children(request))
    await asyncio.sleep(0)
    for child in request.children:
        assert child.params is not None and child.child_input_data is not None
        child.params["task"] = "changed"
        child.child_input_data["input"] = "changed"
        child.child_loop_state["phase"] = "changed"  # type: ignore[index]
        child.intent_event.data["call_id"] = "changed"  # type: ignore[index]
    request.parent_patch.set_fields["iteration"] = 99  # type: ignore[index]
    await task
    await _published(store, original)


# ---------------------------------------------------------------------------------------------
# K2, all or nothing
# ---------------------------------------------------------------------------------------------


@fanout_crash
@pytest.mark.parametrize("after", [0, 1])
@pytest.mark.parametrize("point", ["before_commit", "mid_write"])
async def test_fanout_bind_is_all_or_nothing(
    dendrux_kit: Kit, point: FaultPoint, after: int
) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 3)
    request = _request(parent, calls)
    before = await _snapshot(store, parent)
    outcome = await dendrux_kit.faulted(
        point, lambda: store.bind_managed_children(request), sticky=True, after=after
    )
    if after == 0:
        outcome.expect_fired(point)
    elif not outcome.fired:
        assert outcome.raised is None, "one atomic transaction must finish before the later cut"
        await _published(store, request)
        return
    assert isinstance(outcome.raised, PersistenceError), "an uncommittable bind must fail"
    assert await _snapshot(store, parent) == before
    await _absent(store, request)
    assert (await store.bind_managed_children(request)).outcome == "created"
    await _published(store, request)


# ---------------------------------------------------------------------------------------------
# K3, read-back classification
# ---------------------------------------------------------------------------------------------


@fanout
async def test_fanout_readback_never_grants_a_token(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 3)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    before = await _snapshot(store, parent)
    candidate = _request(parent, calls)
    for replay in (request, candidate):
        existing = await store.bind_managed_children(replay)
        assert existing.outcome == "existing"
        assert existing.parent_wait_id == request.parent_wait.to_dict()["pause_id"]
        assert [(b.tool_call_id, b.child_run_id, b.child_claim) for b in existing.children] == [
            (c.tool_call_id, c.child_run_id, None) for c in request.children
        ]
    await _absent(store, candidate)
    assert await _snapshot(store, parent) == before
    await _published(store, request)


@fanout
async def test_fanout_readback_accepts_an_adoption_in_progress(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    pause_id = request.parent_wait.to_dict()["pause_id"]
    epoch = await _claim_adoption(store, request)
    assert (await store.bind_managed_children(request)).parent_wait_id == pause_id
    await _complete(store, parent, epoch, calls[0], 10, failed=True)
    replay = await store.bind_managed_children(_request(parent, calls))
    assert (replay.outcome, replay.parent_wait_id) == ("existing", pause_id)
    assert [b.child_run_id for b in replay.children] == [c.child_run_id for c in request.children]
    cleared = {"managed_wait": None, "submission": None, "managed_admitted": None}
    patch = LoopStatePatch(remove_pending=(calls[1],), set_fields=cleared)
    await _complete(store, parent, epoch, calls[1], 11, loop_state_patch=patch)
    adopted = await store.bind_managed_children(request)
    assert (adopted.outcome, adopted.parent_wait_id) == ("existing", None)
    assert all(b.child_claim is None for b in adopted.children)


@fanout
async def test_fanout_readback_refuses_an_out_of_order_adoption(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    epoch = await _claim_adoption(store, request)
    await _complete(store, parent, epoch, calls[1], 10)
    before = await _snapshot(store, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_children(request)
    assert refused.value.reason == "binding_conflict"
    assert await _snapshot(store, parent) == before


@fanout
@pytest.mark.parametrize("case", ["partial", "reordered", "claimed_without_marker"])
async def test_fanout_readback_refuses_a_conflicting_batch(dendrux_kit: Kit, case: str) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 3)
    bound = _request(parent, calls[:2])
    await store.bind_managed_children(bound)
    replay = _request(parent, [calls[0], calls[2]])
    if case == "reordered":
        replay = _request(parent, [calls[1], calls[0]])
    elif case == "claimed_without_marker":
        claim = await store.claim_paused_run(parent, expected_status=RunStatus.PAUSED.value)
        assert claim is not False
        replay = bound
    before = await _snapshot(store, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_children(replay)
    assert refused.value.reason == "binding_conflict"
    assert await _snapshot(store, parent) == before
    if replay is not bound:
        await _absent(store, replay)


@fanout
async def test_fanout_readback_on_a_terminal_parent(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    assert await store.finalize_run_if_status_in(
        parent, status=RunStatus.CANCELLED.value, allowed_current_statuses=[RunStatus.PAUSED.value]
    )
    existing = await store.bind_managed_children(_request(parent, calls))
    assert (existing.outcome, existing.parent_wait_id) == ("existing", None)
    assert [b.child_run_id for b in existing.children] == [c.child_run_id for c in request.children]


# ---------------------------------------------------------------------------------------------
# K4, contenders
# ---------------------------------------------------------------------------------------------


@fanout_second
async def test_fanout_bind_contenders_have_one_creator(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 4)
    request, candidate = _request(parent, calls), _request(parent, calls)
    second = await dendrux_kit.second()
    results = await asyncio.gather(
        store.bind_managed_children(request), second.bind_managed_children(candidate)
    )
    assert sorted(r.outcome for r in results) == ["created", "existing"]
    assert sum(b.child_claim is not None for r in results for b in r.children) == 4
    assert results[0].children[0].child_run_id == results[1].children[0].child_run_id
    winner = request if results[0].outcome == "created" else candidate
    await _absent(store, candidate if winner is request else request)
    await _published(store, winner)


# ---------------------------------------------------------------------------------------------
# K5, admission once for the batch
# ---------------------------------------------------------------------------------------------


async def _cap(kit: Kit, root: str, cap: int) -> None:
    state = await kit.store.get_loop_state(root)
    state["managed_limits"] = {"max_managed_runs": cap}
    await kit.store.checkpoint_step(
        root, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )


@fanout
async def test_fanout_admission_counts_the_whole_batch(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 3)
    await _cap(dendrux_kit, parent, 3)  # the root counts: 1 + 3 > 3
    request = _request(parent, calls)
    before = await _snapshot(store, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_children(request)
    assert refused.value.code == "managed_run_limit"
    assert await _snapshot(store, parent) == before
    await _absent(store, request)
    await _cap(dendrux_kit, parent, 4)
    assert (await store.bind_managed_children(request)).outcome == "created"


@fanout
@pytest.mark.parametrize("case", ["over_max_parallel", "one_child", "thirty_three"])
async def test_fanout_limit_refuses_without_writes(dendrux_kit: Kit, case: str) -> None:
    store = dendrux_kit.store
    n = {"over_max_parallel": 3, "one_child": 1, "thirty_three": 33}[case]
    parent, calls = await _parent(dendrux_kit, n)
    request = _request(parent, calls, max_parallel=2 if case == "over_max_parallel" else 32)
    before = await _snapshot(store, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_children(request)
    assert refused.value.reason == _reason("fanout_limit")
    assert await _snapshot(store, parent) == before
    await _absent(store, request)


@fanout_second
async def test_fanout_same_root_binders_share_the_cap(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    root, calls = await _parent(dendrux_kit, 2)
    await _cap(dendrux_kit, root, 5)  # root + 2 children + room for one more batch of 2
    top = _request(root, calls)
    await store.bind_managed_children(top)
    a, b = (c.child_run_id for c in top.children)
    requests = [
        _request(child, await _nest(dendrux_kit, child, 2), root=root, path=("reviewer",))
        for child in (a, b)
    ]
    second = await dendrux_kit.second()
    results = await asyncio.gather(
        store.bind_managed_children(requests[0]),
        second.bind_managed_children(requests[1]),
        return_exceptions=True,
    )
    won = [r for r in results if not isinstance(r, BaseException)]
    lost = [r for r in results if isinstance(r, BaseException)]
    assert len(won) == 1 and won[0].outcome == "created"
    assert len(lost) == 1 and isinstance(lost[0], ManagedStateError)
    assert lost[0].code == "managed_run_limit"
    await _absent(store, requests[results.index(lost[0])])


def _compact_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@fanout
async def test_fanout_bind_refuses_an_oversized_wait_without_writes(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls, provider_kwargs={"model": "m" * (32 * 1024)})
    assert _compact_bytes(request.parent_wait.to_dict()) > 32 * 1024
    before = await _snapshot(store, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_children(request)
    assert refused.value.reason == "wait_invalid"
    assert await _snapshot(store, parent) == before
    await _absent(store, request)


@fanout
async def test_fanout_bind_holds_a_maximum_size_batch(dendrux_kit: Kit) -> None:
    """32 children with ULID call ids, 255-character provider ids and 64-character names."""
    store = dendrux_kit.store
    name = "n" * 64
    parent, calls = await _parent(dendrux_kit, 32, name=name)
    request = _request(parent, calls, name=name, provider_id="p" * 255)
    assert 16 * 1024 < _compact_bytes(request.parent_wait.to_dict()) <= 32 * 1024
    assert (await store.bind_managed_children(request)).outcome == "created"
    await _published(store, request)


# ---------------------------------------------------------------------------------------------
# K6, a cancelled parent or a stopped root refuses the whole batch
# ---------------------------------------------------------------------------------------------


@fanout
@pytest.mark.parametrize(
    "case", [CancelReason.CANCEL_REQUESTED.value, "root_stopped", "stale_epoch", "tenant"]
)
async def test_fanout_bind_refuses_a_stopped_parent_without_writes(
    dendrux_kit: Kit, case: str
) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    expected: type[Exception] = RunCancelledError
    if case == CancelReason.CANCEL_REQUESTED.value:
        assert await store.request_cancel(parent)
    elif case == "root_stopped":
        await store.bind_managed_children(request)
        child = request.children[0].child_run_id
        nested = await _nest(dendrux_kit, child, 2)
        request = _request(child, nested, root=parent, path=("reviewer",))
        await store.mark_cancel_requested([parent], tenant_id="t")
        parent = child
    elif case == "stale_epoch":
        request = replace(request, parent_claim_epoch=1)
        expected = RunFencedError
    else:
        request = replace(request, tenant_id="other")
        expected = RunNotFoundError
    before = await _snapshot(store, parent)
    with pytest.raises(expected):
        await store.bind_managed_children(request)
    assert await _snapshot(store, parent) == before
    await _absent(store, request)


# ---------------------------------------------------------------------------------------------
# K7, the status projection
# ---------------------------------------------------------------------------------------------


@fanout
async def test_managed_children_projection_is_scoped_ordered_and_bounded(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 3)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    c1, c2, c3 = (c.child_run_id for c in request.children)
    other, _ = await _parent(dendrux_kit, 2)
    rows = await store.get_managed_children(parent, [c3, ulid(), c1, c2, other], tenant_id="t")
    assert [r.run_id for r in rows] == [c3, c1, c2]
    for row, child in zip(rows, (request.children[2], *request.children[:2]), strict=True):
        assert (row.parent_run_id, row.tenant_id, row.status) == (
            parent,
            "t",
            RunStatus.RUNNING.value,
        )
        assert ManagedHeader.from_dict(row.header, run_id=row.run_id) == ManagedHeader(
            parent, ("reviewer",), parent, child.tool_call_id
        )
        assert (row.binding, row.loop, row.pause_id, row.pause_reason) == (
            None,
            "react",
            None,
            None,
        )
    assert await store.get_managed_children(parent, [c1], tenant_id="other") == []
    assert await store.get_managed_children(parent, [c1], tenant_id=None) == []
    assert len(await store.get_managed_children(parent, [c1, c2], tenant_id=UNSCOPED)) == 2
    assert await store.get_managed_children(other, [c1], tenant_id="t") == []
    assert await store.get_managed_children(parent, [], tenant_id="t") == []
    for bad in ([ulid() for _ in range(33)], [c1, c1]):
        with pytest.raises(ValueError):
            await store.get_managed_children(parent, bad, tenant_id="t")


@fanout
async def test_managed_children_projection_is_fresh_on_every_call(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 3)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    c1, c2, c3 = ids = [c.child_run_id for c in request.children]
    assert await store.finalize_run(
        c1,
        status=RunStatus.SUCCESS.value,
        expected_current_status=RunStatus.RUNNING.value,
        claim_epoch=0,
    )
    pause_id = ulid()
    await store.pause_run(
        c2,
        status=RunStatus.PAUSED.value,
        pause_data={
            "pause_id": pause_id,
            "pause_reason": PauseReason.REQUESTED.value,
            "pending_tool_calls": [],
        },
        claim_epoch=0,
    )
    nested = await _nest(dendrux_kit, c3, 2)
    inner = _request(c3, nested, root=parent, path=("reviewer",))
    await store.bind_managed_children(inner)
    rows = {r.run_id: r for r in await store.get_managed_children(parent, ids, tenant_id="t")}
    assert rows[c1].status == RunStatus.SUCCESS.value
    assert (rows[c2].status, rows[c2].pause_id, rows[c2].pause_reason) == (
        RunStatus.PAUSED.value,
        pause_id,
        PauseReason.REQUESTED.value,
    )
    inner_wait = inner.parent_wait.to_dict()
    assert (rows[c3].pause_id, rows[c3].pause_reason) == (inner_wait["pause_id"], "managed_wait")
    assert rows[c3].binding == {
        "v": 1,
        "pause_id": inner_wait["pause_id"],
        "bindings": {c.tool_call_id: c.child_run_id for c in inner.children},
    }


@fanout_second
async def test_managed_children_simultaneous_finishers_see_each_other(dendrux_kit: Kit) -> None:
    """Each finisher reads after its own commit; the later one must see both terminal."""
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    ids = [c.child_run_id for c in request.children]
    second = await dendrux_kit.second()

    async def finish(target: Any, child: str) -> set[str]:
        assert await target.finalize_run(
            child,
            status=RunStatus.SUCCESS.value,
            expected_current_status=RunStatus.RUNNING.value,
            claim_epoch=0,
        )
        rows = await target.get_managed_children(parent, ids, tenant_id="t")
        return {r.run_id for r in rows if r.status == RunStatus.SUCCESS.value}

    seen = await asyncio.gather(finish(store, ids[0]), finish(second, ids[1]))
    assert set(ids) in seen, "the last finisher's fresh read must see every sibling terminal"


# ---------------------------------------------------------------------------------------------
# K12, a lost acknowledgement never grants a second creator
# ---------------------------------------------------------------------------------------------


def _lost_ack_outcome(outcome: Any, request: BindManagedChildrenRequest) -> None:
    """A surfaced result names the batch's children; only ``created`` carries tokens."""
    if outcome.raised is not None:
        assert isinstance(outcome.raised, PersistenceError)
        return
    result = outcome.result
    assert [b.child_run_id for b in result.children] == [c.child_run_id for c in request.children]
    if result.outcome == "created":
        claims = [b.child_claim for b in result.children]
        assert all((c.claim_epoch, c.storage_format) == (0, 2) for c in claims)


@fanout_crash
async def test_fanout_bind_lost_ack_grants_only_creator_authority(dendrux_kit: Kit) -> None:
    forms = dendrux_kit.disconnect_forms()
    if not forms:
        store_skip(
            "no disconnect forms: fan-out bind lost acknowledgement untested", "managed_fanout"
        )
    store = dendrux_kit.store
    for form in forms:
        parent, calls = await _parent(dendrux_kit, 3)
        request = _request(parent, calls)
        outcome = await dendrux_kit.faulted(
            "after_commit", partial(store.bind_managed_children, request), form=form
        )
        outcome.expect_fired("after_commit")
        _lost_ack_outcome(outcome, request)
        await _published(store, request)
        # An identical public request carries no private proof of commit ownership.
        for replay in (request, _request(parent, calls)):
            existing = await store.bind_managed_children(replay)
            assert existing.outcome == "existing"
            assert all(b.child_claim is None for b in existing.children)
        await _published(store, request)


@fanout_crash
async def test_fanout_bind_lost_before_commit_is_retried_as_the_same_bind(
    dendrux_kit: Kit,
) -> None:
    """A connection lost before COMMIT proves nothing applied: the store re-enters the bind."""
    forms = dendrux_kit.disconnect_forms()
    if not forms:
        store_skip("no disconnect forms: fan-out bind retry untested", "managed_fanout")
    store = dendrux_kit.store
    for form in forms:
        parent, calls = await _parent(dendrux_kit, 3)
        request = _request(parent, calls)
        outcome = await dendrux_kit.faulted(
            "before_commit", partial(store.bind_managed_children, request), form=form
        )
        outcome.expect_fired("before_commit")
        assert outcome.raised is None, "one lost attempt that applied nothing is retried"
        assert outcome.result.outcome == "created"
        assert all(b.child_claim is not None for b in outcome.result.children)
        await _published(store, request)
