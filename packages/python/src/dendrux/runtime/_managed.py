"""Drive single-child workflows inline through durable bind, adoption and continuation.

Routing, reports and stream cleanup live in the managed helper modules. Fan-out
claim ownership lives in ``_managed_fanout``; the N=1 driver stays in this task.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import TERMINAL_STATUSES, StoredEventType
from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.runtime import (
    _managed_adoption,
    _managed_budget,
    _managed_cancel,
    _managed_definitions,
    _managed_route,
    _managed_stream,
)
from dendrux.runtime._managed_adoption import _FAILURE_CODES as _FAILURE_CODES
from dendrux.runtime._managed_adoption import _FAILURE_TEXT as _FAILURE_TEXT
from dendrux.runtime._managed_adoption import CHILD_CANCELLED as CHILD_CANCELLED
from dendrux.runtime._managed_adoption import CHILD_FAILED as CHILD_FAILED
from dendrux.runtime._managed_adoption import REPORT_INVALID as REPORT_INVALID
from dendrux.runtime._managed_adoption import REPORT_TOO_LARGE as REPORT_TOO_LARGE
from dendrux.runtime._managed_adoption import AcceptedReport, logger
from dendrux.runtime._managed_adoption import _failed_report as _failed_report
from dendrux.runtime._managed_adoption import _report_payload as _report_payload
from dendrux.runtime._managed_adoption import _transformed as _transformed
from dendrux.runtime._managed_adoption import accept_report as accept_report
from dendrux.runtime._managed_adoption import child_report as child_report
from dendrux.runtime._managed_adoption import project_report as project_report
from dendrux.runtime._managed_budget import UNPRICED_ACCEPTED, accepted_count
from dendrux.runtime._managed_definitions import (
    _check_definitions,
    require_bindable,
)
from dendrux.runtime._managed_definitions import (
    require_streamable as require_streamable,
)
from dendrux.runtime._managed_limits import (
    _saved_walk_limit as _saved_walk_limit,
)
from dendrux.runtime._managed_limits import (
    stamp_root_header as stamp_root_header,
)
from dendrux.runtime._managed_route import MANAGED_WAIT_REASON, Route, _Waiting
from dendrux.runtime._managed_route import _bound_child as _bound_child
from dendrux.runtime._managed_route import _intent_call as _intent_call
from dendrux.runtime._managed_route import _outgoing as _outgoing
from dendrux.runtime._managed_route import _projected_pause as _projected_pause
from dendrux.runtime._managed_route import _registered as _registered
from dendrux.runtime._managed_route import _root_result as _root_result
from dendrux.runtime._managed_route import root_view as root_view
from dendrux.runtime._managed_route import walk as walk
from dendrux.runtime._managed_stream import _cancel_unstarted as _cancel_unstarted
from dendrux.runtime._managed_stream import _close_workflow as _close_workflow
from dendrux.runtime._managed_stream import _continuation_stream as _continuation_stream
from dendrux.runtime._managed_stream import _event as _event
from dendrux.runtime._managed_stream import _event_with_error as _event_with_error
from dendrux.runtime._managed_stream import _paused_or_failed as _paused_or_failed
from dendrux.runtime._managed_stream import _Single, _Started, _StreamState, _Terminal
from dendrux.runtime._managed_stream import _stream_failure as _stream_failure
from dendrux.runtime._managed_stream import _structural as _structural
from dendrux.runtime.state import MANAGED_HEADER_KEY, ManagedHeader
from dendrux.types import (
    Limits,
    PauseState,
    RunResult,
    RunStatus,
    ToolCall,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.agent import Agent
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.state import EventWrite, StateStore
    from dendrux.types import RunStream


DESCEND_KEY = "managed_descend"
"""``RunResult.meta`` key: the parent exited at a managed handoff (in memory only)."""
NOT_OWNER_KEY = "managed_not_owner"
"""``RunResult.meta`` key: this caller does not own the next managed step."""

ADMITTED_KEY = "managed_admitted"
"""Checkpoint key: the pending managed call the live batch admitted before its handoff."""


# ---------------------------------------------------------------------------
# Data handed between the parent frame, the child entry and the driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManagedInvocation:
    """One admitted managed call of a parent at its claim epoch (in memory, typed).

    The parent's run frame builds it after its checkpoint is durable; the
    child's run entry turns it into a ``BindManagedChildRequest``.
    """

    store: Any
    tenant_id: str | None
    root_run_id: str
    parent_run_id: str
    parent_epoch: int
    parent_level: int
    parent_agent: Agent
    call: ToolCall
    child: Agent
    child_path: tuple[str, ...]
    wait: PauseState
    iteration: int
    parent_usage: UsageStats
    max_depth: int | None
    child_run_id: str
    unpriced_accepted: int = 0


def is_descend(result: RunResult | None) -> bool:
    return result is not None and DESCEND_KEY in (result.meta or {})


def is_managed_root(agent: Agent) -> bool:
    return bool(getattr(agent, "_subagents", ()))


# ---------------------------------------------------------------------------
# Root entry
# ---------------------------------------------------------------------------


def require_root_entry(agent: Agent, store: StateStore | None, loop: Any) -> None:
    """Refuse a managed root that cannot run durably, before any write."""
    from dendrux.errors import PersistenceNotConfiguredError
    from dendrux.loops.react import ReActLoop
    from dendrux.runtime._capabilities import require_managed_subagents
    from dendrux.runtime.state import unclaimed_store

    if store is None:
        raise PersistenceNotConfiguredError()
    require_managed_subagents(store)
    if not isinstance(loop, ReActLoop):
        raise ValueError("Managed subagents need a ReAct parent loop.")
    _check_definitions(agent, unclaimed_store(store))


async def require_checkpointed_creation(store: Any) -> None:
    """A managed root needs a format-2 (checkpointed) run, checked before creation."""
    from dendrux.runtime.state import unclaimed_store

    reader = getattr(unclaimed_store(store), "_creation_storage_format", None)
    if callable(reader) and await reader() != 2:
        raise ManagedStateError(reason="storage_format")


def raise_unbound(invocation: ManagedInvocation | None, claimed: Any) -> None:
    """Re-raise the active exception for a managed child that never bound (stream path)."""
    if invocation is not None and not claimed:
        raise  # noqa: PLE0704 - called only from an except block to re-raise its active exception


def new_run_id(invocation: ManagedInvocation | None) -> str:
    """A managed child's id is its parent's bind candidate, minted at the handoff."""
    return invocation.child_run_id if invocation is not None else generate_ulid()


def parent_context(invocation: ManagedInvocation | None, current: Callable[[], Any]) -> Any:
    """A managed child never inherits the caller's delegation context."""
    return None if invocation is not None else current()


def parent_link(
    invocation: ManagedInvocation | None, parent_ctx: Any, store: Any
) -> tuple[str | None, int]:
    """The child's parent link and level come from its bound invocation."""
    from dendrux.runtime.context import resolve_parent_link

    if invocation is None:
        return resolve_parent_link(parent_ctx, store)
    return invocation.parent_run_id, invocation.parent_level + 1


async def enter(
    agent: Agent, invocation: ManagedInvocation | None, store: Any, loop: Any
) -> Limits | None:
    """Capture a managed root's limits; refuse one that cannot run durably."""
    if invocation is not None:
        await _managed_cancel.enter_child(invocation, store)
        return None
    if not is_managed_root(agent):
        return None
    require_root_entry(agent, store, loop)
    await require_checkpointed_creation(store)
    return agent._limits


# ---------------------------------------------------------------------------
# The parent's handoff (inside its run frame, before it exits)
# ---------------------------------------------------------------------------


async def prepare_handoff(
    store: StateStore,
    run_id: str,
    result: RunResult,
    recorder: LoopRecorder | None,
    provider_kwargs: dict[str, Any] | None,
) -> RunResult | None:
    """Make the parent's checkpoint durable and return the descend marker.

    Returns None when cancellation was requested: the caller finalizes as for
    any pause. The parent stays RUNNING at its epoch until the child's atomic
    bind parks it.
    """
    from dendrux.loops._helpers import project_message
    from dendrux.loops.react import MANAGED_HANDOFF_KEY
    from dendrux.runtime.context import DelegationDepthExceededError, get_delegation_context

    handoff = result.meta[MANAGED_HANDOFF_KEY]
    # Every bind, live or recovered, checks the child's definition before any effect.
    require_bindable(handoff.parent, handoff.child)
    if await _managed_cancel.cancellation_requested(store, run_id):
        return None
    header = ManagedHeader.from_dict(
        (getattr(store, "loop_state", None) or {}).get(MANAGED_HEADER_KEY), run_id=run_id
    )
    if (ctx := get_delegation_context()) is None or ctx.run_id != run_id:
        # The runner sets this run's own context for every execution; fail closed.
        raise ManagedStateError(reason="header_invalid", run_id=run_id)
    level, max_depth = ctx.delegation_level, ctx.max_delegation_depth
    if max_depth is not None and level + 1 > max_depth:
        raise DelegationDepthExceededError(level + 1, max_depth)
    pause: PauseState = result.meta["pause_state"]
    pause.history[:] = [await project_message(recorder, m) for m in pause.history]
    state = getattr(recorder, "_loop_state", None)
    if isinstance(state, dict):
        # Durable admission evidence: this call passed validation, policy and
        # approval in a live batch; recovery binds a pending call only with it.
        state[ADMITTED_KEY] = handoff.call.id
    checkpoint = getattr(recorder, "checkpoint_outcome", None)
    if checkpoint is not None:
        await checkpoint(result, provider_kwargs)
    state = state or {}
    wait = replace(
        pause,
        pending_tool_calls=[handoff.call],
        pending_targets={handoff.call.id: "agent"},
        pause_reason=MANAGED_WAIT_REASON,
        pause_id=generate_ulid(),
        schema_version=3,
        loop_state_ref=True,
        approval_gates=None,
        pending_decisions=None,
        clarification=None,
        image_view=None,
        image_view_ref=state.get("image_view_ref"),
        provider_kwargs=dict(provider_kwargs or pause.provider_kwargs or {}),
    )
    invocation = ManagedInvocation(
        store=_shared(store),
        tenant_id=ctx.tenant_id,
        root_run_id=header.root_run_id,
        parent_run_id=run_id,
        parent_epoch=int(getattr(store, "claim_epoch", 0)),
        parent_level=level,
        parent_agent=handoff.parent,
        call=handoff.call,
        child=_managed_budget.inherit(handoff.child, handoff.parent),
        child_path=(*header.path, handoff.child.name),
        wait=wait,
        iteration=result.iteration_count,
        parent_usage=result.usage,
        max_depth=max_depth,
        child_run_id=generate_ulid(),
        unpriced_accepted=accepted_count(state.get(UNPRICED_ACCEPTED)),
    )
    return RunResult(
        run_id=run_id,
        status=RunStatus.RUNNING,
        iteration_count=result.iteration_count,
        usage=result.usage,
        meta={DESCEND_KEY: invocation},
    )


def _shared(store: Any) -> Any:
    from dendrux.runtime.state import unclaimed_store

    return unclaimed_store(store)


# ---------------------------------------------------------------------------
# The child's entry: the atomic bind instead of create_run
# ---------------------------------------------------------------------------


def intent_event(invocation: ManagedInvocation, child_run_id: str) -> EventWrite:
    """The parent's delegation-start event: ids, path and kind only, never the task."""
    from dendrux.runtime.state import EventWrite

    call = invocation.call
    return EventWrite(
        generate_ulid(),
        StoredEventType.TOOL_STARTED.value,
        invocation.iteration,
        {
            "tool_name": call.name,
            "call_id": call.id,
            "idempotent": False,
            "target": "agent",
            "child_run_id": child_run_id,
            "agent_path": list(invocation.child_path),
        },
        call.id,
        None,
    )


async def bind_child(
    invocation: ManagedInvocation,
    store: Any,
    run_id: str,
    agent_name: str,
    *,
    loop_state: dict[str, Any] | None,
    **create: Any,
) -> Any:
    """Bind ``run_id`` as the invocation's child in one store transaction."""
    from dendrux.runtime.state import BindManagedChildRequest, LoopStatePatch
    from dendrux.types import ManagedWait

    request = BindManagedChildRequest(
        tenant_id=invocation.tenant_id,
        root_run_id=invocation.root_run_id,
        parent_run_id=invocation.parent_run_id,
        parent_claim_epoch=invocation.parent_epoch,
        tool_call_id=invocation.call.id,
        row_id=generate_ulid(),
        provider_tool_call_id=invocation.call.provider_tool_call_id,
        tool_name=invocation.call.name,
        params=invocation.call.params or None,
        iteration_index=invocation.iteration,
        child_run_id=run_id,
        child_agent_name=agent_name,
        child_path=invocation.child_path,
        child_input_data=create.get("input_data"),
        child_loop_state=_managed_budget.child_state(loop_state, invocation.unpriced_accepted),
        parent_wait=replace(invocation.wait, managed_wait=ManagedWait(invocation.call.id, run_id)),
        parent_patch=LoopStatePatch(set_phase="dispatch"),
        intent_event=intent_event(invocation, run_id),
        child_seed_traces=tuple(create.get("seed_traces") or ()),
        child_model=create.get("model"),
        child_strategy=create.get("strategy"),
        child_meta=create.get("meta"),
        child_pii_mapping=create.get("pii_mapping"),
        parent_iteration_count=invocation.iteration,
        parent_total_usage=invocation.parent_usage,
        max_delegation_depth=invocation.max_depth,
    )
    return await store.bind_managed_child(request)


def not_owner_result(run_id: str) -> RunResult:
    """The child exists but this caller did not create it: never drive it."""
    return RunResult(run_id=run_id, status=RunStatus.RUNNING, meta={NOT_OWNER_KEY: True})


# ---------------------------------------------------------------------------
# The driver (same caller task, explicit trampoline)
# ---------------------------------------------------------------------------


async def drive(result: RunResult, *, notifier: LoopNotifier | None) -> RunResult:
    """Run descended children inline and unwind their outcomes to the root."""
    if not is_descend(result):
        return result
    invocation: ManagedInvocation = result.meta[DESCEND_KEY]
    return await _trampoline(
        result, [], root_run_id=invocation.root_run_id, store=invocation.store, notifier=notifier
    )


async def _trampoline(
    result: RunResult,
    waiting: list[_Waiting],
    *,
    root_run_id: str,
    store: Any,
    notifier: LoopNotifier | None,
) -> RunResult:
    for _ in range(_MAX_STEPS):
        if (result.meta or {}).get(NOT_OWNER_KEY):
            return await _managed_route.root_view(store, root_run_id)
        if is_descend(result):
            invocation: ManagedInvocation = result.meta[DESCEND_KEY]
            result, bound = await _run_child(invocation, notifier)
            if bound:
                waiting.append(_waiting_from(invocation))
            continue
        if not waiting:
            return result
        if result.status not in TERMINAL_STATUSES:
            return await _managed_route._projected_pause(store, root_run_id, waiting, result)
        result = await adopt(waiting.pop(), store=store, notifier=notifier)
    raise ManagedStateError(reason="binding_conflict", run_id=root_run_id)


_MAX_STEPS = 100_000
"""A safety stop for the trampoline; each step is a durable transition."""


def _waiting_from(invocation: ManagedInvocation) -> _Waiting:
    return _Waiting(
        parent_agent=invocation.parent_agent,
        parent_run_id=invocation.parent_run_id,
        child=invocation.child,
        child_path=invocation.child_path,
        call=invocation.call,
        child_run_id=invocation.child_run_id,
    )


async def _run_child(
    invocation: ManagedInvocation, notifier: LoopNotifier | None
) -> tuple[RunResult, bool]:
    """Run the bound child in this task: ``(result, bound)``.

    A child that fails after its bind is adopted as a failed tool result (its
    error stays on the child run). A bind that never happened leaves the
    parent as the result: cancelled when the parent's cancel won, else the
    parent records the failure and the error propagates.
    """
    from dendrux.runtime._subagents import _tighten
    from dendrux.runtime.context import managed_invocation_scope
    from dendrux.runtime.runner import _run_once

    child = invocation.child
    try:
        with managed_invocation_scope(invocation):
            result = await _run_once(
                child,
                provider=child._require_provider(),
                user_input=_task(invocation),
                state_store=invocation.store,
                tenant_id=invocation.tenant_id,
                extra_notifier=notifier,
                # A child's own setting can only tighten the inherited ceiling.
                max_delegation_depth=_tighten(invocation.max_depth, child.max_delegation_depth),
                output_type=getattr(child, "_output_type", None),
            )
    except Exception as exc:
        return await _after_child_error(invocation, exc)
    return result, not (result.meta or {}).get(NOT_OWNER_KEY)


def _task(invocation: ManagedInvocation) -> str:
    return str((invocation.call.params or {}).get("task", ""))


async def _after_child_error(
    invocation: ManagedInvocation,
    exc: Exception,
    *,
    failure_for_storage: Exception | None = None,
) -> tuple[RunResult, bool]:
    """The child's run raised: ``(result, bound)``, or re-raise ``exc``.

    Shared by the run and stream drivers. Bound by another caller: not the
    owner. Never bound: the parent is cancelled when its cancel won, else it
    records the failure and ``exc`` propagates. A bound child that ended
    terminal is adopted like any terminal child.
    """
    from dendrux.errors import RunCancelledError, RunFencedError
    from dendrux.types import RunAlreadyActiveError

    if isinstance(exc, RunAlreadyActiveError):
        return not_owner_result(invocation.parent_run_id), False  # bound by another caller
    record = await invocation.store.get_run(invocation.child_run_id)
    if isinstance(exc, (RunCancelledError, RunFencedError)):
        if record is not None:
            raise exc
        if isinstance(exc, RunFencedError):
            return not_owner_result(invocation.parent_run_id), False
        return await _cancel_parent(invocation), False
    if record is None:
        await _fail_parent(invocation, failure_for_storage or exc)
        raise exc
    if record.status not in {s.value for s in TERMINAL_STATUSES}:
        raise exc
    return await _terminal_child(invocation.store, record.id), True


async def _terminal_child(store: Any, child_run_id: str) -> RunResult:
    from dendrux.runtime.runner import _build_cached_result

    return await _build_cached_result(store, child_run_id)


def _parent_store(invocation: ManagedInvocation) -> Any:
    from dendrux.runtime.state import claimed_store

    return claimed_store(invocation.store, invocation.parent_run_id, invocation.parent_epoch)


async def _cancel_parent(invocation: ManagedInvocation) -> RunResult:
    """The parent's cancel won before its bind: finalize it as for any pause."""
    from dendrux._lifecycle import CancelReason
    from dendrux.runtime.runner import _finalize_loop_cancelled

    parent = RunResult(
        run_id=invocation.parent_run_id,
        status=RunStatus.CANCELLED,
        iteration_count=invocation.iteration,
        usage=invocation.parent_usage,
    )
    return await _finalize_loop_cancelled(
        _parent_store(invocation),
        invocation.parent_run_id,
        parent,
        None,
        {},
        [RunStatus.RUNNING.value],
        {"reason": CancelReason.CANCEL_REQUESTED.value},
    )


async def _fail_parent(invocation: ManagedInvocation, exc: Exception) -> None:
    """A refused bind fails the still-owned parent like any run failure."""
    from dendrux.runtime.runner import _persist_run_failure

    await _persist_run_failure(_parent_store(invocation), invocation.parent_run_id, exc)


# ---------------------------------------------------------------------------
# Adoption: one saved child outcome becomes one parent observation
# ---------------------------------------------------------------------------


async def adopt(waiting: _Waiting, *, store: Any, notifier: LoopNotifier | None) -> RunResult:
    """Accept a terminal child's saved outcome into its parent, then continue the parent."""
    from dendrux.runtime import _reclaim

    claimed = await claim_adoption(waiting, store)
    if claimed is None:
        return not_owner_result(waiting.parent_run_id)
    return await _reclaim.continue_claimed(
        waiting.parent_agent, store, waiting.parent_run_id, claimed, notifier=notifier
    )


async def claim_adoption(waiting: _Waiting, store: Any) -> Any:
    """Durably adopt a terminal child into its parent; the parent's new claim, or None.

    None when the child is not terminal or another caller won the parent's
    claim. The returned claim owns the parent's continuation.
    """
    from dendrux.runtime.state import ClaimResult, claimed_store
    from dendrux.runtime.submit import claim_managed_adoption

    if await _managed_cancel.cancellation_requested(store, waiting.parent_run_id):
        return None
    record = await store.get_run(waiting.child_run_id)
    if record is None:
        raise ManagedStateError(reason="child_missing", run_id=waiting.parent_run_id)
    if record.status not in {s.value for s in TERMINAL_STATUSES}:
        return None
    accepted = await _managed_adoption.accept_report(waiting, record, store)
    claim = await claim_managed_adoption(
        store,
        waiting.parent_run_id,
        agent=waiting.parent_agent,
        call_id=waiting.call_id,
        child_run_id=record.id,
        result=accepted.result,
        blocked=accepted.blocked,
    )
    if not isinstance(claim, ClaimResult):
        return None
    owned = claimed_store(store, waiting.parent_run_id, claim)
    state = await complete_adoption(
        owned,
        waiting.parent_run_id,
        claim,
        accepted,
        child_run_id=record.id,
        path=waiting.child_path,
    )
    return ClaimResult(claim.claim_epoch, claim.storage_format, state, None)


async def complete_adoption(
    owned: Any,
    run_id: str,
    claim: Any,
    accepted: AcceptedReport,
    *,
    child_run_id: str,
    path: tuple[str, ...],
) -> dict[str, Any]:
    """The one completion transaction, under blocked provenance when the parent blocked it."""
    from contextlib import nullcontext

    from dendrux.runtime import _managed_persistence
    from dendrux.runtime._blocked_content import _blocked_evidence

    scope = _blocked_evidence(owned, run_id) if accepted.blocked is not None else nullcontext()
    async with scope:
        return await _managed_persistence.adopt_managed_result(
            owned,
            run_id,
            claim=claim,
            call_id=accepted.result.call_id,
            report=accepted.result,
            child_run_id=child_run_id,
            agent_path=path,
            observation=accepted.observation,
            pii_mapping=accepted.pii_mapping,
        )


# ---------------------------------------------------------------------------
# Root routing: a root-id control reaches the active leaf (0.7, T3)
# ---------------------------------------------------------------------------


async def route(root_agent: Agent, store: Any, run_id: str, *, tenant_id: Any) -> Route:
    """Resolve a control on ``run_id`` to its active leaf; tenant and root checked first.

    A missing or foreign run is not found before anything is disclosed; a
    managed child id is refused (``managed_root_required``). The leaf's Agent
    is resolved from the registered path in ``root_agent``; a missing path
    refuses before any claim.
    """
    from dendrux.errors import ManagedChildControlError
    from dendrux.runtime.state import UNSCOPED

    record = await store.get_run(run_id)
    if record is None or (tenant_id is not UNSCOPED and record.tenant_id != tenant_id):
        raise RunNotFoundError(run_id)
    if await _managed_cancel.managed_child_header(store, record):
        raise ManagedChildControlError(run_id)
    await _managed_budget.route_checkpoint(root_agent, store, record)
    waiting, leaf = await _managed_route.walk(root_agent, store, run_id)
    leaf_agent = waiting[-1].child if waiting else root_agent
    return Route(run_id, record.tenant_id, tuple(waiting), leaf, leaf_agent)


def require_leaf_pause(found: Route, expected_pause_id: str | None) -> None:
    """A routed continuation needs the leaf's external pause and its token.

    A running target (the root's own leg, a running subagent, or one whose
    outcome awaits adoption) has nothing external to continue. Every managed
    continuation needs an explicit ``expected_pause_id`` (strict or not), which
    the leaf's locked claim compares exactly; it is never taken from a later read.
    """
    from dendrux.errors import RunNotPausedError
    from dendrux.runtime._pending import PAUSE_STATUSES
    from dendrux.runtime.submit import _pause_id_required

    status = found.leaf.status
    if not found.routed and status in _TERMINAL_VALUES:
        return  # a finished root: the ordinary path reports it (nothing can be claimed)
    if status not in {s.value for s in PAUSE_STATUSES}:
        # A running target, the root itself included: nothing external to continue.
        # Refusing here keeps a later pause from being taken without its token.
        raise RunNotPausedError(found.root_run_id, RunStatus.RUNNING)
    if expected_pause_id is None:
        _codes.log_code(logger, _codes.MANAGED_PAUSE_ID_REQUIRED, run_id=found.root_run_id)
        raise _pause_id_required()
    _managed_definitions.require_resumable_leaf(found)


async def unwind(found: Route, store: Any, result: RunResult, *, notifier: Any) -> RunResult:
    """Continue from the leaf's result: descend, adopt and continue up to the root."""
    return await _trampoline(
        result,
        list(found.waiting),
        root_run_id=found.root_run_id,
        store=_shared(store),
        notifier=notifier,
    )


async def adopt_terminal_leaf(found: Route, store: Any, *, notifier: Any) -> RunResult | None:
    """Adopt a finished subagent's saved outcome (no token needed); None otherwise."""
    if not found.routed or found.leaf.status not in {s.value for s in TERMINAL_STATUSES}:
        return None
    waiting = list(found.waiting)
    shared = _shared(store)
    result = await adopt(waiting.pop(), store=shared, notifier=notifier)
    return await _trampoline(
        result, waiting, root_run_id=found.root_run_id, store=shared, notifier=notifier
    )


# ---------------------------------------------------------------------------
# Streaming: the same trampoline, advancing each run's generator in this task
# ---------------------------------------------------------------------------

_LIFECYCLE_STARTS = frozenset({"run_started", "run_resumed"})
_TERMINAL_EVENTS = frozenset({"run_completed", "run_paused", "run_error", "run_cancelled"})


_TOOL_EVENTS = frozenset({"tool_use_start", "tool_use_end", "tool_result"})


def _forwarded(
    event: Any,
    run_id: str | None,
    *,
    root_level: bool,
    first: bool,
    managed_names: frozenset[str] = frozenset(),
) -> Any:
    """What the root-facing stream shows of one run's event.

    The root's own events pass as they are, except that a generated subagent
    call is shown structurally (no ``task`` brief, no report). A subagent's
    events are reduced to structural tool progress tagged with its run id:
    ids, tool name, iteration and outcome code, never arguments, results,
    text or lifecycle events. Detail stays on ``RunStore.follow(child_id)``.
    """
    kind = getattr(event.type, "value", event.type)
    if root_level:
        if not first and kind in _LIFECYCLE_STARTS:
            return None
        name = event.tool_name or getattr(event.tool_call, "name", None)
        return _managed_stream._structural(event, event.run_id) if name in managed_names else event
    if kind not in _TOOL_EVENTS:
        return None
    return _managed_stream._structural(event, run_id)


def _child_stream(invocation: ManagedInvocation, notifier: Any) -> Any:
    """The bound child's own stream (created by the atomic bind on first step)."""
    from dendrux.runtime._subagents import _tighten
    from dendrux.runtime.runner import run_stream

    child = invocation.child
    return run_stream(
        child,
        provider=child._require_provider(),
        user_input=_task(invocation),
        state_store=invocation.store,
        tenant_id=invocation.tenant_id,
        extra_notifier=notifier,
        max_delegation_depth=_tighten(invocation.max_depth, child.max_delegation_depth),
        _managed_invocation=invocation,
    )


async def _drain(
    stream: Any, *, root_level: bool, first: bool, managed_names: frozenset[str] = frozenset()
) -> Any:
    """Yield one run's forwarded events; the last item is its terminal event (or None).

    The run's own stream is closed here, in this task, however iteration ends.
    """
    from contextlib import aclosing

    terminal = None
    async with aclosing(stream.__aiter__()) as events:
        async for event in events:
            if getattr(event.type, "value", event.type) in _TERMINAL_EVENTS:
                terminal = event
                continue
            out = _forwarded(
                event,
                stream.run_id,
                root_level=root_level,
                first=first,
                managed_names=managed_names,
            )
            if out is not None:
                yield out
    yield _Terminal(terminal)


@dataclass
class _Leg:
    """One run's stream inside the workflow: its outcome once drained."""

    stream: Any
    invocation: ManagedInvocation | None = None
    terminal: Any = None
    result: Any = None
    bound: bool = False


async def _run_leg(leg: _Leg, state: _StreamState, *, first: bool) -> Any:
    """Forward one run's events and record its outcome on ``leg``.

    A run is the root's own leg only when nothing waits and no child is
    binding. Before its bind a child is not a run: its failure decides for its
    parent, exactly as for ``run()``.
    """
    from contextlib import aclosing

    root_level = not state.waiting and leg.invocation is None
    state.current, state.root_active = leg.stream, root_level
    try:
        names = frozenset(child.name for child in state.root_agent.subagents)
        items = _drain(leg.stream, root_level=root_level, first=first, managed_names=names)
        async with aclosing(items):
            async for item in items:
                state.unstarted = None  # the leg's own stream now owns its cleanup
                if isinstance(item, _Terminal):
                    leg.terminal = item.event
                else:
                    yield item
    except Exception as exc:
        state.unstarted = None
        if leg.invocation is None:
            raise
        leg.result, leg.bound = await _after_child_error(leg.invocation, exc)
        return
    if leg.terminal is None:
        raise ManagedStateError(reason="binding_conflict", run_id=state.root_run_id)
    leg.result, leg.bound = leg.terminal.run_result, leg.invocation is not None


async def _drive_stream(first: Any, state: _StreamState, *, notifier: Any) -> Any:
    """Advance each run of the branch in this task, adopting and unwinding to the root.

    Same transitions as :func:`_trampoline`: a handoff streams the bound
    child; an external leaf pause ends the stream with the root-facing pause;
    a terminal child is adopted and its parent's continuation is streamed.
    Only the root's own completion ends the stream as the root's result.
    """
    state.waiting = state.waiting if state.waiting is not None else []
    leg = _Leg(first)
    for step in range(_MAX_STEPS):
        async for item in _run_leg(leg, state, first=step == 0):
            yield item
        if state.root_run_id is None:
            state.root_run_id = leg.stream.run_id
        if leg.invocation is not None and leg.bound:
            state.waiting.append(_waiting_from(leg.invocation))
        following = await _next_leg(leg, state, notifier)
        if not isinstance(following, _Leg):
            yield following
            return
        leg = following
    raise ManagedStateError(reason="binding_conflict", run_id=state.root_run_id)


async def _next_leg(leg: _Leg, state: _StreamState, notifier: Any) -> Any:
    """The next run to stream, or the workflow stream's final event."""
    result, waiting = leg.result, state.waiting or []
    if (result.meta or {}).get(NOT_OWNER_KEY):
        return _managed_stream._event(
            await _managed_route.root_view(state.store, str(state.root_run_id))
        )
    if is_descend(result):
        invocation = result.meta[DESCEND_KEY]
        state.tenant_id = invocation.tenant_id
        return _Leg(_child_stream(invocation, notifier), invocation)
    if not waiting:
        return leg.terminal if leg.terminal is not None else _managed_stream._event(result)
    if result.status not in TERMINAL_STATUSES:
        return _managed_stream._event(
            await _managed_stream._paused_or_failed(state, waiting, result, leg.terminal)
        )
    claim = await claim_adoption(waiting[-1], state.store)
    if claim is None:
        return _managed_stream._event(
            await _managed_route.root_view(state.store, str(state.root_run_id))
        )
    parent = waiting.pop()
    state.unstarted = (parent.parent_run_id, claim)
    return _Leg(_managed_stream._continuation_stream(parent, state.store, claim, notifier))


def workflow_stream(
    root_agent: Agent,
    open_first: Callable[[_StreamState], Any],
    *,
    run_id: str,
    notifier: Any,
) -> RunStream:
    """A managed workflow's RunStream: one root-facing stream over every run it drives.

    ``open_first`` resolves the store and routing lazily on first iteration and
    returns ``(first stream, first event or None)``; it raises (before anything
    is claimed) for not found, a child id or a missing token. Once driving, a
    failure ends the stream with a root-facing ``RUN_ERROR``. Closing the stream
    early cancels the active run through its own stream and, while the root
    waits on a subagent, cancels the root with its descendants.
    """
    from contextlib import aclosing

    from dendrux.types import RunStream as _RunStream

    state = _StreamState(root_agent)

    async def _generate() -> Any:
        first, opening = await open_first(state)
        if opening is not None:
            yield opening
        driven = _drive_stream(first, state, notifier=notifier)
        try:
            async with aclosing(driven):
                async for event in driven:
                    stream.run_id = state.root_run_id or stream.run_id
                    yield event
        except Exception as exc:
            yield await _managed_stream._stream_failure(state, exc, stream.run_id)

    async def _cleanup() -> RunResult | None:
        return await _managed_stream._close_workflow(state)

    stream = _RunStream(run_id=run_id, generator=_generate(), cleanup=_cleanup)
    return stream


def start_stream(root_agent: Agent, root_stream: Any, *, notifier: Any) -> RunStream:
    """``Agent.stream`` on a managed root: the root's stream, then whatever it hands off."""

    async def open_first(state: _StreamState) -> Any:
        state.store = await root_agent._resolve_state_store()
        return root_stream, None

    return workflow_stream(root_agent, open_first, run_id=root_stream.run_id, notifier=notifier)


def routed_stream(
    root_agent: Agent,
    run_id: str,
    *,
    tenant_id: Any,
    tool_results: Any,
    user_input: Any,
    expected_pause_id: str | None,
    notifier: Any,
    tenant_explicit: bool,
) -> RunStream:
    """``Agent.resume_stream`` on a managed root, routed to the active leaf (0.7).

    Routing errors (not found, a child id, a missing token) raise from the
    first iteration before anything is claimed. A payload-free resume adopts a
    finished subagent without a token.
    """
    from dendrux.runtime.runner import resume_stream
    from dendrux.types import RunEvent, RunEventType

    async def open_first(state: _StreamState) -> Any:
        state.store = store = await root_agent._resolve_state_store()
        found = await route(root_agent, store, run_id, tenant_id=tenant_id)
        state.root_run_id, state.tenant_id = run_id, found.tenant_id
        resumed = RunEvent(type=RunEventType.RUN_RESUMED, run_id=run_id)
        no_payload = tool_results is None and user_input is None
        if no_payload and found.routed and found.leaf.status in _TERMINAL_VALUES:
            waiting = list(found.waiting)
            claim = await claim_adoption(waiting[-1], store)
            if claim is None:
                state.waiting = []
                return _Single(
                    _managed_stream._event(await _managed_route.root_view(store, run_id))
                ), None
            parent = waiting.pop()
            first = _managed_stream._continuation_stream(parent, store, claim, notifier)
            # Cleanup ownership is in place before any event or the claim escapes.
            state.waiting, state.unstarted = waiting, (parent.parent_run_id, claim)
            state.current, state.root_active = first, not waiting
            # The root's own continuation opens with its RUN_RESUMED.
            return first, (resumed if waiting else None)
        require_leaf_pause(found, expected_pause_id)
        leaf = found.leaf_agent
        if found.routed:
            from dendrux.runtime._entry_checks import entry_checks
            from dendrux.runtime.state import UNSCOPED

            entry_checks(
                leaf,
                "resume_stream",
                found.tenant_id,
                default_tenant=UNSCOPED,
                run_id=found.leaf.id,
                tool_results=tool_results,
                user_input=user_input,
            )
        first = resume_stream(
            found.leaf.id,
            agent=leaf,
            provider=leaf._require_provider(),
            state_store=store,
            tool_results=tool_results,
            user_input=user_input,
            extra_notifier=notifier,
            expected_pause_id=expected_pause_id,
            _tenant_explicit=tenant_explicit,
        )
        if not found.routed:
            state.waiting = []
            state.current, state.root_active = first, True
            return first, None
        return await _owned_leaf_stream(state, found, first, resumed)

    return workflow_stream(root_agent, open_first, run_id=run_id, notifier=notifier)


async def _owned_leaf_stream(state: _StreamState, found: Route, first: Any, resumed: Any) -> Any:
    """Validate and claim the leaf in this task before the workflow exposes anything.

    The leaf stream's first step runs its generation check and claim. Only a
    stream that now owns the leaf arms workflow cancellation and opens with the
    root's RUN_RESUMED; a refused or lost claim ends with a root-facing
    RUN_ERROR and changes nothing.
    """
    events = first.__aiter__()
    head = await anext(events, None)
    if head is None or getattr(head.type, "value", head.type) in _TERMINAL_EVENTS:
        await events.aclose()
        state.waiting = []
        view = await _managed_route.root_view(state.store, found.root_run_id)
        error = getattr(head, "error", None) if head is not None else None
        view.error = error
        return _Single(_managed_stream._event_with_error(view, error)), None
    started = _Started(first, events, head)
    state.waiting = list(found.waiting)
    state.current, state.root_active = started, False
    return started, resumed


_TERMINAL_VALUES = frozenset(s.value for s in TERMINAL_STATUSES)


# ---------------------------------------------------------------------------
# Restart: the same adoption and continuation, rebuilt from storage
# ---------------------------------------------------------------------------


async def reconnect(
    root_agent: Agent,
    store: Any,
    root_run_id: str,
    *,
    tenant_id: Any,
    item: Any,
    notifier: LoopNotifier | None,
) -> RunResult:
    """Continue a managed workflow from its root id after a restart.

    Walks the persisted bindings root to leaf. A terminal child is adopted into its
    waiting parent without running it again. A running leaf is reclaimed under lease
    rules (a live owner is never stolen from); a SingleCall leaf never is, on any
    store: ``sweep()`` fails it first. A leaf's external pause is a root view.
    """
    from dendrux.runtime import _reclaim
    from dendrux.runtime.runner import _build_cached_result

    found = await route(root_agent, store, root_run_id, tenant_id=tenant_id)
    waiting, leaf = list(found.waiting), found.leaf
    if leaf.status in {s.value for s in TERMINAL_STATUSES}:
        if not waiting:
            return await _build_cached_result(store, root_run_id)
        result = await adopt(waiting.pop(), store=store, notifier=notifier)
    elif leaf.status == RunStatus.RUNNING.value:
        _managed_definitions.require_claimable(leaf.id, waiting[-1].child_loop if waiting else None)
        leaf_agent = waiting[-1].child if waiting else root_agent
        result = await _reclaim.reclaim(
            leaf_agent,
            store,
            leaf.id,
            tenant_id=tenant_id,
            item=item if not waiting else None,
            notifier=notifier,
            via_root=bool(waiting),
        )
    elif not waiting:
        return await _build_cached_result(store, root_run_id)
    else:
        result = await _build_cached_result(store, leaf.id)
    return await _trampoline(
        result, waiting, root_run_id=root_run_id, store=store, notifier=notifier
    )
