"""Managed subagents (0.7): the inline driver, the bind handoff and durable adoption.

For ``await root.run(task)`` on an Agent with ``subagents=``:

1. The parent's loop returns an in-memory handoff for one admitted subagent call.
   :func:`prepare_handoff` makes the parent's checkpoint durable and returns a
   descend marker; the parent's run frame then exits without pausing.
2. :func:`drive` runs the child in the same caller task. The child's run entry
   takes the invocation and creates the child through the store's atomic
   ``bind_managed_child`` (parent parked as a v3 managed wait, its epoch
   retired). Only the confirmed creator executes the child.
3. When the child is terminal, :func:`adopt` turns its saved outcome into the
   parent's tool observation: ``submit_and_claim`` on the parent's private wait
   generation, then ``complete_tool_call`` with the binding cleared. The parent
   then continues from its checkpoint, never from an old frame.

:func:`reconnect` runs the same adoption and continuation after a restart; no
step exists only in memory.

Root routing (T3): a root-id continuation (``submit_*``, ``resume``,
``resume_stream``) reaches the active leaf through the persisted bindings
(:func:`route`), is validated and claimed with the leaf's own definition, and
:func:`unwind` adopts finished outcomes back up to the root.
:func:`workflow_stream` drives the same transitions for ``stream`` and
``resume_stream``, advancing each run's generator in the caller's task.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import TERMINAL_STATUSES, StoredEventType
from dendrux.errors import ManagedStateError
from dendrux.runtime.state import MANAGED_HEADER_KEY, ManagedHeader
from dendrux.types import (
    PauseState,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.agent import Agent
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.state import EventWrite, RunRecord, StateStore
    from dendrux.types import RunStream

logger = logging.getLogger(__name__)

DESCEND_KEY = "managed_descend"
"""``RunResult.meta`` key: the parent exited at a managed handoff (in memory only)."""
NOT_OWNER_KEY = "managed_not_owner"
"""``RunResult.meta`` key: this caller does not own the next managed step."""

MANAGED_WAIT_REASON = "managed_wait"
ADMITTED_KEY = "managed_admitted"
"""Checkpoint key: the pending managed call the live batch admitted before its handoff."""
_MAX_WALK = 257
"""Bound on a root-to-leaf walk: the definition-node cap plus the root."""

REPORT_TOO_LARGE = "managed_report_too_large"
CHILD_FAILED = "managed_child_failed"
CHILD_CANCELLED = "managed_child_cancelled"
REPORT_INVALID = "managed_report_invalid"


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


@dataclass(frozen=True)
class _Waiting:
    """A parent waiting on one bound child (rebuilt from storage after a restart)."""

    parent_agent: Agent
    parent_run_id: str
    child: Agent
    child_path: tuple[str, ...]
    call: ToolCall
    child_run_id: str

    @property
    def call_id(self) -> str:
        return self.call.id


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


def _check_definitions(root: Agent, store: Any) -> None:
    """Every registered child has a provider and shares the root's store object."""
    stack = [(root, child) for child in root.subagents]
    seen = 0
    while stack and seen < _MAX_WALK:
        parent, child = stack.pop()
        seen += 1
        _require_react(parent, child)
        if child.provider is None:
            raise ValueError(f"Subagent '{child.name}' needs a provider.")
        own = getattr(child, "_state_store", None)
        if (own is not None and own is not store) or (
            own is None and getattr(child, "_database_url", None) is not None
        ):
            raise ValueError(f"Subagent '{child.name}' must use the root's store object or none.")
        stack.extend((child, grandchild) for grandchild in child.subagents)


def require_streamable(root: Agent, entry: str) -> None:
    """Streaming a workflow needs every subagent to be streamable (no guardrails).

    The stream path never scans with guardrails, so a subagent that declares
    them is refused before anything runs, as the root itself is.
    """
    stack = list(root.subagents)
    seen = 0
    while stack and seen < _MAX_WALK:
        child = stack.pop()
        seen += 1
        if child.guardrails or getattr(child, "_output_type", None) is not None:
            raise ValueError(
                f"{entry}() cannot stream subagent '{child.name}': it declares guardrails "
                "or output_type. Use run() for this workflow."
            )
        stack.extend(child.subagents)


def _require_react(parent: Agent, child: Agent) -> None:
    """Managed participants are durable ReAct runs only (owner 2026-11-12)."""
    from dendrux.errors import SubagentDefinitionError
    from dendrux.loops.react import ReActLoop
    from dendrux.runtime._subagents import refuse

    loop = child._effective_loop(None)
    if loop is not None and not isinstance(loop, ReActLoop):
        refuse(
            SubagentDefinitionError(
                parent.name, reason="loop_unsupported", subagent_name=child.name
            )
        )


async def require_checkpointed_creation(store: Any) -> None:
    """A managed root needs a format-2 (checkpointed) run, checked before creation."""
    from dendrux.runtime.state import unclaimed_store

    reader = getattr(unclaimed_store(store), "_creation_storage_format", None)
    if callable(reader) and await reader() != 2:
        raise ManagedStateError(reason="storage_format")


def raise_unbound(invocation: ManagedInvocation | None, claimed: Any) -> None:
    """Re-raise the active exception for a managed child that never bound (stream path)."""
    if invocation is not None and not claimed:
        raise  # noqa: PLE0704 - called only from an ``except`` block


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


async def enter(agent: Agent, invocation: ManagedInvocation | None, store: Any, loop: Any) -> bool:
    """True for a managed root run; refuse one that cannot run durably."""
    if invocation is not None or not is_managed_root(agent):
        return False
    require_root_entry(agent, store, loop)
    await require_checkpointed_creation(store)
    return True


def stamp_root_header(store: Any, run_id: str) -> None:
    """Put the root's managed header into the claimed checkpoint before any bind.

    The recorder copies this state, so the header is written with the root's
    first fenced checkpoint and kept by every later one.
    """
    state = getattr(store, "loop_state", None)
    if not isinstance(state, dict):
        raise ManagedStateError(reason="storage_format", run_id=run_id)
    state[MANAGED_HEADER_KEY] = ManagedHeader(root_run_id=run_id, path=()).to_dict()


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
    # Every bind, live or recovered, admits only a ReAct child (before any effect).
    _require_react(handoff.parent, handoff.child)
    control = await store.get_run_control(run_id)
    if control.cancel_requested:
        return None
    header = ManagedHeader.from_dict(
        (getattr(store, "loop_state", None) or {}).get(MANAGED_HEADER_KEY), run_id=run_id
    )
    ctx = get_delegation_context()
    if ctx is None or ctx.run_id != run_id:
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
        child=handoff.child,
        child_path=(*header.path, handoff.child.name),
        wait=wait,
        iteration=result.iteration_count,
        parent_usage=result.usage,
        max_depth=max_depth,
        child_run_id=generate_ulid(),
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
    from dendrux.runtime.state import BindManagedChildRequest, LoopStatePatch, _initial_loop_state
    from dendrux.types import ManagedWait

    call = invocation.call
    wait = replace(invocation.wait, managed_wait=ManagedWait(call.id, run_id))
    request = BindManagedChildRequest(
        tenant_id=invocation.tenant_id,
        root_run_id=invocation.root_run_id,
        parent_run_id=invocation.parent_run_id,
        parent_claim_epoch=invocation.parent_epoch,
        tool_call_id=call.id,
        row_id=generate_ulid(),
        provider_tool_call_id=call.provider_tool_call_id,
        tool_name=call.name,
        params=call.params or None,
        iteration_index=invocation.iteration,
        child_run_id=run_id,
        child_agent_name=agent_name,
        child_path=invocation.child_path,
        child_input_data=create.get("input_data"),
        child_loop_state=loop_state or _initial_loop_state("ReActLoop") or {},
        parent_wait=wait,
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
            return await root_view(store, root_run_id)
        if is_descend(result):
            invocation: ManagedInvocation = result.meta[DESCEND_KEY]
            result, bound = await _run_child(invocation, notifier)
            if bound:
                waiting.append(_waiting_from(invocation))
            continue
        if not waiting:
            return result
        if result.status not in TERMINAL_STATUSES:
            return await _projected_pause(store, root_run_id, waiting, result)
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
    invocation: ManagedInvocation, exc: Exception
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
        await _fail_parent(invocation, exc)
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


async def root_view(store: Any, root_run_id: str) -> RunResult:
    """The root's persisted, root-facing state, for a caller that does not own the next step.

    A root waiting on a subagent reports its active branch: the leaf's external
    pause, or RUNNING while a subagent runs or awaits adoption.
    """
    from dendrux.runtime.runner import _build_cached_result

    return await _build_cached_result(store, root_run_id)


def _root_result(record: RunRecord, *, status: RunStatus, **fields: Any) -> RunResult:
    return RunResult(
        run_id=record.id,
        status=status,
        iteration_count=record.iteration_count,
        usage=UsageStats(
            input_tokens=record.total_input_tokens,
            output_tokens=record.total_output_tokens,
            total_tokens=record.total_input_tokens + record.total_output_tokens,
            cost_usd=record.total_cost_usd,
            reasoning_tokens=record.total_reasoning_tokens,
        ),
        **fields,
    )


async def _projected_pause(
    store: Any, root_run_id: str, waiting: list[_Waiting], leaf: RunResult
) -> RunResult:
    """A computed root view of a leaf's external pause; the parents keep waiting."""
    record = await store.get_run(root_run_id)
    if record is None:
        raise ManagedStateError(reason="child_missing", run_id=root_run_id)
    pending = leaf.pending
    if pending is not None:
        pending = replace(
            pending,
            run_id=root_run_id,
            agent_name=waiting[-1].child.name,
            agent_path=[record.agent_name, *waiting[-1].child_path],
            origin_run_id=leaf.run_id,
        )
    return _root_result(
        record,
        status=leaf.status,
        pause_id=pending.pause_id if pending is not None else None,
        pending=pending,
    )


# ---------------------------------------------------------------------------
# Adoption: one saved child outcome becomes one parent observation
# ---------------------------------------------------------------------------


def child_report(record: RunRecord, child: Agent, *, limit: int) -> ToolResult:
    """The parent's tool result for a terminal child; codes and counts only on failure."""
    status = record.status
    if status == RunStatus.CANCELLED.value:
        return _failed_report(child.name, CHILD_CANCELLED, record.id)
    if status != RunStatus.SUCCESS.value:
        return _failed_report(child.name, CHILD_FAILED, record.id)
    try:
        payload = _report_payload(record.answer, getattr(child, "_output_type", None))
    except (TypeError, ValueError):
        return _failed_report(child.name, REPORT_INVALID, record.id)
    size = len(payload.encode("utf-8"))
    if size > limit:
        _codes.log_code(
            logger, _codes.MANAGED_REPORT_TOO_LARGE, run_id=record.id, bytes=size, limit=limit
        )
        return _failed_report(child.name, REPORT_TOO_LARGE, record.id, bytes=size, limit=limit)
    return ToolResult(name=child.name, call_id="", payload=payload)


def _report_payload(answer: str | None, output_type: Any) -> str:
    if output_type is None:
        return json.dumps(answer or "", ensure_ascii=False)
    value = json.loads(answer or "")
    output_type.model_validate(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_FAILURE_TEXT = {
    CHILD_FAILED: "The subagent did not finish successfully.",
    CHILD_CANCELLED: "The subagent was cancelled.",
    REPORT_INVALID: "The subagent's output did not match its declared type.",
    REPORT_TOO_LARGE: "The subagent's report is larger than the tool result limit.",
}


_FAILURE_CODES = {
    CHILD_FAILED: _codes.MANAGED_CHILD_FAILED,
    CHILD_CANCELLED: _codes.MANAGED_CHILD_CANCELLED,
    REPORT_INVALID: _codes.MANAGED_REPORT_INVALID,
}


def _failed_report(name: str, code: str, child_run_id: str, **counts: int) -> ToolResult:
    message = _FAILURE_TEXT[code]
    if code in _FAILURE_CODES:
        _codes.log_code(logger, _FAILURE_CODES[code], run_id=child_run_id)
    body = {"error": message, "code": code, "child_run_id": child_run_id, **counts}
    return ToolResult(
        name=name,
        call_id="",
        payload=json.dumps(body),
        success=False,
        error=message,
        meta={"code": code},
    )


@dataclass(frozen=True)
class AcceptedReport:
    """The parent-approved report: transformed, size-checked and projected (in memory)."""

    result: ToolResult
    observation: Any
    pii_mapping: dict[str, str] | None
    blocked: str | None


async def accept_report(waiting: _Waiting, record: RunRecord, store: Any) -> AcceptedReport:
    """Admit a child's saved outcome as the parent admits any tool result.

    The parent's ``tool_result_transform`` runs once with the original call;
    the size refusal applies to the transformed serialized payload; the
    parent's guardrails project it (PII placeholders) or block it. Only this
    representation becomes durable; recovery never repeats the transform.
    """
    parent = waiting.parent_agent
    limit = parent._limits.max_tool_result_bytes
    report = replace(child_report(record, waiting.child, limit=limit), call_id=waiting.call.id)
    report = await _transformed(parent, waiting.call, report, limit, record.id)
    return await project_report(parent, store, waiting.parent_run_id, report)


async def _transformed(
    parent: Agent, call: ToolCall, report: ToolResult, limit: int, child_run_id: str
) -> ToolResult:
    from dendrux.loops.react import _apply_result_transform

    transform = parent.tool_result_transform
    if transform is None or not report.success:
        return report
    try:
        lookups = await parent.get_tool_lookups()
        value = await _apply_result_transform(transform, call, lookups, json.loads(report.payload))
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except Exception:
        return replace(_failed_report(report.name, REPORT_INVALID, child_run_id), call_id=call.id)
    size = len(payload.encode("utf-8"))
    if size > limit:
        _codes.log_code(
            logger, _codes.MANAGED_REPORT_TOO_LARGE, run_id=child_run_id, bytes=size, limit=limit
        )
        failed = _failed_report(
            report.name, REPORT_TOO_LARGE, child_run_id, bytes=size, limit=limit
        )
        return replace(failed, call_id=call.id)
    return replace(report, payload=payload)


async def project_report(
    parent: Agent, store: Any, run_id: str, report: ToolResult
) -> AcceptedReport:
    """The parent's guardrail projection of ``report`` (also used by recovery)."""
    from dendrux.loops._helpers import _project_tool_message
    from dendrux.types import tool_result_message

    message = tool_result_message(report)
    if not parent.guardrails:
        return AcceptedReport(report, message, None, None)
    from dendrux.guardrails import GuardrailEngine

    engine = GuardrailEngine(
        parent.guardrails, pii_mapping=await store.get_pii_mapping(run_id), run_id=run_id
    )
    projected, _findings, blocked = await _project_tool_message(engine, report, message)
    if blocked is not None:
        report = replace(report, payload=projected.content, error=None, meta={}, notes=())
    return AcceptedReport(report, projected, engine.get_pii_mapping(), blocked)


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

    record = await store.get_run(waiting.child_run_id)
    if record is None:
        raise ManagedStateError(reason="child_missing", run_id=waiting.parent_run_id)
    if record.status not in {s.value for s in TERMINAL_STATUSES}:
        return None
    accepted = await accept_report(waiting, record, store)
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

    from dendrux.runtime import persistence
    from dendrux.runtime._blocked_content import _blocked_evidence

    scope = _blocked_evidence(owned, run_id) if accepted.blocked is not None else nullcontext()
    async with scope:
        return await persistence.adopt_managed_result(
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


@dataclass(frozen=True)
class Route:
    """A root-id control resolved to the workflow's active leaf (in memory).

    ``tenant_id`` is the root's actual tenant; every hop carries it. ``waiting``
    lists the parents from the root down to the leaf's parent; the leaf is the
    root itself when nothing is waiting.
    """

    root_run_id: str
    tenant_id: str | None
    waiting: tuple[_Waiting, ...]
    leaf: RunRecord
    leaf_agent: Agent

    @property
    def routed(self) -> bool:
        """The leaf is a subagent, not the root itself."""
        return bool(self.waiting)


async def route(root_agent: Agent, store: Any, run_id: str, *, tenant_id: Any) -> Route:
    """Resolve a control on ``run_id`` to its active leaf; tenant and root checked first.

    A missing or foreign run is not found before anything is disclosed; a
    managed child id is refused (``managed_root_required``). The leaf's Agent
    is resolved from the registered path in ``root_agent``; a missing path
    refuses before any claim.
    """
    from dendrux.errors import ManagedChildControlError, RunNotFoundError
    from dendrux.runtime.state import UNSCOPED

    record = await store.get_run(run_id)
    if record is None or (tenant_id is not UNSCOPED and record.tenant_id != tenant_id):
        raise RunNotFoundError(run_id)
    if await managed_child_header(store, record):
        raise ManagedChildControlError(run_id)
    waiting, leaf = await walk(root_agent, store, run_id)
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


@dataclass
class _StreamState:
    """What the workflow stream's cleanup needs (mutable, one stream)."""

    root_agent: Agent
    store: Any = None
    root_run_id: str | None = None
    tenant_id: Any = None
    current: Any = None
    waiting: list[_Waiting] | None = None
    root_active: bool = True
    """The active leg is the root's own: closing it cancels the root by itself."""
    unstarted: tuple[str, Any] | None = None
    """``(run_id, claim)`` of a claimed continuation whose stream has not started yet."""


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
        return _structural(event, event.run_id) if name in managed_names else event
    if kind not in _TOOL_EVENTS:
        return None
    return _structural(event, run_id)


def _structural(event: Any, run_id: str | None) -> Any:
    """A tool event with ids, name, iteration, phase and outcome code only."""
    from dendrux.types import RunEvent, ToolCall, ToolResult

    call = event.tool_call
    result = event.tool_result
    code = (result.meta or {}).get("code") if result is not None else None
    return RunEvent(
        type=event.type,
        run_id=run_id,
        tool_name=event.tool_name or (call.name if call is not None else None),
        tool_call_id=event.tool_call_id or (call.id if call is not None else None),
        iteration=event.iteration,
        phase=event.phase,
        tool_call=None
        if call is None
        else ToolCall(name=call.name, params={}, id=call.id, provider_tool_call_id=None),
        tool_result=None
        if result is None
        else ToolResult(
            name=result.name,
            call_id=result.call_id,
            payload="",
            success=result.success,
            duration_ms=result.duration_ms,
            meta={"code": code} if isinstance(code, str) else {},
        ),
    )


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


def _continuation_stream(waiting: _Waiting, store: Any, claim: Any, notifier: Any) -> Any:
    """The adopting parent's continuation, streamed from its checkpoint under ``claim``."""
    from dendrux.runtime.runner import resume_stream

    parent = waiting.parent_agent
    return resume_stream(
        waiting.parent_run_id,
        agent=parent,
        provider=parent._require_provider(),
        state_store=store,
        extra_notifier=notifier,
        _claimed=claim,
    )


def _event(result: RunResult) -> Any:
    """The workflow stream's terminal event for a root-facing result."""
    from dendrux.types import RunEvent, RunEventType

    kind = {
        RunStatus.SUCCESS: RunEventType.RUN_COMPLETED,
        RunStatus.MAX_ITERATIONS: RunEventType.RUN_COMPLETED,
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
    }.get(result.status, RunEventType.RUN_PAUSED)
    return RunEvent(type=kind, run_result=result, run_id=result.run_id, error=result.error)


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


@dataclass(frozen=True)
class _Terminal:
    event: Any


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
        return _event(await root_view(state.store, str(state.root_run_id)))
    if is_descend(result):
        invocation = result.meta[DESCEND_KEY]
        state.tenant_id = invocation.tenant_id
        return _Leg(_child_stream(invocation, notifier), invocation)
    if not waiting:
        return leg.terminal if leg.terminal is not None else _event(result)
    if result.status not in TERMINAL_STATUSES:
        return _event(await _paused_or_failed(state, waiting, result, leg.terminal))
    claim = await claim_adoption(waiting[-1], state.store)
    if claim is None:
        return _event(await root_view(state.store, str(state.root_run_id)))
    parent = waiting.pop()
    state.unstarted = (parent.parent_run_id, claim)
    return _Leg(_continuation_stream(parent, state.store, claim, notifier))


async def _paused_or_failed(
    state: _StreamState, waiting: list[_Waiting], result: RunResult, terminal: Any
) -> RunResult:
    """A leaf that stopped short of terminal: its pause through the root, or a failed attempt.

    A continuation attempt that failed and released its claim reports the
    root-facing state with the attempt's error (the leaf keeps waiting).
    """
    root_run_id = str(state.root_run_id)
    if getattr(getattr(terminal, "type", None), "value", None) == "run_error":
        view = await root_view(state.store, root_run_id)
        view.error = result.error
        view.error_info = result.error_info
        return view
    return await _projected_pause(state.store, root_run_id, waiting, result)


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
            yield await _stream_failure(state, exc, stream.run_id)

    async def _cleanup() -> RunResult | None:
        return await _close_workflow(state)

    stream = _RunStream(run_id=run_id, generator=_generate(), cleanup=_cleanup)
    return stream


async def _stream_failure(state: _StreamState, exc: Exception, run_id: str) -> Any:
    """A failure while driving ends the workflow stream with a root-facing RUN_ERROR."""
    from dendrux.runtime.runner import _safe_error
    from dendrux.types import RunEvent, RunEventType

    error = _safe_error(exc)
    if state.root_run_id is None or state.store is None:
        result = RunResult(run_id=run_id, status=RunStatus.ERROR, error=error)
    else:
        result = await root_view(state.store, state.root_run_id)
        result.error = error
    result.meta["error_type"] = type(exc).__name__
    state.root_active = True  # nothing to cancel: the failure is reported, not abandoned
    return RunEvent(
        type=RunEventType.RUN_ERROR, run_result=result, run_id=result.run_id, error=error
    )


async def _close_workflow(state: _StreamState) -> RunResult | None:
    """Abandoned stream: release a claimed but unstarted continuation, close the active
    run's stream, then cancel a waiting root with its descendants."""
    result = None
    if state.unstarted is not None:
        await _cancel_unstarted(state)
    if state.current is not None:
        await state.current.aclose()
        result = state.current.result
    if not state.root_active and state.root_run_id is not None:
        await state.root_agent.cancel_run(
            state.root_run_id, cascade=True, tenant_id=state.tenant_id
        )
        result = await root_view(state.store, state.root_run_id)
    return result


async def _cancel_unstarted(state: _StreamState) -> None:
    """Finalize a continuation this stream claimed but never started, under its epoch.

    Its own stream never ran, so its ordinary close-time cleanup has nothing to
    finalize; without this the claimed run would stay RUNNING with no owner.
    """
    from dendrux._lifecycle import CancelReason
    from dendrux.runtime.runner import _persist_run_cancellation
    from dendrux.runtime.state import claimed_store

    assert state.unstarted is not None
    run_id, claim = state.unstarted
    state.unstarted = None
    owned = claimed_store(state.store, run_id, claim)
    await _persist_run_cancellation(
        owned,
        run_id,
        {"reason": CancelReason.STREAM_CLOSED.value, "partial_output": False},
        on_won=lambda: None,
        emit_event=True,
    )


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
                return _Single(_event(await root_view(store, run_id))), None
            parent = waiting.pop()
            first = _continuation_stream(parent, store, claim, notifier)
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
        view = await root_view(state.store, found.root_run_id)
        error = getattr(head, "error", None) if head is not None else None
        view.error = error
        return _Single(_event_with_error(view, error)), None
    started = _Started(first, events, head)
    state.waiting = list(found.waiting)
    state.current, state.root_active = started, False
    return started, resumed


def _event_with_error(result: RunResult, error: str | None) -> Any:
    from dendrux.types import RunEvent, RunEventType

    return RunEvent(
        type=RunEventType.RUN_ERROR, run_result=result, run_id=result.run_id, error=error
    )


class _Started:
    """A run's stream whose first event was already taken in this task (its claim made)."""

    def __init__(self, stream: Any, events: Any, head: Any) -> None:
        self._stream = stream
        self._events = events
        self._head = head
        self.run_id = stream.run_id

    @property
    def result(self) -> Any:
        return self._stream.result

    async def __aiter__(self) -> Any:
        yield self._head
        async for event in self._events:
            yield event

    async def aclose(self) -> None:
        await self._events.aclose()  # runs the run's own close-time cleanup


_TERMINAL_VALUES = frozenset(s.value for s in TERMINAL_STATUSES)


class _Single:
    """A one-event stream: the root-facing state when nothing can be continued."""

    def __init__(self, event: Any) -> None:
        self._event = event
        self.run_id = event.run_id
        self.result = event.run_result

    async def __aiter__(self) -> Any:
        yield self._event

    async def aclose(self) -> None:
        return None


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
    """Continue a managed workflow from its root id after a restart (narrow, T2b).

    Walks the persisted bindings root to leaf. A terminal child is adopted into
    its waiting parent without running it again; a running leaf is reclaimed
    under ordinary lease rules (a live owner is never stolen from). A leaf's
    external pause is returned as a root view. Full crash and sweep coverage is T4.
    """
    from dendrux.runtime import _reclaim
    from dendrux.runtime.runner import _build_cached_result

    waiting, leaf = await walk(root_agent, store, root_run_id)
    if leaf.status in {s.value for s in TERMINAL_STATUSES}:
        if not waiting:
            return await _build_cached_result(store, root_run_id)
        result = await adopt(waiting.pop(), store=store, notifier=notifier)
    elif leaf.status == RunStatus.RUNNING.value:
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


async def walk(root_agent: Agent, store: Any, root_run_id: str) -> tuple[list[_Waiting], RunRecord]:
    """Follow each v3 wait's exact child id, validating every hop; bounded."""
    record = await store.get_run(root_run_id)
    if record is None:
        from dendrux.errors import RunNotFoundError

        raise RunNotFoundError(root_run_id)
    waiting: list[_Waiting] = []
    agent: Agent = root_agent
    path: tuple[str, ...] = ()
    for _ in range(_MAX_WALK):
        wait = await _outgoing(store, record)
        if wait is None or wait.managed_wait is None:
            return waiting, record
        binding = wait.managed_wait
        child, path = await _bound_child(store, record, binding, root_run_id, path)
        child_agent = _registered(agent, path[-1], record.id)
        intent = await _intent_call(store, record.id, binding.tool_call_id, wait.iteration)
        waiting.append(_Waiting(agent, record.id, child_agent, path, intent, child.id))
        agent, record = child_agent, child
    raise ManagedStateError(reason="binding_conflict", run_id=root_run_id)


async def _outgoing(store: Any, record: RunRecord) -> PauseState | None:
    """The v3 managed wait of a parked parent, else None."""
    if record.status != RunStatus.PAUSED.value:
        return None
    raw = await store.get_pause_state(record.id)
    if not raw or raw.get("schema_version") != 3:
        return None
    return PauseState.from_managed_dict(raw, run_id=record.id)


async def _bound_child(
    store: Any, parent: RunRecord, binding: Any, root_run_id: str, path: tuple[str, ...]
) -> tuple[RunRecord, tuple[str, ...]]:
    child = await store.get_run(binding.child_run_id)
    if child is None:
        raise ManagedStateError(reason="child_missing", run_id=parent.id)
    if child.parent_run_id != parent.id or child.tenant_id != parent.tenant_id:
        raise ManagedStateError(reason="child_mismatch", run_id=parent.id)
    header = ManagedHeader.from_dict(
        (await store.get_loop_state(child.id) or {}).get(MANAGED_HEADER_KEY), run_id=child.id
    )
    if (
        header.root_run_id != root_run_id
        or header.parent_run_id != parent.id
        or header.tool_call_id != binding.tool_call_id
        or header.path[:-1] != path
    ):
        raise ManagedStateError(reason="child_mismatch", run_id=parent.id)
    return child, header.path


async def _intent_call(store: Any, parent_run_id: str, call_id: str, iteration: int) -> ToolCall:
    """The bound call as the parent's model made it (its brief lives on the intent row).

    Read only the wait's own iteration: its rows are one model batch, bounded by
    ``Limits.max_pending_calls``, so the paged read ends; earlier history is never read.
    """
    from dendrux.runtime._bounds import MAX_PAGE

    offset = 0
    while True:
        page = await store.get_tool_calls(
            parent_run_id, iteration_index=iteration, limit=MAX_PAGE, offset=offset
        )
        for row in page:
            if row.tool_call_id == call_id and row.target == "agent":
                return ToolCall(
                    name=row.tool_name,
                    params=dict(row.params or {}),
                    id=row.tool_call_id,
                    provider_tool_call_id=row.provider_tool_call_id,
                )
        if len(page) < MAX_PAGE:
            raise ManagedStateError(reason="binding_missing", run_id=parent_run_id)
        offset += MAX_PAGE


def _registered(parent: Agent, name: str, parent_run_id: str) -> Agent:
    """The parent's registered child by name; a missing path refuses before any claim."""
    for child in parent.subagents:
        if child.name == name:
            return child
    raise ManagedStateError(reason="path_missing", run_id=parent_run_id)


async def managed_child_header(store: Any, record: RunRecord) -> bool:
    """True when ``record`` is a managed child (its header names an invocation)."""
    if getattr(record, "parent_run_id", None) is None:
        return False
    header = (await store.get_loop_state(record.id) or {}).get(MANAGED_HEADER_KEY)
    return isinstance(header, dict) and "invocation" in header
