"""Atomic persistence of accepted managed subagent results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import StoredEventType
from dendrux.errors import ManagedStateError
from dendrux.runtime import _managed_budget, _managed_cancel, _trace_shared
from dendrux.runtime.durability import retry_critical
from dendrux.runtime.state import EventWrite, LoopStatePatch, _apply_loop_patch
from dendrux.types import generate_ulid

# isort: split
from dendrux.types import tool_result_message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.runtime.state import StateStore
    from dendrux.types import Message, ToolResult


@_managed_budget.propagate_fault
async def adopt_managed_result(
    store: StateStore,
    run_id: str,
    *,
    claim: Any,
    call_id: str,
    report: ToolResult,
    child_run_id: str,
    agent_path: Sequence[str],
    observation: Message | None = None,
    pii_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Complete a parent's managed tool intent with its accepted report (0.7).

    One ``complete_tool_call`` transaction under the adoption claim's epoch:
    started intent to completed or failed (its binding metadata kept), the
    reserved observation, the outcome event, the call removed from pending and
    the outgoing binding and submission marker cleared. Ordinary
    ``consume_submission`` is never also called for this observation. A retry
    after a lost acknowledgement finds the completed intent and adds nothing.
    Returns the checkpoint after the change.
    """
    state = dict(claim.loop_state or {})
    binding = state.get("managed_wait") or {}
    if binding.get("tool_call_id") != call_id or binding.get("child_run_id") != child_run_id:
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    return await _complete(
        store,
        run_id,
        _Completion(claim.claim_epoch, state, call_id, report, child_run_id, agent_path),
        _managed_budget.adoption_patch(state, call_id),
        observation=observation,
        pii_mapping=pii_mapping,
    )


async def adopt_fanout_result(
    store: StateStore,
    run_id: str,
    *,
    claim_epoch: int,
    state: dict[str, Any],
    call_id: str,
    report: ToolResult,
    child_run_id: str,
    agent_path: Sequence[str],
    provenance: tuple[Any, int],
    observation: Message | None = None,
    pii_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Complete one call of a claimed fan-out adoption with its accepted report (0.8).

    ``state`` is the parent checkpoint after the previous completion (the claim's
    checkpoint for the first); it is not mutated. ``provenance`` is the child's
    ``(fault, accepted unpriced count)``: counts only rise, a fault sticks. One
    ``complete_tool_call`` transaction under ``claim_epoch``: the started intent
    resolved, the reserved observation, the event, the call removed from pending.
    The completion of the last pending bound call also clears the wait, the
    admitted calls and the adoption marker; earlier ones keep them so recovery
    still recognizes the adoption. Returns the checkpoint after the change.

    Raises:
        ManagedStateError: ``binding_conflict`` when the checkpoint's fan-out
            wait does not bind ``call_id`` to ``child_run_id``; ``pending_missing``
            when the call is no longer pending.
    """
    bindings = (state.get("managed_wait") or {}).get("bindings")
    if not isinstance(bindings, dict) or bindings.get(call_id) != child_run_id:
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    fault, accepted = provenance
    carried = _managed_budget.carry_fault(state, fault, accepted, child_run_id)
    remaining = [
        p.get("call_id")
        for p in carried.get("pending", [])
        if p.get("call_id") in bindings and p.get("call_id") != call_id
    ]
    patch = (
        _managed_budget.progress_patch(carried, call_id)
        if remaining
        else _managed_budget.adoption_patch(carried, call_id)
    )
    return await _complete(
        store,
        run_id,
        _Completion(claim_epoch, carried, call_id, report, child_run_id, agent_path),
        patch,
        observation=observation,
        pii_mapping=pii_mapping,
    )


@dataclass(frozen=True)
class _Completion:
    """One managed call's accepted report, under the adopting claim's epoch and checkpoint."""

    claim_epoch: int
    state: dict[str, Any]
    call_id: str
    report: ToolResult
    child_run_id: str
    agent_path: Sequence[str]


async def _complete(
    store: StateStore,
    run_id: str,
    done: _Completion,
    patch: LoopStatePatch,
    *,
    observation: Message | None,
    pii_mapping: dict[str, str] | None,
) -> dict[str, Any]:
    """The one ``complete_tool_call`` of a managed adoption; the checkpoint after it."""
    state, call_id, report = done.state, done.call_id, done.report
    pending = next((p for p in state.get("pending", []) if p.get("call_id") == call_id), None)
    if pending is None:
        raise ManagedStateError(reason="pending_missing", run_id=run_id)
    iteration = int(state.get("iteration", 0))
    trace = _trace_shared.message_trace(
        observation or tool_result_message(report), pending["order_index"]
    )
    trace.meta["iteration"] = iteration
    event = EventWrite(
        generate_ulid(),
        StoredEventType.TOOL_COMPLETED.value,
        iteration,
        {
            "tool_name": report.name,
            "target": "agent",
            "success": report.success,
            "duration_ms": report.duration_ms,
            "child_run_id": done.child_run_id,
            "agent_path": list(done.agent_path),
            "managed_outcome": _managed_cancel.outcome_code(report),
            **_managed_budget.adoption_event(report, state),
        },
        call_id,
        None,
    )

    async def complete() -> None:
        await store.complete_tool_call(
            run_id,
            claim_epoch=done.claim_epoch,
            tool_call_id=call_id,
            outcome="completed" if report.success else "failed",
            result_payload=report.payload,
            success=report.success,
            duration_ms=report.duration_ms,
            error_message=report.error,
            replayed=False,
            observation=trace,
            event=event,
            loop_state_patch=patch,
            meta=None,
            pii_mapping=pii_mapping,
        )

    await retry_critical(complete, label="complete_tool_call", run_id=run_id)
    return _apply_loop_patch(state, patch)
