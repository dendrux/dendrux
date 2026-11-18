"""Atomic persistence of accepted managed subagent results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import StoredEventType
from dendrux.errors import ManagedStateError
from dendrux.runtime import _managed_budget, _managed_cancel, _trace_shared
from dendrux.runtime.durability import retry_critical
from dendrux.runtime.state import EventWrite, _apply_loop_patch
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
    pending = next((p for p in state.get("pending", []) if p.get("call_id") == call_id), None)
    if binding.get("tool_call_id") != call_id or binding.get("child_run_id") != child_run_id:
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    if pending is None:
        raise ManagedStateError(reason="pending_missing", run_id=run_id)
    iteration = int(state.get("iteration", 0))
    trace = _trace_shared.message_trace(
        observation or tool_result_message(report), pending["order_index"]
    )
    trace.meta["iteration"] = iteration
    outcome = "completed" if report.success else "failed"
    event = EventWrite(
        generate_ulid(),
        StoredEventType.TOOL_COMPLETED.value,
        iteration,
        {
            "tool_name": report.name,
            "target": "agent",
            "success": report.success,
            "duration_ms": report.duration_ms,
            "child_run_id": child_run_id,
            "agent_path": list(agent_path),
            "managed_outcome": _managed_cancel.outcome_code(report),
            **_managed_budget.adoption_event(report, state),
        },
        call_id,
        None,
    )
    patch = _managed_budget.adoption_patch(state, call_id)

    async def complete() -> None:
        await store.complete_tool_call(
            run_id,
            claim_epoch=claim.claim_epoch,
            tool_call_id=call_id,
            outcome=outcome,
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
