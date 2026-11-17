"""Managed-subagent adoption and recovery helpers for explicit reclaim (0.7).

Moved out of ``dendrux.runtime._reclaim``, which imports every name back so the
reclaim flow and its patch seams are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux.runtime.state import MANAGED_BINDING_KEY
from dendrux.types import RunResult, RunStatus, ToolCall

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime.runner import _ResumeContext
    from dendrux.runtime.state import ClaimResult, StateStore


def _pending_adoption(state: dict[str, Any] | None) -> bool:
    """A running parent whose checkpoint still holds a claimed managed adoption."""
    state = state or {}
    return state.get(MANAGED_BINDING_KEY) is not None and bool(state.get("submission"))


async def _finish_saved_adoption(
    agent: Agent, owned: StateStore, run_id: str, claim: ClaimResult
) -> ClaimResult:
    """Complete the managed tool intent from the durably accepted report, once."""
    from dendrux.errors import ManagedStateError
    from dendrux.runtime._managed import complete_adoption, project_report
    from dendrux.runtime.state import ClaimResult, ManagedHeader
    from dendrux.runtime.submit import _submission_payload, deserialize_tool_results

    state = claim.loop_state or {}
    binding = state[MANAGED_BINDING_KEY]
    raw = await owned.get_pause_state(run_id) or {}
    try:
        payload = _submission_payload(raw)
        results = deserialize_tool_results(payload["submitted_tool_results"])
    except (KeyError, TypeError, ValueError):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id) from None
    if len(results) != 1 or results[0].call_id != binding.get("tool_call_id"):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    header = ManagedHeader.from_dict(state.get("managed"), run_id=run_id)
    tool = next(
        (p.get("tool") for p in state.get("pending", []) if p.get("call_id") == results[0].call_id),
        results[0].name,
    )
    # The saved report is already transformed and admitted; only its projection
    # into the conversation is redone, from the saved mapping.
    accepted = await project_report(agent, owned, run_id, results[0])
    if payload.get("managed_blocked"):
        accepted = replace(accepted, blocked=accepted.blocked or "blocked")
    after = await complete_adoption(
        owned,
        run_id,
        claim,
        accepted,
        child_run_id=binding.get("child_run_id"),
        path=(*header.path, str(tool)),
    )
    return ClaimResult(claim.claim_epoch, claim.storage_format, after, None)


def _refuse_recovered_managed_results(
    run_id: str,
    descriptor: dict[str, Any],
    raw: dict[str, Any],
    state: dict[str, Any],
    payload: dict[str, Any],
    lookups: Any,
) -> None:
    """A saved caller result never answers a generated subagent call (0.7).

    Earlier unreleased builds could pause a subagent call for a client and claim
    a submitted result for it. Recovery checks the saved results against the
    durable pending targets (pause and checkpoint) and the generated calls before
    projecting or consuming anything. Approval submissions are runtime denials
    and are not checked here.
    """
    from dendrux.loops.react import SubagentTool
    from dendrux.runtime.submit import deserialize_tool_results, refuse_managed_results

    if descriptor["kind"] != "tool_results":
        return
    results = deserialize_tool_results(payload.get("submitted_tool_results") or [])
    targets = {
        **(raw.get("pending_targets") or {}),
        **{p["call_id"]: p.get("target") for p in state.get("pending", [])},
        **{r.call_id: "agent" for r in results if isinstance(lookups.fn.get(r.name), SubagentTool)},
    }
    refuse_managed_results(run_id, targets, results)


@dataclass(frozen=True)
class _ManagedPending:
    """The one generated subagent call pending in a supported recovered batch (0.7)."""

    call: ToolCall
    admitted: bool


def _refuse_bound_pending(found: _ManagedPending | None, run_id: str, rows: dict[str, Any]) -> None:
    """A pending managed call that already has a tool row is invalid state (0.7).

    A bound call's parent is parked, never reclaimed with it pending.
    """
    from dendrux.errors import ManagedStateError

    if found is not None and found.call.id in rows:
        raise ManagedStateError(reason="binding_outstanding", run_id=run_id)


def _managed_handoff(
    agent: Agent, ctx: _ResumeContext, run_id: str, call: ToolCall, lookups: Any
) -> RunResult:
    """Hand the admitted managed call to the durable bind, after the server calls."""
    from dendrux.loops.react import MANAGED_HANDOFF_KEY, ManagedHandoff

    pause = replace(
        ctx.pause_state,
        history=ctx.history,
        pending_tool_calls=[call],
        pending_targets={call.id: "agent"},
        approval_gates=None,
        pending_decisions=None,
    )
    return RunResult(
        run_id=run_id,
        status=RunStatus.RUNNING,
        steps=pause.steps,
        iteration_count=pause.iteration,
        usage=pause.usage,
        meta={
            "pause_state": pause,
            MANAGED_HANDOFF_KEY: ManagedHandoff(
                call=call, child=lookups.fn[call.name].agent, parent=agent
            ),
        },
    )


def _handed_off(found: _ManagedPending | None, held: list[ToolCall]) -> bool:
    """Only the admitted managed call is left: it goes to the bind (0.7)."""
    return found is not None and [tc.id for tc in held] == [found.call.id]
