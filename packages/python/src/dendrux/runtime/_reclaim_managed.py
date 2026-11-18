"""Managed-subagent adoption and recovery helpers for explicit reclaim (0.7).

Moved out of ``dendrux.runtime._reclaim``, which imports every name back so the
reclaim flow and its patch seams are unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux.errors import ManagedStateError, RunNotFoundError, RunNotReclaimableError
from dendrux.loops import _managed_batch
from dendrux.runtime._fanout_adoption_state import (
    adoption_payload,
    fanout_adoption,
    resolved_prefix,
)
from dendrux.runtime._managed_adoption import (
    _check_bound_rows,
    _intent_rows,
    _iteration_rows,
    finish_fanout_adoption,
)
from dendrux.runtime._managed_claim import scoped_run
from dendrux.runtime.state import MANAGED_BINDING_KEY, UNSCOPED
from dendrux.types import PauseState, RunResult, RunStatus, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.agent import Agent
    from dendrux.runtime.runner import _ResumeContext
    from dendrux.runtime.state import ClaimResult, StateStore, TenantScope
    from dendrux.types import Message


def _pending_adoption(state: dict[str, Any] | None, *, run_id: str | None = None) -> bool:
    """A running parent whose checkpoint still holds a claimed managed adoption."""
    state = state or {}
    if fanout_adoption(state, run_id=run_id) is not None:
        return True
    return state.get(MANAGED_BINDING_KEY) is not None and bool(state.get("submission"))


async def _finish_saved_adoption(
    agent: Agent,
    owned: StateStore,
    run_id: str,
    claim: ClaimResult,
    *,
    tenant_id: TenantScope = UNSCOPED,
) -> ClaimResult:
    """Complete the managed tool intent from the durably accepted report, once."""
    state = claim.loop_state or {}
    if fanout_adoption(state, run_id=run_id) is not None:
        record = await scoped_run(owned, run_id, tenant_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return await finish_fanout_adoption(agent, owned, run_id, claim, tenant_id=record.tenant_id)
    return await _finish_legacy_adoption(agent, owned, run_id, claim)


async def _finish_legacy_adoption(
    agent: Agent, owned: StateStore, run_id: str, claim: ClaimResult
) -> ClaimResult:
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
    """The ordered managed calls captured before recovery consumes a submission."""

    call: ToolCall
    admitted: bool
    calls: tuple[ToolCall, ...] = ()

    @property
    def ordered(self) -> tuple[ToolCall, ...]:
        return self.calls or (self.call,)

    @property
    def call_ids(self) -> tuple[str, ...]:
        return tuple(call.id for call in self.ordered)

    @classmethod
    def from_calls(
        cls, calls: Sequence[ToolCall], *, admitted: object, run_id: str
    ) -> _ManagedPending:
        if not calls:
            raise ManagedStateError(reason="binding_conflict", run_id=run_id)
        return cls(
            calls[0],
            admitted=admitted == [call.id for call in calls]
            or (len(calls) == 1 and admitted == calls[0].id),
            calls=tuple(calls),
        )


_ManagedPending.__module__ = "dendrux.runtime._reclaim"


def _refuse_bound_pending(found: _ManagedPending | None, run_id: str, rows: dict[str, Any]) -> None:
    """A pending managed call that already has a tool row is invalid state (0.7).

    A bound call's parent is parked, never reclaimed with it pending.
    """

    if found is not None and any(call.id in rows for call in found.ordered):
        raise ManagedStateError(reason="binding_outstanding", run_id=run_id)


def _managed_handoff(
    agent: Agent, ctx: _ResumeContext, run_id: str, calls: list[ToolCall], lookups: Any
) -> RunResult:
    """Hand the admitted managed call to the durable bind, after the server calls."""
    from dendrux.loops.react import MANAGED_HANDOFF_KEY, ManagedHandoff

    pause = replace(
        ctx.pause_state,
        history=ctx.history,
        pending_tool_calls=calls,
        pending_targets={call.id: "agent" for call in calls},
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
                calls=calls, child=lookups.fn[calls[0].name].agent, parent=agent
            ),
        },
    )


def _handed_off(found: _ManagedPending | None, held: list[ToolCall]) -> bool:
    """A nonempty admitted subset retains the captured managed call order."""
    return (
        found is not None
        and bool(held)
        and _ordered_subset(found.call_ids, [call.id for call in held])
    )


def _ordered_subset(order: Sequence[str], subset: Sequence[str]) -> bool:
    remaining = iter(order)
    return all(any(candidate == call_id for candidate in remaining) for call_id in subset)


def _pending_managed_ids(entries: Any, run_id: str) -> set[str]:
    if not isinstance(entries, list):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    pending: set[str] = set()
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ManagedStateError(reason="binding_conflict", run_id=run_id)
        call_id, phase = entry.get("call_id"), entry.get("phase")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in seen
            or not isinstance(phase, str)
            or phase not in {"client", "gated", "dispatchable", "approved", "rejected", "invalid"}
        ):
            raise ManagedStateError(reason="binding_conflict", run_id=run_id)
        seen.add(call_id)
        if phase not in {"rejected", "invalid"}:
            pending.add(call_id)
    return pending


def _surviving_managed(
    found: _ManagedPending | None, state: dict[str, Any], run_id: str
) -> _ManagedPending | None:
    if found is None:
        return None
    pending = _pending_managed_ids(state.get("pending", []), run_id)
    calls = [call for call in found.ordered if call.id in pending]
    if not calls:
        return None
    admitted = state.get("managed_admitted")
    if isinstance(admitted, list):
        ids = [call.id for call in calls]
        if not _ordered_subset(found.call_ids, admitted) or not _ordered_subset(admitted, ids):
            raise ManagedStateError(reason="binding_conflict", run_id=run_id)
        admitted = ids
    return _ManagedPending.from_calls(calls, admitted=admitted, run_id=run_id)


async def _check_wait_format(
    store: StateStore, run_id: str, state: dict[str, Any], *, tenant_id: str | None
) -> None:
    raw = await store.get_pause_state(run_id) or {}
    binding = state.get(MANAGED_BINDING_KEY)
    version = raw.get("schema_version", 1)
    if version == 4:
        wait = PauseState.from_managed_dict(raw, run_id=run_id)
        if binding is not None:
            await _validate_saved_fanout(store, run_id, state, raw, wait, tenant_id)
        return
    legacy = (
        isinstance(binding, dict)
        and binding.get("v", 1) == 1
        and set(binding) - {"v"} == {"tool_call_id", "child_run_id", "pause_id"}
        and all(isinstance(v, str) and v for k, v in binding.items() if k != "v")
    )
    if (isinstance(version, int) and version > 4) or (binding is not None and not legacy):
        _codes.log_code(logging.getLogger(__name__), _codes.PAUSE_FORMAT_UNSUPPORTED, run_id=run_id)
        raise RunNotReclaimableError(run_id, "pause_format_unsupported")


async def _validate_saved_fanout(
    store: StateStore,
    run_id: str,
    state: dict[str, Any],
    raw: dict[str, Any],
    wait: PauseState,
    tenant_id: str | None,
) -> None:
    fanout = wait.managed_fanout
    if (
        fanout is None
        or wait.pause_id is None
        or state.get(MANAGED_BINDING_KEY) != fanout.to_checkpoint(pause_id=wait.pause_id)
    ):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    if not state.get("submission"):
        return
    ids = fanout_adoption(state, run_id=run_id)
    bindings = fanout.bindings
    ordered = tuple(binding.tool_call_id for binding in bindings)
    envelope = raw.get("submission")
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if ids != ordered or payload != adoption_payload(ordered):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    rows = await _iteration_rows(store, run_id, wait.iteration, tenant_id)
    _check_bound_rows(rows, set(ordered), run_id)
    done = resolved_prefix(
        _intent_rows(rows), [(b.tool_call_id, b.child_run_id) for b in bindings], run_id=run_id
    )
    pending = state.get("pending")
    if (
        done == len(ordered)
        or not isinstance(pending, list)
        or [item.get("call_id") if isinstance(item, dict) else None for item in pending]
        != list(ordered[done:])
    ):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)


def _refused_batch(
    history: list[Message], refused: set[str], pending: list[ToolCall]
) -> list[ToolCall]:
    return next(
        (
            message.tool_calls
            for message in reversed(history)
            if message.tool_calls and refused.issubset({call.id for call in message.tool_calls})
        ),
        pending,
    )


def _saved_refusal(
    history: list[Message],
    refused: set[str],
    decision: _managed_batch.BatchDecision,
    limit: int,
) -> tuple[_managed_batch.BatchDecision, int]:
    for message in history:
        if message.role.value != "tool" or message.call_id not in refused:
            continue
        try:
            body = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(body, dict) or "error" not in body:
            continue
        reason = body.get("reason")
        if reason in (None, "client_call", "exclusive"):
            return _managed_batch.BatchDecision("refuse", decision.count, reason), limit
        saved_limit, count = body.get("limit"), body.get("requested")
        if reason == "fanout_limit" and type(saved_limit) is int and type(count) is int:
            return _managed_batch.BatchDecision("refuse", count, reason), saved_limit
    return decision, limit


async def _admit_recovered_call(
    found: _ManagedPending, outcome: Any, lookups: Any, batch: Any
) -> list[ToolCall]:
    ready = [tc for tc in outcome.pending_calls if tc.id in found.call_ids]
    if not ready or outcome.approval_gates:
        return list(outcome.pending_calls)
    return await _managed_batch._admit_managed_briefs(ready, lookups, batch)
