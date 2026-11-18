"""Managed subagents: turning terminal children's saved outcomes into their parent's reports.

The parent admits a child's outcome as it admits any tool result: the report
is built from the saved terminal record (codes and counts only on failure),
passed through the parent's ``tool_result_transform``, size-checked and then
projected through the parent's guardrails. For one child (0.7) the claim and
completion stay in :mod:`dendrux.runtime._managed`; for a fan-out wait (0.8)
this module claims first and then completes each call in call order.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import TERMINAL_STATUSES
from dendrux._managed_wait import MANAGED_FANOUT_SCHEMA_VERSION
from dendrux.errors import ManagedStateError
from dendrux.runtime import _managed_budget, _managed_cancel, _managed_persistence
from dendrux.runtime._blocked_content import _blocked_evidence
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime._fanout_adoption_state import (
    RESOLVED_OUTCOMES,
    IntentRow,
    adoption_payload,
    fanout_adoption,
    resolved_prefix,
)
from dendrux.runtime._managed_claim import claim_fanout_adoption
from dendrux.runtime._managed_route import (
    _registered,
    _Waiting,
)
from dendrux.runtime.state import MANAGED_HEADER_KEY, ClaimResult, ManagedHeader, claimed_store
from dendrux.types import PauseState, RunStatus, ToolCall, ToolResult, ToolTarget

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime._managed_tree import WaitNode
    from dendrux.runtime.state import RunRecord, ToolCallReadRecord

logger = logging.getLogger("dendrux.runtime._managed")
_TERMINAL = frozenset(status.value for status in TERMINAL_STATUSES)

REPORT_TOO_LARGE = "managed_report_too_large"
CHILD_FAILED = "managed_child_failed"
CHILD_CANCELLED = "managed_child_cancelled"
REPORT_INVALID = "managed_report_invalid"


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
    return await _accept(
        waiting.parent_agent, waiting.call, waiting.child, record, store, waiting.parent_run_id
    )


async def _accept(
    parent: Agent, call: ToolCall, child: Agent, record: RunRecord, store: Any, run_id: str
) -> AcceptedReport:
    """Report, transform, size check and projection of one terminal child, in that order."""
    limit = parent._limits.max_tool_result_bytes
    report = replace(child_report(record, child, limit=limit), call_id=call.id)
    report = await _transformed(parent, call, report, limit, record.id)
    return await project_report(parent, store, run_id, report)


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


AcceptedReport.__module__ = "dendrux.runtime._managed"


# ---------------------------------------------------------------------------
# 0.8: adopting a fan-out wait's N terminal children (spec §7.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FanoutCall:
    """One call of a fan-out wait as adoption completes it: the call and its bound child."""

    call: ToolCall
    child: Agent
    child_path: tuple[str, ...]
    child_run_id: str


@dataclass(frozen=True)
class _FanoutParent:
    """The adopting parent and its wait's calls, in call order."""

    agent: Agent
    run_id: str
    root_run_id: str
    calls: tuple[FanoutCall, ...]


_RESOLVED_BY_CONTRACT = "complete_tool_call"
"""Stands in for a resolved row's content digest, which the tool-call read model omits.

A row leaves ``started`` only through ``complete_tool_call``, which always writes the
digest (store contract K9). The row's ``meta.managed`` binding is not in the read model
either; each child's own header is checked against its call before its completion.
"""


async def adopt_settled(store: Any, node: WaitNode) -> ClaimResult | None:
    """Claim a settled v4 wait, then complete each call in call order; the claim, or None.

    The caller has read every child of ``node`` terminal in a transaction that
    began after its own terminal commit (spec §7.2). Claim first: no report is
    built, transformed or projected before the claim, and a caller that does not
    win it (the parent is cancelled, moved on, or another caller adopted) runs
    nothing and returns None. The returned claim owns the parent's continuation.
    """
    parent = _parent_of(node)
    if await _managed_cancel.cancellation_requested(store, node.run_id):
        return None
    claim = await claim_fanout_adoption(
        store, node.run_id, agent=node.agent, pause_id=node.pause_id, bindings=node.bindings
    )
    if not isinstance(claim, ClaimResult):
        return None
    return await _adopt(store, parent, claim, node.iteration)


async def adopt_fanout(store: Any, node: WaitNode, claim: ClaimResult) -> ClaimResult:
    """Complete a claimed v4 adoption's unresolved rows in call order (``AdoptFanout``).

    Runs under ``claim``, the parent's adoption claim. Returns the claim with the
    checkpoint after the last completion: wait, admitted calls and marker cleared.

    Raises:
        ManagedStateError: ``binding_conflict`` when ``claim`` is not this wait's
            v4 adoption or its rows are not a resolved prefix; ``child_mismatch``
            when a child is not terminal or is bound elsewhere.
    """
    return await _adopt(store, _parent_of(node), claim, node.iteration)


async def finish_fanout_adoption(
    agent: Agent, owned: Any, run_id: str, claim: ClaimResult
) -> ClaimResult:
    """Recovery: complete a claimed v4 adoption from its persisted state, rows still started only.

    ``owned`` is bound to ``claim``, a new claim of the parent whose checkpoint
    holds the v4 adoption marker (``fanout_adoption``). Call order comes from the
    claimed pause's ``bindings`` list, which must equal the marker's calls and the
    claim payload's; each call is rebuilt from its intent row and the parent's
    registered subagents. Resolved rows are never recomputed.
    """
    state = claim.loop_state or {}
    ids = fanout_adoption(state, run_id=run_id)
    if ids is None:
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    wait = await _claimed_wait(owned, run_id, ids)
    header = ManagedHeader.from_dict(state.get(MANAGED_HEADER_KEY), run_id=run_id)
    rows = await _iteration_rows(owned, run_id, wait.iteration)
    calls = _recovered_calls(agent, header.path, wait, rows, run_id)
    parent = _FanoutParent(agent, run_id, header.root_run_id, calls)
    return await _complete_calls(claimed_store(owned, run_id, claim), parent, claim, rows)


_CONFLICT = "binding_conflict"


def _parent_of(node: WaitNode) -> _FanoutParent:
    if node.version != MANAGED_FANOUT_SCHEMA_VERSION:
        raise ManagedStateError(reason=_CONFLICT, run_id=node.run_id)
    calls = tuple(
        FanoutCall(child.call, child.agent, child.agent_path, child.run_id)
        for child in node.children
        if child.call is not None
    )
    return _FanoutParent(node.agent, node.run_id, node.root_run_id, calls)


async def _adopt(
    store: Any, parent: _FanoutParent, claim: ClaimResult, iteration: int
) -> ClaimResult:
    owned = claimed_store(store, parent.run_id, claim)
    rows = await _iteration_rows(owned, parent.run_id, iteration)
    return await _complete_calls(owned, parent, claim, rows)


async def _complete_calls(
    owned: Any, parent: _FanoutParent, claim: ClaimResult, rows: list[ToolCallReadRecord]
) -> ClaimResult:
    """Every call after the resolved prefix, in call order, each from the previous checkpoint."""
    run_id = parent.run_id
    bindings = [(item.call.id, item.child_run_id) for item in parent.calls]
    if fanout_adoption(claim.loop_state, run_id=run_id) != tuple(c for c, _ in bindings):
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    done = resolved_prefix(_intent_rows(rows, parent.calls), bindings, run_id=run_id)
    if done == len(bindings):  # the last completion clears the marker in its own transaction
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    state = dict(claim.loop_state or {})
    for item in parent.calls[done:]:
        state = await _complete_call(owned, parent, item, claim.claim_epoch, state)
    return ClaimResult(claim.claim_epoch, claim.storage_format, state, None)


async def _complete_call(
    owned: Any, parent: _FanoutParent, item: FanoutCall, epoch: int, state: dict[str, Any]
) -> dict[str, Any]:
    """Read, report, transform, project and commit one call; the parent checkpoint after it.

    The projection's engine starts from the parent's persisted PII mapping, which
    holds every earlier completion's placeholders. A blocked report marks the
    parent in this call's own completion transaction.
    """
    record, child_state = await _terminal_child(owned, parent, item)
    accepted = await _accept(parent.agent, item.call, item.child, record, owned, parent.run_id)
    provenance = _managed_budget.child_fault(child_state, record, accepted.result, record.id)
    blocked = accepted.blocked is not None
    async with _blocked_evidence(owned, parent.run_id) if blocked else nullcontext():
        return await _managed_persistence.adopt_fanout_result(
            owned,
            parent.run_id,
            claim_epoch=epoch,
            state=state,
            call_id=item.call.id,
            report=accepted.result,
            child_run_id=item.child_run_id,
            agent_path=item.child_path,
            provenance=provenance,
            observation=accepted.observation,
            pii_mapping=accepted.pii_mapping,
        )


async def _terminal_child(
    owned: Any, parent: _FanoutParent, item: FanoutCall
) -> tuple[RunRecord, dict[str, Any]]:
    """The child's terminal record and checkpoint; it must answer this call of this parent."""
    record = await owned.get_run(item.child_run_id)
    state = await owned.get_loop_state(item.child_run_id)
    if record is None or state is None:
        raise ManagedStateError(reason="child_missing", run_id=parent.run_id)
    header = ManagedHeader.from_dict(state.get(MANAGED_HEADER_KEY), run_id=item.child_run_id)
    if (
        record.status not in _TERMINAL
        or record.parent_run_id != parent.run_id
        or (header.root_run_id, header.parent_run_id, header.tool_call_id)
        != (parent.root_run_id, parent.run_id, item.call.id)
    ):
        raise ManagedStateError(reason="child_mismatch", run_id=parent.run_id)
    return record, state


async def _claimed_wait(owned: Any, run_id: str, ids: tuple[str, ...]) -> PauseState:
    """The claimed v4 wait; its bindings and the claim payload name ``ids``, in call order."""
    raw = await owned.get_pause_state(run_id) or {}
    try:
        wait = PauseState.from_managed_dict(raw, run_id=run_id)
    except ManagedStateError:
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id) from None
    fanout = wait.managed_fanout
    order = () if fanout is None else tuple(b.tool_call_id for b in fanout.bindings)
    submission = raw.get("submission")
    payload = submission.get("payload") if isinstance(submission, dict) else None
    if order != ids or payload != adoption_payload(ids):
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    return wait


def _recovered_calls(
    agent: Agent,
    parent_path: tuple[str, ...],
    wait: PauseState,
    rows: list[ToolCallReadRecord],
    run_id: str,
) -> tuple[FanoutCall, ...]:
    """Each binding's call as its intent row recorded it, with the registered child."""
    by_call = {row.tool_call_id: row for row in rows if row.target == ToolTarget.AGENT.value}
    calls: list[FanoutCall] = []
    for binding in wait.managed_fanout.bindings if wait.managed_fanout else ():
        row = by_call.get(binding.tool_call_id)
        if row is None:
            raise ManagedStateError(reason="binding_missing", run_id=run_id)
        call = ToolCall(
            name=row.tool_name,
            params=dict(row.params or {}),
            id=row.tool_call_id,
            provider_tool_call_id=row.provider_tool_call_id,
        )
        child = _registered(agent, row.tool_name, run_id)
        calls.append(FanoutCall(call, child, (*parent_path, row.tool_name), binding.child_run_id))
    return tuple(calls)


def _intent_rows(
    records: list[ToolCallReadRecord], calls: tuple[FanoutCall, ...]
) -> list[IntentRow]:
    """The prefix check's view of the iteration's rows, from the public read model."""
    bound = {item.call.id: item.child_run_id for item in calls}
    return [
        IntentRow(
            record.tool_call_id,
            record.outcome,
            _RESOLVED_BY_CONTRACT if record.outcome in RESOLVED_OUTCOMES else None,
            bound.get(record.tool_call_id) if record.target == ToolTarget.AGENT.value else None,
        )
        for record in records
    ]


async def _iteration_rows(store: Any, run_id: str, iteration: int) -> list[ToolCallReadRecord]:
    """The wait's own iteration rows (one model batch, bounded by ``max_pending_calls``)."""
    rows: list[ToolCallReadRecord] = []
    while True:
        page = await store.get_tool_calls(
            run_id, iteration_index=iteration, limit=MAX_PAGE, offset=len(rows)
        )
        rows.extend(page)
        if len(page) < MAX_PAGE:
            return rows
