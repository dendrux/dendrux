"""Shared race-safe helpers for the Agent submit methods.

The agent methods :meth:`Agent.submit_tool_results`,
:meth:`Agent.submit_input`, and :meth:`Agent.submit_approval` all follow
the same persist-first-then-CAS-claim pattern. This module keeps that
pattern in one place so the guarantees (persist before claim, atomic
transition, stable errors) stay consistent across methods.

Helpers raise public exceptions from :mod:`dendrux.errors`; callers do
not need to interpret store return values.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.types import ApprovalDecision, PauseState, RunStatus, ToolCall, ToolResult, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dendrux.runtime.state import StateStore

__all__ = [
    "DEFAULT_DENIAL_REASON",
    "approval_gates_of",
    "build_rejection_results",
    "claim_or_raise",
    "deserialize_decisions",
    "is_client_call",
    "normalize_decisions",
    "preflight_status",
    "raise_for_non_paused_status",
    "serialize_decisions",
    "serialize_tool_results",
    "validate_approval_decisions",
    "validate_tool_results",
]

DEFAULT_DENIAL_REASON = "User declined to run this tool."


_TERMINAL_STATUSES = frozenset(
    {
        RunStatus.SUCCESS,
        RunStatus.ERROR,
        RunStatus.CANCELLED,
        RunStatus.MAX_ITERATIONS,
    }
)

_PAUSE_STATUSES = frozenset(
    {
        RunStatus.WAITING_CLIENT_TOOL,
        RunStatus.WAITING_HUMAN_INPUT,
        RunStatus.WAITING_APPROVAL,
    }
)


async def raise_for_non_paused_status(store: StateStore, run_id: str) -> NoReturn:
    """Raise the most specific exception for a run that has no pause state."""
    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    current = RunStatus(record.status)
    if current in _TERMINAL_STATUSES:
        raise RunAlreadyTerminalError(run_id, current)
    raise RunNotPausedError(run_id, current)


async def validate_tool_results(store: StateStore, run_id: str, results: list[ToolResult]) -> None:
    """Verify ``results`` align with the pending tool calls on ``run_id``.

    Raises on any mismatch — no partial acceptance. Rejects duplicate
    ``call_id`` values (one result per pending call) as well as
    missing/unexpected ids.
    """
    raw_pause = await store.get_pause_state(run_id)
    if raw_pause is None:
        await raise_for_non_paused_status(store, run_id)

    pause_state = PauseState.from_dict(raw_pause)
    pending_ids = {tc.id for tc in pause_state.pending_tool_calls}
    provided_counts = Counter(r.call_id for r in results)
    duplicates = sorted(k for k, v in provided_counts.items() if v > 1)
    provided_ids = set(provided_counts)

    parts: list[str] = []
    if duplicates:
        parts.append(f"duplicate call_ids={duplicates}")
    missing = sorted(pending_ids - provided_ids)
    extra = sorted(provided_ids - pending_ids)
    if missing:
        parts.append(f"missing call_ids={missing}")
    if extra:
        parts.append(f"unexpected call_ids={extra}")

    if parts:
        raise InvalidToolResultError(run_id, "; ".join(parts))


async def preflight_status(store: StateStore, run_id: str, expected_status: RunStatus) -> None:
    """Raise the most specific error unless ``run_id`` is in ``expected_status``.

    Also refuses a run whose cancellation was requested but not yet observed:
    it is on the path to CANCELLED and resuming it would bounce off the next
    checkpoint.
    """
    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    current = RunStatus(record.status)
    if current != expected_status:
        if current in _TERMINAL_STATUSES:
            raise RunAlreadyTerminalError(run_id, current)
        if current in _PAUSE_STATUSES:
            raise PauseStatusMismatchError(run_id, current, expected_status)
        # RUNNING, PENDING, or anything else non-paused.
        raise RunNotPausedError(run_id, current)
    if record.cancel_requested:
        raise RunAlreadyTerminalError(run_id, RunStatus.CANCELLED)


async def claim_or_raise(
    store: StateStore,
    run_id: str,
    *,
    expected_status: RunStatus,
    submitted_data: dict[str, Any] | None = None,
) -> None:
    """Perform a status preflight, then persist-first handoff + CAS claim.

    The preflight reads the current status before attempting the CAS so
    that a never-paused running row does not get misclassified as a
    concurrent claim race. If the preflight observes ``expected_status``,
    the CAS runs; any subsequent failure is attributable to a genuine
    race.

    If ``submitted_data`` is provided, merges it into ``pause_data`` and
    transitions from ``expected_status`` to RUNNING atomically. Otherwise
    just claims (used for approval-approve).

    Raises:
        RunNotFoundError: ``run_id`` does not exist.
        RunAlreadyTerminalError: run is already in a terminal state.
        RunAlreadyClaimedError: run was paused at preflight but another
            submit won the CAS.
        PauseStatusMismatchError: run is paused for a different reason.
        RunNotPausedError: run is in some other non-paused state (e.g.
            already running, never paused).
    """
    await preflight_status(store, run_id, expected_status)

    # --- Claim: run was expected_status at preflight, attempt CAS. ---
    if submitted_data is not None:
        won = await store.submit_and_claim(
            run_id,
            expected_status=expected_status.value,
            submitted_data=submitted_data,
        )
    else:
        won = await store.claim_paused_run(run_id, expected_status=expected_status.value)

    if won:
        return

    # CAS lost after preflight succeeded — something transitioned the run
    # between the two reads. Re-check to pick the most specific error.
    record = await store.get_run(run_id)
    if record is None:
        # Extremely unlikely — run vanished mid-submit.
        raise RunNotFoundError(run_id)

    current = RunStatus(record.status)
    if current in _TERMINAL_STATUSES:
        raise RunAlreadyTerminalError(run_id, current)
    if current == RunStatus.RUNNING:
        # We just observed ``expected_status``; RUNNING now means a
        # concurrent submit claimed it.
        raise RunAlreadyClaimedError(run_id)
    if current in _PAUSE_STATUSES:
        # Unusual — paused status changed between preflight and CAS.
        raise PauseStatusMismatchError(run_id, current, expected_status)
    raise RunNotPausedError(run_id, current)


def build_rejection_results(
    pause_state: PauseState, rejection_reason: str | None = None
) -> list[ToolResult]:
    """Build synthetic failed :class:`ToolResult` entries for every pending call.

    Used by :meth:`Agent.submit_approval` when ``approved=False`` so the
    rejection is visible to the LLM as failed tool output and the agent
    decides the next step (retry, alternative, give up).

    The rejection reason is encoded in ``payload`` as a JSON string so the
    LLM sees it as tool-message content. ``error`` carries the same value
    for observability.
    """
    import json

    reason = rejection_reason or DEFAULT_DENIAL_REASON
    payload = json.dumps(reason)
    return [
        ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=payload,
            success=False,
            error=reason,
            duration_ms=0,
        )
        for tc in pause_state.pending_tool_calls
    ]


def serialize_tool_results(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Match the wire shape expected by ``submitted_tool_results`` in pause_data."""
    return [
        {
            "name": r.name,
            "call_id": r.call_id,
            "payload": r.payload,
            "success": r.success,
            "error": r.error,
            "duration_ms": r.duration_ms,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Per-call approval decisions
# ---------------------------------------------------------------------------


def is_client_call(
    pause_state: PauseState,
    tc: ToolCall,
    target_lookup: Mapping[str, ToolTarget] | None = None,
) -> bool:
    """True when a pending call must never execute on the server.

    Fail-safe routing: the call is client-side if either its persisted
    pause-time target or the current tool definition says non-server.
    """
    persisted = pause_state.pending_targets.get(tc.id)
    if persisted is not None and persisted != ToolTarget.SERVER.value:
        return True
    current = (target_lookup or {}).get(tc.name)
    return current is not None and current != ToolTarget.SERVER


def approval_gates_of(
    pause_state: PauseState,
    target_lookup: Mapping[str, ToolTarget] | None = None,
) -> dict[str, dict[str, str | None]]:
    """Approval-gated call ids of a WAITING_APPROVAL pause.

    Uses only the snapshot persisted at pause time; policy is never
    recomputed. Pauses persisted before per-call approvals carry no
    snapshot: every server-side pending call is then treated as gated
    (source ``"legacy"``), so nothing runs without an explicit decision.
    """
    if pause_state.approval_gates is not None:
        return {k: dict(v) for k, v in pause_state.approval_gates.items()}
    return {
        tc.id: {"source": "legacy", "reason": None}
        for tc in pause_state.pending_tool_calls
        if not is_client_call(pause_state, tc, target_lookup)
    }


def normalize_decisions(decisions: object) -> list[tuple[str, ApprovalDecision]]:
    """Type-check a public ``decisions`` mapping into ``(call_id, decision)`` pairs."""
    if not isinstance(decisions, Mapping):
        raise ValueError(
            "decisions must be a mapping of tool call id to ApprovalDecision, "
            f"got {type(decisions).__name__}."
        )
    pairs: list[tuple[str, ApprovalDecision]] = []
    for call_id, decision in decisions.items():
        if not isinstance(call_id, str):
            raise ValueError(f"decisions keys must be tool call id strings, got {call_id!r}.")
        if not isinstance(decision, ApprovalDecision):
            raise ValueError(
                f"decisions[{call_id!r}] must be an ApprovalDecision, "
                f"got {type(decision).__name__}."
            )
        pairs.append((call_id, decision))
    return pairs


def validate_approval_decisions(
    run_id: str,
    pause_state: PauseState,
    pairs: Iterable[tuple[str, ApprovalDecision]],
    target_lookup: Mapping[str, ToolTarget] | None = None,
) -> dict[str, ApprovalDecision]:
    """Check decisions cover exactly the approval-gated calls of the pause.

    Raises ``ValueError`` on duplicate, missing or unknown call ids — no
    partial acceptance. Returns the decisions keyed by call id.
    """
    pair_list = list(pairs)
    gated = set(approval_gates_of(pause_state, target_lookup))
    counts = Counter(call_id for call_id, _ in pair_list)
    problems: list[str] = []
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    if duplicates:
        problems.append(f"duplicate call_ids={duplicates}")
    missing = sorted(gated - set(counts))
    if missing:
        problems.append(f"missing call_ids={missing}")
    unknown = sorted(set(counts) - gated)
    if unknown:
        problems.append(f"unknown call_ids={unknown}")
    if problems:
        raise ValueError(
            f"Approval decisions for run '{run_id}' must cover exactly the "
            f"approval-gated calls: {'; '.join(problems)}."
        )
    return dict(pair_list)


def serialize_decisions(
    pause_state: PauseState, decisions: Mapping[str, ApprovalDecision]
) -> list[dict[str, Any]]:
    """Wire shape of ``submitted_decisions`` in pause_data, in pending order."""
    order = {tc.id: i for i, tc in enumerate(pause_state.pending_tool_calls)}
    return [
        {"call_id": call_id, "approved": d.approved, "reason": d.reason}
        for call_id, d in sorted(decisions.items(), key=lambda kv: order.get(kv[0], len(order)))
    ]


def deserialize_decisions(raw: object) -> list[tuple[str, ApprovalDecision]]:
    """Parse persisted ``submitted_decisions`` back into pairs (duplicates kept)."""
    if not isinstance(raw, list):
        raise ValueError("submitted_decisions must be a list.")
    pairs: list[tuple[str, ApprovalDecision]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("call_id"), str):
            raise ValueError("submitted_decisions entries need a string call_id.")
        reason = item.get("reason")
        pairs.append(
            (
                item["call_id"],
                ApprovalDecision(
                    approved=item.get("approved") is True,
                    reason=reason if isinstance(reason, str) else None,
                ),
            )
        )
    return pairs
