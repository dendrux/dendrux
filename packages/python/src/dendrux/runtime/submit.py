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
from typing import TYPE_CHECKING, Any, NoReturn

from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.types import PauseState, RunStatus, ToolResult

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore

__all__ = [
    "build_rejection_results",
    "claim_or_raise",
    "raise_for_non_paused_status",
    "serialize_tool_results",
    "validate_tool_results",
]


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
    # --- Preflight: get the current status BEFORE we touch CAS. ---
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

    # If cancellation has been requested but the runner has not yet
    # observed it, refuse the submit — the run is on the path to
    # CANCELLED and resuming it would just bounce off the next checkpoint.
    if record.cancel_requested:
        raise RunAlreadyTerminalError(run_id, RunStatus.CANCELLED)

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

    reason = rejection_reason or "User declined to run this tool."
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
