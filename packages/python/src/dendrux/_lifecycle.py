"""Shared persisted lifecycle vocabulary.

Stored event names use dots; the streaming RunEventType vocabulary uses
underscores and remains in types. Pause-event sets answer different questions
and stay with their consumers, built from these shared names. Migrations retain
historical literals.
"""

from enum import StrEnum
from typing import Literal

from dendrux.types import GovernanceEventType as GovernanceEventType
from dendrux.types import RunStatus

_TERMINAL_ORDER = (
    RunStatus.SUCCESS,
    RunStatus.ERROR,
    RunStatus.CANCELLED,
    RunStatus.MAX_ITERATIONS,
)
_WAITING_ORDER = (
    RunStatus.WAITING_CLIENT_TOOL,
    RunStatus.WAITING_HUMAN_INPUT,
    RunStatus.WAITING_APPROVAL,
)
_PAUSE_ORDER = (*_WAITING_ORDER, RunStatus.PAUSED)
TERMINAL_STATUSES = frozenset(_TERMINAL_ORDER)
WAITING_STATUSES = frozenset(_WAITING_ORDER)
PAUSE_STATUSES = frozenset(_PAUSE_ORDER)
# Active means nonterminal, including pending and durably paused runs.
ACTIVE_STATUSES = frozenset(RunStatus) - TERMINAL_STATUSES
TERMINAL_STATUS_VALUES = tuple(status.value for status in _TERMINAL_ORDER)
WAITING_STATUS_VALUES = tuple(status.value for status in _WAITING_ORDER)
PAUSE_STATUS_VALUES = tuple(status.value for status in _PAUSE_ORDER)
ACTIVE_STATUS_VALUES = tuple(status.value for status in RunStatus if status in ACTIVE_STATUSES)
# Terminal results that close an OpenTelemetry run span with ERROR status.
FAILED_STATUSES = TERMINAL_STATUSES - {RunStatus.SUCCESS}
# Human approval/input pauses may contain a question, never a final answer.
HIDDEN_ANSWER_STATUSES = frozenset({RunStatus.WAITING_APPROVAL, RunStatus.WAITING_HUMAN_INPUT})
STATUSES_WITH_ANSWER_TEXT = frozenset(RunStatus) - HIDDEN_ANSWER_STATUSES
# These pauses display model arguments with placeholders still intact.
PLACEHOLDER_PAUSE_STATUSES = HIDDEN_ANSWER_STATUSES | {RunStatus.PAUSED}
# Preserve the runner's existing cancellation CAS: waiting or running, without paused.
CANCEL_FINALIZE_STATUS_VALUES = (*WAITING_STATUS_VALUES, RunStatus.RUNNING.value)
# Pauses a direct resume may ingest images for: client-tool or approval results, or user input.
RESUME_RESULTS_STATUS_VALUES = (
    RunStatus.WAITING_CLIENT_TOOL.value,
    RunStatus.WAITING_APPROVAL.value,
)
RESUME_INPUT_STATUS_VALUES = (RunStatus.WAITING_HUMAN_INPUT.value,)

PendingKind = Literal["client_tool", "approval", "input", "paused"]
PENDING_KIND_BY_STATUS: dict[str, PendingKind] = {
    RunStatus.WAITING_CLIENT_TOOL: "client_tool",
    RunStatus.WAITING_APPROVAL: "approval",
    RunStatus.WAITING_HUMAN_INPUT: "input",
    RunStatus.PAUSED: "paused",
}


class StoredEventType(StrEnum):
    """Structural event names stored in run_events.event_type."""

    RUN_STARTED = "run.started"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"
    RUN_RESUME_ABORTED = "run.resume_aborted"
    RUN_COMPLETED = "run.completed"
    RUN_ERROR = "run.error"
    RUN_CANCELLED = "run.cancelled"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_ABANDONED = "run.abandoned"
    RUN_RECLAIMED = "run.reclaimed"
    RUN_LEASE_EXPIRED = "run.lease_expired"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_REPLAYED = "tool.replayed"
    TOOLS_DECLARED = "tools.declared"
    LLM_COMPLETED = "llm.completed"
    LLM_FAILED = "llm.failed"


TERMINAL_EVENT_TYPES = frozenset(
    {
        StoredEventType.RUN_COMPLETED.value,
        StoredEventType.RUN_ERROR.value,
        StoredEventType.RUN_CANCELLED.value,
    }
)


class FailureReason(StrEnum):
    """Stable failure_reason values written by loops, recovery and sweeps."""

    STALE_RUNNING = "stale_running"
    NEVER_STARTED = "never_started"
    ABANDONED_WAITING = "abandoned_waiting"
    STALE_RUNNING_LEGACY = "stale_running_legacy"
    RECLAIM_LIMIT = "reclaim_limit"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    PENDING_LIMIT = "pending_limit"
    BUDGET_EXCEEDED = "budget_exceeded"
    LEASE_LOST = "lease_lost"
    HISTORY_TOO_LARGE = "history_too_large"
    RECOVERY_CALL_MISSING = "recovery_call_missing"
    RECOVERY_STATE_MISSING = "recovery_state_missing"
    RECOVERY_STATE_INVALID = "recovery_state_invalid"


class CancelReason(StrEnum):
    """Stable reasons recorded on run cancellation events."""

    CANCEL_REQUESTED = "cancel_requested"
    TASK_CANCELLED = "task_cancelled"
    STREAM_CLOSED = "stream_closed"
    BUDGET_EXCEEDED = FailureReason.BUDGET_EXCEEDED.value


class PauseReason(StrEnum):
    """Cooperative and budget pause reasons; user messages remain arbitrary text."""

    REQUESTED = "requested"
    BUDGET = "budget"


class ApprovalReason(StrEnum):
    """Built-in reasons for requiring approval."""

    REQUIRES_APPROVAL = "requires_approval"
    REPLAY_AFTER_CRASH = "replay_after_crash"


class DenialReason(StrEnum):
    """Built-in denial reasons safe to expose in a redacted event."""

    DENIED_BY_POLICY = "denied_by_policy"
    USER_DECLINED = "User declined to run this tool."
    HOOK_ERROR = "policy hook error"
    CLIENT_APPROVAL_UNSUPPORTED = "approval is not supported for client tools"


STORAGE_FORMAT_CHECKPOINTED = 2
