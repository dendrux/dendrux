"""Pin lifecycle vocabulary and keep the consumers' distinct questions intact."""

import ast
from pathlib import Path

import dendrux
from dendrux.types import RunStatus

ROOT = Path(dendrux.__file__).parent
EVENT_NAMES = {
    "run.started",
    "run.paused",
    "run.resumed",
    "run.resume_aborted",
    "run.completed",
    "run.error",
    "run.cancelled",
    "run.interrupted",
    "run.abandoned",
    "run.reclaimed",
    "run.lease_expired",
    "tool.started",
    "tool.completed",
    "tool.replayed",
    "tools.declared",
    "llm.completed",
    "llm.failed",
}


def test_stored_event_names_have_one_definition():
    copies = []
    for path in ROOT.rglob("*.py"):
        if path.name == "_lifecycle.py" or "migrations" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (
                    node.value in EVENT_NAMES
                    or any(f"'{name}'" in node.value for name in EVENT_NAMES)
                )
            ):
                copies.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not copies, "Stored event literals outside lifecycle home: " + ", ".join(copies)


def test_status_partitions_match_existing_public_values():
    from dendrux.runtime import _pending, replay, submit
    from dendrux.store import _follow

    terminal = {"success", "error", "cancelled", "max_iterations"}
    pause = {"waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"}
    assert (
        _follow.TERMINAL_STATUSES
        == replay._TERMINAL_STATUSES
        == submit._TERMINAL_STATUSES
        == terminal
    )
    assert _follow.PAUSE_STATUSES == _pending.PAUSE_STATUSES == submit._PAUSE_STATUSES == pause
    assert set(RunStatus) - terminal - pause == {"pending", "running"}


def test_pause_sets_keep_their_existing_differences():
    from dendrux.store._follow import _TERMINAL_EVENT_TYPES
    from dendrux.store._redact import PAUSE_CHANGING_EVENTS

    assert {"run.completed", "run.error", "run.cancelled"} == _TERMINAL_EVENT_TYPES
    assert {
        "run.paused",
        "approval.requested",
        "run.resume_aborted",
        "run.resumed",
        "run.completed",
        "run.error",
        "run.cancelled",
        "run.interrupted",
        "run.abandoned",
    } == PAUSE_CHANGING_EVENTS


def test_reason_allowlists_keep_their_existing_values():
    from dendrux.store import _redact

    assert {
        "stale_running",
        "never_started",
        "abandoned_waiting",
        "guardrail_blocked",
        "budget_exceeded",
        "reclaim_limit",
        "stale_running_legacy",
        "pending_limit",
        "lease_lost",
        "history_too_large",
        "recovery_call_missing",
        "recovery_state_missing",
        "recovery_state_invalid",
    } == _redact._FAILURE_REASONS
    assert {
        "cancel_requested",
        "task_cancelled",
        "stream_closed",
        "budget_exceeded",
    } == _redact._CANCEL_REASONS
    assert {"requested", "budget"} == _redact._PAUSE_REASONS
    assert {"requires_approval", "replay_after_crash"} == _redact._CONSTANT_APPROVAL_REASONS
    assert {
        "denied_by_policy",
        "User declined to run this tool.",
        "policy hook error",
        "approval is not supported for client tools",
    } == _redact._CONSTANT_DENIAL_REASONS


def test_lifecycle_partitions_and_sql_values():
    from dendrux import _lifecycle as lifecycle
    from dendrux.db.enums import AgentRunStatus
    from dendrux.runtime._pending import _KIND_BY_STATUS

    assert {s.value for s in AgentRunStatus} == {s.value for s in RunStatus}
    assert lifecycle.TERMINAL_STATUSES.isdisjoint(lifecycle.ACTIVE_STATUSES)
    assert set(RunStatus) == lifecycle.TERMINAL_STATUSES | lifecycle.ACTIVE_STATUSES
    assert lifecycle.WAITING_STATUSES | {RunStatus.PAUSED} == lifecycle.PAUSE_STATUSES
    assert lifecycle.TERMINAL_STATUSES - {RunStatus.SUCCESS} == lifecycle.FAILED_STATUSES
    assert (
        set(RunStatus)
        - {
            RunStatus.WAITING_APPROVAL,
            RunStatus.WAITING_HUMAN_INPUT,
        }
        == lifecycle.STATUSES_WITH_ANSWER_TEXT
    )
    for states, values in (
        (lifecycle.TERMINAL_STATUSES, lifecycle.TERMINAL_STATUS_VALUES),
        (lifecycle.WAITING_STATUSES, lifecycle.WAITING_STATUS_VALUES),
        (lifecycle.PAUSE_STATUSES, lifecycle.PAUSE_STATUS_VALUES),
        (lifecycle.ACTIVE_STATUSES, lifecycle.ACTIVE_STATUS_VALUES),
    ):
        assert set(values) == states
        assert all(type(value) is str for value in values)
    assert lifecycle.CANCEL_FINALIZE_STATUS_VALUES == (
        "waiting_client_tool",
        "waiting_human_input",
        "waiting_approval",
        "running",
    )
    assert _KIND_BY_STATUS == {
        "waiting_client_tool": "client_tool",
        "waiting_approval": "approval",
        "waiting_human_input": "input",
        "paused": "paused",
    }
    assert lifecycle.STORAGE_FORMAT_CHECKPOINTED == 2
    assert {event.value for event in lifecycle.StoredEventType} == EVENT_NAMES


def test_pause_window_and_invalidation_sets_remain_distinct():
    from dendrux.runtime._pending import PAUSE_WINDOW_EVENT_TYPES
    from dendrux.runtime.stores.sql.base import PAUSE_INVALIDATION_EVENT_TYPES

    assert PAUSE_INVALIDATION_EVENT_TYPES == (
        "run.resumed",
        "run.completed",
        "run.error",
        "run.cancelled",
    )
    assert ("run.paused", *PAUSE_INVALIDATION_EVENT_TYPES) == PAUSE_WINDOW_EVENT_TYPES


def test_reason_enums_are_the_redaction_allowlists():
    from dendrux import _lifecycle as lifecycle
    from dendrux.loops._budget import BUDGET_EXCEEDED
    from dendrux.runtime.submit import DEFAULT_DENIAL_REASON
    from dendrux.store import _redact

    for name, enum in (
        ("_FAILURE_REASONS", lifecycle.FailureReason),
        ("_CANCEL_REASONS", lifecycle.CancelReason),
        ("_PAUSE_REASONS", lifecycle.PauseReason),
        ("_CONSTANT_APPROVAL_REASONS", lifecycle.ApprovalReason),
        ("_CONSTANT_DENIAL_REASONS", lifecycle.DenialReason),
    ):
        assert getattr(_redact, name) == frozenset(enum)
    assert lifecycle.FailureReason.BUDGET_EXCEEDED.value == BUDGET_EXCEEDED
    assert lifecycle.CancelReason.BUDGET_EXCEEDED.value == BUDGET_EXCEEDED
    assert lifecycle.DenialReason.USER_DECLINED.value == DEFAULT_DENIAL_REASON


def _format_literal_comparison_lines(source: str) -> list[int]:
    """Line numbers of comparisons that mention storage_format and a literal 2."""
    if "storage_format" not in source:
        return []
    lines = source.splitlines()

    def segment(node: ast.AST) -> str:
        first, last = node.lineno - 1, node.end_lineno - 1
        if first == last:
            return lines[first][node.col_offset : node.end_col_offset]
        # ast offsets are UTF-8 byte offsets, as ast.get_source_segment treats them.
        head = lines[first].encode()[node.col_offset :].decode()
        tail = lines[last].encode()[: node.end_col_offset].decode()
        return "\n".join([head, *lines[first + 1 : last], tail])

    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Compare)
        and "storage_format" in segment(node)
        and any(
            isinstance(child, ast.Constant) and child.value in (2, "2")
            for child in ast.iter_child_nodes(node)
        )
    ]


def test_checkpointed_format_comparisons_use_the_home():
    copies = []
    for path in ROOT.rglob("*.py"):
        if "migrations" in path.parts:
            continue
        copies.extend(
            f"{path.relative_to(ROOT)}:{lineno}"
            for lineno in _format_literal_comparison_lines(path.read_text())
        )
    assert not copies, "Checkpoint format literals outside lifecycle home: " + ", ".join(copies)


def test_format_comparison_scan_flags_literals_and_ignores_clean_code():
    assert _format_literal_comparison_lines(
        "def f(record):\n    if record.storage_format == 2:\n        return 1\n"
    ) == [2]
    multiline = (
        "def f(record):\n    if (\n        record.storage_format\n        == '2'\n"
        "    ):\n        pass\n"
    )
    assert _format_literal_comparison_lines(multiline) == [3]
    assert _format_literal_comparison_lines("x = 1\nif x == 2:\n    pass\n") == []
    assert (
        _format_literal_comparison_lines("def f(r):\n    return r.storage_format == CURRENT\n")
        == []
    )


def test_compatibility_literals_and_log_enums_match_lifecycle():
    from typing import get_args

    from dendrux import _codes
    from dendrux._lifecycle import CancelReason, FailureReason, PauseReason
    from dendrux.runtime.state import SweptRun
    from dendrux.types import PauseRequestOutcome

    # Public signature syntax and Literal typing cannot use computed strings.
    assert SweptRun.__dataclass_fields__["status"].default == RunStatus.ERROR.value
    assert PauseReason.REQUESTED.value in get_args(PauseRequestOutcome)
    # _codes is below lifecycle in the layers and cannot import it.
    assert frozenset(FailureReason) & _codes._ENUMS["reason"] == {
        FailureReason.RECLAIM_LIMIT,
        FailureReason.HISTORY_TOO_LARGE,
        FailureReason.RECOVERY_CALL_MISSING,
        FailureReason.RECOVERY_STATE_MISSING,
        FailureReason.RECOVERY_STATE_INVALID,
    }
    assert frozenset(FailureReason) & _codes._ENUMS["category"] == {
        FailureReason.STALE_RUNNING,
        FailureReason.ABANDONED_WAITING,
    }
    assert CancelReason.CANCEL_REQUESTED in _codes._ENUMS["reason"]


def test_governance_events_use_the_existing_shared_names():
    from dendrux.types import GovernanceEventType

    values = {event.value for event in GovernanceEventType}
    copies = []
    for path in ROOT.rglob("*.py"):
        if path.name in {"types.py", "_codes.py"} or "migrations" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in values
            ):
                copies.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not copies, "Governance event literals outside existing enum: " + ", ".join(copies)


def test_checkpointed_cli_format_uses_the_home():
    import inspect

    from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
    from dendrux.cli.runs import StorageFormat

    assert StorageFormat.checkpointed.value == str(STORAGE_FORMAT_CHECKPOINTED)
    literals = {
        node.value
        for node in ast.walk(ast.parse(inspect.getsource(StorageFormat)))
        if isinstance(node, ast.Constant)
    }
    assert "2" not in literals, "CLI format enum must derive its value from lifecycle"


def test_single_status_literals_use_shared_names():
    values = {
        "running",
        "max_iterations",
        "waiting_approval",
        "waiting_human_input",
        "waiting_client_tool",
    }
    copies = []
    for path in ROOT.rglob("*.py"):
        if path.name in {"types.py", "_lifecycle.py", "_codes.py"} or "migrations" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in values
            ):
                copies.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not copies, "Status literals outside lifecycle vocabulary: " + ", ".join(copies)
