"""Pure recovery decisions; no side effect is authorized by an absent intent alone."""

import pytest

from dendrux.runtime import persistence


@pytest.mark.parametrize(
    "phase,row,expected",
    [
        ("dispatchable", None, "dispatch"),
        ("gated", None, "approval"),
        ("approved", None, "dispatch"),
        ("rejected", None, "resolve"),
        ("invalid", None, "resolve"),
        ("client", None, "wait_client"),
        ("dispatchable", {"outcome": "started", "idempotent": True}, "replay"),
        ("dispatchable", {"outcome": "started", "idempotent": False}, "approval_replay"),
        ("dispatchable", {"outcome": "completed"}, "observe"),
        ("dispatchable", {"outcome": "failed"}, "observe"),
        ("dispatchable", {"outcome": "unknown"}, "observe"),
    ],
)
def test_each_pending_resolution_row(phase, row, expected):
    resolve = getattr(persistence, "resolve_pending", None)
    assert resolve is not None, "reclaim needs a pure resolution table"
    pending = {
        "call_id": "call",
        "phase": phase,
        "order_index": 17,
        "tool": "search",
        "target": "server",
    }
    rows = [{"tool_call_id": "call", **row}] if row else []
    state = {"pending": [pending]}
    actions = resolve(state, rows)
    assert [(a.kind, a.call_id, a.order_index) for a in actions] == [(expected, "call", 17)]
    assert state == {"pending": [pending]}


def test_submission_has_priority_and_resolved_pending_never_replays():
    resolve = getattr(persistence, "resolve_pending", None)
    assert resolve is not None, "submission consumption precedes dispatch"
    actions = resolve({"submission": {"submission_id": "s"}, "pending": [{"call_id": "c"}]}, [])
    assert [a.kind for a in actions] == ["consume_submission"]
    assert resolve({"pending": []}, [{"tool_call_id": "c", "outcome": "completed"}]) == []
