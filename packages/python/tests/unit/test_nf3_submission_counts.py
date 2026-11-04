"""Submission patches keep counters separate from approval decisions."""

from types import SimpleNamespace

from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import generate_ulid


def test_only_consumed_client_calls_increment_once():
    recorder = PersistenceRecorder(SimpleNamespace(), generate_ulid())
    recorder._loop_state = {"tool_counts": {"read": 1, "server": 3}}
    pending = {
        "a": {"tool": "read", "phase": "client"},
        "b": {"tool": "read", "phase": "client"},
        "s": {"tool": "server", "phase": "gated"},
    }
    assert recorder._submitted_client_counts(pending, ["a", "a", "b", "s", "missing"]) == {
        "read": 3,
        "server": 3,
    }
    assert recorder._loop_state == {"tool_counts": {"read": 1, "server": 3}}
    assert recorder._submitted_client_counts(pending, []) == {"read": 1, "server": 3}
