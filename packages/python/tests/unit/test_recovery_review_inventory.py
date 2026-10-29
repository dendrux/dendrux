"""Public redaction must preserve the enumerated recovery failure reasons."""


def test_recovery_failure_reason_inventory():
    from dendrux.store._redact import _FAILURE_REASONS

    assert {
        "reclaim_limit",
        "stale_running_legacy",
        "pending_limit",
        "lease_lost",
    } <= _FAILURE_REASONS


def test_unpickling_recovery_errors_does_not_repeat_failure_logs(caplog):
    import pickle

    from dendrux.errors import RunCancelledError, RunNotReclaimableError

    for error in (
        RunNotReclaimableError("run", "loop_unknown"),
        RunCancelledError("run", "parent"),
    ):
        before = len(caplog.records)
        restored = pickle.loads(pickle.dumps(error))
        assert restored.args == error.args and restored.__dict__ == error.__dict__
        assert len(caplog.records) == before
