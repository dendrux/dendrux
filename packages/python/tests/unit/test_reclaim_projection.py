"""Recovery projections share the browser DTO and safe event vocabulary."""

from datetime import UTC, datetime

from dendrux._wire import result_to_dict
from dendrux.store import StoredEvent
from dendrux.store._redact import redact_event
from dendrux.types import RecoveryInfo, RunResult, RunStatus


def test_wire_recovery_matches_result():
    result = RunResult(
        run_id="run",
        status=RunStatus.SUCCESS,
        recovery=RecoveryInfo(2, datetime(2026, 10, 29, tzinfo=UTC), ["call"], False),
    )
    assert result_to_dict(result) == result.to_dict()
    assert result_to_dict(result)["recovery"]["last_reclaimed_at"].endswith("Z")


def test_recovery_approval_safe_vocabulary():
    view = redact_event(
        StoredEvent(
            0,
            0,
            "approval.requested",
            None,
            {
                "reason": "replay_after_crash",
                "source": "recovery",
                "params": {"private": "sentinel"},
            },
            None,
        )
    ).data
    assert view["reason"] == "replay_after_crash"
    assert view["source"] == "recovery"
    assert "sentinel" not in str(view)
