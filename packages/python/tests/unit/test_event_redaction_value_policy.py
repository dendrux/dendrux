"""Shape is not provenance.

A redacted field survives only when its value is in an enumerated Dendrux set,
is a number, bool, generated id or digest, or is a developer-declared name the
writer copies from configuration. Text that merely looks like a code is hidden.

- Guardrail finding kinds: Dendrux's built-in kinds are named; any other kind
  (a custom guardrail's, which may come from scanned content) is
  ``"<custom_kind>"``, with counts kept and merged.
- ``llm.completed.cost_source`` keeps only Dendrux's cost sources;
  ``tool.completed.parse_error`` and ``provider.retry.endpoint`` are hidden;
  ``retry_after`` is a bounded number.
- Every former shape-checked field now holds only enumerated values.
- Every non-object ``event.data`` (also ``[]``, ``None``, ``0``, ``False``,
  ``""``) reads as ``{}`` for redacted readers.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dendrux.store import StoredEvent, redact_event

SECRET = "probe-secret-5831"
LOOKS_LIKE_A_CODE = "probe_secret_5831"


def _shown(event_type: str, data: Any, **kw: Any) -> Any:
    ev = StoredEvent(
        sequence_index=0,
        iteration_index=0,
        event_type=event_type,
        correlation_id="c1",
        data=data,
        created_at=None,
    )
    return redact_event(ev, **kw).data


def test_custom_guardrail_kinds_are_a_placeholder() -> None:
    detected = _shown(
        "guardrail.detected",
        {"direction": "incoming", "findings_count": 3, "entities": [SECRET, "EMAIL_ADDRESS"]},
    )
    assert detected["entities"] == ["<custom_kind>", "EMAIL_ADDRESS"]
    blocked = _shown(
        "guardrail.blocked",
        {
            "direction": "incoming",
            "code": "guardrail_blocked",
            "kinds": [SECRET, "US_SSN", "OTHER"],
            "counts": {SECRET: 1, "OTHER": 2, "US_SSN": 4},
        },
    )
    assert blocked["kinds"] == ["<custom_kind>", "US_SSN"]
    assert blocked["counts"] == {"<custom_kind>": 3, "US_SSN": 4}
    skipped = _shown("guardrail.restore_skipped", {"entities": {SECRET: 2}, "count": 2})
    assert skipped == {"entities": {"<custom_kind>": 2}, "count": 2}
    state = _shown(
        "image.state",
        {"guardrail_verdicts": [{"sha256": "ab" * 32, "action": "block", "entities": [SECRET]}]},
    )
    assert state["guardrail_verdicts"][0]["entities"] == ["<custom_kind>"]
    trusted = _shown("guardrail.detected", {"entities": [SECRET]}, include_payloads=True)
    assert trusted["entities"] == [SECRET]


@pytest.mark.parametrize(
    ("event_type", "data", "key"),
    [
        ("llm.completed", {"cost_source": LOOKS_LIKE_A_CODE, "input_tokens": 1}, "cost_source"),
        (
            "tool.completed",
            {"tool_name": "t", "success": True, "parse_error": "x at char 5"},
            "parse_error",
        ),
        ("provider.retry", {"endpoint": "/v1/messages", "status_code": 429}, "endpoint"),
        ("provider.retry", {"retry_after": "99999999"}, "retry_after"),
        ("run.paused", {"status": LOOKS_LIKE_A_CODE, "reason": LOOKS_LIKE_A_CODE}, "status"),
        ("run.paused", {"reason": LOOKS_LIKE_A_CODE}, "reason"),
        ("run.completed", {"status": "success", "stop_reason": LOOKS_LIKE_A_CODE}, "stop_reason"),
        ("run.cancelled", {"reason": LOOKS_LIKE_A_CODE}, "reason"),
        ("run.interrupted", {"failure_reason": LOOKS_LIKE_A_CODE}, "failure_reason"),
        ("approval.requested", {"source": LOOKS_LIKE_A_CODE}, "source"),
        ("approval.decided", {"decision": LOOKS_LIKE_A_CODE}, "decision"),
        ("policy.hook_error", {"tool_name": "t", "error_type": "ProbeSecret5831"}, "error_type"),
        ("guardrail.blocked", {"direction": LOOKS_LIKE_A_CODE}, "direction"),
        (
            "guardrail.unmapped_placeholder",
            {"placeholders": ["<<PROBE_SECRET_5831>>"]},
            "placeholders",
        ),
        ("skill.activated", {"name": "s", "origin": LOOKS_LIKE_A_CODE}, "origin"),
        ("skill.script_failed", {"name": "s", "error_type": "ProbeSecret5831"}, "error_type"),
        ("mcp.error", {"code": LOOKS_LIKE_A_CODE, "error_type": "Probe5831"}, "code"),
        ("mcp.error", {"error_type": "Probe5831"}, "error_type"),
        ("budget.exceeded", {"metric": LOOKS_LIKE_A_CODE, "used": 1}, "metric"),
        ("governance.changed", {"changed": [LOOKS_LIKE_A_CODE]}, "changed"),
        ("image.degraded", {"reason": LOOKS_LIKE_A_CODE, "count": 1}, "reason"),
        ("image.warning", {"warning": LOOKS_LIKE_A_CODE}, "warning"),
    ],
)
def test_code_shaped_text_is_hidden(event_type: str, data: dict[str, Any], key: str) -> None:
    shown = _shown(event_type, data)
    assert "5831" not in json.dumps(shown) and "99999999" not in json.dumps(shown)
    if key == "changed":
        assert shown[key] == []
    elif key == "placeholders":
        assert shown == {"count": 1}
    else:
        assert key not in shown


def test_enumerated_values_survive() -> None:
    assert _shown("llm.completed", {"cost_source": "table"})["cost_source"] == "table"
    assert _shown("provider.retry", {"retry_after": "30"})["retry_after"] == 30
    assert _shown("run.completed", {"status": "success", "stop_reason": "end_turn"}) == {
        "status": "success",
        "stop_reason": "end_turn",
    }
    assert _shown("run.interrupted", {"failure_reason": "stale_running"}) == {
        "failure_reason": "stale_running"
    }
    assert _shown("mcp.error", {"code": "connection_failed", "failure_mode": "strict"}) == {
        "code": "connection_failed",
        "failure_mode": "strict",
    }
    assert _shown("budget.exceeded", {"metric": "tokens", "action": "pause"}) == {
        "metric": "tokens",
        "action": "pause",
    }
    assert _shown("skill.script_failed", {"name": "s", "error_type": "timeout"})["error_type"] == (
        "timeout"
    )


@pytest.mark.parametrize("bad", [[], None, 0, False, "", ["bad"], "x", 7, True])
def test_every_non_object_data_reads_as_empty(bad: Any) -> None:
    for event_type in ("run.resumed", "run.paused", "tool.completed", "custom.x"):
        assert _shown(event_type, bad) == {}
        assert _shown(event_type, bad, active_pause=True) == {}
        assert _shown(event_type, bad, include_payloads=True) == bad
