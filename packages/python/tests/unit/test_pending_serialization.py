"""Pending serializers preserve ordinary pauses and expose managed pause origins."""

import pytest

from dendrux import _wire
from dendrux.store import PendingPause, PendingToolCall
from dendrux.types import RunResult, RunStatus
from tests.conformance import harness

ORIGIN = {
    "agent_name": "reviewer",
    "agent_path": ["lead", "reviewer"],
    "origin_run_id": "child",
}
WIRE_KEYS = {
    "run_id",
    "status",
    "kind",
    "pause_id",
    "pending_tool_calls",
    "question",
    "options",
    "reason",
    "message",
    "exceeded",
}


@pytest.mark.parametrize("managed", [False, True], ids=["ordinary", "managed"])
@pytest.mark.parametrize("serializer", ["dto", "result", "overflow", "snapshot"])
def test_pending_origin_shape(
    managed: bool, serializer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = PendingPause(
        "root",
        RunStatus.WAITING_APPROVAL.value,
        "approval",
        "pause",
        tool_calls=[PendingToolCall("call", "publish", "server", {"text": "draft"})],
        agent_name=ORIGIN["agent_name"] if managed else None,
        agent_path=ORIGIN["agent_path"] if managed else None,
        origin_run_id=ORIGIN["origin_run_id"] if managed else None,
    )
    expected_keys = WIRE_KEYS
    if serializer == "snapshot":
        shown = harness.raw(pending)
        expected_keys = WIRE_KEYS - {"pending_tool_calls", "exceeded"} | {
            "tool_calls",
            "budget_exceeded",
        }
        assert shown["tool_calls"][0] == {
            "id": "call",
            "name": "publish",
            "target": "server",
            "params": {"text": "draft"},
            "approval": None,
            "args_hidden": False,
        }
    elif serializer == "dto":
        shown = pending.to_dict()
    else:
        if serializer == "overflow":
            monkeypatch.setattr(_wire, "_redacted", lambda data, mapping: None)
        shown = RunResult("root", RunStatus.WAITING_APPROVAL, pending=pending).to_dict()["pending"]
    assert set(shown) == expected_keys | (set(ORIGIN) if managed else set())
    assert (shown["run_id"], shown["pause_id"]) == ("root", "pause")
    assert shown["reason"] is None
    if managed:
        assert {key: shown[key] for key in ORIGIN} == ORIGIN
    else:
        assert (pending.agent_name, pending.agent_path, pending.origin_run_id) == (None, None, None)
