"""A bounded pause display retains enough identity to request server resolution."""

import json

import pytest

from dendrux.store import PendingPause, PendingToolCall
from dendrux.types import RunResult, RunStatus


@pytest.mark.parametrize("overflow", ["work", "mapping", "depth"])
def test_pending_overflow_hides_arguments_without_losing_pause_identity(overflow):
    mapping = {f"<<EMAIL_{i}>>": f"a{i:03d}@ex.io" for i in range(990)}
    params = {"blob": "a" * 840000, "private": "PRIVATE_TOOL_PAYLOAD"}
    if overflow == "mapping":
        mapping.update({f"<<MORE_{i}>>": str(i) for i in range(20)})
    if overflow == "depth":
        mapping = {}
        params = {"secret": "PRIVATE_TOOL_PAYLOAD"}
        for _ in range(40):
            params = {"nested": params}
    pause = PendingPause(
        "r",
        "waiting_approval",
        "approval",
        "pause-token",
        tool_calls=[PendingToolCall("c", "approve", "server", params)],
        question="PRIVATE_QUESTION",
    )
    wire = RunResult(
        "r", RunStatus.WAITING_APPROVAL, pending=pause, meta={"pii_mapping": mapping}
    ).to_dict()
    assert wire["pending"] is not None, "display overflow must preserve the pause envelope"
    assert wire["pending"]["pause_id"] == "pause-token"
    assert wire["pending"]["status"] == "waiting_approval"
    assert wire["pending"]["kind"] == "approval"
    call = wire["pending"]["pending_tool_calls"][0]
    assert (call["id"], call["name"], call["target"]) == ("c", "approve", "server")
    assert call["params"] is None and call["args_hidden"] is True
    assert "PRIVATE_" not in json.dumps(wire)
