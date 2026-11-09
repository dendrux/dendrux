"""Recovery recognition uses durable unresolved calls without weakening intent checks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.runtime._reclaim import needs_resolution
from dendrux.types import PauseState, ToolCall, UsageStats


@pytest.mark.parametrize(
    "version,reason,phase,target,matching,started,expected,reads",
    [
        (2, "requested", "gated", "server", True, False, True, 0),
        (2, "requested", "gated", "server", False, False, False, 1),
        (2, "requested", "approved", "server", True, False, False, 1),
        (2, "requested", "rejected", "server", True, False, False, 1),
        (2, "requested", "dispatchable", "server", True, False, True, 0),
        (2, "requested", "client", "client", True, False, False, 0),
        (2, "requested", "gated", "client", True, False, False, 0),
        (2, None, "gated", "server", True, False, False, 1),
        (1, "requested", "gated", "server", True, False, False, 0),
        (2, "requested", None, "server", True, False, False, 1),
        (2, "requested", None, "server", True, True, True, 1),
    ],
)
async def test_061_requested_pause_recognition(
    version, reason, phase, target, matching, started, expected, reads
):
    call = ToolCall("write_region")
    pause = PauseState(
        "agent", [call], [], [], 1, 0, UsageStats(), schema_version=version, pause_reason=reason
    )
    store = SimpleNamespace(
        get_tool_calls=AsyncMock(
            return_value=[SimpleNamespace(tool_call_id=call.id, outcome="started")]
            if started
            else []
        )
    )
    checkpoint = (
        {
            "pending": [
                {"call_id": call.id if matching else "other", "phase": phase, "target": target}
            ]
        }
        if phase
        else None
    )
    assert await needs_resolution(store, "run", pause, checkpoint=checkpoint) is expected
    assert store.get_tool_calls.await_count == reads


@pytest.mark.parametrize("recovery_gate", [False, True])
async def test_061_empty_or_recovery_marked_pause_needs_no_read(recovery_gate):
    pause = PauseState("agent", [], [], [], 1, 0, UsageStats(), schema_version=2)
    if recovery_gate:
        pause.approval_gates = {"call": {"source": "recovery", "reason": "replay_after_crash"}}
    store = SimpleNamespace(get_tool_calls=AsyncMock())
    assert await needs_resolution(store, "run", pause) is recovery_gate
    store.get_tool_calls.assert_not_awaited()


@pytest.mark.parametrize("decision_id,expected,reads", [(None, True, 0), ("decision", False, 1)])
async def test_061_requested_dispatchable_decision_is_preserved(decision_id, expected, reads):
    call = ToolCall("write_region")
    pause = PauseState(
        "agent", [call], [], [], 1, 0, UsageStats(), schema_version=2, pause_reason="requested"
    )
    store = SimpleNamespace(get_tool_calls=AsyncMock(return_value=[]))
    checkpoint = {
        "pending": [
            {
                "call_id": call.id,
                "phase": "dispatchable",
                "target": "server",
                "decision_id": decision_id,
            }
        ]
    }
    assert await needs_resolution(store, "run", pause, checkpoint=checkpoint) is expected
    assert store.get_tool_calls.await_count == reads
