"""Prepared calls only become recovery gates when approval is unresolved."""

from types import SimpleNamespace

import pytest

from dendrux.runtime._reclaim import _hold_started_outcome, recovery_gate
from dendrux.types import ToolCall


@pytest.mark.parametrize("outcome", [None, "started", "completed"])
@pytest.mark.parametrize("pending_approval", [False, True])
def test_061_prepared_sibling_keeps_its_gate_identity(outcome, pending_approval):
    call = ToolCall("write")
    row = SimpleNamespace(outcome=outcome) if outcome is not None else None
    prepared = SimpleNamespace(pending_calls=[call], pending_approval=pending_approval)
    original = {"source": "static", "reason": "requires_approval"}
    gates = {call.id: original} if pending_approval else {}
    _hold_started_outcome(row, prepared, gates)
    assert gates == (
        {call.id: recovery_gate() if outcome == "started" else original} if pending_approval else {}
    )


async def test_061_prepared_dispatch_missing_identity_fails_safely():
    from dendrux.errors import PersistenceError
    from dendrux.runtime._reclaim import _dispatch_prepared_batch

    with pytest.raises(PersistenceError) as caught:
        await _dispatch_prepared_batch(
            None,
            None,
            "run",
            None,
            [SimpleNamespace(call_id=None)],
            [SimpleNamespace(id=None)],
            {},
            {},
            None,
            {},
            None,
        )
    assert caught.value.code == "recovery_submission_missing"


async def test_061_fresh_recovery_gate_clears_prior_evaluated_batch():
    from dendrux.runtime._reclaim import _recovery_pause_result, needs_resolution
    from dendrux.types import PauseState, UsageStats

    call = ToolCall("write")
    decisions = {"old": {"approved": False, "reason": "not today"}}
    pause = PauseState(
        "agent",
        [call],
        [],
        [],
        1,
        0,
        UsageStats(),
        schema_version=2,
        pause_reason="requested",
        pending_decisions=decisions,
    )
    result = await _recovery_pause_result(
        None,
        "run",
        SimpleNamespace(history=[], pause_state=pause),
        {"phase": "awaiting_approval"},
        None,
        [call],
        {call.id: recovery_gate()},
    )
    fresh = result.meta["pause_state"]
    assert fresh.pending_decisions is None
    assert await needs_resolution(None, "run", fresh)
    assert pause.pending_decisions == decisions
