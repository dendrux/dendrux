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
