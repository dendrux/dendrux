"""T0 move evidence; later owners may delete location checks when reshaping code."""

from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import ManagedStateError
from dendrux.runtime import _trace_shared, persistence
from dendrux.types import ToolResult


def test_managed_adoption_keeps_the_original_import_alias():
    function = persistence.adopt_managed_result
    assert Path(unwrap(function).__code__.co_filename).name == "_managed_persistence.py"
    from dendrux.runtime import _managed_persistence

    assert function is _managed_persistence.adopt_managed_result


@pytest.mark.parametrize("success", [True, False])
async def test_managed_adoption_keeps_the_trace_patch_seam_and_v3_write(monkeypatch, success):
    state = {
        "v": 3,
        "iteration": 2,
        "managed_wait": {"tool_call_id": "call", "child_run_id": "child"},
        "pending": [{"call_id": "call", "order_index": 4}],
        "submission": {"call_ids": ["call"]},
    }
    claim = SimpleNamespace(loop_state=state, claim_epoch=7)
    report = ToolResult(name="reviewer", call_id="call", payload="accepted", success=success)
    store = SimpleNamespace(complete_tool_call=AsyncMock())
    trace = _trace_shared.message_trace
    seen = []

    def message_trace(*args):
        seen.append(args)
        return trace(*args)

    monkeypatch.setattr(_trace_shared, "message_trace", message_trace)
    result = await unwrap(persistence.adopt_managed_result)(
        store,
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        claim=claim,
        call_id="call",
        report=report,
        child_run_id="child",
        agent_path=("01ARZ3NDEKTSV4RRFFQ69G5FAV", "child"),
        pii_mapping={"token": "value"},
    )
    store.complete_tool_call.assert_awaited_once()
    args, fields = store.complete_tool_call.call_args
    assert args == ("01ARZ3NDEKTSV4RRFFQ69G5FAV",)
    assert fields["claim_epoch"] == 7
    assert fields["outcome"] == ("completed" if success else "failed")
    assert fields["result_payload"] == "accepted"
    assert fields["pii_mapping"] == {"token": "value"}
    assert fields["event"].event_type == "tool.completed"
    assert fields["event"].data["agent_path"] == ["01ARZ3NDEKTSV4RRFFQ69G5FAV", "child"]
    assert fields["observation"].order_index == 4
    assert fields["observation"].meta["iteration"] == 2
    assert len(seen) == 1 and seen[0][1] == 4
    assert result["pending"] == [] and result["managed_wait"] is None
    assert result["submission"] is None and result["phase"] == "model_next"
    assert state["pending"] == [{"call_id": "call", "order_index": 4}]


async def test_managed_adoption_raises_the_existing_binding_exception():
    store = SimpleNamespace(complete_tool_call=AsyncMock())
    claim = SimpleNamespace(loop_state={"managed_wait": {}}, claim_epoch=7)
    with pytest.raises(ManagedStateError) as caught:
        await unwrap(persistence.adopt_managed_result)(
            store,
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            claim=claim,
            call_id="call",
            report=ToolResult(name="reviewer", call_id="call", payload="accepted"),
            child_run_id="child",
            agent_path=("01ARZ3NDEKTSV4RRFFQ69G5FAV", "child"),
        )
    assert caught.value.reason == "binding_conflict"
    store.complete_tool_call.assert_not_awaited()
