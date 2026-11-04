"""Rejected tool observations are marked before their recorder callback."""

import pytest

from dendrux._meta_keys import NOT_EXECUTED_META_KEY
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import _ToolBatch
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import ToolCall, ToolResult


@pytest.mark.parametrize("method", ["reject", "reject_listed_first"])
async def test_rejection_marker_precedes_persistence(method):
    recorded = []

    class Recorder(BaseRecorder):
        async def on_message_appended(self, run_id, message, iteration):
            recorded.append(dict(message.meta))

    batch = _ToolBatch(
        strategy=NativeToolCalling(),
        history=[],
        recorder=Recorder(),
        notifier=None,
        run_id="run",
        iteration=1,
        warnings=[],
    )
    call = ToolCall("work", id="call")
    result = ToolResult("work", payload="", call_id=call.id, success=False, error="rejected")
    await getattr(batch, method)(call, result)
    assert len(recorded) == 1
    assert recorded[0][NOT_EXECUTED_META_KEY] is True
