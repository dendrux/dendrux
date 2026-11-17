"""Failure propagation and tool completion preserve their observable ordering."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.loops import react
from dendrux.loops._helpers import _BlockedObservationError
from dendrux.runtime import _run_events, runner
from dendrux.types import Message, Role, ToolCall, ToolResult


@pytest.mark.parametrize("strict", [False, True])
async def test_failure_persistence_preserves_strict_write_error(monkeypatch, strict):
    error = RuntimeError("write_failed")
    emit = AsyncMock()
    monkeypatch.setattr(_run_events, "_finalize_running_event", AsyncMock(side_effect=error))
    monkeypatch.setattr(_run_events, "_emit_event", emit)
    if strict:
        with pytest.raises(RuntimeError) as caught:
            await runner._persist_run_failure(
                object(), "run", None, payload={}, finalize={}, strict=True
            )
        assert caught.value is error
    else:
        assert not await runner._persist_run_failure(object(), "run", None, payload={}, finalize={})
    emit.assert_not_awaited()


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
async def test_recorded_tool_completion_notifies_the_retained_result(monkeypatch, tracked, blocked):
    call = ToolCall("lookup")
    raw = ToolResult(call_id=call.id, name=call.name, payload="value")
    safe = ToolResult(call_id=call.id, name=call.name, payload="[blocked: 1 finding]")
    tracker = SimpleNamespace(result=None, recorded=False) if tracked else None
    rejection = _BlockedObservationError(
        Message(Role.TOOL, safe.payload, name=call.name, call_id=call.id), [], "blocked", safe
    )
    events = []

    async def record(*args):
        events.append("record")
        if tracker is not None:
            assert tracker.result is raw and tracker.recorded is False
        if blocked:
            raise rejection

    async def notify(notifier, run_id, tc, result, iteration, warnings):
        events.append("notify")
        assert result is (safe if blocked else raw)
        if tracker is not None:
            assert tracker.result is result and tracker.recorded is True

    monkeypatch.setattr(react, "_record_tool_started", AsyncMock())
    monkeypatch.setattr(react, "_notify_tool_started", AsyncMock())
    monkeypatch.setattr(react, "_execute_tool", AsyncMock(return_value=raw))
    monkeypatch.setattr(react, "_record_tool", record)
    monkeypatch.setattr(react, "_notify_tool", notify)
    invoke = react._execute_record_notify(call, None, None, None, "run", 2, [], tracker=tracker)
    if blocked:
        with pytest.raises(_BlockedObservationError) as caught:
            await invoke
        assert caught.value is rejection
    else:
        assert await invoke == (call, raw)
    assert events == ["record", "notify"]
