"""T0 move evidence; later owners may delete location checks when reshaping code."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.loops import _managed_batch, _pause_shared, react
from dendrux.runtime import persistence
from dendrux.types import RunStatus, ToolCall, ToolTarget, UsageStats


@pytest.mark.parametrize(
    "name", ["_pause_result", "_build_pause", "_snapshot_usage", "_commentary"]
)
def test_pause_dependencies_live_below_the_facade(name):
    assert Path(getattr(react, name).__code__.co_filename).name == "_pause_shared.py"


@pytest.mark.parametrize("name", ["message_trace", "_message_trace", "_tool_call_evidence"])
def test_trace_dependencies_live_below_the_facade(name):
    assert Path(getattr(persistence, name).__code__.co_filename).name == "_trace_shared.py"


def test_handoff_calls_a_wrapped_owner_pause_patch_once(monkeypatch):
    call = ToolCall(name="child", params={"task": "go"})
    child = object()
    context = SimpleNamespace(
        lookups=SimpleNamespace(
            fn={"child": react.SubagentTool(child)}, target={"child": ToolTarget.AGENT}
        ),
        agent=SimpleNamespace(name="parent"),
        state=SimpleNamespace(history=[], steps=[], usage=UsageStats()),
        budget=None,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        pause_view=lambda: None,
        meta=lambda: {},
    )
    original = _pause_shared._pause_result
    calls = []

    def pause(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(_pause_shared, "_pause_result", pause)
    result = react._managed_handoff(context, react._ToolCallOutcome([], [call]), 3)
    assert result.status == RunStatus.RUNNING
    assert result.meta["pause_state"].pending_tool_calls == [call]
    assert len(calls) == 1


async def test_brief_admission_reads_the_owner_logger_patch(monkeypatch, caplog):
    assert react.logger.name == "dendrux.loops.react"
    assert persistence.logger.name == "dendrux.runtime.persistence"
    assert _managed_batch.logger is react.logger
    patched = logging.getLogger("dendrux.tests.managed_brief_patch")
    monkeypatch.setattr(_managed_batch, "logger", patched)
    call = ToolCall(name="child", params={"task": "too large"})
    child = SimpleNamespace(_limits=SimpleNamespace(max_input_bytes=1))
    lookups = SimpleNamespace(fn={"child": react.SubagentTool(child)})
    batch = SimpleNamespace(run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV", reject=AsyncMock())
    assert await react._admit_managed_briefs([call], lookups, batch) == []
    batch.reject.assert_awaited_once()
    assert [r.name for r in caplog.records if r.code == "input_size_limit"] == [patched.name]
