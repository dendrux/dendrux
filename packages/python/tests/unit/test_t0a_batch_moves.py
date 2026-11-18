"""T0 move evidence; later owners may delete location checks when reshaping code."""

from pathlib import Path

import pytest

from dendrux.loops import _pause_shared, react


@pytest.mark.parametrize(
    "name",
    [
        "_is_managed",
        "_managed_batch_unsupported",
        "_managed_batch_result",
        "_admit_managed_briefs",
        "_brief_too_large",
        "_managed_handoff",
    ],
)
def test_managed_batch_functions_live_in_the_extracted_module(name):
    function = getattr(react, name)
    assert Path(function.__code__.co_filename).name == "_managed_batch.py"
    from dendrux.loops import _managed_batch

    assert function is getattr(_managed_batch, name)


def test_managed_markers_keep_identity_and_original_class_metadata():
    assert Path(react.SubagentTool.__init__.__code__.co_filename).name == "_managed_batch.py"
    from dendrux.loops import _managed_batch

    assert react.SubagentTool is _managed_batch.SubagentTool
    assert react.ManagedHandoff is _managed_batch.ManagedHandoff
    assert react.SubagentTool.__module__ == "dendrux.loops.react"
    assert react.ManagedHandoff.__module__ == "dendrux.loops.react"


def test_managed_handoff_reads_the_owner_pause_patch_seam(monkeypatch):
    from types import SimpleNamespace

    from dendrux.types import RunResult, RunStatus, ToolCall, ToolTarget, UsageStats

    call = ToolCall(name="child", params={"task": "go"})
    child, parent = object(), SimpleNamespace(name="parent")
    context = SimpleNamespace(
        lookups=SimpleNamespace(
            fn={"child": react.SubagentTool(child)}, target={"child": ToolTarget.AGENT}
        ),
        agent=parent,
        state=SimpleNamespace(history=[], steps=[], usage=UsageStats()),
        budget=None,
        run_id="parent",
        pause_view=lambda: None,
        meta=lambda: {},
    )
    result = RunResult(run_id="parent", status=RunStatus.RUNNING)
    seen = []

    def pause(*args, **kwargs):
        seen.append((args, kwargs))
        return result

    monkeypatch.setattr(_pause_shared, "_pause_result", pause)
    assert react._managed_handoff(context, react._ToolCallOutcome([], [call]), 3) is result
    assert seen == [((context, RunStatus.RUNNING, 3), {"pending_calls": [call]})]
    assert result.meta[react.MANAGED_HANDOFF_KEY] == react.ManagedHandoff(call, child, parent)
