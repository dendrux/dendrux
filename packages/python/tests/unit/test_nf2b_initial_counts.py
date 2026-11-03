"""Initialize the resumed loop independently of a recorder's later binding hooks."""

from types import SimpleNamespace

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.runtime.runner import _initial_call_counts_kwargs, _resumed_loop_kwargs
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import LLMResponse, Message, PauseState, Role, RunStatus, ToolCall, UsageStats


class SavedCountsRecorder(BaseRecorder):
    def __init__(self, counts):
        self.counts = counts

    def execution_counts(self):
        return self.counts


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("saved", [None, {}, {"work": 1}])
@pytest.mark.parametrize("history_count", [0, 1])
async def test_resumed_loop_enforces_saved_counts_without_late_binding(
    streaming, saved, history_count
):
    effects = []

    @tool(max_calls_per_run=1)
    async def work() -> str:
        """Record an execution."""
        effects.append(True)
        return "ok"

    history = [Message(Role.TOOL, "prior", name="work", call_id="prior")] * history_count
    pause = PauseState(
        agent_name="Agent",
        pending_tool_calls=[],
        history=history,
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
    )
    recorder = SavedCountsRecorder(saved)
    original = None if saved is None else dict(saved)
    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")])
    loop = ReActLoop()
    ctx = SimpleNamespace(
        pause_state=pause,
        history=history,
        recorder=recorder,
        notifier=None,
        resolved_loop=loop,
        resolved_strategy=NativeToolCalling(),
    )
    method = loop.run_stream if streaming else loop.run
    kwargs = _resumed_loop_kwargs(
        Agent(prompt="p", tools=[work]), provider, None, "run", ctx, None, method
    )
    if streaming:
        events = [event async for event in method(**kwargs)]
        result = events[-1].run_result
    else:
        result = await method(**kwargs)
    assert result.status == RunStatus.SUCCESS
    count = history_count if original is None else original.get("work", 0)
    assert effects == ([] if count else [True])
    assert recorder.counts == original


def test_initial_count_kwargs_keep_legacy_loop_and_recorder_compatibility():
    def legacy_loop(*, initial_history=None):
        pass

    def forwarding_loop(**kwargs):
        pass

    recorder = SavedCountsRecorder({"work": 1})
    assert _initial_call_counts_kwargs(legacy_loop, recorder) == {}
    assert _initial_call_counts_kwargs(forwarding_loop, recorder) == {
        "_initial_call_counts": {"work": 1}
    }
    assert _initial_call_counts_kwargs(ReActLoop().run, BaseRecorder()) == {
        "_initial_call_counts": None
    }
