"""Characterize the policy differences retained across dispatch paths."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux import Agent, Skill, tool
from dendrux._meta_keys import NOT_EXECUTED_META_KEY
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import _hook_tenant_id, _init_loop_state, _process_tool_calls
from dendrux.runtime._reclaim import _dispatch_recovery_call, _recovery_denial
from dendrux.runtime.runner import _approval_call_counts, _execute_approved_tools
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    AgentStep,
    ApprovalDecision,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunStatus,
    ToolCall,
    ToolCallContext,
    ToolDecision,
    ToolResult,
    ToolTarget,
    UsageStats,
)


def pause(calls=(), history=()):
    return PauseState(
        agent_name="Agent",
        pending_tool_calls=list(calls),
        history=list(history),
        steps=[],
        iteration=3,
        trace_order_offset=0,
        usage=UsageStats(),
    )


class RestoredRecorder(BaseRecorder):
    storage_format = 2
    guardrail_engine = None

    def __init__(self, counts):
        self.counts = counts
        self._replay_calls = set()

    def execution_counts(self):
        return self.counts


async def dispatch(
    agent,
    tc,
    *,
    phase="dispatchable",
    kind="dispatch",
    counts=None,
    history=None,
    recorder=None,
    on_work=None,
):
    ctx = SimpleNamespace(
        recorder=recorder or RestoredRecorder({}),
        notifier=None,
        resolved_strategy=NativeToolCalling(),
        pause_state=pause([tc]),
        history=history if history is not None else [],
    )
    return await _dispatch_recovery_call(
        agent,
        ctx,
        "run",
        tc,
        kind,
        phase,
        await agent.get_tool_lookups(),
        counts if counts is not None else {},
        None,
        "tenant",
        on_work,
    )


async def test_static_denial_exact_result_event_and_execution_flag():
    effects = []

    @tool()
    async def work():
        effects.append(True)
        return "done"

    agent = Agent(prompt="test", tools=[work], deny=["work"])
    tc = ToolCall("work", id="call")
    recorder = RestoredRecorder({})
    recorder.on_governance_event = AsyncMock()
    recorder.on_tool_completed = AsyncMock()
    history = []
    live = await _process_tool_calls(
        step=AgentStep(reasoning=None, action=tc),
        call_counts={},
        lookups=await agent.get_tool_lookups(),
        strategy=NativeToolCalling(),
        recorder=recorder,
        notifier=None,
        run_id="run",
        iteration=3,
        history=history,
        warnings=[],
        deny=agent.deny,
        tenant_id="tenant",
    )
    state = pause([tc])
    approved = await _execute_approved_tools(agent, state, recorder, None, "run")
    recovered_history = []
    recovered = await dispatch(agent, tc, history=recovered_history, recorder=recorder)
    message = "Tool 'work' is denied by policy."
    expected = ToolResult(
        name="work",
        call_id="call",
        payload=json.dumps({"denied": message}),
        success=False,
        error=message,
    )
    assert live.all_results == approved.results == recovered.all_results == [(tc, expected)]
    assert history[-1].content == state.history[-1].content == recovered_history[-1].content
    assert history[-1].content == expected.payload
    assert NOT_EXECUTED_META_KEY not in history[-1].meta
    assert NOT_EXECUTED_META_KEY not in recovered_history[-1].meta
    assert state.history[-1].meta[NOT_EXECUTED_META_KEY] is True
    assert recorder.on_governance_event.await_count == 3
    for event in recorder.on_governance_event.call_args_list:
        assert event.args == (
            "run",
            "policy.denied",
            3,
            {"tool_name": "work", "call_id": "call", "reason": "denied_by_policy"},
        )
        assert event.kwargs == {"correlation_id": "call"}
    recorder.on_tool_completed.assert_not_awaited()
    assert effects == []


@pytest.mark.parametrize("reason", [None, 'human says "no"'])
def test_recovery_decision_denial_wraps_reason(reason):
    message = _recovery_denial(ToolCall("work", id="call"), reason)
    assert message.content == json.dumps({"denied": reason})
    assert message.name == "work" and message.call_id == "call"
    assert message.meta["is_error"] is True


@pytest.mark.parametrize("restored", [None, {}, {"work": 1}])
async def test_restored_counts_win_only_in_approved_path(restored):
    history = [
        Message(Role.USER, "input"),
        Message(Role.TOOL, "unnamed", name="unused", call_id="u"),
        Message(Role.TOOL, "a", name="work", call_id="a"),
        Message(Role.TOOL, "b", name="work", call_id="b", meta={NOT_EXECUTED_META_KEY: False}),
        Message(Role.TOOL, "denied", name="work", call_id="b", meta={NOT_EXECUTED_META_KEY: True}),
        Message(Role.TOOL, "empty", name="unused", call_id="e"),
    ]
    # Exercise the defensive predicates on malformed/legacy history, too.
    object.__setattr__(history[1], "name", None)
    object.__setattr__(history[-1], "name", "")
    recorder = RestoredRecorder(restored)
    live = await _init_loop_state(
        user_input="",
        recorder=recorder,
        notifier=None,
        run_id="run",
        initial_history=history,
        initial_steps=None,
        initial_usage=None,
    )
    assert live.call_counts == {"work": 2, "": 1}
    approved = _approval_call_counts(recorder, pause(history=history))
    assert approved == ({"work": 2} if restored is None else restored)
    if restored is not None:
        assert approved is restored
    assert _approval_call_counts(BaseRecorder(), pause(history=history)) == {"work": 2}


async def test_approved_execution_uses_restored_count_instead_of_history():
    effects = []

    @tool(max_calls_per_run=2)
    async def work():
        effects.append(True)
        return "ok"

    state = pause([ToolCall("work")], [Message(Role.TOOL, "old", name="work", call_id="old")] * 3)
    recorder = RestoredRecorder({"work": 1})
    result = await _execute_approved_tools(
        Agent(prompt="p", tools=[work]), state, recorder, None, "run"
    )
    assert result.results[0][1].success
    assert effects == [True] and recorder.counts == {"work": 2}


async def test_approved_skill_dispatch_keeps_policy_and_strategy_choices(monkeypatch):
    from dendrux.loops import react
    from dendrux.skills._session import SkillTool

    @tool()
    async def work():
        return "ok"

    hook = Mock()
    agent = Agent(prompt="p", tools=[work], deny=["work"], before_tool=hook)
    lookups = await agent.get_tool_lookups()
    lookups.fn["work"] = SkillTool(SimpleNamespace(), "work")
    monkeypatch.setattr(agent, "get_tool_lookups", AsyncMock(return_value=lookups))
    process = AsyncMock(return_value=SimpleNamespace(all_results=[]))
    monkeypatch.setattr(react, "_process_tool_calls", process)
    await _execute_approved_tools(agent, pause([ToolCall("work")]), BaseRecorder(), None, "run")
    process.assert_awaited_once()
    kwargs = process.call_args.kwargs
    assert kwargs["deny"] == frozenset({"work"})
    assert kwargs["tenant_id"] is None
    assert type(kwargs["strategy"]) is NativeToolCalling
    assert kwargs["warnings"] == []
    assert kwargs["approved_skills"] is True
    assert not kwargs.get("require_approval")
    assert kwargs.get("before_tool") is None
    hook.assert_not_called()


@pytest.mark.parametrize("storage_format", [1, 2])
async def test_approved_decision_denial_keeps_wrapped_reason(storage_format):
    @tool()
    async def work():
        pytest.fail("a rejected decision must not execute")

    tc = ToolCall("work", id="call")
    recorder = RestoredRecorder({})
    recorder.on_tool_started = AsyncMock()
    result = await _execute_approved_tools(
        Agent(prompt="p", tools=[work]),
        pause([tc]),
        recorder,
        None,
        "run",
        decisions={tc.id: ApprovalDecision(approved=False, reason="human_denied")},
        state_store=SimpleNamespace(storage_format=storage_format),
    )
    expected = json.dumps({"denied": "human_denied"})
    assert result.results[0][1].payload == expected
    assert _recovery_denial(tc, "human_denied").content == expected
    assert recorder.on_tool_started.await_count == (storage_format == 1)


@pytest.mark.parametrize("barrier", [RunStatus.CANCELLED, RunStatus.PAUSED])
async def test_approved_barrier_keeps_remaining_calls(monkeypatch, barrier):
    from dendrux.runtime import runner

    @tool()
    async def work():
        pytest.fail("the barrier must run before execution")

    monkeypatch.setattr(runner, "_resume_barrier", AsyncMock(return_value=barrier))
    calls = [ToolCall("work", id="first"), ToolCall("work", id="second")]
    result = await _execute_approved_tools(
        Agent(prompt="p", tools=[work]),
        pause(calls),
        BaseRecorder(),
        None,
        "run",
    )
    assert result.barrier == barrier and result.results == []
    assert [item[0] for item in result.remaining] == calls


async def test_invalid_skill_activation_precedes_policy():
    hook = Mock()
    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("use_skill", params={"name": []})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        before_tool=hook,
        skills=[Skill(name="demo", description="d", body="b")],
    )
    result = await agent.run("go")
    assert result.status == RunStatus.SUCCESS
    message = next(m for m in provider.call_history[-1]["messages"] if m.role == Role.TOOL)
    assert message.meta["is_error"] is True
    hook.assert_not_called()


@pytest.mark.parametrize("has_hook,has_store", [(False, False), (False, True), (True, False)])
async def test_hook_tenant_lookup_is_lazy(has_hook, has_store):
    store = SimpleNamespace(get_run=AsyncMock())
    assert (
        await _hook_tenant_id(Mock() if has_hook else None, store if has_store else None, "run")
        is None
    )
    store.get_run.assert_not_awaited()


@pytest.mark.parametrize("record,expected", [(None, None), (SimpleNamespace(tenant_id="t"), "t")])
async def test_hook_tenant_lookup_reads_once(record, expected):
    store = SimpleNamespace(get_run=AsyncMock(return_value=record))
    assert await _hook_tenant_id(Mock(), store, "run") == expected
    store.get_run.assert_awaited_once_with("run")


@pytest.mark.parametrize(
    "phase,kind,hooked,gated",
    [
        ("dispatchable", "dispatch", True, False),
        ("dispatchable", "approval", True, True),
        ("approved", "approval", False, False),
        ("approved", "replay", False, False),
    ],
)
async def test_direct_recovery_dispatch_context_and_approval_rules(phase, kind, hooked, gated):
    contexts, effects = [], []

    @tool()
    async def work(value: str):
        effects.append(value)
        return "ok"

    def policy(ctx):
        contexts.append(ctx)
        return ToolDecision.allow()

    agent = Agent(
        prompt="p",
        tools=[work],
        before_tool=policy,
        require_approval=["work"] if phase == "approved" else [],
    )
    tc = ToolCall("work", params={"value": "x"}, id="call")
    recorder = RestoredRecorder({})
    on_work = Mock()
    outcome = await dispatch(agent, tc, phase=phase, kind=kind, recorder=recorder, on_work=on_work)
    expected = ToolCallContext(
        run_id="run",
        call_id="call",
        tool_name="work",
        params={"value": "x"},
        target=ToolTarget.SERVER,
        source="local",
        meta={},
        iteration=3,
        tenant_id="tenant",
        images=(),
    )
    assert contexts == ([expected] if hooked else [])
    assert outcome.pending_approval is gated
    assert effects == ([] if gated else ["x"])
    assert recorder._replay_calls == ({"call"} if kind == "replay" else set())
    on_work.assert_called_once_with()
