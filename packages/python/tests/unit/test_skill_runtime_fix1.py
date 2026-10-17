"""Regression contracts from the independent Skills runtime review."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, Skill, SkillExecutionResult, SkillExecutor, tool
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.runner import _execute_approved_tools
from dendrux.skills._session import SkillSession, current
from dendrux.types import LLMResponse, Message, PauseState, Role, ToolCall, UsageStats


def package():
    return Skill(
        name="demo",
        description="description",
        body="body",
        files={"ref": "resource", "scripts/a.py": "code"},
    )


class Reads(MockLLM):
    def __init__(self):
        super().__init__([])
        self.seen = []
        self.path = "ref"

    async def complete(self, messages, tools=None, **kwargs):
        results = [m for m in messages if m.role == Role.TOOL and m.name == "read_skill_resource"]
        if results:
            self.seen.append((kwargs["run_id"], results[-1].meta.get("is_error", False)))
            return LLMResponse(text="done")
        return LLMResponse(
            tool_calls=[
                ToolCall(name="read_skill_resource", params={"skill": "demo", "path": self.path})
            ]
        )


@pytest.mark.parametrize("different_agents", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
async def test_interleaved_streams_own_session(different_agents, reverse):
    p = Reads()
    a = Agent(
        prompt="test",
        provider=p,
        skills=[Skill(name="demo", description="d", body="b", files={"ref": "resource"})],
    )
    b = Agent(prompt="other", provider=p, skills=a._explicit_skills) if different_agents else a
    streams = [a.stream("A", activate_skills=["demo"]), b.stream("B")]
    iterators = [s.__aiter__() for s in streams]
    stale = []
    done = set()
    try:
        while len(done) != 2:
            for i in [1, 0] if reverse else [0, 1]:
                if i in done:
                    continue
                try:
                    await anext(iterators[i])
                    stale.append((current(), get_delegation_context()))
                except StopAsyncIteration:
                    done.add(i)
        assert dict(p.seen) == {streams[0].run_id: False, streams[1].run_id: True}
        assert all(s is None and d is None for s, d in stale)
        assert current() is None and get_delegation_context() is None
    finally:
        for stream in streams:
            await stream.aclose()


@pytest.mark.parametrize("name", ["use_skill", "read_skill_resource", "run_skill_script"])
async def test_non_skill_reserved_name_is_developer_tool(name):
    async def fn(value: str):
        return f"developer {value}"

    fn.__name__ = name
    p = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name=name, params={"value": "x"})]),
            LLMResponse(text="done"),
        ]
    )
    a = Agent(prompt="test", provider=p, tools=[tool()(fn)])
    await a.run("input")
    result = next(m for m in p.call_history[-1]["messages"] if m.role == Role.TOOL)
    assert not result.meta.get("is_error") and json.loads(result.content) == "developer x"


@pytest.mark.parametrize("policy", ["plain", "deny", "max_calls"])
async def test_approved_non_skill_calls_match_main(policy):
    order = []

    @tool(max_calls_per_run=1)
    async def srv():
        order.append("srv-start")
        await asyncio.sleep(0)
        order.append("srv-end")
        return "server"

    @tool(target="client")
    async def cli():
        order.append("cli")
        return "client-stub"

    calls = [ToolCall(name="srv", params={}), ToolCall(name="cli", params={})]
    history = (
        [Message(role=Role.TOOL, name="srv", call_id="prior", content='"prior"')]
        if policy == "max_calls"
        else []
    )
    state = PauseState(
        agent_name="Agent",
        pending_tool_calls=calls,
        history=history,
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
    )
    a = Agent(prompt="test", tools=[srv, cli], deny=["srv"] if policy == "deny" else None)
    approval = await _execute_approved_tools(a, state, BaseRecorder(), None, "run")
    assert [(tc.name, tr.success, json.loads(tr.payload)) for tc, tr in approval.results] == [
        ("srv", True, "server"),
    ]
    # A client-target call is never executed server-side on approval (A-P0-1).
    assert [tc.name for tc in approval.client_pending] == ["cli"]
    assert order == ["srv-start", "srv-end"]
    assert [m.name for m in state.history[-1:]] == ["srv"]


@pytest.mark.parametrize("outcome", [None, {"stdout": "x"}, "own_timeout"])
async def test_bad_executor_outcome_is_tool_failure(outcome):
    class Executor(SkillExecutor):
        async def execute(self, request):
            if outcome == "own_timeout":
                raise TimeoutError("executor timeout")
            return outcome

    a = Agent(prompt="test", skills=[package()], skill_executor=Executor())
    s = SkillSession.prepare(a, explicit=["demo"])
    s.sink = AsyncMock()
    result = await s.execute(
        ToolCall(
            name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
        )
    )
    assert not result.success
    assert s.sink.call_args.args[1]["error_type"] == (
        "TimeoutError" if outcome == "own_timeout" else "invalid_result"
    )


async def test_unadvertised_script_has_no_unavailable_event():
    with pytest.warns(UserWarning):
        a = Agent(prompt="test", skills=[package()])
    s = SkillSession.prepare(a, explicit=["demo"])
    s.sink = AsyncMock()
    result = await s.execute(
        ToolCall(
            name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
        )
    )
    assert not result.success
    assert not s.sink.call_args_list


async def test_request_event_once_per_call_id():
    class Executor(SkillExecutor):
        async def execute(self, request):
            return SkillExecutionResult()

    s = SkillSession.prepare(
        Agent(prompt="test", skills=[package()], skill_executor=Executor()), explicit=["demo"]
    )
    s.sink = AsyncMock()
    call = ToolCall(
        name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
    )
    await s.preflight(call, requested=True)
    await s.preflight(call, requested=True)
    assert s.sink.call_count == 1


async def test_activation_failure_label_is_bounded_and_console_safe(capsys):
    from dendrux.notifiers.console import ConsoleNotifier

    s = SkillSession.prepare(
        Agent(prompt="test", skills=[Skill(name="demo", description="d", body="b")])
    )
    s.sink = AsyncMock()
    await s.execute(ToolCall(name="use_skill", params={"name": ["[/x]" * 1000]}))
    data = s.sink.call_args.args[1]
    assert isinstance(data["name"], str) and len(data["name"]) <= 256
    await ConsoleNotifier().on_governance_event(
        "run", "skill.activation_failed", 0, {"name": "[/x]", "reason": "not_found"}
    )
    assert "[/x]" in capsys.readouterr().out


def test_restore_unions_explicit_multiple_successes_and_reactivation():
    a = Agent(
        prompt="test",
        skills=[
            Skill(name=n, description="d", body="b") for n in ["explicit", "one", "two", "failed"]
        ],
    )
    s = SkillSession.prepare(a, explicit=["explicit"])
    calls = [
        ToolCall(name="use_skill", params={"name": n}, id=str(i))
        for i, n in enumerate(["one", "two", "failed", "one"])
    ]
    history = [Message(role=Role.ASSISTANT, content="", tool_calls=calls)]
    history += [
        Message(
            role=Role.TOOL,
            name="use_skill",
            call_id=c.id,
            content="unparsed result",
            meta={"is_error": c.params["name"] == "failed"},
        )
        for c in calls
    ]
    s.restore(history)
    assert s.activated == {"explicit", "one", "two"}


async def test_singlecall_run_override_rejects_before_row():
    from dendrux.loops.single import SingleCall

    store = AsyncMock()
    a = Agent(
        prompt="test",
        provider=MockLLM([]),
        state_store=store,
        skills=[Skill(name="demo", description="d", body="b", files={"ref": "x"})],
    )
    with pytest.raises(ValueError, match="SingleCall"):
        await a.run("input", loop=SingleCall())
    store.create_run.assert_not_called()


async def test_denied_catalogue_is_hidden_from_actual_model_request():
    skills = [
        Skill(name=n, description=f"description-{n}", body="body", files={"ref": "resource"})
        for n in ["demo", "other"]
    ]
    p = MockLLM([LLMResponse(text="done")])
    a = Agent(prompt="base", provider=p, skills=skills, deny=["use_skill"])
    await a.run("input", activate_skills=["demo"])
    call = p.call_history[0]
    assert call["messages"][0].content == "base"
    assert {td.name for td in call["tools"]} == {"read_skill_resource"}
    assert call["tools"][0].parameters["properties"]["skill"]["enum"] == ["demo"]


async def test_stream_close_from_another_task_restores_context():
    p = MockLLM([LLMResponse(text="done")])
    a = Agent(prompt="test", provider=p, skills=[Skill(name="demo", description="d", body="b")])
    stream = a.stream("input", activate_skills=["demo"])
    iterator = stream.__aiter__()
    await anext(iterator)
    assert current() is None and get_delegation_context() is None
    await asyncio.create_task(stream.aclose())
    assert current() is None and get_delegation_context() is None
