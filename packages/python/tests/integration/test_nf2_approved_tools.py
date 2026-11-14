"""Approved observations retain the resumed strategy and notifier diagnostics."""

import pytest

from dendrux import tool
from dendrux.loops.base import BaseNotifier
from dendrux.runtime import _run_resume_prep
from dendrux.strategies.base import Strategy
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    ApprovalDecision,
    LLMResponse,
    Message,
    Role,
    RunStatus,
    ToolCall,
)
from tests.integration.test_skills_v2 import agent

pytestmark = pytest.mark.area_runtime


class ObservationStrategy(Strategy):
    def build_messages(self, **kwargs):
        return NativeToolCalling().build_messages(**kwargs)

    def parse_response(self, response):
        return NativeToolCalling().parse_response(response)

    def format_tool_result(self, result):
        return Message(Role.USER, f"Observation {result.name}: {result.payload}")


class FailingNotifier(BaseNotifier):
    def __init__(self):
        self.finished = []

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
        raise RuntimeError("notifier unavailable")

    async def on_run_finished(self, run_id, result):
        self.finished.append(result)


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_approval"])
@pytest.mark.parametrize("kind", ["skill", "server", "denied", "invalid"])
async def test_approved_results_use_resolved_strategy(store, entry, kind, monkeypatch):
    @tool()
    async def work():
        return "server result"

    call = ToolCall("use_skill", params={"name": "demo"}) if kind == "skill" else ToolCall("work")
    a = agent(
        store,
        [LLMResponse(tool_calls=[call])],
        tools=[work],
        require_approval=[call.name],
    )
    paused = await a.run("input")
    if kind == "invalid":
        # A definition can change while approval is pending. The saved call
        # must become a strategy-formatted validation failure, never execute.
        @tool()
        async def work(value: str):
            pytest.fail("a saved call missing the new required argument must not execute")

    from dendrux.runtime import runner

    prepare = runner._prepare_resume

    async def with_strategy(*args, **kwargs):
        # Agent's facade does not expose runner's strategy keyword. Exercise
        # each public entry with the real resumed context at this seam.
        kwargs["strategy"] = ObservationStrategy()
        return await prepare(*args, **kwargs)

    monkeypatch.setattr(_run_resume_prep, "_prepare_resume", with_strategy)
    b = agent(store, [LLMResponse(text="done")], tools=[work])
    scope = {"expected_pause_id": paused.pause_id}
    if kind == "denied":
        # Decisions are submitted through the public per-call approval path.
        if entry != "submit_approval":
            pytest.skip("per-call denials enter through submit_approval")
        result = await b.submit_approval(
            paused.run_id, decisions={call.id: ApprovalDecision.deny("no")}, **scope
        )
    elif entry == "resume_stream":
        async with b.resume_stream(paused.run_id, **scope) as stream:
            async for _ in stream:
                pass
        result = stream.result
    elif entry == "resume":
        result = await b.resume(paused.run_id, **scope)
    else:
        result = await b.submit_approval(paused.run_id, approved=True, **scope)
    assert result.status == RunStatus.SUCCESS
    messages = b._provider.call_history[-1]["messages"]
    observation = next(
        (m for m in messages if m.content.startswith(f"Observation {call.name}:")), None
    )
    assert observation is not None
    assert observation.role == Role.USER
    traces = await store.get_traces(paused.run_id)
    assert any(t.role == "user" and t.content == observation.content for t in traces)


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_approval"])
@pytest.mark.parametrize("skill", [False, True])
async def test_approved_notifier_warnings_reach_result(store, entry, skill):
    @tool()
    async def work():
        return "server result"

    call = ToolCall("use_skill", params={"name": "demo"}) if skill else ToolCall("work")
    a = agent(
        store,
        [LLMResponse(tool_calls=[call])],
        tools=[work],
        require_approval=[call.name],
    )
    paused = await a.run("input")
    # A later loop warning must be preserved alongside the approved-tool warning.
    b = agent(
        store,
        [LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")],
        tools=[work],
    )
    notifier = FailingNotifier()
    scope = {"expected_pause_id": paused.pause_id, "notifier": notifier}
    if entry == "resume_stream":
        async with b.resume_stream(paused.run_id, **scope) as stream:
            async for _ in stream:
                pass
        result = stream.result
    elif entry == "resume":
        result = await b.resume(paused.run_id, **scope)
    else:
        result = await b.submit_approval(paused.run_id, approved=True, **scope)
    assert result.status == RunStatus.SUCCESS
    warnings = result.meta.get("notifier_warnings", [])
    assert len(warnings) == 2
    assert all(warning.startswith("on_tool_completed failed at iteration ") for warning in warnings)
    assert notifier.finished[-1].meta["notifier_warnings"] == warnings
