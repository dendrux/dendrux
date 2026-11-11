"""Managed subagents (0.7, T2b step 1): batch preflight and the loop handoff.

A generated subagent call (``ToolTarget.AGENT`` from ``Agent(subagents=...)``) never
enters the parallel server group. Two managed calls, or a managed call plus a
client pending target, are refused before any effect with one stable
``managed_batch_unsupported`` observation per call. A supported batch keeps the
ordinary approval gate; its server calls finish first, then the loop hands the
single managed call to the runner (no client pause).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import MANAGED_HANDOFF_KEY, ManagedHandoff, ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolDecision
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-brief-7f2e"

calls: dict[str, int] = {}
running: dict[str, int] = {"now": 0, "peak": 0}


def _reset() -> None:
    calls.clear()
    running.update(now=0, peak=0)


async def _work(name: str) -> str:
    calls[name] = calls.get(name, 0) + 1
    running["now"] += 1
    running["peak"] = max(running["peak"], running["now"])
    await asyncio.sleep(0.01)
    running["now"] -= 1
    return name


@tool()
async def fetch_a(x: int) -> str:
    """Fetch A."""
    return await _work("fetch_a")


@tool()
async def fetch_b(x: int) -> str:
    """Fetch B."""
    return await _work("fetch_b")


@tool(target="client")
async def read_sheet(sheet: str) -> str:
    """Client tool."""
    return "never"


def _child(name: str = "reviewer") -> Agent:
    return Agent(
        name=name,
        description="Reviews one draft.",
        prompt="Review.",
        provider=ScriptedLLM([LLMResponse(text="child never runs here")]),
    )


def _parent(*subagents: Agent, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="Lead.",
        tools=kwargs.pop("tools", [fetch_a, fetch_b, read_sheet]),
        subagents=list(subagents),
        **kwargs,
    )


def _call(name: str, params: dict[str, Any], pid: str) -> ToolCall:
    return ToolCall(name=name, params=params, provider_tool_call_id=pid)


async def _run(agent: Agent, responses: list[LLMResponse]) -> tuple[Any, ScriptedLLM]:
    _reset()
    llm = ScriptedLLM(responses)
    result = await ReActLoop().run(
        agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
    )
    return result, llm


def _observations(llm: ScriptedLLM, call: int) -> dict[str, dict[str, Any]]:
    out = {}
    for message in llm.messages[call]:
        if message.role.value == "tool":
            out[message.call_id] = json.loads(message.content)
    return out


async def test_subagent_is_offered_and_handed_off_not_paused() -> None:
    child = _child()
    response = LLMResponse(tool_calls=[_call("reviewer", {"task": BRIEF}, "p1")])
    agent = _parent(child)
    offered: list[Any] = []

    class Capturing(ScriptedLLM):
        async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
            offered.extend(tools or [])
            return await super().complete(messages, tools, **kwargs)

    llm = Capturing([response])
    _reset()
    result = await ReActLoop().run(
        agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
    )
    assert result.status == RunStatus.RUNNING
    handoff = result.meta[MANAGED_HANDOFF_KEY]
    assert isinstance(handoff, ManagedHandoff)
    assert handoff.child is child
    assert handoff.call.name == "reviewer"
    assert handoff.call.params == {"task": BRIEF}
    pause = result.meta["pause_state"]
    assert [tc.id for tc in pause.pending_tool_calls] == [handoff.call.id]
    assert pause.pending_targets == {handoff.call.id: "agent"}
    reviewer = next(td for td in offered if td.name == "reviewer")
    assert reviewer.target.value == "agent"
    assert reviewer.parameters["required"] == ["task"]


async def test_agent_target_never_enters_parallel_dispatch() -> None:
    """Spec test 6: server calls keep their group, run once, then the one managed call."""
    child = _child()
    batch = LLMResponse(
        tool_calls=[
            _call("reviewer", {"task": BRIEF}, "p1"),
            _call("fetch_a", {"x": 1}, "p2"),
            _call("fetch_b", {"x": 2}, "p3"),
        ]
    )
    result, llm = await _run(_parent(child), [batch])
    assert calls == {"fetch_a": 1, "fetch_b": 1}
    # The two parallel server calls ran together; the managed call was not in the group.
    assert running["peak"] == 2
    assert result.status == RunStatus.RUNNING
    handoff = result.meta[MANAGED_HANDOFF_KEY]
    assert handoff.call.name == "reviewer"
    history = result.meta["pause_state"].history
    observed = [m.call_id for m in history if m.role.value == "tool"]
    assert len(observed) == 2 and handoff.call.id not in observed
    assert llm.calls == 1


@pytest.mark.parametrize(
    "batch",
    [
        [("reviewer", {"task": BRIEF}), ("reviewer", {"task": BRIEF}), ("fetch_a", {"x": 1})],
        [("reviewer", {"task": BRIEF}), ("auditor", {"task": BRIEF})],
        [("reviewer", {"task": BRIEF}), ("read_sheet", {"sheet": "S"}), ("fetch_a", {"x": 1})],
    ],
    ids=["two-same", "two-different", "managed-plus-client"],
)
async def test_unsupported_batch_refused_per_call_with_zero_effects(
    batch: list[tuple[str, dict[str, Any]]],
) -> None:
    hooked: list[str] = []

    async def before_tool(ctx: Any) -> ToolDecision | None:
        hooked.append(ctx.tool_name)
        return None

    agent = _parent(_child("reviewer"), _child("auditor"), before_tool=before_tool)
    first = LLMResponse(tool_calls=[_call(n, p, f"p{i}") for i, (n, p) in enumerate(batch)])
    result, llm = await _run(agent, [first, LLMResponse(text="ok")])
    assert result.status == RunStatus.SUCCESS
    assert MANAGED_HANDOFF_KEY not in result.meta
    assert calls == {} and hooked == []
    seen = _observations(llm, 1)
    assert len(seen) == len(batch)
    for observation in seen.values():
        assert set(observation) == {"error"} and "subagent" in observation["error"]
    assert BRIEF not in json.dumps(seen)


async def test_unsupported_batch_observations_carry_the_stable_code() -> None:
    recorded: list[tuple[str, Any]] = []

    class Recorder(BaseRecorder):
        async def on_tool_completed(
            self, run_id: str, tool_call: ToolCall, tool_result: Any, iteration: int
        ) -> None:
            recorded.append((tool_call.name, tool_result))

    agent = _parent(_child("reviewer"), _child("auditor"))
    first = LLMResponse(
        tool_calls=[
            _call("reviewer", {"task": BRIEF}, "p1"),
            _call("auditor", {"task": BRIEF}, "p2"),
        ]
    )
    _reset()
    llm = ScriptedLLM([first, LLMResponse(text="ok")])
    await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=Recorder(),
    )
    assert [name for name, _ in recorded] == ["reviewer", "auditor"]
    for _, result in recorded:
        assert result.success is False
        assert result.meta["code"] == "managed_batch_unsupported"


async def test_approval_gate_still_stops_the_whole_supported_batch() -> None:
    agent = _parent(_child(), require_approval=["fetch_a"])
    batch = LLMResponse(
        tool_calls=[_call("fetch_a", {"x": 1}, "p1"), _call("reviewer", {"task": BRIEF}, "p2")]
    )
    result, _ = await _run(agent, [batch])
    assert result.status == RunStatus.WAITING_APPROVAL
    assert MANAGED_HANDOFF_KEY not in result.meta
    assert calls == {}
    pending = result.meta["pause_state"].pending_tool_calls
    assert sorted(tc.name for tc in pending) == ["fetch_a", "reviewer"]


async def test_managed_call_denied_by_hook_is_an_ordinary_denial() -> None:
    async def before_tool(ctx: Any) -> ToolDecision | None:
        return ToolDecision.deny("no") if ctx.tool_name == "reviewer" else None

    agent = _parent(_child(), before_tool=before_tool)
    batch = LLMResponse(tool_calls=[_call("reviewer", {"task": BRIEF}, "p1")])
    result, llm = await _run(agent, [batch, LLMResponse(text="ok")])
    assert result.status == RunStatus.SUCCESS
    assert MANAGED_HANDOFF_KEY not in result.meta
    assert llm.calls == 2


async def test_invalid_brief_is_rejected_before_handoff() -> None:
    agent = _parent(_child())
    batch = LLMResponse(tool_calls=[_call("reviewer", {"task": 3}, "p1")])
    result, llm = await _run(agent, [batch, LLMResponse(text="ok")])
    assert result.status == RunStatus.SUCCESS
    assert MANAGED_HANDOFF_KEY not in result.meta


async def test_tool_defs_offer_the_subagent_as_an_agent_target() -> None:
    agent = _parent(_child())
    lookups = await agent.get_tool_lookups()
    assert lookups.target["reviewer"].value == "agent"
    assert lookups.parallel["reviewer"] is False
    with pytest.raises(RuntimeError):
        lookups.fn["reviewer"]()
