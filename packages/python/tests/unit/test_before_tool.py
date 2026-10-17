"""Tests for the ``before_tool`` policy hook (Track A2 + A-P1-5)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from dendrux import ToolCallContext, ToolDecision
from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop, _call_before_tool, _tool_source
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.tools import build_tool_lookups
from dendrux.types import (
    LLMResponse,
    RunEventType,
    RunStatus,
    ToolCall,
    ToolDef,
    ToolTarget,
)

EXECUTED: list[tuple[str, Any]] = []


@tool()
async def send_email(to: list[str], body: dict) -> str:
    """Send an email."""
    EXECUTED.append(("send_email", {"to": to, "body": body}))
    return "sent"


@tool()
async def search(query: str) -> str:
    """Search."""
    EXECUTED.append(("search", query))
    return f"results for {query}"


@tool()
async def refund(order_id: int) -> str:
    """Refund an order."""
    EXECUTED.append(("refund", order_id))
    return "refunded"


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Client-side read."""
    EXECUTED.append(("read_range", sheet))
    return "client"


@pytest.fixture(autouse=True)
def _reset() -> None:
    EXECUTED.clear()


class SpyRecorder(BaseRecorder):
    def __init__(self) -> None:
        self.governance: list[tuple[str, dict[str, Any]]] = []
        self.tools: list[tuple[ToolCall, Any]] = []

    async def on_governance_event(
        self, run_id, event_type, iteration, data, correlation_id=None
    ) -> None:
        self.governance.append((event_type, data))

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
        self.tools.append((tool_call, tool_result))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [d for e, d in self.governance if e == name]


def _agent(hook, **kw) -> Agent:
    defaults: dict[str, Any] = {
        "prompt": "test",
        "tools": [send_email, search, refund, read_range],
        "before_tool": hook,
    }
    defaults.update(kw)
    return Agent(**defaults)


def _llm(*calls: ToolCall, final: str = "done") -> MockLLM:
    return MockLLM([LLMResponse(tool_calls=list(calls)), LLMResponse(text=final)])


async def _run(agent: Agent, llm: MockLLM, recorder: SpyRecorder | None = None):
    return await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=recorder,
    )


def _tool_msgs(llm: MockLLM, call_index: int = 1) -> list[Any]:
    return [m for m in llm.call_history[call_index]["messages"] if m.role.value == "tool"]


# ------------------------------------------------------------------
# Construction + types
# ------------------------------------------------------------------


class TestConstruction:
    def test_non_callable_rejected(self) -> None:
        with pytest.raises(ValueError, match="before_tool must be callable"):
            _agent("nope")

    def test_single_call_rejected(self) -> None:
        with pytest.raises(ValueError, match="SingleCall"):
            Agent(prompt="p", loop=SingleCall(), before_tool=lambda c: None)

    def test_property(self) -> None:
        def hook(call):
            return None

        assert _agent(hook).before_tool is hook
        assert Agent(prompt="p", tools=[search]).before_tool is None

    def test_decision_constructors(self) -> None:
        assert ToolDecision.allow().kind == "allow"
        assert ToolDecision.deny("x") == ToolDecision(kind="deny", reason="x")
        assert ToolDecision.require_approval().reason is None
        assert ToolDecision.substitute({"a": 1}).result == {"a": 1}
        with pytest.raises(ValueError):
            ToolDecision.deny("")


# ------------------------------------------------------------------
# Decisions
# ------------------------------------------------------------------


class TestDecisions:
    async def test_none_and_allow_execute(self) -> None:
        for decision in (None, ToolDecision.allow()):
            EXECUTED.clear()
            llm = _llm(ToolCall(name="search", params={"query": "q"}))
            result = await _run(_agent(lambda c, d=decision: d), llm)
            assert result.status == RunStatus.SUCCESS
            assert EXECUTED == [("search", "q")]

    async def test_deny(self) -> None:
        rec = SpyRecorder()
        llm = _llm(ToolCall(name="search", params={"query": "q"}))
        result = await _run(_agent(lambda c: ToolDecision.deny("Tools locked")), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert EXECUTED == []
        msg = _tool_msgs(llm)[0]
        assert json.loads(msg.content) == {"denied": "Tools locked"}
        [event] = rec.events("policy.denied")
        assert event["source"] == "hook"
        assert event["reason"] == "Tools locked"
        assert "params" not in event
        # Like static deny: not an execution, no tool_calls row.
        assert rec.tools == []

    async def test_require_approval_server_pauses(self) -> None:
        rec = SpyRecorder()
        tc_search = ToolCall(name="search", params={"query": "q"})
        tc_mail = ToolCall(name="send_email", params={"to": ["a"] * 11, "body": {}})
        llm = MockLLM([LLMResponse(tool_calls=[tc_search, tc_mail])])

        def hook(call: ToolCallContext):
            if call.tool_name == "send_email" and len(call.params["to"]) > 10:
                return ToolDecision.require_approval("More than 10 recipients")
            return None

        result = await _run(_agent(hook), llm, rec)
        assert result.status == RunStatus.WAITING_APPROVAL
        assert EXECUTED == []
        pause = result.meta["pause_state"]
        assert [tc.name for tc in pause.pending_tool_calls] == ["search", "send_email"]
        assert pause.approval_gates == {
            tc_mail.id: {"source": "hook", "reason": "More than 10 recipients"}
        }
        [event] = rec.events("approval.requested")
        assert event == {
            "tool_name": "send_email",
            "call_id": tc_mail.id,
            "reason": "More than 10 recipients",
            "source": "hook",
        }
        assert pause.to_dict()["approval_gates"] == pause.approval_gates

    async def test_require_approval_on_client_is_deny_with_warning(self) -> None:
        rec = SpyRecorder()
        llm = _llm(ToolCall(name="read_range", params={"sheet": "S"}))
        result = await _run(_agent(lambda c: ToolDecision.require_approval()), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert EXECUTED == []
        assert json.loads(_tool_msgs(llm)[0].content) == {
            "denied": "approval is not supported for client tools"
        }
        assert rec.events("policy.hook_warning")[0]["tool_name"] == "read_range"
        assert rec.events("policy.denied")[0]["source"] == "hook"
        assert rec.events("approval.requested") == []

    async def test_substitute(self) -> None:
        rec = SpyRecorder()
        llm = _llm(ToolCall(name="refund", params={"order_id": 1}))
        hook = lambda c: ToolDecision.substitute({"status": "held_for_review"})  # noqa: E731
        result = await _run(_agent(hook), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert EXECUTED == []
        assert json.loads(_tool_msgs(llm)[0].content) == {"status": "held_for_review"}
        [(tc, tr)] = rec.tools
        assert tr.success is True
        assert tr.meta == {"substituted": True}
        assert rec.events("policy.substituted") == [{"tool_name": "refund", "call_id": tc.id}]

    async def test_substitute_non_json_fails_closed(self) -> None:
        rec = SpyRecorder()
        llm = _llm(ToolCall(name="refund", params={"order_id": 1}))
        result = await _run(_agent(lambda c: ToolDecision.substitute(object())), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert EXECUTED == []
        assert json.loads(_tool_msgs(llm)[0].content) == {"denied": "policy hook error"}
        assert rec.events("policy.hook_error") == [
            {
                "tool_name": "refund",
                "call_id": rec.events("policy.denied")[0]["call_id"],
                "error_type": "TypeError",
            }
        ]
        assert rec.events("policy.substituted") == []
        assert rec.tools == []


# ------------------------------------------------------------------
# Fail-closed + isolation
# ------------------------------------------------------------------


class TestFailClosed:
    async def test_hook_raises_denies_and_continues(self) -> None:
        rec = SpyRecorder()
        tc1 = ToolCall(name="refund", params={"order_id": 1})
        tc2 = ToolCall(name="search", params={"query": "q"})
        llm = _llm(tc1, tc2)

        def hook(call):
            if call.tool_name == "refund":
                raise ValueError("secret detail must not leak")
            return None

        result = await _run(_agent(hook), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert EXECUTED == [("search", "q")]
        [err] = rec.events("policy.hook_error")
        assert err == {"tool_name": "refund", "call_id": tc1.id, "error_type": "ValueError"}
        assert "secret" not in json.dumps(rec.governance)
        contents = {m.call_id: json.loads(m.content) for m in _tool_msgs(llm)}
        assert contents[tc1.id] == {"denied": "policy hook error"}

    async def test_wrong_return_type_fails_closed(self) -> None:
        rec = SpyRecorder()
        llm = _llm(ToolCall(name="search", params={"query": "q"}))
        result = await _run(_agent(lambda c: "allow"), llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert EXECUTED == []
        assert rec.events("policy.hook_error")[0]["error_type"] == "TypeError"

    async def test_cancelled_error_not_swallowed(self) -> None:
        def hook(call):
            raise asyncio.CancelledError

        llm = _llm(ToolCall(name="search", params={"query": "q"}))
        with pytest.raises(asyncio.CancelledError):
            await _run(_agent(hook), llm)
        assert EXECUTED == []

    async def test_keyboard_interrupt_not_swallowed(self) -> None:
        def hook(call):
            raise KeyboardInterrupt

        llm = _llm(ToolCall(name="search", params={"query": "q"}))
        with pytest.raises(KeyboardInterrupt):
            await _run(_agent(hook), llm)

    async def test_mutation_does_not_affect_execution(self) -> None:
        seen_meta: list[dict] = []

        def hook(call: ToolCallContext):
            call.params["to"].append("attacker@example.com")
            call.params["body"]["text"] = "rewritten"
            call.params["extra"] = 1
            call.meta["poisoned"] = True
            seen_meta.append(call.meta)
            return None

        tc = ToolCall(name="send_email", params={"to": ["a@x"], "body": {"text": "hi"}})
        llm = _llm(tc)
        agent = _agent(hook)
        await _run(agent, llm)
        assert EXECUTED == [("send_email", {"to": ["a@x"], "body": {"text": "hi"}})]
        assert tc.params == {"to": ["a@x"], "body": {"text": "hi"}}
        lookups = await agent.get_tool_lookups()
        assert "poisoned" not in lookups.meta["send_email"]

    async def test_async_hook(self) -> None:
        async def hook(call):
            await asyncio.sleep(0)
            return ToolDecision.deny("async no")

        llm = _llm(ToolCall(name="search", params={"query": "q"}))
        await _run(_agent(hook), llm)
        assert EXECUTED == []
        assert json.loads(_tool_msgs(llm)[0].content) == {"denied": "async no"}


# ------------------------------------------------------------------
# Static policy precedence
# ------------------------------------------------------------------


class TestStaticPrecedence:
    async def test_static_deny_first_hook_not_called(self) -> None:
        calls: list[str] = []

        def hook(call):
            calls.append(call.tool_name)
            return ToolDecision.allow()

        llm = _llm(ToolCall(name="refund", params={"order_id": 1}))
        await _run(_agent(hook, deny=["refund"]), llm)
        assert calls == []
        assert EXECUTED == []

    async def test_static_gate_survives_allow_and_substitute(self) -> None:
        for decision in (ToolDecision.allow(), ToolDecision.substitute("x")):
            tc = ToolCall(name="refund", params={"order_id": 1})
            llm = MockLLM([LLMResponse(tool_calls=[tc])])
            agent = _agent(lambda c, d=decision: d, require_approval=["refund"])
            result = await _run(agent, llm)
            assert result.status == RunStatus.WAITING_APPROVAL
            gates = result.meta["pause_state"].approval_gates
            assert gates == {tc.id: {"source": "static", "reason": "requires_approval"}}

    async def test_hook_deny_overrides_static_gate(self) -> None:
        rec = SpyRecorder()
        llm = _llm(ToolCall(name="refund", params={"order_id": 1}))
        agent = _agent(lambda c: ToolDecision.deny("locked"), require_approval=["refund"])
        result = await _run(agent, llm, rec)
        assert result.status == RunStatus.SUCCESS
        assert rec.events("approval.requested") == []
        assert EXECUTED == []

    async def test_static_approval_event_payload_unchanged(self) -> None:
        rec = SpyRecorder()
        tc = ToolCall(name="refund", params={"order_id": 1})
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        await _run(_agent(lambda c: None, require_approval=["refund"]), llm, rec)
        assert rec.events("approval.requested") == [
            {"tool_name": "refund", "call_id": tc.id, "reason": "requires_approval"}
        ]


# ------------------------------------------------------------------
# Context contents + sources
# ------------------------------------------------------------------


class TestContext:
    async def test_context_fields(self) -> None:
        seen: list[ToolCallContext] = []

        def hook(call):
            seen.append(call)
            return None

        tc = ToolCall(name="read_range", params={"sheet": "S"})
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        result = await ReActLoop().run(
            agent=_agent(hook),
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            run_id="run-1",
        )
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        [ctx] = seen
        assert (ctx.run_id, ctx.call_id, ctx.tool_name) == ("run-1", tc.id, "read_range")
        assert ctx.target == ToolTarget.CLIENT == "client"
        assert ctx.source == "local"
        assert ctx.iteration == 1
        assert ctx.tenant_id is None
        assert ctx.params == {"sheet": "S"}

    def test_tool_source_classification(self) -> None:
        mcp_def = ToolDef(
            name="gh__create_issue",
            description="",
            parameters={"type": "object", "properties": {}},
            meta={"mcp_tool_name": "create_issue", "annotations": {"destructiveHint": True}},
        )

        async def mcp_exec(**kw):
            return "ok"

        lookups = build_tool_lookups(
            [search], mcp_executors={"gh__create_issue": mcp_exec}, mcp_tool_defs=[mcp_def]
        )
        assert _tool_source("search", lookups) == "local"
        assert _tool_source("gh__create_issue", lookups) == "mcp"

    async def test_meta_snapshot_includes_mcp_annotations(self) -> None:
        mcp_def = ToolDef(
            name="gh__create_issue",
            description="",
            parameters={"type": "object", "properties": {}},
            meta={"mcp_tool_name": "create_issue", "annotations": {"destructiveHint": True}},
        )

        async def mcp_exec(**kw):
            return "ok"

        lookups = build_tool_lookups(
            [], mcp_executors={"gh__create_issue": mcp_exec}, mcp_tool_defs=[mcp_def]
        )
        seen: list[ToolCallContext] = []

        def hook(call):
            seen.append(call)
            call.meta["annotations"]["destructiveHint"] = False
            return None

        decision, err = await _call_before_tool(
            hook,
            ToolCall(name="gh__create_issue", params={}),
            lookups,
            run_id="r",
            iteration=3,
            tenant_id="t1",
        )
        assert (decision, err) == (None, None)
        assert seen[0].source == "mcp"
        assert seen[0].tenant_id == "t1"
        assert lookups.meta["gh__create_issue"]["annotations"]["destructiveHint"] is True


class TestMCPAndSkills:
    async def test_hook_applies_to_mcp_tools(self) -> None:
        from dendrux.mcp import MCPServer

        calls: list[dict] = []

        async def mcp_exec(**kw):
            calls.append(kw)
            return "created"

        agent = Agent(
            prompt="p",
            tool_sources=[MCPServer("gh", command=["echo"])],
            before_tool=lambda c: ToolDecision.deny("mcp locked") if c.source == "mcp" else None,
        )
        agent._discovered_tool_defs = [
            ToolDef(
                name="gh__create_issue",
                description="",
                parameters={"type": "object", "properties": {"title": {"type": "string"}}},
                meta={"mcp_tool_name": "create_issue"},
            )
        ]
        agent._mcp_executors = {"gh__create_issue": mcp_exec}
        llm = _llm(ToolCall(name="gh__create_issue", params={"title": "x"}))
        await _run(agent, llm)
        assert calls == []
        assert json.loads(_tool_msgs(llm)[0].content) == {"denied": "mcp locked"}

    async def test_hook_applies_to_skill_tools(self) -> None:
        from pathlib import Path

        from dendrux.skills import Skill

        skill = Skill.from_dir(
            Path(__file__).parent.parent / "fixtures" / "skills" / "pdf-processing"
        )
        seen: list[tuple[str, str]] = []

        def hook(call):
            seen.append((call.tool_name, call.source))
            return ToolDecision.deny("no skills today")

        agent = Agent(prompt="p", tools=[search], skills=[skill], before_tool=hook)
        llm = _llm(ToolCall(name="use_skill", params={"name": skill.name}))
        await _run(agent, llm)
        assert seen == [("use_skill", "skill")]
        assert json.loads(_tool_msgs(llm)[0].content) == {"denied": "no skills today"}


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------


class TestStreaming:
    async def test_stream_deny_and_substitute(self) -> None:
        def hook(call):
            if call.tool_name == "refund":
                return ToolDecision.substitute({"held": True})
            return ToolDecision.deny("nope")

        llm = _llm(
            ToolCall(name="refund", params={"order_id": 1}),
            ToolCall(name="search", params={"query": "q"}),
        )
        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=_agent(hook),
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="go",
            )
        ]
        assert EXECUTED == []
        results = {
            e.tool_call.name: e.tool_result for e in events if e.type == RunEventType.TOOL_RESULT
        }
        assert json.loads(results["refund"].payload) == {"held": True}
        assert json.loads(results["search"].payload) == {"denied": "nope"}

    async def test_stream_require_approval_pauses(self) -> None:
        tc = ToolCall(name="refund", params={"order_id": 1})
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=_agent(lambda c: ToolDecision.require_approval("big")),
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="go",
            )
        ]
        paused = events[-1]
        assert paused.type == RunEventType.RUN_PAUSED
        assert paused.run_result.status == RunStatus.WAITING_APPROVAL
        assert paused.run_result.meta["pause_state"].approval_gates == {
            tc.id: {"source": "hook", "reason": "big"}
        }
