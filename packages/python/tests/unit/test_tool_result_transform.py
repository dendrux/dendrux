"""Tests for Agent(tool_result_transform=...)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dendrux import ToolResultContext, ToolResultTransform
from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop, _execute_tool
from dendrux.mcp._server import MCPServer
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import LLMResponse, Message, RunEventType, RunStatus, ToolCall, ToolTarget


@tool()
async def get_rows(limit: int) -> list[dict[str, Any]]:
    """Return tabular rows."""
    return [{"x": i} for i in range(limit)]


@tool()
def sync_rows(limit: int) -> list[dict[str, Any]]:
    """Return tabular rows (sync)."""
    return [{"y": i} for i in range(limit)]


@tool()
async def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


@tool()
async def returns_none() -> None:
    """Returns None."""
    return None


def _tag(ctx: ToolResultContext) -> Any:
    return {"result_id": f"r-{ctx.call_id}", "rows": ctx.result}


async def _async_tag(ctx: ToolResultContext) -> Any:
    return {"result_id": "async", "rows": ctx.result}


def _tool_messages(llm: MockLLM, call_index: int) -> list[Message]:
    msgs = llm.call_history[call_index]["messages"]
    return [m for m in msgs if m.role.value == "tool"]


def _llm_for(name: str, params: dict[str, Any]) -> tuple[MockLLM, ToolCall]:
    tc = ToolCall(name=name, params=params, provider_tool_call_id="p1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
    return llm, tc


class _CaptureRecorder(BaseRecorder):
    def __init__(self) -> None:
        self.completed: list[Any] = []

    async def on_message_appended(self, run_id: str, message: Any, iteration: int) -> None:
        pass

    async def on_llm_call_completed(self, run_id: str, response: Any, iteration: int, **kw: Any):
        pass

    async def on_tool_completed(self, run_id: str, tool_call: Any, tool_result: Any, iteration):
        self.completed.append(tool_result)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_default_is_none(self) -> None:
        agent = Agent(prompt="t", tools=[get_rows])
        assert agent.tool_result_transform is None

    def test_accepts_callable(self) -> None:
        agent = Agent(prompt="t", tools=[get_rows], tool_result_transform=_tag)
        assert agent.tool_result_transform is _tag

    def test_rejects_non_callable(self) -> None:
        with pytest.raises(ValueError, match="tool_result_transform must be callable"):
            Agent(prompt="t", tools=[get_rows], tool_result_transform="nope")  # type: ignore[arg-type]

    def test_public_exports(self) -> None:
        import dendrux

        assert "ToolResultContext" in dendrux.__all__
        assert "ToolResultTransform" in dendrux.__all__
        transform: ToolResultTransform = _tag
        assert callable(transform)

    def test_context_is_frozen(self) -> None:
        ctx = ToolResultContext(
            tool_name="t", call_id="c", params={}, result=1, target=ToolTarget.SERVER, meta={}
        )
        with pytest.raises(AttributeError):
            ctx.result = 2  # type: ignore[misc]


# ------------------------------------------------------------------
# _execute_tool unit behavior
# ------------------------------------------------------------------


class TestExecuteTool:
    async def test_no_transform_unchanged(self) -> None:
        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(ToolCall(name="get_rows", params={"limit": 2}), lookups)
        assert json.loads(tr.payload) == [{"x": 0}, {"x": 1}]

    async def test_sync_transform_applied(self) -> None:
        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tc = ToolCall(name="get_rows", params={"limit": 1})
        tr = await _execute_tool(tc, lookups, transform=_tag)
        assert tr.success
        assert json.loads(tr.payload) == {"result_id": f"r-{tc.id}", "rows": [{"x": 0}]}

    async def test_async_transform_applied(self) -> None:
        agent = Agent(prompt="t", tools=[sync_rows])
        lookups = await agent.get_tool_lookups()
        tc = ToolCall(name="sync_rows", params={"limit": 1})
        tr = await _execute_tool(tc, lookups, transform=_async_tag)
        assert json.loads(tr.payload) == {"result_id": "async", "rows": [{"y": 0}]}

    async def test_context_fields(self) -> None:
        seen: list[ToolResultContext] = []

        def capture(ctx: ToolResultContext) -> Any:
            seen.append(ctx)
            return ctx.result

        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tc = ToolCall(name="get_rows", params={"limit": 1})
        tr = await _execute_tool(tc, lookups, transform=capture)

        assert json.loads(tr.payload) == [{"x": 0}]
        (ctx,) = seen
        assert ctx.tool_name == "get_rows"
        assert ctx.call_id == tc.id
        assert ctx.params == {"limit": 1}
        assert ctx.result == [{"x": 0}]
        assert ctx.target == ToolTarget.SERVER
        assert ctx.meta == {}

    async def test_none_result_is_passed_through(self) -> None:
        seen: list[Any] = []

        def capture(ctx: ToolResultContext) -> Any:
            seen.append(ctx.result)
            return ctx.result

        agent = Agent(prompt="t", tools=[returns_none])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(
            ToolCall(name="returns_none", params={}), lookups, transform=capture
        )
        assert seen == [None]
        assert tr.success and tr.payload == "null"

    async def test_transform_returning_none_replaces_result(self) -> None:
        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(
            ToolCall(name="get_rows", params={"limit": 1}), lookups, transform=lambda ctx: None
        )
        assert tr.success and tr.payload == "null"

    async def test_transform_raises_fails_call(self) -> None:
        def bad(ctx: ToolResultContext) -> Any:
            raise ValueError("bad shape")

        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(
            ToolCall(name="get_rows", params={"limit": 1}), lookups, transform=bad
        )
        assert tr.success is False
        assert "tool_result_transform" in (tr.error or "")
        assert "bad shape" in (tr.error or "")
        assert json.loads(tr.payload) == {"error": tr.error}

    async def test_async_transform_raises_fails_call(self) -> None:
        async def bad(ctx: ToolResultContext) -> Any:
            raise ValueError("async bad")

        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(
            ToolCall(name="get_rows", params={"limit": 1}), lookups, transform=bad
        )
        assert tr.success is False
        assert "async bad" in (tr.error or "")

    async def test_transform_returning_non_json_fails_call(self) -> None:
        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(
            ToolCall(name="get_rows", params={"limit": 1}), lookups, transform=lambda ctx: object()
        )
        assert tr.success is False
        assert "non-JSON-serializable" in (tr.error or "")

    async def test_not_applied_on_tool_error(self) -> None:
        calls: list[Any] = []

        def capture(ctx: ToolResultContext) -> Any:
            calls.append(ctx)
            return "changed"

        agent = Agent(prompt="t", tools=[explode])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(ToolCall(name="explode", params={}), lookups, transform=capture)
        assert tr.success is False
        assert tr.error == "boom"
        assert calls == []

    async def test_not_applied_on_unknown_tool(self) -> None:
        calls: list[Any] = []
        agent = Agent(prompt="t", tools=[get_rows])
        lookups = await agent.get_tool_lookups()
        tr = await _execute_tool(
            ToolCall(name="missing", params={}), lookups, transform=lambda c: calls.append(c)
        )
        assert tr.success is False
        assert calls == []


# ------------------------------------------------------------------
# Loop behavior: model + recorder see the transformed result
# ------------------------------------------------------------------


class TestLoop:
    @pytest.mark.parametrize("transform", [_tag, _async_tag])
    async def test_model_and_recorder_see_transformed(self, transform: Any) -> None:
        llm, _ = _llm_for("get_rows", {"limit": 2})
        agent = Agent(provider=llm, prompt="t", tools=[get_rows], tool_result_transform=transform)
        recorder = _CaptureRecorder()
        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            recorder=recorder,
        )
        assert result.status == RunStatus.SUCCESS
        (msg,) = _tool_messages(llm, 1)
        body = json.loads(msg.content)
        assert "result_id" in body
        assert body["rows"] == [{"x": 0}, {"x": 1}]
        (recorded,) = recorder.completed
        assert recorded.payload == msg.content

    async def test_parallel_calls_all_transformed(self) -> None:
        tcs = [
            ToolCall(name="get_rows", params={"limit": 1}, provider_tool_call_id="a"),
            ToolCall(name="sync_rows", params={"limit": 1}, provider_tool_call_id="b"),
        ]
        llm = MockLLM([LLMResponse(tool_calls=tcs), LLMResponse(text="done")])
        agent = Agent(
            provider=llm, prompt="t", tools=[get_rows, sync_rows], tool_result_transform=_tag
        )
        await agent.run("go")
        msgs = _tool_messages(llm, 1)
        assert len(msgs) == 2
        assert {json.loads(m.content)["result_id"] for m in msgs} == {f"r-{tc.id}" for tc in tcs}

    async def test_transform_failure_reaches_model_as_error(self) -> None:
        def bad(ctx: ToolResultContext) -> Any:
            raise KeyError("missing")

        llm, _ = _llm_for("get_rows", {"limit": 1})
        agent = Agent(provider=llm, prompt="t", tools=[get_rows], tool_result_transform=bad)
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        (msg,) = _tool_messages(llm, 1)
        assert msg.meta.get("is_error") is True
        assert "tool_result_transform" in json.loads(msg.content)["error"]

    async def test_stream_applies_transform(self) -> None:
        llm, _ = _llm_for("get_rows", {"limit": 1})
        agent = Agent(prompt="t", tools=[get_rows], tool_result_transform=_tag)
        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
            )
        ]
        results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert len(results) == 1
        assert "result_id" in json.loads(results[0].tool_result.payload)
        (msg,) = _tool_messages(llm, 1)
        assert "result_id" in json.loads(msg.content)


# ------------------------------------------------------------------
# MCP tools: same hook, real SDK stdio server
# ------------------------------------------------------------------


class TestMCPTool:
    async def test_transform_applies_to_real_mcp_tool(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "mcp_echo_server.py"
        server = MCPServer(
            "echo",
            command=[sys.executable, str(fixture)],
            connect_timeout=10.0,
            call_timeout=10.0,
        )
        seen: list[ToolResultContext] = []

        def tag(ctx: ToolResultContext) -> Any:
            seen.append(ctx)
            return {**ctx.result, "result_id": "r1"}

        tc = ToolCall(name="echo__echo", params={"message": "hi"}, provider_tool_call_id="p1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="t", tool_sources=[server], tool_result_transform=tag)
        try:
            result = await agent.run("go")
        finally:
            await agent.close()

        assert result.status == RunStatus.SUCCESS
        (msg,) = _tool_messages(llm, 1)
        assert json.loads(msg.content) == {"result": "hi", "result_id": "r1"}
        (ctx,) = seen
        assert ctx.tool_name == "echo__echo"
        assert ctx.params == {"message": "hi"}
        assert ctx.result == {"result": "hi"}
        assert ctx.meta["source_name"] == "echo"
