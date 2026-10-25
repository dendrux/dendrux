"""Integration: tool_result_transform output is what gets persisted and sent to the model."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall

if TYPE_CHECKING:
    from dendrux import ToolResultContext


@tool()
async def get_rows(limit: int) -> list[dict[str, Any]]:
    """Return tabular rows."""
    return [{"x": i} for i in range(limit)]


@tool()
async def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


def _tag(ctx: ToolResultContext) -> Any:
    return {"result_id": f"r-{ctx.tool_name}", "rows": ctx.result}


@pytest.fixture
def db_store(engine):
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def run_store(engine):
    return RunStore.from_engine(engine)


def _model_tool_contents(llm: MockLLM, call_index: int) -> list[str]:
    msgs = llm.call_history[call_index]["messages"]
    return [m.content for m in msgs if m.role.value == "tool"]


async def test_persisted_payload_matches_model_input(db_store, run_store) -> None:
    tc = ToolCall(name="get_rows", params={"limit": 2}, provider_tool_call_id="p1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        tools=[get_rows],
        state_store=db_store,
        tool_result_transform=_tag,
    )
    result = await agent.run("go")
    assert result.status == RunStatus.SUCCESS

    expected = {"result_id": "r-get_rows", "rows": [{"x": 0}, {"x": 1}]}
    (model_content,) = _model_tool_contents(llm, 1)
    assert json.loads(model_content) == expected

    (inv,) = await run_store.get_tool_invocations(result.run_id)
    assert inv.success is True
    assert inv.params == {"limit": 2}
    assert inv.result == expected

    traces = await run_store.get_traces(result.run_id)
    tool_traces = [t for t in traces if t.role == "tool"]
    assert [json.loads(t.content) for t in tool_traces] == [expected]


async def test_transform_failure_is_persisted_as_failed_call(db_store, run_store) -> None:
    def bad(ctx: ToolResultContext) -> Any:
        raise ValueError("bad shape")

    tc = ToolCall(name="get_rows", params={"limit": 1}, provider_tool_call_id="p1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
    agent = Agent(
        provider=llm, prompt="t", tools=[get_rows], state_store=db_store, tool_result_transform=bad
    )
    result = await agent.run("go")

    (inv,) = await run_store.get_tool_invocations(result.run_id)
    assert inv.success is False
    assert "tool_result_transform" in (inv.error or "")
    assert "bad shape" in (inv.error or "")


async def test_not_applied_when_tool_raises(db_store, run_store) -> None:
    calls: list[Any] = []

    def capture(ctx: ToolResultContext) -> Any:
        calls.append(ctx)
        return ctx.result

    tc = ToolCall(name="explode", params={}, provider_tool_call_id="p1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        tools=[explode],
        state_store=db_store,
        tool_result_transform=capture,
    )
    result = await agent.run("go")

    assert calls == []
    (inv,) = await run_store.get_tool_invocations(result.run_id)
    assert inv.success is False
    assert inv.error == "boom"


async def test_applied_on_approval_resume_path(db_store, run_store) -> None:
    tc = ToolCall(name="get_rows", params={"limit": 1}, provider_tool_call_id="p1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        tools=[get_rows],
        require_approval=["get_rows"],
        state_store=db_store,
        tool_result_transform=_tag,
    )
    paused = await agent.run("go")
    assert paused.status == RunStatus.WAITING_APPROVAL

    result = await agent.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )
    assert result.status == RunStatus.SUCCESS

    expected = {"result_id": "r-get_rows", "rows": [{"x": 0}]}
    (model_content,) = _model_tool_contents(llm, 1)
    assert json.loads(model_content) == expected
    (inv,) = await run_store.get_tool_invocations(paused.run_id)
    assert inv.result == expected
