"""Persistence and approval-resume coverage for tool parameter validation."""

from __future__ import annotations

import json

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall

REFUNDS: list[int] = []


@tool()
async def refund(order_id: int) -> str:
    """Refund an order."""
    REFUNDS.append(order_id)
    return f"refunded {order_id}"


def _approval_agent(llm: MockLLM, store) -> Agent:
    return Agent(
        provider=llm,
        prompt="Support agent.",
        tools=[refund],
        require_approval=["refund"],
        state_store=store,
    )


async def test_invalid_call_persisted_as_failed_invocation(store):
    REFUNDS.clear()
    tc = ToolCall(name="refund", params={"order_id": "seven"}, provider_tool_call_id="t1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="ok")])
    agent = Agent(provider=llm, prompt="p", tools=[refund], state_store=store)

    result = await agent.run("refund")

    assert REFUNDS == []
    (inv,) = await RunStore(store).get_tool_invocations(result.run_id)
    assert inv.success is False
    assert inv.error == "refund argument 'order_id' must be integer, got string."
    assert inv.result == {"error": inv.error, "code": "invalid_params"}
    events = [e.event_type for e in await store.get_run_events(result.run_id)]
    assert "tool.completed" in events


async def test_approval_resume_validates_before_executing(store):
    REFUNDS.clear()
    tc = ToolCall(name="refund", params={"orderid": 7}, provider_tool_call_id="t1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="acknowledged")])
    agent = _approval_agent(llm, store)
    paused = await agent.run("refund 7")
    assert paused.status == RunStatus.WAITING_APPROVAL

    result = await agent.submit_approval(paused.run_id, approved=True)

    assert result.status == RunStatus.SUCCESS
    assert REFUNDS == []
    tool_msgs = [m for m in llm.call_history[1]["messages"] if m.role.value == "tool"]
    assert json.loads(tool_msgs[0].content) == {
        "error": "refund has no argument 'orderid'; refund is missing required "
        "argument(s): order_id. Valid arguments: order_id.",
        "code": "invalid_params",
    }
    (inv,) = await RunStore(store).get_tool_invocations(result.run_id)
    assert inv.success is False
    events = [e.event_type for e in await store.get_run_events(result.run_id)]
    assert "tool.started" not in events


async def test_approval_resume_valid_call_still_executes(store):
    REFUNDS.clear()
    tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="processed")])
    agent = _approval_agent(llm, store)
    paused = await agent.run("refund 7")

    result = await agent.submit_approval(paused.run_id, approved=True)

    assert result.answer == "processed"
    assert REFUNDS == [7]
