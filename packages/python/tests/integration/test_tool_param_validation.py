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


BAD_REFUND_ARGS = '{"order_id": 7,}'
BAD_REFUND_ERROR = "Expecting property name enclosed in double quotes at char 15"


def _bad_refund() -> ToolCall:
    from dendrux.llm._helpers import build_tool_call

    return build_tool_call(
        "refund", BAD_REFUND_ARGS, provider_tool_call_id="t1", provider="test", model="m"
    )


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Read a sheet range on the client."""
    return ""


async def test_malformed_call_never_pauses_for_approval(store):
    REFUNDS.clear()
    llm = MockLLM([LLMResponse(tool_calls=[_bad_refund()]), LLMResponse(text="retry later")])
    agent = _approval_agent(llm, store)

    result = await agent.run("refund 7")

    assert result.status == RunStatus.SUCCESS
    assert REFUNDS == []
    events = [e.event_type for e in await store.get_run_events(result.run_id)]
    assert "approval.requested" not in events
    (inv,) = await RunStore(store).get_tool_invocations(result.run_id)
    assert inv.result["code"] == "invalid_json"


async def test_approval_resume_rejects_malformed_pending_call(store):
    REFUNDS.clear()
    tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
    llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="acknowledged")])
    agent = _approval_agent(llm, store)
    paused = await agent.run("refund 7")
    pause_data = await store.get_pause_state(paused.run_id)
    pause_data["pending_tool_calls"][0]["parse_error"] = "Expecting value at char 14"
    await store.pause_run(paused.run_id, status=paused.status.value, pause_data=pause_data)

    result = await agent.submit_approval(paused.run_id, approved=True)

    assert result.status == RunStatus.SUCCESS
    assert REFUNDS == []
    tool_msgs = [m for m in llm.call_history[1]["messages"] if m.role.value == "tool"]
    assert json.loads(tool_msgs[0].content)["code"] == "invalid_json"
    events = await store.get_run_events(result.run_id)
    assert "tool.started" not in [e.event_type for e in events]
    completed = [e for e in events if e.event_type == "tool.completed"]
    assert completed[-1].data["parse_error"] == "Expecting value at char 14"


async def test_parse_error_survives_client_tool_pause_and_resume(store):
    from dendrux.types import PauseState, ToolResult

    REFUNDS.clear()
    client_call = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t2")
    llm = MockLLM([LLMResponse(tool_calls=[_bad_refund(), client_call]), LLMResponse(text="done")])
    agent = Agent(provider=llm, prompt="p", tools=[refund, read_range], state_store=store)

    paused = await agent.run("go")

    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    raw_pause = await store.get_pause_state(paused.run_id)
    assert BAD_REFUND_ARGS not in json.dumps(raw_pause)
    state = PauseState.from_dict(raw_pause)
    assistant = next(m for m in state.history if m.tool_calls)
    assert assistant.tool_calls[0].parse_error == BAD_REFUND_ERROR
    assert [tc.name for tc in state.pending_tool_calls] == ["read_range"]

    result = await agent.submit_tool_results(
        paused.run_id,
        [ToolResult(name="read_range", call_id=client_call.id, payload='"A1"')],
    )

    assert result.status == RunStatus.SUCCESS
    assert REFUNDS == []
    tool_msgs = [m for m in llm.call_history[1]["messages"] if m.role.value == "tool"]
    assert json.loads(tool_msgs[0].content)["code"] == "invalid_json"
    resumed = next(m for m in llm.call_history[1]["messages"] if m.tool_calls)
    assert resumed.tool_calls[0].parse_error == BAD_REFUND_ERROR
