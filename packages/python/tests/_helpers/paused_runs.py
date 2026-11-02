"""Persisted pause fixtures shared by runtime and tenancy tests."""

from dataclasses import dataclass
from typing import Any

from dendrux import Agent, tool
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@tool()
async def refund(order_id: int) -> str:
    return f"Refunded {order_id}"


@dataclass
class Paused:
    agent: Agent
    run_id: str
    provider: ScriptedLLM


async def _pause(kind: str, store: Any, tenant: str | None, *, before_tool=None) -> Paused:
    """Build an agent and run it to a pause of ``kind`` under ``tenant``."""
    holder: dict[str, Agent] = {}

    @tool()
    async def step() -> str:
        ctx = get_delegation_context()
        assert ctx is not None
        await holder["agent"].request_pause(ctx.run_id)
        return "stepped"

    if kind == "approval":
        first = tool_call("refund", {"order_id": 7}, pid="t1")
        tools = [refund]
    elif kind == "paused":
        first = tool_call("step", pid="t1")
        tools = [step]
    else:
        first = tool_call("read_range", {"sheet": "S1"}, pid="t1")
        tools = [read_range]
    llm = ScriptedLLM([first, LLMResponse(text="done")])
    agent = Agent(
        strict_tenancy=False,
        provider=llm,
        prompt="Test.",
        tools=tools,
        require_approval=["refund"] if kind == "approval" else None,
        state_store=store,
        before_tool=before_tool,
    )
    holder["agent"] = agent
    paused = await agent.run("go", tenant_id=tenant)
    if kind == "input":
        await store.pause_run(
            paused.run_id,
            status=RunStatus.WAITING_HUMAN_INPUT.value,
            pause_data=await store.get_pause_state(paused.run_id),
        )
    expected = {
        "client": RunStatus.WAITING_CLIENT_TOOL,
        "input": RunStatus.WAITING_HUMAN_INPUT,
        "approval": RunStatus.WAITING_APPROVAL,
        "paused": RunStatus.PAUSED,
    }[kind]
    record = await store.get_run(paused.run_id)
    assert record.status == expected.value
    return Paused(agent, paused.run_id, llm)
