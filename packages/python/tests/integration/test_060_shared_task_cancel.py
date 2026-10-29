"""Cancelling a nested child preserves its same-task parent."""

from __future__ import annotations

import asyncio

import pytest

from dendrux.agent import Agent
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio


async def _only(store, parent=None):
    runs = await store.list_runs(parent_run_id=parent) if parent else await store.list_runs()
    if parent is None:
        runs = [r for r in runs if r.parent_run_id is None]
    assert len(runs) == 1
    return str(runs[0].id)


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("cascade", [False, True])
async def test_cancel_child_leaves_same_task_parent_running(store, cascade):
    holder = {}

    @tool()
    async def delegate() -> str:
        return (await holder["agent"].run("sub")).status.value

    llm = ScriptedLLM(
        [
            tool_call("delegate", pid="p1"),
            LLMResponse(text="child"),
            LLMResponse(text="parent done"),
        ],
        gate_at=1,
    )
    agent = Agent(provider=llm, prompt="T.", tools=[delegate], state_store=store)
    holder["agent"] = agent
    task = asyncio.create_task(agent.run("go"))
    await asyncio.wait_for(llm.entered.wait(), 5)
    parent_id = await _only(store)
    child_id = await _only(store, parent=parent_id)
    await agent.cancel_run(child_id, cascade=cascade)
    result = await asyncio.wait_for(task, 5)
    print("parent status:", result.status, "child:", (await store.get_run(child_id)).status)
    assert (await store.get_run(child_id)).status == "cancelled"
    assert result.status == RunStatus.SUCCESS, f"parent ended {result.status}"
    assert (await store.get_run(parent_id)).status == "success"
