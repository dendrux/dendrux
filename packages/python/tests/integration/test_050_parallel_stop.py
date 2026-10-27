"""on a real store: a run-ending failure in a parallel batch stops siblings.

The parallel execution scenario (parallel tool execution): a tool-call row
write fails for one call while its sibling is still running. Before 0.5 the
sibling kept running, applied its side effect and wrote its row after
``run()`` had already raised.
"""

from __future__ import annotations

import asyncio

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, ToolCall

LOG: list[str] = []


@tool()
async def fast(x: int) -> int:
    """Fast."""
    return x


@tool()
async def slow_side_effect(x: int) -> int:
    """Slow, with a side effect."""
    await asyncio.sleep(0.3)
    LOG.append("slow_side_effect executed")
    return x


async def test_failed_row_write_stops_the_running_sibling(store):
    LOG.clear()
    original = store.save_tool_call

    async def failing(run_id, *, tool_name, **kwargs):
        if tool_name == "fast":
            raise RuntimeError("constraint violation")
        return await original(run_id, tool_name=tool_name, **kwargs)

    store.save_tool_call = failing
    calls = [
        ToolCall(name="fast", params={"x": 1}, provider_tool_call_id="a"),
        ToolCall(name="slow_side_effect", params={"x": 2}, provider_tool_call_id="b"),
    ]
    llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="ok")])
    agent = Agent(prompt="p", tools=[fast, slow_side_effect], provider=llm, state_store=store)

    with pytest.raises(RuntimeError, match="constraint violation"):
        await agent.run("go")

    run_id = (await store.list_runs())[0].id
    rows_at_raise = await store.get_tool_calls(run_id)
    await asyncio.sleep(0.6)
    assert LOG == []
    rows = await store.get_tool_calls(run_id)
    # The sibling was recorded as cancelled before the failure propagated,
    # and nothing was written afterwards.
    assert [(r.tool_name, r.success) for r in rows] == [("slow_side_effect", False)]
    assert len(rows) == len(rows_at_raise)
    assert rows[0].result is not None and rows[0].result["code"] == "cancelled"
    run = await store.get_run(run_id)
    assert run is not None and run.status != RunStatus.SUCCESS.value
