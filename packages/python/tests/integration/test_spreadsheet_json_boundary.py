"""Spreadsheet-sized JSON fits the client and model argument byte budgets."""

import json

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunStatus, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool(target="client")
async def read_cells() -> list:
    return []


@tool()
async def write_cells(cells: list) -> int:
    return len(cells)


@pytest.mark.parametrize("shape", ["flat", "grid"])
async def test_client_spreadsheet_result_round_trips(store, shape):
    cells = (
        list(range(20000)) if shape == "flat" else [[r * c for c in range(200)] for r in range(100)]
    )
    provider = ScriptedLLM([tool_call("read_cells", pid="read"), LLMResponse(text="done")])
    async with Agent(
        prompt="p", tools=[read_cells], provider=provider, state_store=store, strict_tenancy=False
    ) as agent:
        paused = await agent.run("read", tenant_id="A")
        payload = json.dumps(cells)
        result = await agent.submit_tool_results(
            paused.run_id,
            [
                ToolResult.from_dict(
                    dict(
                        name="read_cells", call_id=paused.pending.tool_calls[0].id, payload=payload
                    )
                )
            ],
            tenant_id="A",
        )
        assert result.status == RunStatus.SUCCESS
        persisted = await store.get_tool_calls(result.run_id)
        assert persisted[0].result == cells
        assert payload in provider.contents(1)


@pytest.mark.parametrize("rows", [4000, 6000])
async def test_model_spreadsheet_arguments_round_trip(store, rows):
    cells = [[i, i + 1] for i in range(rows)]
    provider = ScriptedLLM(
        [tool_call("write_cells", {"cells": cells}, pid="write"), LLMResponse(text="done")]
    )
    async with Agent(
        prompt="p", tools=[write_cells], provider=provider, state_store=store, strict_tenancy=False
    ) as agent:
        result = await agent.run("write", tenant_id="A")
        assert result.status == RunStatus.SUCCESS
        persisted = await store.get_tool_calls(result.run_id)
        assert persisted[0].success is True
        assert persisted[0].params == {"cells": cells}
