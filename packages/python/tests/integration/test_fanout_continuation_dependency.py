"""Developer ContextVars propagate to child tools on both managed paths."""

from __future__ import annotations

from contextvars import ContextVar

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
request_id: ContextVar[str] = ContextVar("request_id", default="MISSING")


@pytest.mark.parametrize("n", [1, 2])
async def test_two_call_run_continues_only_with_the_v4_reader(store, n):
    seen = []

    @tool()
    async def whoami() -> str:
        seen.append(request_id.get())
        return request_id.get()

    children = [
        Agent(
            name=f"c{i}",
            description="d",
            prompt="p",
            tools=[whoami],
            provider=ScriptedLLM(
                [LLMResponse(tool_calls=[ToolCall("whoami", {})]), LLMResponse(text="ok")]
            ),
        )
        for i in range(n)
    ]
    calls = [ToolCall(c.name, {"task": "go"}) for c in children]
    parent = Agent(
        name="lead",
        prompt="p",
        state_store=store,
        subagents=children,
        provider=ScriptedLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")]),
    )
    token = request_id.set("req-42")
    try:
        result = await parent.run("go", tenant_id="t1")
        assert result.status == RunStatus.SUCCESS
        assert seen == ["req-42"] * n
    finally:
        request_id.reset(token)
        await parent.close()
        for c in children:
            await c.close()
