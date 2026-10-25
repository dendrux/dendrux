"""Redacted feeds over real runs show only declared tool names (SQLite + Postgres).

A model that calls a tool the agent does not have gets the call rejected;
the redacted feed shows a placeholder instead of the model's text. A declared
tool that fails keeps its name. Trusted readers see the real names.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.store import FollowEnd, FollowHeartbeat, RunStore, StoredEvent
from dendrux.tool import tool
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio

SECRET = "probe-secret-5831"


@tool()
async def flaky(n: int) -> str:
    raise RuntimeError("boom")


async def _follow(run_store: RunStore, run_id: str, **kw: Any) -> list[StoredEvent]:
    out: list[StoredEvent] = []
    async with run_store.follow(run_id, tenant_id="A", max_duration_s=10.0, **kw) as f:
        async for item in f:
            if isinstance(item, FollowEnd):
                break
            if not isinstance(item, FollowHeartbeat):
                out.append(item)
    return out


async def test_unknown_tool_name_is_a_placeholder(store: Any, engine: Any) -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    tool_call(SECRET, {"x": 1}, pid="u").tool_calls[0],
                    tool_call("flaky", {"n": 1}, pid="f").tool_calls[0],
                ]
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(provider=llm, prompt="t", state_store=store, tools=[flaky], strict_tenancy=False)
    result = await agent.run("go", tenant_id="A")
    assert result.status.value == "success"
    # Positive evidence that "flaky" is declared (the writer seam proposed in
    # round 6; written here by hand). The model's invented name is not in it.
    await store.save_run_event(
        result.run_id, event_type="tools.declared", data={"local_tool_names": ["flaky"]}
    )
    run_store = RunStore.from_engine(engine)
    events = await _follow(run_store, result.run_id)
    completed = [e.data for e in events if e.event_type == "tool.completed"]
    names = sorted(d["tool_name"] for d in completed)
    assert names == ["<unknown_tool>", "flaky"]
    assert SECRET not in json.dumps([e.data for e in events])
    views = [
        await run_store.get_event_view(result.run_id, e, tenant_id="A")
        for e in await run_store.get_events(result.run_id, tenant_id="A")
    ]
    assert SECRET not in json.dumps([v.data for v in views])
    trusted = await _follow(run_store, result.run_id, include_payloads=True)
    assert SECRET in json.dumps([e.data for e in trusted])
