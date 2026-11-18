"""Real-driver nested fan-in, routed unwind and reconnect after process death."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.runtime import _managed_fanout
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.scripted_provider import ScriptedLLM

TENANT = "nested-recovery"
pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


@tool(target="client")
async def read() -> str:
    """Read a value in the caller's client."""
    raise AssertionError("client tool executed on server")


def calls(*names):
    return LLMResponse(tool_calls=[ToolCall(name, {"task": "go"}) for name in names])


def workflow(store, effects, *, both=False, again=False, root_width=2, nested_width=2):
    def agent(name, script, *, children=(), tools=()):
        return Agent(
            name=name,
            description="Reviews",
            prompt="Review",
            state_store=store if name == "root" else None,
            provider=effects.provider(name, ScriptedLLM(script)),
            subagents=list(children),
            tools=list(tools),
            tool_result_transform=effects.transform() if children else None,
        )

    client = LLMResponse(tool_calls=[ToolCall("read", {})])
    end = LLMResponse(text="done")
    a1 = agent("a1", [client, end] if both else [end, end], tools=[read])
    a2 = agent("a2", [client, end, end], tools=[read])
    names = ("a1", "a2") if nested_width == 2 else ("a2",)
    script = [calls(*names), *([calls("a1", "a2")] if again else []), end]
    a = agent("a", script, children=[a1, a2])
    b = agent("b", [end])
    root = agent(
        "root", [calls(*(("a", "b") if root_width == 2 else ("a",))), end], children=[a, b]
    )
    return root


async def records(engine):
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(AgentRun))).scalars())


async def answer(root, run_id, pending):
    call = pending.tool_calls[0]
    return await root.submit_tool_results(
        run_id,
        [ToolResult(call_id=call.id, name=call.name, payload="value")],
        tenant_id=TENANT,
        expected_pause_id=pending.pause_id,
    )


@pytest.mark.parametrize("again", [False, True])
async def test_nested_submit_unwinds_and_may_delegate_again(store, engine, again):
    effects = FanoutEffects()
    root = workflow(store, effects, again=again)
    paused = await root.run("go", tenant_id=TENANT)
    assert [p.agent_path for p in paused.pendings] == [["root", "a", "a2"]]
    before = {
        row.id: await store.get_run(row.id)
        for row in await records(engine)
        if row.status == "success"
    }
    assert {row.agent_name for row in before.values()} == {"a1", "b"}
    result = await answer(root, paused.run_id, paused.pending)
    assert result.status == RunStatus.SUCCESS and result.run_id == paused.run_id
    assert result.pending is None and result.pendings == () and result.pause_id is None
    after = await records(engine)
    assert all(row.status == "success" for row in after)
    assert len(after) == (7 if again else 5)
    assert all([(await store.get_run(key)) == value for key, value in before.items()])
    counts = effects.counts()
    assert dict(counts.provider) == {
        "root": 2,
        "a": 3 if again else 2,
        "a1": 2 if again else 1,
        "a2": 3 if again else 2,
        "b": 1,
    }
    assert counts.transform_calls() == (6 if again else 4)
    for name, width in (("root", 2), ("a", 4 if again else 2)):
        row = next(row for row in after if row.agent_name == name)
        intents = await store.get_tool_calls(row.id)
        assert len(intents) == width and all(intent.outcome == "completed" for intent in intents)
    await root.close()


async def test_nested_concurrent_submits_adopt_each_wait_once(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent claims require Postgres")
    effects = FanoutEffects()
    root = workflow(store, effects, both=True)
    paused = await root.run("go", tenant_id=TENANT)
    a = next(row for row in await records(engine) if row.agent_name == "a")
    ready, arrivals = asyncio.Event(), 0
    original = _managed_fanout.settle_parent

    async def settle(store, node, **kwargs):
        nonlocal arrivals
        if node.run_id == a.id:
            arrivals += 1
            if arrivals == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 10)
        return await original(store, node, **kwargs)

    monkeypatch.setattr(_managed_fanout, "settle_parent", settle)
    results = await asyncio.gather(*(answer(root, paused.run_id, p) for p in paused.pendings))
    assert arrivals == 2 and any(result.status == RunStatus.SUCCESS for result in results)
    assert all(row.status == "success" for row in await records(engine))
    counts = effects.counts()
    assert dict(counts.provider) == {"root": 2, "a": 2, "a1": 2, "a2": 2, "b": 1}
    assert counts.transform_calls() == 4
    for run_id in (a.id, paused.run_id):
        rows = await store.get_tool_calls(run_id)
        assert len(rows) == 2 and all(row.outcome == "completed" for row in rows)
        events = await store.get_run_events(run_id)
        assert sum(event.event_type == "run.completed" for event in events) == 1
    await root.close()


@pytest.mark.parametrize(("root_width", "nested_width"), [(1, 2), (2, 1), (1, 1)])
async def test_reconnect_carries_fanout_policy_through_single_child_waits(
    store, root_width, nested_width
):
    effects = FanoutEffects()
    root = workflow(store, effects, root_width=root_width, nested_width=nested_width)
    paused = await root.run("go", tenant_id=TENANT)
    before = effects.counts()
    result = await root.reclaim(paused.run_id, tenant_id=TENANT)
    assert result.status == RunStatus.WAITING_CLIENT_TOOL
    assert result.pendings == paused.pendings
    assert effects.counts() == before
    assert (await answer(root, paused.run_id, result.pending)).status == RunStatus.SUCCESS
    assert effects.counts().transform_calls() == root_width + nested_width
    await root.close()
