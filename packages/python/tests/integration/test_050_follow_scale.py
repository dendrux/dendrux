"""``RunStore.follow`` cost and fairness at scale (SQLite + Postgres).

- Replaying a run's history costs each follower a small, fixed number of
  statements, however many past pauses the run had: the active-pause lookup
  is shared by every follower of the run, not repeated per event.
- One tenant cannot take every follower slot: ``EventFeed`` has a
  per-tenant cap (a fifth of ``max_followers`` by default, overridable).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import event

from dendrux.agent import Agent
from dendrux.store import EventFeed, FollowCapacityError, FollowEnd, FollowHeartbeat, RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@tool()
async def gated(n: int) -> str:
    return "ok"


async def _run_with_pauses(store: Any, pauses: int) -> str:
    """A finished run with ``pauses`` client pauses and ``pauses`` approval pauses."""
    script = []
    for i in range(pauses):
        script.append(tool_call("read_range", {"sheet": f"S{i}"}, pid=f"c{i}"))
        script.append(tool_call("gated", {"n": i}, pid=f"g{i}"))
    script.append(LLMResponse(text="done"))
    agent = Agent(
        provider=ScriptedLLM(script),
        prompt="t",
        state_store=store,
        tools=[read_range, gated],
        require_approval=["gated"],
        strict_tenancy=False,
        max_iterations=100,
    )
    result = await agent.run("go", tenant_id="A")
    while result.status.value != "success":
        assert result.pending is not None
        if result.pending.kind == "client_tool":
            call = result.pending.tool_calls[0]
            result = await agent.submit_tool_results(
                result.run_id,
                [ToolResult(name="read_range", call_id=call.id, payload='"x"')],
                tenant_id="A",
                expected_pause_id=result.pause_id,
            )
        else:
            result = await agent.submit_approval(
                result.run_id, approved=True, tenant_id="A", expected_pause_id=result.pause_id
            )
    return result.run_id


async def _statements_for(engine: Any, run_store: RunStore, run_id: str, followers: int) -> int:
    count = [0]

    def on_execute(*_a: Any) -> None:
        count[0] += 1

    feed = EventFeed(run_store, poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=5.0)

    async def one() -> int:
        seen = 0
        async with feed.follow(run_id, tenant_id="A", max_duration_s=20.0) as f:
            async for item in f:
                if isinstance(item, FollowEnd):
                    assert item.reason == "terminal"
                    break
                if not isinstance(item, FollowHeartbeat):
                    seen += 1
        return seen

    event.listen(engine.sync_engine, "before_cursor_execute", on_execute)
    try:
        seen = await asyncio.gather(*[one() for _ in range(followers)])
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", on_execute)
        await feed.aclose()
    assert len(set(seen)) == 1 and seen[0] > 0
    return count[0]


async def test_backfill_cost_per_follower_is_flat(store: Any, engine: Any) -> None:
    run_store = RunStore.from_engine(engine)
    small = await _run_with_pauses(store, 1)
    large = await _run_with_pauses(store, 6)
    followers = 20
    small_cost = await _statements_for(engine, run_store, small, followers)
    large_cost = await _statements_for(engine, run_store, large, followers)
    per_follower = large_cost / followers
    print(f"statements: {followers} followers, 2 vs 12 past pauses: {small_cost} vs {large_cost}")
    # Fixed per follower: scope check, drained check, one events batch. The
    # head, poller and active-pause reads are shared by concurrent followers
    # (a follower arriving after the feed closed pays them again: still fixed).
    assert per_follower <= 6, (small_cost, large_cost)
    # Twelve more past pauses add no statements per follower.
    assert large_cost - small_cost <= 4, (small_cost, large_cost)


# ---------------------------------------------------------------------------
# Per-tenant follower cap
# ---------------------------------------------------------------------------


async def _paused_run(store: Any, tenant: str | None) -> str:
    agent = Agent(
        provider=ScriptedLLM(
            [tool_call("read_range", {"sheet": "S"}, pid="c"), LLMResponse(text="d")]
        ),
        prompt="t",
        state_store=store,
        tools=[read_range],
        strict_tenancy=False,
    )
    return (await agent.run("go", tenant_id=tenant)).run_id


async def test_one_tenant_cannot_take_every_slot(store: Any, engine: Any) -> None:
    run_store = RunStore.from_engine(engine)
    run_a = await _paused_run(store, "A")
    run_b = await _paused_run(store, "B")
    feed = EventFeed(run_store, max_followers=10, max_followers_per_tenant=4)
    held = []
    try:
        for _ in range(4):
            f = feed.follow(run_a, tenant_id="A")
            await f.start()
            held.append(f)
        hostile = feed.follow(run_a, tenant_id="A")
        with pytest.raises(FollowCapacityError) as info:
            await hostile.start()
        assert info.value.limit == 4 and info.value.per_tenant is True
        assert feed.followers == 4
        victim = feed.follow(run_b, tenant_id="B")
        await victim.start()
        held.append(victim)
        await held.pop(0).aclose()  # a slot freed by A is A's again
        again = feed.follow(run_a, tenant_id="A")
        await again.start()
        held.append(again)
    finally:
        for f in held:
            await f.aclose()
    assert feed.followers == 0


async def test_default_cap_is_a_fifth_and_untenanted_share_the_global_cap(
    store: Any, engine: Any
) -> None:
    run_store = RunStore.from_engine(engine)
    assert EventFeed(run_store).max_followers_per_tenant == 100
    assert EventFeed(run_store, max_followers=3).max_followers_per_tenant == 1
    assert EventFeed(run_store, max_followers_per_tenant=None).max_followers_per_tenant is None
    run_none = await _paused_run(store, None)
    feed = EventFeed(run_store, max_followers=5)  # per tenant: 1
    held = []
    try:
        for _ in range(5):  # untenanted followers are not one tenant
            f = feed.follow(run_none, tenant_id=None)
            await f.start()
            held.append(f)
        with pytest.raises(FollowCapacityError) as info:
            await feed.follow(run_none, tenant_id=None).start()
        assert info.value.per_tenant is False and info.value.limit == 5
    finally:
        for f in held:
            await f.aclose()


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
async def test_per_tenant_cap_is_validated(engine: Any, bad: Any) -> None:
    with pytest.raises(ValueError):
        EventFeed(RunStore.from_engine(engine), max_followers_per_tenant=bad)
