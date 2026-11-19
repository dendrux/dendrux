"""Root budget enforcement uses flushed, shared usage throughout a fan-out tree."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from typing import Any

import pytest
from sqlalchemy import event

from dendrux import Agent, tool
from dendrux.errors import BudgetConfigurationError, PersistenceError
from dendrux.loops.single import SingleCall
from dendrux.runtime import _managed_budget
from dendrux.runtime.context import get_delegation_context
from dendrux.types import Budget, Limits, LLMResponse, RunStatus, ToolCall, UsageStats
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.gates import wait_for_gate
from tests.integration.test_budget import _CostReportingLLM
from tests.integration.test_fanout_cancel import TENANT, running_root_id

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


def response(cost: float | None = 0.0, *names: str) -> LLMResponse:
    return LLMResponse(
        text=None if names else "done",
        tool_calls=[ToolCall(n, {} if n == "work" else {"task": "work"}) for n in names] or None,
        usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=cost),
    )


def parent(
    store: Any,
    children: list[Agent],
    budget: Budget | None,
    *,
    cost: float = 1,
    limits: Limits | None = None,
) -> tuple[Agent, Any]:
    provider = _CostReportingLLM([response(cost, *(c.name for c in children)), response()])
    root = Agent(
        name="lead",
        prompt="Lead",
        provider=provider,
        subagents=children,
        state_store=store,
        budget=budget,
        limits=limits,
    )
    return root, provider


def worker(
    name: str, effects: FanoutEffects, *, single: bool = False, cost: float | None = 3
) -> tuple[Agent, Any]:
    @tool()
    def work() -> str:
        """Do one unit of work."""
        effects.record_tool(name, "work")
        return "done"

    provider = _CostReportingLLM(
        [response(cost)] if single else [response(cost, "work"), response()], gate_at=0
    )
    return Agent(
        name=name,
        description="Worker",
        prompt="Work",
        provider=effects.provider(name, provider),
        tools=[] if single else [work],
        loop=SingleCall() if single else None,
    ), provider


@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_g1_spent_budget_refuses_the_entire_batch_before_bind(
    store: Any, action: str
) -> None:
    effects = FanoutEffects()
    children = [worker(f"w{i}", effects, single=i % 2 == 0)[0] for i in range(4)]
    root, provider = parent(store, children, Budget(max_cost_usd=1, on_exceed=action))
    result = await root.run("go", tenant_id=TENANT)
    assert result.status.value == ("paused" if action == "pause" else "cancelled")
    assert provider.calls == 1 and effects.counts().provider == {}
    assert len((await store.get_subtree(result.run_id, tenant_id=TENANT)).nodes) == 1
    assert (await store.get_loop_state(result.run_id)).get("managed_wait") is None


@pytest.mark.parametrize("shape", ["react", "mixed", "single", "nested"])
@pytest.mark.parametrize("action", ["pause", "cancel", "warn"])
async def test_g2_g3_one_unchecked_iteration_per_leaf_and_root_enforcement(
    store: Any,
    monkeypatch: Any,
    shape: str,
    action: str,
) -> None:
    effects = FanoutEffects()
    pairs = [
        worker(
            f"w{i}",
            effects,
            single=shape == "single" or shape in ("mixed", "nested") and i % 2 == 0,
        )
        for i in range(4)
    ]
    children = [c for c, _ in pairs]
    if shape == "nested":
        children = [
            Agent(
                name=f"m{i}",
                description="Delegates",
                prompt="Delegate",
                provider=_CostReportingLLM(
                    [response(0, *[c.name for c in children[i * 2 : i * 2 + 2]]), response()]
                ),
                subagents=children[i * 2 : i * 2 + 2],
            )
            for i in range(2)
        ]
    root, provider = parent(store, children, Budget(max_cost_usd=10, on_exceed=action))
    # Every leaf flushes before any ReAct leaf measures its next checkpoint.
    flushed = asyncio.Barrier(4)
    original = _managed_budget.record_interaction
    written = set()

    async def flush(recorder: Any, write: Any) -> None:
        await original(recorder, write)
        path = recorder._loop_state.get("managed", {}).get("path", [])
        if path and path[-1].startswith("w") and recorder._run_id not in written:
            written.add(recorder._run_id)
            await asyncio.wait_for(flushed.wait(), 10)

    monkeypatch.setattr(_managed_budget, "record_interaction", flush)
    run = asyncio.create_task(root.run("go", tenant_id=TENANT))
    try:
        for _, p in pairs:
            await wait_for_gate(p.entered, task=run)
        rid = await running_root_id(store)
        assert (await store.subtree_usage(rid)).known_cost_usd == 1
        for _, p in pairs:
            p.gate.set()
        result = await asyncio.wait_for(run, 60)
        assert (
            result.status.value
            == {"pause": "paused", "cancel": "cancelled", "warn": "success"}[action]
        )
        assert provider.calls == (2 if action == "warn" else 1)
        usage = await store.subtree_usage(rid)
        assert usage.known_cost_usd == 13
        assert sum(p.calls for _, p in pairs) <= (8 if action == "warn" else 4)
        if action != "warn":
            assert sum(effects.counts().tools.values()) == (
                4 if shape == "react" else 0 if shape == "single" else 2
            )
        if action == "pause":
            expected = 4 if shape == "react" else 1 if shape == "single" else 2
            assert len(result.pendings) == expected
            assert all(p.reason == "budget" for p in result.pendings)
            assert all(p.budget_exceeded[0]["used"] == 13 for p in result.pendings)
        if action == "cancel":
            assert result.meta["budget_exceeded"][0]["used"] == 13
    finally:
        for _, p in pairs:
            p.gate.set()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await root.close()


@pytest.mark.parametrize("placement", ["child", "grandchild", "root_scope"])
async def test_g4_only_one_root_budget_is_allowed(store: Any, placement: str) -> None:
    leaf = Agent(
        name="leaf",
        description="Worker",
        prompt="Work",
        provider=_CostReportingLLM([]),
        budget=Budget(max_tokens=10) if placement != "root_scope" else None,
    )
    with pytest.raises(BudgetConfigurationError) as raised:
        children = [leaf]
        if placement == "grandchild":
            children = [
                Agent(
                    name="middle",
                    description="Delegates",
                    prompt="Delegate",
                    provider=_CostReportingLLM([]),
                    subagents=[leaf],
                )
            ]
        parent(store, children, Budget(max_tokens=100, include_children=placement != "root_scope"))
    assert raised.value.code == "managed_budget_invalid"


@pytest.mark.parametrize("failure", ["read", "truncated", "write"])
async def test_g5_fault_in_one_branch_fails_closed_after_adoption(
    store: Any,
    monkeypatch: Any,
    caplog: Any,
    failure: str,
) -> None:
    effects = FanoutEffects()
    pairs = [worker(f"w{i}", effects, single=True, cost=1) for i in range(2)]
    root, provider = parent(
        store, [c for c, _ in pairs], Budget(max_cost_usd=100, on_exceed="cancel")
    )
    read, write = store.subtree_usage, store.save_llm_interaction

    def faulty_child() -> bool:
        ctx = get_delegation_context()
        return ctx is not None and ctx.delegation_level == 1 and ctx.run_id == origin[0]

    async def broken_read(*args: Any, **kwargs: Any) -> Any:
        if faulty_child() and failure != "write":
            if failure == "read":
                raise PersistenceError()
            raise BudgetConfigurationError(reason="subtree_limit")
        return await read(*args, **kwargs)

    @wraps(write)
    async def broken_write(*args: Any, **kwargs: Any) -> Any:
        if faulty_child() and failure == "write":
            raise PersistenceError()
        return await write(*args, **kwargs)

    monkeypatch.setattr(store, "subtree_usage", broken_read)
    monkeypatch.setattr(store, "save_llm_interaction", broken_write)
    origin = [""]
    run = asyncio.create_task(root.run("go", tenant_id=TENANT))
    try:
        for _, p in pairs:
            await wait_for_gate(p.entered, task=run)
        rid = await running_root_id(store)
        subtree = await store.get_subtree(rid, tenant_id=TENANT)
        origin[0] = next(n.run.id for n in subtree.nodes if n.run.agent_name == "w0")
        for _, p in pairs:
            p.gate.set()
        result = await asyncio.wait_for(run, 20)
        assert result.status == RunStatus.PAUSED and provider.calls == 1
        state = await store.get_loop_state(rid)
        assert state[_managed_budget.UNKNOWN] == {
            "code": "managed_usage_unknown",
            "origin_run_id": origin[0],
        }
        assert (await store.get_pause_state(rid))["pause_message"] == "managed_usage_unknown"
        assert effects.counts().provider == {"w0": 1, "w1": 1}
        assert any(
            getattr(r, "code", None) == "managed_usage_unknown" and r.run_id == origin[0]
            for r in caplog.records
        )
        before = effects.counts()
        resumed = await root.submit_resume(rid, tenant_id=TENANT, expected_pause_id=result.pause_id)
        assert resumed.status == RunStatus.SUCCESS and provider.calls == 2
        assert effects.counts() == before
    finally:
        for _, p in pairs:
            p.gate.set()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await root.close()


@pytest.mark.parametrize("action", ["pause", "cancel", "warn"])
async def test_g6_unpriced_single_calls_warn_and_root_enforces_known_unpriced_policy(
    store: Any,
    action: str,
) -> None:
    effects = FanoutEffects()
    pairs = [worker(f"w{i}", effects, single=True, cost=None) for i in range(2)]
    for _, p in pairs:
        p.gate.set()
    root, provider = parent(
        store, [c for c, _ in pairs], Budget(max_cost_usd=100, on_exceed=action)
    )
    result = await root.run("go", tenant_id=TENANT)
    assert (
        result.status.value == {"pause": "paused", "cancel": "cancelled", "warn": "success"}[action]
    )
    assert provider.calls == (2 if action == "warn" else 1)
    tree = await store.get_subtree(result.run_id, tenant_id=TENANT)
    assert tree.usage.known_cost_usd == 1 and tree.usage.unpriced_calls == 2
    assert [n.run.status for n in tree.nodes if n.depth == 1] == ["success", "success"]
    assert _managed_budget.UNKNOWN not in (await store.get_loop_state(result.run_id))
    events = [e for n in tree.nodes for e in await store.get_run_events(n.run.id)]
    assert sum(e.event_type == "budget.unpriced" for e in events) >= 1
    if action == "pause":
        assert (await store.get_pause_state(result.run_id)).get("pause_message") is None
        before = effects.counts()
        resumed = await root.submit_resume(
            result.run_id, tenant_id=TENANT, expected_pause_id=result.pause_id
        )
        assert resumed.status == RunStatus.SUCCESS
        assert effects.counts() == before
        assert (await store.get_loop_state(result.run_id))[_managed_budget.UNPRICED_ACCEPTED] == 2


async def test_g7_eight_leaf_measurements_have_bounded_query_cost(
    store: Any,
    engine: Any,
    monkeypatch: Any,
    record_property: Any,
) -> None:
    effects = FanoutEffects()
    pairs = [worker(f"w{i}", effects) for i in range(8)]
    for _, p in pairs:
        p.gate.set()
    children = [c for c, _ in pairs]
    root, _ = parent(
        store, children, Budget(max_cost_usd=1000), limits=Limits(max_parallel_subagents=8)
    )
    selected: ContextVar[list[str] | None] = ContextVar("measurement_sql", default=None)
    measurements = []
    real = store.subtree_usage

    def capture(conn: Any, cursor: Any, sql: str, params: Any, ctx: Any, many: Any) -> None:
        bucket = selected.get()
        if bucket is not None:
            bucket.append(sql.split()[0].upper())

    async def measured(*args: Any, **kwargs: Any) -> Any:
        bucket: list[str] = []
        token = selected.set(bucket)
        try:
            return await real(*args, **kwargs)
        finally:
            selected.reset(token)
            measurements.append(Counter(bucket))

    monkeypatch.setattr(store, "subtree_usage", measured)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await root.run("go", tenant_id=TENANT)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert result.status == RunStatus.SUCCESS
    assert effects.counts().provider == {f"w{i}": 2 for i in range(8)}
    assert measurements and max(m["SELECT"] for m in measurements) <= 4  # D=1, D+3
    record_property("measurement_count", len(measurements))
    record_property("measurement_selects", sum(m["SELECT"] for m in measurements))
