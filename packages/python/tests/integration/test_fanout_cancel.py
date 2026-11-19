"""Fan-out root controls preserve claims, scoped membership and terminal outcomes."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from functools import wraps
from typing import Any

import pytest
from sqlalchemy import select, update

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.loops.single import SingleCall
from dendrux.types import (
    Budget,
    Limits,
    LLMResponse,
    RunStatus,
    ToolCall,
    ToolTarget,
    generate_ulid,
)
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.gates import wait_for_gate
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_adoption import Child, lead, parked
from tests.integration.test_fanout_store_races import batch, fanout_parent, nest
from tests.integration.test_managed_cancel import parallel_stores
from tests.integration.test_managed_cancel import paused_workflow as v3_workflow

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
TENANT = "t1"


def batch_response(*names: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name, {"task": "work"}) for name in names])


def root_agent(store: Any, children: list[Agent], **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="Lead",
        state_store=store,
        subagents=children,
        provider=ScriptedLLM(
            [batch_response(*(c.name for c in children)), LLMResponse(text="done")]
        ),
        **kwargs,
    )


async def running_root_id(store: Any) -> str:
    async with store._session_factory() as session:
        return (
            await session.execute(
                select(AgentRun.id).where(
                    AgentRun.parent_run_id.is_(None), AgentRun.tenant_id == TENANT
                )
            )
        ).scalar_one()


def gated_child(
    name: str, effects: FanoutEffects, *, single: bool = False
) -> tuple[Agent, ScriptedLLM]:
    @tool()
    def work() -> str:
        """Finish one unit of work."""
        effects.record_tool(name, "work")
        return "done"

    responses = (
        [LLMResponse(text="done")]
        if single
        else [LLMResponse(tool_calls=[ToolCall("work", {})]), LLMResponse(text="done")]
    )
    provider = ScriptedLLM(responses, gate_at=0)
    child = Agent(
        name=name,
        description="Worker",
        prompt="Work",
        provider=effects.provider(name, provider),
        tools=[] if single else [work],
        loop=SingleCall() if single else None,
    )
    return child, provider


async def invoke(root: Agent, entry: str) -> Any:
    if entry == "run":
        return await root.run("go", tenant_id=TENANT)
    async with root.stream("go", tenant_id=TENANT) as stream:
        async for _ in stream:
            pass
    return stream.result


@pytest.mark.parametrize("nested", ["flat", "fanout", "mixed"])
@pytest.mark.parametrize("cascade", [False, True])
@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_a9_pause_reaches_every_react_leaf_without_revoking_claims(
    store: Any,
    monkeypatch: Any,
    nested: str,
    cascade: bool,
    entry: str,
) -> None:
    effects = FanoutEffects()
    one, p1 = gated_child("one", effects)
    two, p2 = gated_child("two", effects)
    single, ps = gated_child("single", effects, single=True)
    children = [one, two, single]
    if nested != "flat":
        middle = Agent(
            name="middle",
            description="Delegates",
            prompt="Delegate",
            subagents=[one, two, single] if nested == "mixed" else [one, two],
            provider=ScriptedLLM(
                [
                    batch_response("one", "two", "single")
                    if nested == "mixed"
                    else batch_response("one", "two"),
                    LLMResponse(text="done"),
                ]
            ),
        )
        children = [middle] if nested == "mixed" else [middle, single]
    root = root_agent(store, children)
    run = asyncio.create_task(invoke(root, entry))
    try:
        for provider in (p1, p2, ps):
            await wait_for_gate(provider.entered, task=run)
        root_id = await running_root_id(store)
        before = await store.get_subtree(root_id, tenant_id=TENANT)
        epochs = {n.run.id: n.run.claim_epoch for n in before.nodes}
        requested = []
        real = store.request_pause

        async def record(rid: str, **kwargs: Any) -> bool:
            requested.append(rid)
            return await real(rid, **kwargs)

        monkeypatch.setattr(store, "request_pause", record)
        outcome = await root.request_pause(root_id, tenant_id=TENANT, cascade=cascade)
        depths = {n.run.id: n.depth for n in before.nodes}
        assert [depths[r] for r in requested] == sorted(depths.values(), reverse=True)
        assert {rid for rid, _ in outcome.affected} == set(epochs)
        for rid, epoch in epochs.items():
            assert (await store.get_run(rid)).claim_epoch == epoch
        for provider in (p1, p2, ps):
            provider.gate.set()
        result = await asyncio.wait_for(run, 15)
        assert result.status == RunStatus.PAUSED
        assert {p.agent_name for p in result.pendings} == {"one", "two"}
        assert all(p.reason == "requested" for p in result.pendings)
        subtree = await store.get_subtree(root_id, tenant_id=TENANT)
        assert (
            next(n.run.status for n in subtree.nodes if n.run.agent_name == "single") == "success"
        )
        assert effects.counts().provider == {"one": 1, "two": 1, "single": 1}
    finally:
        for provider in (p1, p2, ps):
            provider.gate.set()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await root.close()


@pytest.mark.parametrize("single", [False, True])
async def test_x2_remote_cancel_stops_checkpoint_and_preserves_committed_success(
    engine: Any,
    single: bool,
) -> None:
    async with AsyncExitStack() as stack:
        owner_store, remote_store = await parallel_stores(engine, stack, 2)
        effects = FanoutEffects()
        slow, provider = gated_child("slow", effects, single=single)
        fast = Agent(
            name="fast",
            description="Worker",
            prompt="Work",
            loop=SingleCall(),
            provider=effects.provider("fast", ScriptedLLM([LLMResponse(text="done")])),
        )
        root = root_agent(owner_store, [fast, slow])
        remote = root_agent(remote_store, [fast, slow])
        committed = asyncio.Event()
        finalize = owner_store.finalize_with_event

        @wraps(finalize)
        async def finalized(rid: str, **kwargs: Any) -> Any:
            result = await finalize(rid, **kwargs)
            if kwargs.get("status") == "success":
                committed.set()
            return result

        owner_store.finalize_with_event = finalized
        run = asyncio.create_task(root.run("go", tenant_id=TENANT))
        try:
            await wait_for_gate(provider.entered, task=run)
            await wait_for_gate(committed, task=run)
            rid = await running_root_id(remote_store)
            await remote.cancel_run(rid, tenant_id=TENANT)
            assert not provider.cancelled and not run.done()
            provider.gate.set()
            result = await asyncio.wait_for(run, 15)
            assert result.status == RunStatus.CANCELLED
            subtree = await remote_store.get_subtree(rid, tenant_id=TENANT)
            assert {n.run.agent_name: n.run.status for n in subtree.nodes} == {
                "lead": "cancelled",
                "fast": "success",
                "slow": "success" if single else "cancelled",
            }
            assert effects.counts().provider == {"fast": 1, "slow": 1}
            assert effects.counts().tools == {}
        finally:
            provider.gate.set()
            if not run.done():
                run.cancel()
            await asyncio.gather(run, return_exceptions=True)
            await root.close()
            await remote.close()


async def test_x3_cancel_one_paused_two_running_and_foreign_sibling(store: Any) -> None:
    root = lead(state_store=store)
    fanout = await parked(store, root, [Child(status="running") for _ in range(3)])
    paused, *running = fanout.child_ids
    await store.pause_run(paused, status="paused", pause_data={})
    foreign = generate_ulid()
    await store.create_run(foreign, "foreign", tenant_id="other", parent_run_id=fanout.run_id)
    result = await root.cancel_run(fanout.run_id, tenant_id=TENANT)
    assert result.status == RunStatus.CANCELLED
    assert (await store.get_run(paused)).status == "cancelled"
    assert all([(await store.get_run(r)).cancel_requested for r in running])
    assert not (await store.get_run(foreign)).cancel_requested
    assert result.meta["cascade_count"] == 4
    assert {r["run_id"] for r in result.meta["cascade"]} == {fanout.run_id, *fanout.child_ids}


async def test_x3_live_cancel_settles_paused_and_running_owners(
    store: Any, monkeypatch: Any
) -> None:
    effects = FanoutEffects()

    @tool(target=ToolTarget.CLIENT)
    def answer() -> str:
        """Ask the client."""
        raise AssertionError("client tool must not execute")

    paused_child = Agent(
        name="paused",
        description="Worker",
        prompt="Work",
        tools=[answer],
        provider=effects.provider(
            "paused", ScriptedLLM([LLMResponse(tool_calls=[ToolCall("answer", {})])])
        ),
    )
    pairs = [gated_child(name, effects) for name in ("one", "two")]
    root = root_agent(store, [paused_child, *(c for c, _ in pairs)])
    controller = lead(state_store=store)
    paused = asyncio.Event()
    pause = store.pause_with_event

    @wraps(pause)
    async def saved(*args: Any, **kwargs: Any) -> Any:
        result = await pause(*args, **kwargs)
        if kwargs.get("status") == "waiting_client_tool":
            paused.set()
        return result

    monkeypatch.setattr(store, "pause_with_event", saved)
    run = asyncio.create_task(root.run("go", tenant_id=TENANT))
    try:
        for _, p in pairs:
            await wait_for_gate(p.entered, task=run)
        await wait_for_gate(paused, task=run)
        rid = await running_root_id(store)
        result = await controller.cancel_run(rid, tenant_id=TENANT)
        assert result.status == RunStatus.CANCELLED
        for _, p in pairs:
            p.gate.set()
        assert (await asyncio.wait_for(run, 15)).status == RunStatus.CANCELLED
        tree = await store.get_subtree(rid, tenant_id=TENANT)
        assert [n.run.status for n in tree.nodes] == ["cancelled"] * 4
        assert effects.counts().provider == {"paused": 1, "one": 1, "two": 1}
        assert effects.counts().tools == {}
    finally:
        for _, p in pairs:
            p.gate.set()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await root.close()
        await controller.close()


async def test_x5_truncated_fanout_cascade_is_repeatable(store: Any) -> None:
    root, calls = await fanout_parent(store, 3)
    request = batch(root, calls)
    await store.bind_managed_children(request)
    all_children = [c.child_run_id for c in request.children]
    nested_calls = await nest(store, all_children[0], 2)
    nested = batch(all_children[0], nested_calls, root=root, path=("reviewer",))
    await store.bind_managed_children(nested)
    all_children.extend(c.child_run_id for c in nested.children)
    for child in all_children:
        if child != request.children[0].child_run_id:
            await store.pause_run(child, status="paused", pause_data={})
    first = await lead(
        state_store=store, budget=Budget(max_tokens=100, max_subtree_nodes=2)
    ).cancel_run(root, tenant_id=TENANT)
    assert first.meta["cascade_truncated"] and first.meta["cascade_count"] == 2
    assert any([(await store.get_run(c)).status == "paused" for c in all_children])
    again = await lead(state_store=store).cancel_run(root, tenant_id=TENANT)
    assert not again.meta["cascade_truncated"] and again.meta["cascade_count"] == 6
    assert [(await store.get_run(c)).status for c in all_children] == ["cancelled"] * 5
    third = await lead(state_store=store).cancel_run(root, tenant_id=TENANT)
    assert third.meta["cascade_count"] == 6


async def test_a9_pause_refuses_foreign_root_and_corrupt_binding_before_writes(
    store: Any,
    monkeypatch: Any,
) -> None:
    root = lead(state_store=store)
    fanout = await parked(store, root, [Child(status="running"), Child(status="running")])
    with pytest.raises(RunNotFoundError):
        await root.request_pause(fanout.run_id, tenant_id="other")
    real = store.get_managed_children

    async def missing(*args: Any, **kwargs: Any) -> Any:
        return (await real(*args, **kwargs))[:1]

    monkeypatch.setattr(store, "get_managed_children", missing)
    with pytest.raises(ManagedStateError):
        await root.request_pause(fanout.run_id, tenant_id=TENANT)
    assert not (await store.get_run(fanout.run_id)).pause_requested
    assert not any([(await store.get_run(c)).pause_requested for c in fanout.child_ids])


async def test_a9_single_child_keeps_v3_wait_bytes_and_leaf_first_order(store: Any) -> None:
    effects = FanoutEffects()
    child, provider = gated_child("one", effects)
    root = root_agent(store, [child])
    run = asyncio.create_task(root.run("go", tenant_id=TENANT))
    try:
        await wait_for_gate(provider.entered, task=run)
        rid = await running_root_id(store)
        pause = await store.get_pause_state(rid)
        before = json.dumps(pause, sort_keys=True)
        assert pause["schema_version"] == 3
        assert isinstance((await store.get_loop_state(rid))["managed_admitted"], str)
        result = await root.request_pause(rid, tenant_id=TENANT)
        assert result.affected[-1][0] == rid and len(result.affected) == 2
        assert json.dumps(await store.get_pause_state(rid), sort_keys=True) == before
        provider.gate.set()
        paused = await asyncio.wait_for(run, 15)
        assert paused.status == RunStatus.PAUSED and len(paused.pendings) == 1
        assert effects.counts().provider_calls("one") == 1
    finally:
        provider.gate.set()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await root.close()


async def test_v3_pause_uses_saved_bindings_without_requiring_the_registered_definition(
    store: Any,
) -> None:
    original, rid, child_id, _ = await v3_workflow(store)
    different = Agent(
        name="different", description="Worker", prompt="Work", provider=ScriptedLLM([])
    )
    controller = root_agent(store, [different])
    try:
        outcome = await controller.request_pause(rid, tenant_id=TENANT)
        assert outcome.affected == [(child_id, "already_waiting"), (rid, "already_paused")]
    finally:
        await controller.close()
        await original.close()


async def test_v3_pause_keeps_invalid_checkpoint_binding_refusal(store: Any) -> None:
    root, rid, child_id, _ = await v3_workflow(store)
    state = await store.get_loop_state(rid)
    state["managed_wait"]["tool_call_id"] = 123
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == rid, AgentRun.tenant_id == TENANT)
            .values(loop_state=state)
        )
        await session.commit()
    try:
        with pytest.raises(ManagedStateError) as raised:
            await root.request_pause(rid, tenant_id=TENANT)
        assert raised.value.reason == "wait_invalid" and raised.value.run_id == rid
        assert not (await store.get_run(child_id)).pause_requested
    finally:
        await root.close()


async def test_pause_honors_saved_run_cap_above_the_pending_reader_bound(store: Any) -> None:
    rid, calls = await fanout_parent(store, 16, cap=300)
    request = batch(rid, calls)
    await store.bind_managed_children(request)
    for child in request.children:
        state = await store.get_loop_state(child.child_run_id)
        state["loop"] = "react"
        await store.checkpoint_step(
            child.child_run_id, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
        )
        below = await nest(store, child.child_run_id, 16)
        await store.bind_managed_children(
            batch(child.child_run_id, below, root=rid, path=("reviewer",))
        )
    leaf = Agent(name="reviewer", description="Worker", prompt="Work", provider=ScriptedLLM([]))
    middle = Agent(
        name="reviewer",
        description="Delegates",
        prompt="Delegate",
        provider=ScriptedLLM([]),
        subagents=[leaf],
    )
    root = root_agent(
        store, [middle], limits=Limits(max_parallel_subagents=32, max_managed_runs=300)
    )
    try:
        outcome = await root.request_pause(rid, tenant_id=TENANT)
        assert len(outcome.affected) == 273 and outcome.affected[-1][0] == rid
        tree = await store.get_subtree(rid, tenant_id=TENANT)
        assert len(tree.nodes) == 273
        assert all(n.run.pause_requested for n in tree.nodes if n.depth == 2)
    finally:
        await root.close()
