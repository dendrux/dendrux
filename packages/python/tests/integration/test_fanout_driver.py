"""T4b: owner custody, task identity, all-settled waits and the fan-in boundary."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from time import perf_counter

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, tool
from dendrux.errors import ManagedStateError, PersistenceError
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.runtime._managed_tree import BranchSpec, FanoutInvocation
from dendrux.runtime.state import ClaimResult
from dendrux.types import (
    Limits,
    LLMResponse,
    RunEventType,
    RunResult,
    RunStatus,
    ToolCall,
    ToolTarget,
    UsageStats,
    generate_ulid,
)
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.gated_provider import GatedStreamLLM
from tests._helpers.gates import wait_for_gate
from tests.integration.test_fanout_store_races import fanout_parent, fanout_wait

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


def driver():
    name = "dendrux.runtime._managed_fanout"
    assert importlib.util.find_spec(name) is not None, "fan-out owner driver is required"
    return importlib.import_module(name)


async def plan(store, n=2, *, gated=False, single=True):
    effects = FanoutEffects()
    root, calls = await fanout_parent(store, n)
    providers = [GatedStreamLLM() for _ in calls]
    children = tuple(
        Agent(
            name="reviewer",
            description="Review one task",
            prompt="p",
            provider=effects.provider(str(i), p),
            loop=SingleCall() if single else None,
        )
        for i, p in enumerate(providers)
    )
    parent = Agent(
        name="lead", prompt="p", provider=MockLLM([]), limits=Limits(max_parallel_subagents=32)
    )
    branches = tuple(
        BranchSpec(ToolCall("reviewer", {"task": "go"}, id=c), a, ("reviewer",), generate_ulid())
        for c, a in zip(calls, children, strict=True)
    )
    wait = fanout_wait([(b.call.id, b.child_run_id) for b in branches])
    invocation = FanoutInvocation(
        store, "t1", root, root, 0, 0, parent, wait, 1, UsageStats(), 10, branches
    )
    for provider in providers:
        provider.gate_complete = gated
    return invocation, providers, effects


async def close_plan(invocation):
    for b in invocation.branches:
        await b.child.close()
    await invocation.parent_agent.close()


async def test_p2_t1_t3_real_overlap_and_isolated_owner_tasks(store):
    module = driver()
    invocation, providers, effects = await plan(store, gated=True)
    caller = asyncio.current_task()
    run = await module.start_fanout(invocation)
    try:
        for p in providers:
            await wait_for_gate(p.first_sent)
        assert not run.owners_done()
        assert len(set(p.complete_tasks[0] for p in providers)) == 2
        assert all(p.complete_tasks[0] is not caller for p in providers)
        contexts = [p.complete_contexts[0] for p in providers]
        assert len({id(c) for c in contexts}) == 2
        assert len({id(c._lease_lock) for c in contexts}) == 2
        for p in providers:
            p.gate.set()
        outcomes = await asyncio.wait_for(run.wait(), 5)
        assert [o.kind for o in outcomes] == ["terminal", "terminal"]
        for i, p in enumerate(providers):
            assert p.complete_tasks == [run.owners[i]] * 3
            assert p.complete_closed
            assert effects.counts().provider_calls(str(i)) == 1
            assert (await store.get_run(invocation.branches[i].child_run_id)).status == "success"
    finally:
        for p in providers:
            p.gate.set()
        await run.aexit()
        await close_plan(invocation)


@pytest.mark.parametrize("n", [2, 8])
async def test_q1_zero_event_children_complete_without_a_queue_waiter(store, monkeypatch, n):
    module = driver()
    invocation, _, effects = await plan(store, n)
    calls = []
    original = store._creation_storage_format

    async def read():
        calls.append(1)
        return await original()

    monkeypatch.setattr(store, "_creation_storage_format", read)
    try:
        run = await module.start_fanout(invocation)
        outcomes = await asyncio.wait_for(run.wait(), 5)
        assert calls == [1]
        assert run.queue is None
        assert run.owners_done()
        assert [o.kind for o in outcomes] == ["terminal"] * n
        assert sum(effects.counts().provider.values()) == n
        assert all(t.done() for t in run.owners)
    finally:
        await close_plan(invocation)


@pytest.mark.parametrize("k", [0, 1, 2])
async def test_x12_partial_task_creation_finalizes_every_claim(store, monkeypatch, k):
    module = driver()
    invocation, _, effects = await plan(store, 3)
    real = asyncio.create_task
    created = []
    error = ValueError("SENTINEL-startup")

    def create(coro, *args, **kwargs):
        if getattr(coro, "cr_code", None).co_name == "run_branch":
            if len(created) == k:
                raise error
            task = real(coro, *args, **kwargs)
            created.append(task)
            return task
        return real(coro, *args, **kwargs)

    monkeypatch.setattr(module.asyncio, "create_task", create)
    try:
        with pytest.raises(ValueError) as caught:
            await module.start_fanout(invocation)
        assert caught.value is error
        assert all(t.done() for t in created)
        assert sum(effects.counts().provider.values()) == 0
        for branch in invocation.branches:
            assert (await store.get_run(branch.child_run_id)).status == "cancelled"
    finally:
        await close_plan(invocation)


async def test_x6_t2_caller_cancel_joins_owners_and_preserves_first_cancel(store):
    module = driver()
    invocation, providers, effects = await plan(store, gated=True)
    run = await module.start_fanout(invocation)
    waiter = asyncio.create_task(run.wait())
    try:
        for p in providers:
            await wait_for_gate(p.first_sent)
        waiter.cancel("first")
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(waiter, 5)
        assert caught.value.args == ("first",)
        assert run.owners_done()
        for i, b in enumerate(invocation.branches):
            assert providers[i].complete_closed
            assert effects.counts().provider_calls(str(i)) == 1
            assert (await store.get_run(b.child_run_id)).status == "cancelled"
    finally:
        for p in providers:
            p.gate.set()
        await run.aexit()
        await close_plan(invocation)


@pytest.mark.parametrize("read_fails", [False, True])
async def test_p17_errors_are_collected_in_call_order_after_siblings_settle(
    store, monkeypatch, caplog, read_fails
):
    module = driver()
    invocation, _, effects = await plan(store, 3)
    real = module._run_prepared
    errors = [ValueError("SENTINEL-error-one"), TypeError("SENTINEL-error-two")]

    async def execute(branch, prepared, claim, **kwargs):
        i = [b.call.id for b in invocation.branches].index(branch.call.id)
        if i < 2:
            if i == 0:
                await asyncio.sleep(0)
            raise errors[i]
        return await real(branch, prepared, claim, **kwargs)

    monkeypatch.setattr(module, "_run_prepared", execute)
    original_read = store.get_run

    async def read(run_id):
        if read_fails and run_id == invocation.branches[1].child_run_id:
            raise PersistenceError()
        return await original_read(run_id)

    monkeypatch.setattr(store, "get_run", read)
    try:
        run = await module.start_fanout(invocation)
        with pytest.raises(ValueError) as caught:
            await asyncio.wait_for(run.wait(), 5)
        assert caught.value is errors[0]
        assert run.owners_done()
        assert effects.counts().provider_calls("2") == 1
        assert (await store.get_run(invocation.branches[2].child_run_id)).status == "success"
        logs = [
            r for r in caplog.records if getattr(r, "code", None) == "managed_fanout_branch_error"
        ]
        assert len(logs) == (3 if read_fails else 2)
        assert {r.run_id for r in logs} == {b.child_run_id for b in invocation.branches[:2]}
        assert "SENTINEL" not in caplog.text
    finally:
        await close_plan(invocation)


async def test_owned_only_existing_bind_runs_no_child(store):
    module = driver()
    invocation, _, effects = await plan(store)
    try:
        first = await module.start_fanout(invocation)
        await first.wait()
        before = effects.counts()
        second = await module.start_fanout(invocation)
        assert second.owners == []
        assert second.ledger.states == ()
        assert await second.wait() == ()
        assert effects.counts() == before
    finally:
        await close_plan(invocation)


async def test_fan_in_uses_fresh_children_read_and_adopt_protocol(store):
    module = driver()
    invocation, _, effects = await plan(store)
    seen = []

    async def claim_parent(store, node):
        seen.append(("claim", node.bindings))
        return ClaimResult(2, 2, {}, None)

    # Minimal AdoptFanout implementation: no report writes, transforms or continuation.
    async def adopter(store, node, claim):
        seen.append(("adopt", node.bindings))
        return claim

    try:
        run = await module.start_fanout(invocation)
        node = module.wait_node(replace(invocation, branches=tuple(reversed(invocation.branches))))
        assert (
            await module.settle_parent(store, node, claim_parent=claim_parent, adopt_fanout=adopter)
            is None
        )
        assert seen == []
        await run.wait()
        result = await module.settle_parent(
            store, node, claim_parent=claim_parent, adopt_fanout=adopter
        )
        assert result.claim_epoch == 2
        assert [kind for kind, _ in seen] == ["claim", "adopt"]
        assert seen[0][1] == invocation.bindings == seen[1][1]
        assert sum(effects.counts().provider.values()) == 2
    finally:
        await close_plan(invocation)


async def test_x8_x9_repeated_cancel_retains_finalize_and_join(store, monkeypatch):
    module = driver()
    invocation, providers, effects = await plan(store, 3, gated=True)
    run = await module.start_fanout(invocation)
    for owner in run.owners[1:]:
        owner.cancel()
    await wait_for_gate(providers[0].first_sent, task=run.owners[0])
    entered = {b.child_run_id: asyncio.Event() for b in invocation.branches}
    release = {b.child_run_id: asyncio.Event() for b in invocation.branches}
    original = store.finalize_with_event
    writes = []

    async def finalize(run_id, *, claim_epoch=None, **kwargs):
        writes.append((run_id, claim_epoch))
        entered[run_id].set()
        await release[run_id].wait()
        return await original(run_id, claim_epoch=claim_epoch, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", finalize)
    waiter = asyncio.create_task(run.wait())
    await asyncio.sleep(0)
    waiter.cancel("first")
    try:
        ids = [b.child_run_id for b in invocation.branches]
        await wait_for_gate(entered[ids[1]], task=waiter)
        waiter.cancel("second")
        release[ids[1]].set()
        await wait_for_gate(entered[ids[2]], task=waiter)
        waiter.cancel("third")
        release[ids[2]].set()
        await wait_for_gate(entered[ids[0]], task=waiter)
        waiter.cancel("fourth")
        release[ids[0]].set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(waiter, 5)
        assert caught.value.args == ("first",)
        assert run.ledger.states == ("accepted", "finalized", "finalized")
        assert sorted(writes) == sorted((child, 0) for child in ids)
        assert all(owner.done() and owner.cancelling() == 1 for owner in run.owners)
        assert [r.status for r in await asyncio.gather(*(store.get_run(c) for c in ids))] == [
            "cancelled"
        ] * 3
        assert sum(effects.counts().provider.values()) == 1
    finally:
        for gate in release.values():
            gate.set()
        await asyncio.gather(waiter, return_exceptions=True)
        await run.aexit()
        await close_plan(invocation)


async def test_t2_context_set_reset_stays_in_owner(store, monkeypatch):
    from dendrux.runtime import context

    module = driver()
    invocation, _, _ = await plan(store)
    pairs = []
    original_set, original_reset = context.set_delegation_context, context.reset_delegation_context

    def set_context(ctx):
        token = original_set(ctx)
        pairs.append(("set", asyncio.current_task(), token))
        return token

    def reset_context(token):
        pairs.append(("reset", asyncio.current_task(), token))
        original_reset(token)

    monkeypatch.setattr(context, "set_delegation_context", set_context)
    monkeypatch.setattr(context, "reset_delegation_context", reset_context)
    try:
        run = await module.start_fanout(invocation)
        await run.wait()
        for owner in run.owners:
            owned = [p for p in pairs if p[1] is owner]
            assert [p[0] for p in owned] == ["set", "reset"]
            assert owned[0][2] is owned[1][2]
        assert context.get_delegation_context() is None
    finally:
        await close_plan(invocation)


async def test_event_sink_receives_structural_tool_objects_in_owner(store):
    module = driver()
    invocation, _, effects = await plan(store)
    received = []

    @tool()
    async def echo() -> str:
        """Return a test result."""
        effects.record_tool("0", "echo")
        return "SENTINEL-tool-result"

    child = Agent(
        name="reviewer",
        description="Review one task",
        prompt="p",
        tools=[echo],
        provider=effects.provider(
            "react",
            MockLLM([LLMResponse(tool_calls=[ToolCall("echo", {})]), LLMResponse(text="finished")]),
        ),
    )
    invocation = replace(
        invocation, branches=(replace(invocation.branches[0], child=child), invocation.branches[1])
    )

    async def sink(event):
        received.append((event, asyncio.current_task()))

    try:
        run = await module.start_fanout(invocation, event_sink=sink)
        await run.wait()
        assert [event.type for event, _ in received] == [
            RunEventType.TOOL_USE_START,
            RunEventType.TOOL_USE_END,
            RunEventType.TOOL_RESULT,
        ]
        assert all(task is run.owners[0] for _, task in received)
        assert "SENTINEL" not in repr([e for e, _ in received])
        assert effects.counts().tool("0", "echo") == 1
        assert effects.counts().provider_calls("react") == 2
    finally:
        await close_plan(invocation)


async def test_n32_bind_ack_join_and_finalize_deadline(store, engine, monkeypatch, record_property):
    from dendrux.runtime import durability
    from tests.integration.test_fanout_store_races import batch

    if engine.dialect.name != "postgresql":
        pytest.skip("five-second ownership deadline measurement requires Postgres")
    root, calls = await fanout_parent(store, 32)
    request = batch(root, calls)
    committed, release = asyncio.Event(), asyncio.Event()
    original_commit = AsyncSession.commit
    armed = True

    async def commit(session):
        nonlocal armed
        await original_commit(session)
        if armed:
            armed = False
            committed.set()
            await release.wait()

    monkeypatch.setattr(AsyncSession, "commit", commit)
    caller = asyncio.create_task(
        durability._fanout_ownership_handoff(
            store, lambda: store.bind_managed_children(request), run_id=root
        )
    )
    try:
        await wait_for_gate(committed, task=caller)
        begin = perf_counter()
        caller.cancel("first")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        elapsed = perf_counter() - begin
        record_property("n32_ownership_seconds", elapsed)
        print(f"N32_OWNERSHIP_SECONDS={elapsed:.6f}")
        assert elapsed < 5
        records = await asyncio.gather(*(store.get_run(c.child_run_id) for c in request.children))
        assert all(r.status == "cancelled" and r.claim_epoch == 0 for r in records)
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.parametrize("damage", ["header_parent", "header_path", "duplicate_row"])
async def test_settle_refuses_mismatched_child_identity(store, monkeypatch, damage):
    module = driver()
    invocation, _, effects = await plan(store)
    original = store.get_managed_children
    touched = []

    async def read(*args, **kwargs):
        rows = await original(*args, **kwargs)
        header = deepcopy(rows[0].header)
        if damage == "header_parent":
            header["invocation"]["parent_run_id"] = generate_ulid()
        elif damage == "header_path":
            header["path"] = ["other"]
        else:
            return [*rows, rows[0]]
        return [replace(rows[0], header=header), *rows[1:]]

    async def claim_parent(store, node):
        touched.append("claim")
        return None

    async def adopter(store, node, claim):
        touched.append("adopt")
        return claim

    try:
        run = await module.start_fanout(invocation)
        await run.wait()
        before = effects.counts()
        monkeypatch.setattr(store, "get_managed_children", read)
        with pytest.raises(ManagedStateError):
            await module.settle_parent(
                store, module.wait_node(invocation), claim_parent=claim_parent, adopt_fanout=adopter
            )
        assert touched == []
        assert effects.counts() == before
    finally:
        await close_plan(invocation)


async def test_p1_p3_parent_continues_in_caller_after_call_order_adoption(store, monkeypatch):
    from dendrux.runtime import _reclaim

    module = driver()
    invocation, providers, effects = await plan(store, gated=True)
    observed = []
    caller_task = None
    claimed = ClaimResult(2, 2, {}, None)

    async def claim_parent(store, node):
        records = await asyncio.gather(
            *(store.get_run(b.child_run_id) for b in invocation.branches)
        )
        assert all(r.status == "success" for r in records)
        return claimed

    # This minimal AdoptFanout supplies ordering only, not T4c's durable completions.
    async def adopter(store, node, claim):
        observed.extend(binding.tool_call_id for binding in node.bindings)
        assert asyncio.current_task() is caller_task
        return claim

    async def continuation(agent, store, run_id, claim, *, notifier):
        assert observed == list(invocation.call_ids)
        assert claim is claimed
        assert asyncio.current_task() is caller_task
        return RunResult(run_id=run_id, status=RunStatus.SUCCESS)

    async def drive():
        nonlocal caller_task
        caller_task = asyncio.current_task()
        return await module.drive_fanout(
            invocation, claim_parent=claim_parent, adopt_fanout=adopter
        )

    monkeypatch.setattr(_reclaim, "continue_claimed", continuation)
    runner = asyncio.create_task(drive())
    try:
        for p in providers:
            await wait_for_gate(p.first_sent, task=runner)
        providers[1].gate.set()
        # Wait on a durable second-child terminal event rather than a timing guess.
        for _ in range(500):
            if (await store.get_run(invocation.branches[1].child_run_id)).status == "success":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("second child did not commit")
        assert not runner.done()
        assert observed == []
        providers[0].gate.set()
        result = await asyncio.wait_for(runner, 5)
        assert result.status == RunStatus.SUCCESS
        assert observed == list(invocation.call_ids)
        assert sum(effects.counts().provider.values()) == 2
    finally:
        for p in providers:
            p.gate.set()
        await asyncio.gather(runner, return_exceptions=True)
        await close_plan(invocation)


async def test_t4_single_child_provider_generator_stays_in_caller(store):
    provider = GatedStreamLLM()
    child = Agent(
        name="reviewer",
        description="Review one task",
        prompt="p",
        provider=provider,
        loop=SingleCall(),
    )
    root = Agent(
        name="lead",
        prompt="p",
        subagents=[child],
        state_store=store,
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("reviewer", {"task": "go"})]),
                LLMResponse(text="done"),
            ]
        ),
    )
    try:
        result = await root.run("go", tenant_id="t1")
        assert result.status == RunStatus.SUCCESS
        assert provider.complete_tasks == [asyncio.current_task()] * 3
        assert provider.complete_closed
    finally:
        await root.close()
        await child.close()


@pytest.mark.parametrize("n,failed", [(8, 0), (8, 2), (30, 0)])
async def test_p9_p10_s1_s2_structured_children_have_distinct_invocations(store, n, failed):
    from pydantic import BaseModel

    from dendrux.runtime import _managed_adoption

    class Report(BaseModel):
        index: int

    class StructuredMock(MockLLM):
        capabilities = replace(MockLLM.capabilities, supports_structured_output=True)

    module = driver()
    invocation, _, effects = await plan(store, n)
    branches = []
    for i, branch in enumerate(invocation.branches):
        await branch.child.close()
        child = Agent(
            name="reviewer",
            description="Review one task",
            prompt="p",
            loop=SingleCall(),
            output_type=Report,
            provider=effects.provider(
                str(i),
                StructuredMock(
                    [LLMResponse(text="invalid" if i < failed else json.dumps({"index": i}))]
                ),
            ),
        )
        branches.append(replace(branch, child=child))
    invocation = replace(invocation, branches=tuple(branches))
    try:
        run = await module.start_fanout(invocation)
        await run.wait()
        records = await asyncio.gather(*(store.get_run(b.child_run_id) for b in branches))
        assert len({r.id for r in records}) == n
        headers = [((await store.get_loop_state(r.id))["managed"]) for r in records]
        assert len({h["invocation"]["tool_call_id"] for h in headers}) == n
        assert all(h["path"] == ["reviewer"] for h in headers)
        assert sum(r.status == "success" for r in records) == n - failed
        reports = [
            _managed_adoption.child_report(r, b.child, limit=32768)
            for r, b in zip(records, branches, strict=True)
        ]
        assert sum(report.success for report in reports) == n - failed
        assert all(effects.counts().provider_calls(str(i)) == 1 for i in range(n))
    finally:
        await close_plan(invocation)


async def test_p11_depth_refusal_is_confined_to_its_branch(store):
    module = driver()
    invocation, _, effects = await plan(store)
    grandchild = Agent(
        name="grandchild",
        description="Do one task",
        prompt="p",
        loop=SingleCall(),
        provider=effects.provider("grandchild", MockLLM([])),
    )
    child = Agent(
        name="reviewer",
        description="Review one task",
        prompt="p",
        subagents=[grandchild],
        max_delegation_depth=1,
        provider=effects.provider(
            "nested", MockLLM([LLMResponse(tool_calls=[ToolCall("grandchild", {"task": "go"})])])
        ),
    )
    await invocation.branches[0].child.close()
    invocation = replace(
        invocation, branches=(replace(invocation.branches[0], child=child), invocation.branches[1])
    )
    try:
        run = await module.start_fanout(invocation)
        outcomes = await run.wait()
        assert [o.result.status.value for o in outcomes] == ["error", "success"]
        assert effects.counts().provider_calls("grandchild") == 0
        assert effects.counts().provider_calls("nested") == 1
        assert effects.counts().provider_calls("1") == 1
    finally:
        await close_plan(invocation)
        await grandchild.close()


async def test_concurrent_exit_waits_for_the_same_retained_cleanup(store, monkeypatch):
    module = driver()
    invocation, _, _ = await plan(store)
    run = await module.start_fanout(invocation)
    for owner in run.owners:
        owner.cancel()
    entered, release = asyncio.Event(), asyncio.Event()
    original = run.ledger.finalize

    async def finalize(store, owned):
        entered.set()
        await release.wait()
        await original(store, owned)

    monkeypatch.setattr(run.ledger, "finalize", finalize)
    closer = asyncio.create_task(run.aexit())
    waiter = None
    try:
        await wait_for_gate(entered, task=closer)
        waiter = asyncio.create_task(run.wait())
        await asyncio.sleep(0)
        assert not waiter.done(), "wait returned before its retained finalization completed"
        release.set()
        await closer
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert run.owners_done()
        assert run.ledger.states == ("finalized", "finalized")
    finally:
        release.set()
        await asyncio.gather(closer, return_exceptions=True)
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        await close_plan(invocation)


async def test_x1_remote_cancel_interrupts_four_in_process_owners(store):
    module = driver()
    invocation, providers, effects = await plan(store, 4, gated=True)
    invocation.parent_agent._state_store = store
    run = await module.start_fanout(invocation)
    try:
        for p, owner in zip(providers, run.owners, strict=True):
            await wait_for_gate(p.first_sent, task=owner)
        cancelled = await invocation.parent_agent.cancel_run(
            invocation.root_run_id, tenant_id="t1", cascade=True
        )
        assert cancelled.status.value == "cancelled"
        outcomes = await asyncio.wait_for(run.wait(), 5)
        assert [o.result.status.value for o in outcomes] == ["cancelled"] * 4
        assert run.owners_done()
        assert all(p.complete_closed for p in providers)
        assert sum(effects.counts().provider.values()) == 4
        assert (await store.get_run(invocation.root_run_id)).status == "cancelled"
    finally:
        for p in providers:
            p.gate.set()
        await run.aexit()
        await close_plan(invocation)


async def test_external_pause_settles_without_delaying_its_sibling(store):
    module = driver()
    invocation, _, effects = await plan(store)

    @tool(target=ToolTarget.CLIENT)
    def answer() -> str:
        """Ask the client."""
        raise AssertionError("a client tool must not execute")

    child = Agent(
        name="reviewer",
        prompt="p",
        tools=[answer],
        provider=effects.provider(
            "paused", MockLLM([LLMResponse(tool_calls=[ToolCall("answer", {})])])
        ),
    )
    await invocation.branches[0].child.close()
    invocation = replace(
        invocation, branches=(replace(invocation.branches[0], child=child), invocation.branches[1])
    )
    try:
        run = await module.start_fanout(invocation)
        outcomes = await run.wait()
        assert [o.kind for o in outcomes] == ["paused", "terminal"]
        paused = outcomes[0].result
        assert paused.pause_id is not None
        assert paused.pending is not None
        assert paused.pending.pause_id == paused.pause_id
        assert (await store.get_pause_state(paused.run_id))["pause_id"] == paused.pause_id
        assert effects.counts().provider_calls("paused") == 1
        assert effects.counts().provider_calls("1") == 1
    finally:
        await close_plan(invocation)


async def test_p13_subtree_usage_counts_children_without_adoption_spend(store):
    module = driver()
    invocation, _, effects = await plan(store)
    branches = []
    for i, b in enumerate(invocation.branches):
        await b.child.close()
        child = Agent(
            name="reviewer",
            prompt="p",
            loop=SingleCall(),
            provider=effects.provider(
                str(i),
                MockLLM(
                    [
                        LLMResponse(
                            text="done",
                            usage=UsageStats(
                                input_tokens=i + 2,
                                output_tokens=1,
                                total_tokens=i + 3,
                                cost_usd=0.1,
                            ),
                        )
                    ]
                ),
            ),
        )
        branches.append(replace(b, child=child))
    invocation = replace(invocation, branches=tuple(branches))
    try:
        run = await module.start_fanout(invocation)
        await run.wait()
        usage = await store.subtree_usage(invocation.root_run_id)
        assert (usage.input_tokens, usage.output_tokens, usage.llm_calls) == (5, 2, 2)
        before = effects.counts()

        async def claim_parent(store, node):
            return ClaimResult(2, 2, {}, None)

        async def adopter(store, node, claim):
            return claim

        await module.settle_parent(
            store, module.wait_node(invocation), claim_parent=claim_parent, adopt_fanout=adopter
        )
        assert await store.subtree_usage(invocation.root_run_id) == usage
        assert effects.counts() == before
    finally:
        await close_plan(invocation)


async def test_p15_eight_children_release_two_connection_pool_before_provider(engine, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from dendrux.runtime.stores._factory import store_for_engine

    pools = []
    original = ThreadPoolExecutor.__init__

    def thread_pool(self, *args, **kwargs):
        pools.append(kwargs.get("max_workers"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", thread_pool)
    pool = create_async_engine(engine.url, pool_size=2, max_overflow=0)
    invocation = None
    run = None
    providers = []
    try:
        store = store_for_engine(pool)
        invocation, providers, effects = await plan(store, 8, gated=True)
        run = await driver().start_fanout(invocation)
        for p, owner in zip(providers, run.owners, strict=True):
            await wait_for_gate(p.first_sent, task=owner)
        assert pool.pool.checkedout() == 0
        assert pools == []
        for p in providers:
            p.gate.set()
        assert len(await asyncio.wait_for(run.wait(), 5)) == 8
        assert sum(effects.counts().provider.values()) == 8
        assert pool.pool.checkedout() == 0
    finally:
        for p in providers:
            p.gate.set()
        if run is not None:
            await run.aexit()
        if invocation is not None:
            await close_plan(invocation)
        await pool.dispose()


@pytest.mark.parametrize("where", ["preparation", "provider"])
async def test_callback_exception_payload_is_not_persisted(store, monkeypatch, caplog, where):
    module = driver()
    invocation, providers, effects = await plan(store)
    error = ValueError("SENTINEL-private-callback-payload")
    try:
        if where == "preparation":

            async def prepare(*args, **kwargs):
                raise error

            monkeypatch.setattr(module, "prepare_child_creation", prepare)
            with pytest.raises(ValueError) as caught:
                await module.start_fanout(invocation)
            assert caught.value is error
            record = await store.get_run(invocation.parent_run_id)
            assert sum(effects.counts().provider.values()) == 0
        else:
            providers[0].complete_error = error
            run = await module.start_fanout(invocation)
            outcomes = await run.wait()
            assert [o.result.status.value for o in outcomes] == ["error", "success"]
            record = await store.get_run(invocation.branches[0].child_run_id)
            assert sum(effects.counts().provider.values()) == 2
        assert record.status == "error"
        events = await store.get_run_events(record.id)
        assert "SENTINEL" not in record.error
        assert "SENTINEL" not in repr(events)
        assert "SENTINEL" not in caplog.text
    finally:
        await close_plan(invocation)


async def test_x12_startup_preserves_first_cancel_during_retained_exit(store, monkeypatch):
    module = driver()
    invocation, _, _ = await plan(store)
    create_task = asyncio.create_task
    original = store.finalize_with_event
    entered, release = asyncio.Event(), asyncio.Event()

    def create(coro, *args, **kwargs):
        if getattr(coro, "cr_code", None).co_name == "run_branch":
            raise asyncio.CancelledError("first")
        return create_task(coro, *args, **kwargs)

    @wraps(original)
    async def finalize(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(module.asyncio, "create_task", create)
    monkeypatch.setattr(store, "finalize_with_event", finalize)
    caller = create_task(module.start_fanout(invocation))
    try:
        await wait_for_gate(entered, task=caller)
        caller.cancel("second")
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await caller
        assert caught.value.args == ("first",)
        for b in invocation.branches:
            assert (await store.get_run(b.child_run_id)).status == "cancelled"
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await close_plan(invocation)


async def test_fenced_owner_returns_not_owner_without_reexecuting(store, engine):
    from dendrux.db.models import AgentRun

    invocation, providers, effects = await plan(store, gated=True)
    run = await driver().start_fanout(invocation)
    try:
        for p, owner in zip(providers, run.owners, strict=True):
            await wait_for_gate(p.first_sent, task=owner)
        child = invocation.branches[0].child_run_id
        async with AsyncSession(engine) as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == child).values(claim_epoch=1)
            )
            await session.commit()
        for p in providers:
            p.gate.set()
        outcomes = await run.wait()
        assert [o.kind for o in outcomes] == ["not_owner", "terminal"]
        assert (await store.get_run(child)).status == "running"
        assert sum(effects.counts().provider.values()) == 2
    finally:
        for p in providers:
            p.gate.set()
        await run.aexit()
        await close_plan(invocation)


async def test_cancel_during_error_read_finalizes_the_accepted_claim(store, monkeypatch):
    module = driver()
    invocation, _, effects = await plan(store)
    entered = asyncio.Event()
    real_runner, real_read = module._run_prepared, store.get_run
    armed = True

    async def execute(branch, prepared, claim, **kwargs):
        if branch.call.id == invocation.call_ids[0]:
            raise PersistenceError()
        return await real_runner(branch, prepared, claim, **kwargs)

    async def read(run_id):
        nonlocal armed
        if armed and run_id == invocation.branches[0].child_run_id:
            armed = False
            entered.set()
            await asyncio.Event().wait()
        return await real_read(run_id)

    monkeypatch.setattr(module, "_run_prepared", execute)
    run = await module.start_fanout(invocation)
    monkeypatch.setattr(store, "get_run", read)
    waiter = asyncio.create_task(run.wait())
    try:
        await wait_for_gate(entered, task=run.owners[0])
        await run.owners[1]
        waiter.cancel("first")
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert (await real_read(invocation.branches[0].child_run_id)).status == "cancelled"
        assert run.owners_done()
        assert effects.counts().provider_calls("0") == 0
        assert effects.counts().provider_calls("1") == 1
    finally:
        await asyncio.gather(waiter, return_exceptions=True)
        await run.aexit()
        await close_plan(invocation)
