"""Routing races on Postgres (spec §16.2 R4, R6), with deterministic schedules.

Each "process" is its own engine and connection pool over the same private test database,
with its own Agent instance. Schedules use events and barriers, never timing luck. The
adoption step is the fake ``settle_parent`` of the routing tests; its claim is the store's
own compare-and-set, so the "one adoption" assertions hold on real concurrency.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.runtime import _managed_budget, _managed_route
from dendrux.runtime._reclaim import continue_claimed
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse, RunResult, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_routing import (
    ROOT_TEXT,
    TENANT,
    FakeSettle,
    approver,
    fan_out,
    install_fake_settle,
    pause_of,
    published,
    root_agent,
    start_root,
)

DEADLINE = 30


@pytest.fixture(autouse=True)
def _postgres_only(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent connections require Postgres")
    published.clear()


async def _processes(stack: AsyncExitStack, engine: Any, n: int) -> list[Any]:
    """``n`` independent engines (one per simulated process) over the same database."""
    stores = []
    for _ in range(n):
        pool = create_async_engine(engine.url, pool_size=4, max_overflow=0)
        stack.push_async_callback(pool.dispose)
        stores.append(store_for_engine(pool))
    return stores


def finisher_held(name: str) -> tuple[Agent, ScriptedLLM]:
    """A child whose one model call is held until the schedule releases it."""
    llm = ScriptedLLM([LLMResponse(text=f"{name} done")], gate_at=0)
    agent = Agent(name=name, description=f"{name} answers.", prompt="Answer.", provider=llm)
    return agent, llm


async def _owner_finishes(
    store: Any, root: Agent, child: Agent, child_id: str, claim: Any, fake: FakeSettle
) -> RunResult | None:
    """An owner task: run the bound child to its end, then settle the parent as the last
    finisher (a fresh read after its own terminal commit) and continue it if it adopts."""
    root_id = (await store.get_run(child_id)).parent_run_id
    await continue_claimed(
        _managed_budget.inherit(child, root), store, child_id, claim, notifier=None
    )
    tree = await _managed_route.walk_tree(root, store, root_id, tenant_id=TENANT)
    adopted = await fake(store, tree.root)  # type: ignore[arg-type]
    if adopted is None:
        return None
    return await continue_claimed(root, store, root_id, adopted, notifier=None)


@pytest.mark.parametrize("schedule", ["approval_reads_first", "both_read_all_terminal"])
async def test_r4_approval_submit_against_a_siblings_terminal_commit(
    engine: Any, monkeypatch: pytest.MonkeyPatch, schedule: str
) -> None:
    """R4: child 2's routed approval races child 1's terminal commit and settle.

    ``approval_reads_first``: child 1 is held in its model call while the approval runs
    to its settle, which reads child 1 RUNNING; then child 1 finishes and adopts.
    ``both_read_all_terminal``: both settles wait at a barrier until both children are
    terminal, then both read and claim at once. Either way: the leaf is claimed once, one
    adoption, the root continues once.
    """
    barrier = asyncio.Barrier(2)

    async def both_terminal() -> None:
        await barrier.wait()

    fake = FakeSettle(gate=both_terminal if schedule == "both_read_all_terminal" else None)
    install_fake_settle(monkeypatch, fake)
    async with AsyncExitStack() as stack:
        store_a, store_b = await _processes(stack, engine, 2)
        one, one_llm = finisher_held("one")
        two = approver("two")
        root_a, root_b = root_agent(store_a, one, two), root_agent(store_b, one, two)
        root_id = await start_root(store_a, root_a)
        fan = await fan_out(store_a, root_a, root_id, [one, two], run=(1,))
        approval_epoch = (await store_a.get_run(fan.child(1))).claim_epoch

        owner = asyncio.create_task(
            _owner_finishes(store_a, root_a, one, fan.child(0), fan.claims[fan.child(0)], fake)
        )
        await asyncio.wait_for(one_llm.entered.wait(), DEADLINE)
        submit = asyncio.create_task(
            root_b.submit_approval(
                root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 1)
            )
        )
        if schedule == "approval_reads_first":
            routed = await asyncio.wait_for(submit, DEADLINE)
            one_llm.gate.set()
            continued = await asyncio.wait_for(owner, DEADLINE)
            assert routed.status == RunStatus.RUNNING and routed.pendings == ()
            assert fake.calls == [(root_id, False), (root_id, True)]
            assert continued is not None and continued.status == RunStatus.SUCCESS
        else:
            one_llm.gate.set()
            routed, continued = await asyncio.wait_for(asyncio.gather(submit, owner), DEADLINE)
            assert fake.calls == [(root_id, True), (root_id, True)]

        assert fake.adoptions == 1
        assert published == ["two"]
        leaf = await store_a.get_run(fan.child(1))
        assert (leaf.status, leaf.claim_epoch) == (RunStatus.SUCCESS.value, approval_epoch + 1)
        final = await root_a.get_result(root_id, tenant_id=TENANT)
        assert (final.status, final.answer) == (RunStatus.SUCCESS, ROOT_TEXT)
        assert root_a.provider.calls + root_b.provider.calls == 1  # type: ignore[union-attr]


async def test_r6_submits_to_two_leaves_from_two_processes(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6: both callers look the tree up before either claims; each answer applies once to
    its own leaf, nothing crosses, nothing deadlocks, and the root adopts once."""
    fake = install_fake_settle(monkeypatch)
    looked_up = asyncio.Barrier(2)
    walk = _managed_route._walk

    async def walk_then_wait(*args: Any) -> Any:
        walked = await walk(*args)
        await looked_up.wait()
        return walked

    monkeypatch.setattr(_managed_route, "_walk", walk_then_wait)
    async with AsyncExitStack() as stack:
        store_a, store_b = await _processes(stack, engine, 2)
        one, two = approver("one"), approver("two")
        root_a, root_b = root_agent(store_a, one, two), root_agent(store_b, one, two)
        root_id = await start_root(store_a, root_a)
        fan = await fan_out(store_a, root_a, root_id, [one, two])
        epochs = [(await store_a.get_run(c)).claim_epoch for c in fan.children]

        results = await asyncio.wait_for(
            asyncio.gather(
                root_a.submit_approval(
                    root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
                ),
                root_b.submit_approval(
                    root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 1)
                ),
            ),
            DEADLINE,
        )

        assert sorted(published) == ["one", "two"]
        for child, epoch in zip(fan.children, epochs, strict=True):
            record = await store_a.get_run(child)
            assert (record.status, record.claim_epoch) == (RunStatus.SUCCESS.value, epoch + 1)
        assert one.provider.calls == 2 and two.provider.calls == 2  # type: ignore[union-attr]
        assert fake.adoptions == 1
        assert [r.run_id for r in results] == [root_id, root_id]
        assert RunStatus.SUCCESS in {r.status for r in results}
        final = await root_a.get_result(root_id, tenant_id=TENANT)
        assert (final.status, final.answer) == (RunStatus.SUCCESS, ROOT_TEXT)
        assert root_a.provider.calls + root_b.provider.calls == 1  # type: ignore[union-attr]
