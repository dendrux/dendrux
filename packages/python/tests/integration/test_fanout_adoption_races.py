"""0.8 fan-out adoption races (T4c, spec §16.2 R2 and the adoption part of R5), Postgres only.

Each "process" is its own single-connection pool, store and Agent objects. Schedules are forced
with barriers and gates, never timing luck:

- R2: the last two finishers each read a sibling non-terminal, then both commit, then each
  applies the last-finisher rule (a fresh read after its own commit, spec §7.2) and both
  contend for the claim. Exactly one adopts. The same schedule with the read moved before the
  commit adopts nothing, which shows the schedule really is the lost-wake one.
- R5: cancel before the claim, cancel between the pre-claim reads and the claim, and cancel
  while the winner completes rows. A cancelled root is never revived.

The settle step here is a minimal in-test reading of §7.2 (a fresh ``get_managed_children``
read, then ``adopt_settled``); the driver's ``settle_parent`` is T4b's.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime import _reclaim
from dendrux.runtime._managed_adoption import adopt_settled
from dendrux.runtime.state import ClaimResult
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_adoption import (
    TENANT,
    Child,
    Fanout,
    assert_adopted_once,
    intents,
    lead,
    node_of,
    observations,
    parent_row,
    parked,
    transform_counts,
)


def _postgres_only(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("independent-connection races run on Postgres; SQLite shares a connection")


async def _processes(stack: AsyncExitStack, engine: Any, n: int) -> list[Any]:
    """``n`` independent stores, each on its own single-connection pool."""
    stores = []
    for i in range(n):
        pool = create_async_engine(
            engine.url,
            pool_size=1,
            max_overflow=0,
            connect_args={"server_settings": {"application_name": f"t4c-{i}"}},
        )
        stack.push_async_callback(pool.dispose)
        stores.append(store_for_engine(pool))
    return stores


def _gate_claims(store: Any, barrier: asyncio.Barrier) -> None:
    """Hold this store's claim until every contender has reached its own claim."""
    real = store.submit_and_claim

    async def claim(*args: Any, **kwargs: Any) -> Any:
        await asyncio.wait_for(barrier.wait(), timeout=10)
        return await real(*args, **kwargs)

    store.submit_and_claim = claim


# ---------------------------------------------------------------------------
# R2, the forced lost-wake schedule
# ---------------------------------------------------------------------------


async def _finisher(
    store: Any,
    fanout: Fanout,
    child: int,
    effects: FanoutEffects,
    barriers: tuple[asyncio.Barrier, asyncio.Barrier],
    *,
    read_after_commit: bool,
) -> ClaimResult | None:
    """Finish one child, then settle its parent: the read either follows or precedes the commit."""
    ids = list(fanout.child_ids)
    before = await store.get_managed_children(fanout.run_id, ids, tenant_id=TENANT)
    assert any(r.status == RunStatus.RUNNING.value for r in before if r.run_id != ids[child])
    await asyncio.wait_for(barriers[0].wait(), timeout=10)  # both read a sibling running
    assert await store.finalize_run(
        ids[child],
        status=RunStatus.SUCCESS.value,
        answer=f"report {child}",
        claim_epoch=0,
        expected_current_status=RunStatus.RUNNING.value,
    )
    await asyncio.wait_for(barriers[1].wait(), timeout=10)  # both committed
    rows = (
        await store.get_managed_children(fanout.run_id, ids, tenant_id=TENANT)
        if read_after_commit
        else before
    )
    if len(rows) != len(ids) or any(r.status != RunStatus.SUCCESS.value for r in rows):
        return None
    root = lead(tool_result_transform=effects.transform())
    return await adopt_settled(store, node_of(fanout, root))


async def _lost_wake(engine: Any, *, read_after_commit: bool) -> tuple[Fanout, Any, list[Any]]:
    effects = FanoutEffects()
    fanout = await parked(
        store_for_engine(engine),
        lead(),
        [
            Child("report 0"),
            Child(status=RunStatus.RUNNING.value),
            Child(status=RunStatus.RUNNING.value),
        ],
    )
    async with AsyncExitStack() as stack:
        one, two = await _processes(stack, engine, 2)
        claims = asyncio.Barrier(2)
        _gate_claims(one, claims)
        _gate_claims(two, claims)
        barriers = (asyncio.Barrier(2), asyncio.Barrier(2))
        outcomes = await asyncio.gather(
            _finisher(one, fanout, 1, effects, barriers, read_after_commit=read_after_commit),
            _finisher(two, fanout, 2, effects, barriers, read_after_commit=read_after_commit),
        )
    return fanout, effects, list(outcomes)


async def test_r2_the_last_finishers_adopt_exactly_once(engine: Any) -> None:
    _postgres_only(engine)
    fanout, effects, outcomes = await _lost_wake(engine, read_after_commit=True)
    winners = [o for o in outcomes if o is not None]
    assert len(winners) == 1, "both finishers contend; one claim wins"
    assert transform_counts(effects, fanout) == [1, 1, 1]
    await assert_adopted_once(engine, fanout, winners[0])
    assert (await parent_row(engine, fanout.run_id)).claim_epoch == 2


async def test_r2_revert_reading_before_the_commit_loses_the_wake(engine: Any) -> None:
    """Revert check of the rule: with the read before the commit nobody adopts (forced schedule)."""
    _postgres_only(engine)
    fanout, effects, outcomes = await _lost_wake(engine, read_after_commit=False)
    assert outcomes == [None, None]
    assert effects.counts().transform_calls() == 0
    row = await parent_row(engine, fanout.run_id)
    assert (row.status, row.claim_epoch) == (RunStatus.PAUSED.value, 1)
    assert [r.outcome for r in await intents(engine, fanout.run_id, fanout)] == ["started"] * 3


# ---------------------------------------------------------------------------
# R5 (adoption part), cancel vs the adoption claim
# ---------------------------------------------------------------------------


async def _cancel(store: Any, run_id: str) -> None:
    await lead(state_store=store).cancel_run(run_id, tenant_id=TENANT)


async def _assert_never_adopted(engine: Any, fanout: Fanout, effects: FanoutEffects) -> None:
    row = await parent_row(engine, fanout.run_id)
    assert row.status == RunStatus.CANCELLED.value
    assert "submission" not in (row.pause_data or {})
    assert effects.counts().transform_calls() == 0
    assert [r.outcome for r in await intents(engine, fanout.run_id, fanout)] == ["started"] * 2
    assert await observations(engine, fanout.run_id) == []


async def test_r5_cancel_before_the_claim_adopts_nothing(engine: Any) -> None:
    _postgres_only(engine)
    effects = FanoutEffects()
    fanout = await parked(store_for_engine(engine), lead(), [Child("a"), Child("b")])
    async with AsyncExitStack() as stack:
        adopter, canceller = await _processes(stack, engine, 2)
        await _cancel(canceller, fanout.run_id)
        root = lead(tool_result_transform=effects.transform())
        assert await adopt_settled(adopter, node_of(fanout, root)) is None
    await _assert_never_adopted(engine, fanout, effects)


async def test_r5_cancel_between_the_reads_and_the_claim_wins(engine: Any) -> None:
    """The cancel commits after the adopter's pre-claim reads; the claim's CAS then loses."""
    _postgres_only(engine)
    effects = FanoutEffects()
    fanout = await parked(store_for_engine(engine), lead(), [Child("a"), Child("b")])
    async with AsyncExitStack() as stack:
        adopter, canceller = await _processes(stack, engine, 2)
        reached, release = asyncio.Event(), asyncio.Event()
        real = adopter.submit_and_claim

        async def held(*args: Any, **kwargs: Any) -> Any:
            reached.set()
            await asyncio.wait_for(release.wait(), timeout=10)
            return await real(*args, **kwargs)

        adopter.submit_and_claim = held
        root = lead(tool_result_transform=effects.transform())
        adoption = asyncio.ensure_future(adopt_settled(adopter, node_of(fanout, root)))
        await asyncio.wait_for(reached.wait(), timeout=10)
        await _cancel(canceller, fanout.run_id)
        release.set()
        assert await adoption is None
    await _assert_never_adopted(engine, fanout, effects)


async def test_r5_cancel_during_the_completions_stops_the_continuation(engine: Any) -> None:
    """Claimed first: rows keep completing under the claim; the continuation applies the cancel."""
    _postgres_only(engine)
    effects = FanoutEffects()
    entered, release = asyncio.Event(), asyncio.Event()

    async def gated(ctx: Any) -> Any:
        if ctx.call_id == fanout.calls[0].id:
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=10)
        return ctx.result

    fanout = await parked(store_for_engine(engine), lead(), [Child("a"), Child("b")])
    async with AsyncExitStack() as stack:
        adopter, canceller = await _processes(stack, engine, 2)
        root = lead(tool_result_transform=effects.transform(gated))
        adoption = asyncio.ensure_future(adopt_settled(adopter, node_of(fanout, root)))
        await asyncio.wait_for(entered.wait(), timeout=10)
        await _cancel(canceller, fanout.run_id)
        assert (await parent_row(engine, fanout.run_id)).cancel_requested is True
        release.set()
        claim = await adoption
        assert isinstance(claim, ClaimResult)
        await assert_adopted_once(engine, fanout, claim)
        assert transform_counts(effects, fanout) == [1, 1]
        llm = ScriptedLLM([LLMResponse(text="never")])
        result = await _reclaim.continue_claimed(
            lead(llm), adopter, fanout.run_id, claim, notifier=None
        )
    assert result.status == RunStatus.CANCELLED and llm.calls == 0
    assert (await parent_row(engine, fanout.run_id)).status == RunStatus.CANCELLED.value
