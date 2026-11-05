"""Public followers keep safety progress under hints; all timing uses a fake clock."""

import asyncio
import math
from statistics import median

import pytest

from dendrux.store import EventFeed, FollowEnd, RunStore, StoredEvent, _follow
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, FakeState, ticks
from tests.unit.test_follow_coordinator import clock as clock


@pytest.fixture(autouse=True)
def deterministic_schedule(monkeypatch):
    monkeypatch.setattr(_follow.random, "uniform", lambda low, high: (low + high) / 2)


async def advance(clock, live, hot, seconds, step=0.05):
    for _ in range(round(seconds / step)):
        for offset in range(0, len(hot), 500):
            live.hint(*hot[offset : offset + 500])
        clock.advance(step)
        await ticks(40)


async def consume_terminal(follow, delivered, rid):
    async for item in follow:
        if isinstance(item, StoredEvent) and item.event_type == "run.completed":
            delivered.append(rid)
        if isinstance(item, FollowEnd):
            return item


@pytest.mark.parametrize(
    "state_type,hot_count,quiet_count",
    [
        (FakeState, 100, 1),
        (FakeState, 100, 7),
        (BatchedState, 4500, 1),
        (BatchedState, 4500, 7),
        (BatchedState, 4500, 401),
    ],
)
async def test_missed_terminal_hints_progress_across_tenants(
    clock, state_type, hot_count, quiet_count
):
    state, live = state_type(), FakeLive()
    state.live_updates = live
    reader = RunStore(state)
    feed = EventFeed(reader, max_followers=5000, max_followers_per_tenant=None)
    quiet = [f"quiet{i}" for i in range(quiet_count)]
    hot = [f"hot{i}" for i in range(hot_count)]
    follows, consumers, delivered = [], [], []
    try:
        for rid in quiet + hot:
            tenant = "quiet-tenant" if rid in quiet else "hot-tenant"
            state.create(rid, tenant=tenant)
            state.write(rid)
            follow = feed.follow(rid, tenant_id=tenant, until_terminal=rid in quiet)
            await follow.start()
            follows.append(follow)
        await advance(clock, live, [], 20)
        assert live.healthy
        assert set(quiet + hot) <= {rid for ids in state.event_reads for rid in ids}
        for rid, follow in zip(quiet, follows, strict=False):
            first = await anext(follow)
            assert isinstance(first, StoredEvent) and first.sequence_index == 0
            state.write(rid, "run.completed")
            state.statuses[rid] = "success"
            consumers.append(asyncio.create_task(consume_terminal(follow, delivered, rid)))
        before = len(state.event_reads)
        queries_before = query_count(state)
        started = clock.now
        for _ in range(1200):
            await advance(clock, live, hot, 0.05)
            if all(task.done() for task in consumers):
                break
        elapsed = clock.now - started
        print(
            f"{state_type.__name__}: hot={hot_count}, quiet={quiet_count}, terminal={elapsed:.2f}s"
        )
        assert all(task.done() for task in consumers), (
            "missed terminal hint starved behind hot tenant"
        )
        assert all(task.result().reason == "terminal" for task in consumers)
        assert delivered == quiet
        first_reads = list(
            dict.fromkeys(rid for ids in state.event_reads[before:] for rid in ids if rid in quiet)
        )
        assert first_reads == quiet
        # Token bucket: 20 initial tokens plus 20 per fake second, including status reads.
        assert query_count(state) - queries_before <= 20 + math.ceil(20 * elapsed)
    finally:
        for task in consumers:
            task.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
        for follow in follows:
            await follow.aclose()
        await reader.close()
        await live.close()


def query_count(state):
    return len(state.event_reads) + len(state.status_reads) + len(state.per_run_status_reads)


@pytest.mark.parametrize("state_type", [FakeState, BatchedState])
async def test_normal_hints_remain_prompt(clock, state_type):
    state, live = state_type(), FakeLive()
    state.live_updates = live
    reader = RunStore(state)
    state.create("hot", tenant="tenant")
    follow = reader.follow("hot", tenant_id="tenant", until_terminal=False)
    consumer = None
    latencies = []
    try:
        await follow.start()
        await ticks(40)
        assert live.healthy
        for _ in range(30):
            state.write("hot")
            started = clock.now
            consumer = asyncio.create_task(anext(follow))
            live.hint("hot")
            for _ in range(20):
                clock.advance(0.01)
                await ticks(40)
                if consumer.done():
                    break
            assert consumer.done()
            assert isinstance(consumer.result(), StoredEvent)
            latencies.append(clock.now - started)
            await advance(clock, live, [], 0.2)
        print(
            f"{state_type.__name__}: fake hint median={median(latencies):.3f}s "
            f"max={max(latencies):.3f}s"
        )
        assert max(latencies) <= 0.15
    finally:
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        await follow.aclose()
        await reader.close()
        await live.close()


@pytest.mark.parametrize("state_type", [FakeState, BatchedState])
@pytest.mark.parametrize("count", [0, 1, 25])
async def test_idle_and_all_hinted_sets_do_not_create_fairness_wakeups(clock, state_type, count):
    state, live = state_type(), FakeLive()
    state.live_updates = live
    reader = RunStore(state)
    follows = []
    ids = [f"r{i}" for i in range(count)]
    try:
        for rid in ids:
            state.create(rid)
            follow = reader.follow(rid, tenant_id=None, until_terminal=False)
            await follow.start()
            follows.append(follow)
        await advance(clock, live, [], 5)
        for rid in ids:
            state.write(rid)
        await advance(clock, live, ids, 5)
        for follow in follows:
            item = await anext(follow)
            assert isinstance(item, StoredEvent) and item.sequence_index == 0
        await advance(clock, live, [], 3)
        coordinator = reader._follow_coordinator()
        before = query_count(state)
        # An exhausted allowance must not wake an otherwise idle coordinator.
        coordinator._reads_until_oldest = 0
        assert coordinator._next_wake(clock.now) > clock.now
        await advance(clock, live, [], 1)
        assert query_count(state) == before
        for follow in follows:
            await follow.aclose()
        assert coordinator._next_wake(clock.now) > clock.now
    finally:
        for follow in follows:
            await follow.aclose()
        await reader.close()
        await live.close()


async def test_oldest_healthy_turn_precedes_retry_jobs_after_budget_boundary(clock):
    from dendrux.store._follow_recovery import _Isolation
    from tests.unit.test_follow_recovery import watch

    state = FakeState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    quiet = watch(reader, state, "quiet")
    hot = watch(reader, state, "hot")
    hot.reason = "hint"
    coordinator._reads_until_oldest = 1
    try:
        assert await coordinator._read_due(clock.now)
        assert state.event_reads == [("hot",)]
        assert coordinator._reads_until_oldest == 0
        retry = watch(reader, state, "retry")
        job = _Isolation([retry], status=False)
        job.queue()
        coordinator._isolations.append(job)
        assert await coordinator._read_due(clock.now)
        assert state.event_reads == [("hot",), (quiet.run_id,)]
        assert list(coordinator._isolations) == [job]
    finally:
        await reader.close()


async def test_reserved_turn_on_event_only_batch_store_costs_at_most_two_reads(clock, monkeypatch):
    from tests.unit.test_follow_recovery import watch

    monkeypatch.setattr(_follow, "MAX_READS_PER_S", 1e9)
    state = BatchedState()
    state.get_run_statuses = None
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    for i in range(200):
        watch(reader, state, f"r{i}")
    coordinator._reads_until_oldest = 0
    try:
        assert await coordinator._read_due(clock.now)
        assert state.event_reads == [("r0",)]
        assert query_count(state) == 2
    finally:
        await reader.close()
