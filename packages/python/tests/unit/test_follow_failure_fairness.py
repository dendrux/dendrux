"""Failed batches must relinquish their age without consuming every healthy turn."""

import asyncio

import pytest

from dendrux.store import EventFeed, RunStore, StoredEvent
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, FakeState
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_healthy_fairness import advance
from tests.unit.test_follow_healthy_fairness import deterministic_schedule as deterministic_schedule
from tests.unit.test_follow_recovery import watch


@pytest.mark.parametrize(
    "state_type,phase", [(FakeState, "events"), (BatchedState, "events"), (BatchedState, "status")]
)
async def test_quiet_followers_progress_while_failed_runs_keep_hinting(clock, state_type, phase):
    state, live = state_type(), FakeLive()
    state.live_updates = live
    reader = RunStore(state)
    feed = EventFeed(reader, max_followers=1000, max_followers_per_tenant=None)
    hot = [f"hot{i}" for i in range(500)]
    bad = [f"bad{i}" for i in range(10)]
    quiet = [f"quiet{i}" for i in range(3)]
    follows, consumers = [], []
    try:
        for rid in hot + bad + quiet:
            state.create(rid, tenant="quiet" if rid in quiet else "busy")
            state.write(rid)
            follow = feed.follow(rid, tenant_id=state.tenants[rid], until_terminal=False)
            await follow.start()
            follows.append(follow)
        await advance(clock, live, [], 60)
        assert live.healthy
        for follow in follows[-3:]:
            assert (await anext(follow)).sequence_index == 0
        if phase == "events":
            state.fail_runs.update(bad)
        else:
            statuses = state.get_run_statuses

            async def failing_statuses(ids):
                result = await statuses(ids)
                if set(ids).intersection(bad):
                    raise OSError("unavailable status")
                return result

            state.get_run_statuses = failing_statuses
        for rid, follow in zip(quiet, follows[-3:], strict=True):
            state.write(rid)
            consumers.append(asyncio.create_task(next_event(follow)))
        started = clock.now
        for _ in range(3600):
            await advance(clock, live, hot + bad, 0.05)
            if all(task.done() for task in consumers):
                break
        print(f"{state_type.__name__} {phase}: quiet delivery {clock.now - started:.2f}s")
        assert all(task.done() for task in consumers), (
            "failed hinted batches starved quiet followers"
        )
        assert all(
            isinstance(task.result(), StoredEvent) and task.result().sequence_index == 1
            for task in consumers
        )
    finally:
        for task in consumers:
            task.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
        for follow in follows:
            await follow.aclose()
        await reader.close()
        await live.close()


async def next_event(follow):
    async for item in follow:
        if isinstance(item, StoredEvent):
            return item
    raise AssertionError("follower ended before its event")


@pytest.mark.parametrize("status", [False, True])
@pytest.mark.parametrize("suspect", [False, True])
async def test_failed_group_advances_only_its_healthy_failed_phase(clock, status, suspect):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    runs = [watch(reader, state, str(i)) for i in range(2)]
    failed = "status_due" if status else "events_due"
    other = "events_due" if status else "status_due"
    for run in runs:
        setattr(run, failed, clock.now - 10)
        setattr(run, other, clock.now + 5)
        if suspect:
            coordinator._suspects.mark(run)
    orders = [run.order for run in runs]
    try:
        coordinator._queue_isolation(runs, status)
        for run, order in zip(runs, orders, strict=True):
            assert getattr(run, failed) == (clock.now - 10 if suspect else clock.now)
            assert getattr(run, other) == clock.now + 5
            assert (run.order == order) if suspect else (run.order > order)
    finally:
        await reader.close()


async def test_hints_keep_most_turns_ahead_of_a_quiet_backlog(clock, monkeypatch):
    state = FakeState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    # Drive the coordinator explicitly so each request and budget boundary is observable.
    monkeypatch.setattr(coordinator, "revive", lambda: None)
    for i in range(40):
        watch(reader, state, f"quiet{i}")
    for i in range(20):
        watch(reader, state, f"hot{i}")
    try:
        for i in range(20):
            coordinator.on_hints((f"hot{i}",))
            assert await coordinator._read_due(clock.now)
            clock.advance(0.1)
        hot_reads = sum(ids[0].startswith("hot") for ids in state.event_reads)
        quiet_reads = len(state.event_reads) - hot_reads
        assert hot_reads >= 18, "oldest reservations consumed more than a tenth of these turns"
        assert quiet_reads >= 1, "the quiet backlog still needs reserved turns"
    finally:
        await reader.close()


async def test_reserved_batches_exclude_due_suspects(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    suspect = watch(reader, state, "suspect")
    healthy = watch(reader, state, "healthy")
    coordinator._suspects.mark(suspect)
    coordinator._reads_until_oldest = 0
    try:
        assert coordinator._oldest_healthy_chunk([suspect, healthy]) == [healthy]
    finally:
        await reader.close()


async def test_empty_healthy_pool_preserves_the_next_reserved_turn(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    suspect = watch(reader, state, "suspect")
    coordinator._suspects.mark(suspect)
    coordinator._reads_until_oldest = 0
    try:
        assert coordinator._oldest_healthy_chunk([suspect]) == []
        healthy = watch(reader, state, "healthy")
        assert coordinator._oldest_healthy_chunk([suspect, healthy]) == [healthy]
    finally:
        await reader.close()
