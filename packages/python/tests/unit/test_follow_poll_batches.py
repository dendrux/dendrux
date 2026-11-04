"""Quiet safety reads stay batched; removed watches never consume queued reads."""

import asyncio
import random

from dendrux.store import RunStore, _follow
from tests.unit.test_follow_coordinator import BatchedState, Clock, FakeState
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_coordinator import fast_budget as fast_budget


def watched(c, state, count):
    for i in range(count):
        rid = str(i)
        state.create(rid)
        run = _follow._WatchedRun(c, rid, None, i)
        run.followers.add(_follow._Follower(False, 10))
        run.events_due = run.status_due = c._now()
        c.runs[rid] = run


async def test_thousand_idle_runs_share_default_jitter_batches(clock: Clock, monkeypatch):
    state = BatchedState()
    reader = RunStore(state)
    c = reader._follow_coordinator()
    monkeypatch.setattr(c, "_healthy", lambda: True)
    monkeypatch.setattr(_follow.random, "uniform", random.Random(20261105).uniform)
    watched(c, state, 1000)
    start = clock.now
    steady_reads = 0
    while clock.now < start + 60:
        chunk = c._due_chunk(clock.now)
        if chunk:
            before = len(state.event_reads) + len(state.status_reads)
            await c._cycle(chunk, clock.now)
            if clock.now >= start + 20:
                steady_reads += len(state.event_reads) + len(state.status_reads) - before
            clock.advance(0.001)
        else:
            clock.now = c._next_wake(clock.now)
    assert steady_reads <= 60  # <=1.5 reads/s at 1,000 idle runs, default ±20% jitter.
    assert all(run.reads >= 4 for run in c.runs.values())
    await reader.close()


async def test_removed_runs_skip_token_wait_and_status_phase(monkeypatch):
    state = FakeState()
    reader = RunStore(state)
    c = reader._follow_coordinator()
    watched(c, state, 200)
    waiting, release = asyncio.Event(), asyncio.Event()
    tokens = 0

    async def token():
        nonlocal tokens
        tokens += 1
        if tokens == 2:
            waiting.set()
            await release.wait()

    monkeypatch.setattr(c, "_token", token)
    task = asyncio.create_task(c._cycle(list(c.runs.values()), c._now()))
    await waiting.wait()
    for run in list(c.runs.values()):
        run.followers.clear()
        c.unwatch(run)
    release.set()
    await task
    assert len(state.event_reads) == 1
    assert state.per_run_status_reads == []
    assert tokens == 2
    await reader.close()
