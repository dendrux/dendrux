"""C2: scheduler-tick join wake and burst coalescing, with a fake loop clock."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from dendrux.store import EventFeed, RunStore, StoredEvent
from dendrux.store._follow import _Follower, _RunFeed


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
async def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    loop = asyncio.get_running_loop()
    clock = Clock(loop.time())
    monkeypatch.setattr(loop, "time", lambda: clock.now)
    return clock


class Store:
    def __init__(self) -> None:
        self.events: list[StoredEvent] = []
        self.polls = 0
        self.status_reads = 0
        self.block: asyncio.Event | None = None
        self.fetching = asyncio.Event()

    async def get_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None,
        limit: int,
    ) -> list[StoredEvent]:
        self.polls += 1
        self.fetching.set()
        if self.block is not None:
            await self.block.wait()
        return [
            e
            for e in self.events
            if after_sequence_index is None or e.sequence_index > after_sequence_index
        ][:limit]

    async def _status(self, run_id: str, *args: Any) -> str:
        self.status_reads += 1
        return "running"

    def write(self) -> StoredEvent:
        event = StoredEvent(
            sequence_index=len(self.events),
            iteration_index=0,
            event_type="test.event",
            correlation_id=None,
            data={},
            created_at=None,
        )
        self.events.append(event)
        return event


def setup() -> tuple[Store, EventFeed, _RunFeed]:
    store = Store()
    owner = EventFeed(cast("RunStore", store), max_followers=2000, max_followers_per_tenant=None)
    run = _RunFeed(owner, "run", None)
    owner._feeds["run"] = run
    run.add(_Follower(until_terminal=False, limit=10))
    return store, owner, run


async def ticks(count: int = 1) -> None:
    for _ in range(count):
        await asyncio.sleep(0)  # Yield the scheduler; never wait for wall time.


async def test_join_sees_already_written_event_in_one_tick(clock: Clock) -> None:
    store, owner, run = setup()
    try:
        await ticks()
        assert store.polls == 1
        event = store.write()
        follower = _Follower(until_terminal=False, limit=10)
        run.add(follower)
        await ticks()
        assert list(follower.buffer) == [event]
        assert store.polls == 2
        assert store.status_reads == 1
    finally:
        await owner.aclose()


async def test_thousand_concurrent_joins_coalesce_to_one_extra_poll(clock: Clock) -> None:
    store, owner, run = setup()
    try:
        await ticks()
        event = store.write()
        followers = [_Follower(until_terminal=False, limit=10) for _ in range(1000)]

        async def join(follower: _Follower) -> None:
            run.add(follower)

        await asyncio.gather(*(join(follower) for follower in followers))
        await ticks()
        assert all(list(f.buffer) == [event] for f in followers)
        assert store.polls == 2  # One initial poll + one poll for the entire burst.
        await ticks(10)
        assert store.polls == 2
    finally:
        await owner.aclose()


async def test_no_extra_polls_without_joins_and_backoff_is_preserved(clock: Clock) -> None:
    store, owner, _ = setup()
    try:
        await ticks()
        await ticks(10)
        assert store.polls == 1
        clock.advance(0.501)
        await ticks(3)
        assert store.polls == 2
        clock.advance(0.501)
        await ticks(3)
        assert store.polls == 2  # Idle interval has doubled to one second.
        clock.advance(0.501)
        await ticks(3)
        assert store.polls == 3
    finally:
        await owner.aclose()


async def test_join_during_inflight_poll_is_not_lost(clock: Clock) -> None:
    store, owner, run = setup()
    try:
        await ticks()
        store.block = asyncio.Event()
        store.fetching.clear()
        clock.advance(0.501)
        await ticks(3)
        assert store.fetching.is_set()
        followers = [_Follower(until_terminal=False, limit=10) for _ in range(1000)]
        for follower in followers:
            run.add(follower)
        event = store.write()
        store.block.set()
        await ticks(3)
        assert all(list(f.buffer) == [event] for f in followers)
        assert store.polls == 3  # Initial + in-flight + one coalesced pending wake.
        await ticks(10)
        assert store.polls == 3
    finally:
        await owner.aclose()


async def test_join_only_wakes_its_own_run(clock: Clock) -> None:
    store, owner, run = setup()
    other = _RunFeed(owner, "other", None)
    owner._feeds["other"] = other
    other_follower = _Follower(until_terminal=False, limit=10)
    other.add(other_follower)
    try:
        await ticks()
        assert store.polls == 2
        event = store.write()
        joining = _Follower(until_terminal=False, limit=10)
        run.add(joining)
        await ticks()
        assert list(joining.buffer) == [event]
        assert not other_follower.buffer
        assert store.polls == 3
    finally:
        await owner.aclose()
