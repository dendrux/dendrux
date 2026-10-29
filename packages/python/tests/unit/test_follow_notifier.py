"""Notification wake hints preserve polling and in-flight wakeups."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import cast

import pytest

from dendrux.db._notify import NotifierState, Subscription
from dendrux.store import EventFeed, RunStore
from dendrux.store._follow import _Follower, _RunFeed
from tests.unit.test_follow_wake import Clock, Store, ticks


class Notifier:
    def __init__(self):
        self.state = NotifierState(True, 0)
        self.event = asyncio.Event()
        self.started = self.stopped = 0
        self.subscribers = 0

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    @asynccontextmanager
    async def subscribe(self, run_id):
        self.subscribers += 1
        try:
            yield Subscription(self.event, self.state)
        finally:
            self.subscribers -= 1

    def health(self, healthy):
        self.state.healthy = healthy
        self.state.generation += 1
        self.event.set()


@pytest.fixture
async def clock(monkeypatch):
    loop = asyncio.get_running_loop()
    value = Clock(loop.time())
    monkeypatch.setattr(loop, "time", lambda: value.now)
    return value


def setup():
    store, notifier = Store(), Notifier()
    owner = EventFeed(cast("RunStore", store), notifier=notifier, poll_healthy_s=30)
    run = _RunFeed(owner, "run", None)
    owner._feeds["run"] = run
    follower = _Follower(until_terminal=False, limit=10)
    run.add(follower)
    return store, notifier, owner, run, follower


async def test_notification_during_fetch_is_pending_and_coalesces(clock):
    store, notifier, owner, run, follower = setup()
    try:
        await ticks(3)
        assert notifier.subscribers == 1
        assert notifier.started == 0  # Borrowed lifecycle belongs to the developer.
        store.block = asyncio.Event()
        store.fetching.clear()
        notifier.event.set()
        await ticks(3)
        assert store.fetching.is_set()
        for _ in range(1000):
            notifier.event.set()
        event = store.write()
        store.block.set()
        await ticks(5)
        assert list(follower.buffer) == [event]
        assert store.polls == 3
        await ticks(10)
        assert store.polls == 3
    finally:
        await owner.aclose()
    assert notifier.subscribers == 0 and notifier.stopped == 0


async def test_health_changes_switch_intervals_immediately_both_ways(clock):
    store, notifier, owner, run, follower = setup()
    try:
        await ticks(3)
        clock.advance(29)
        await ticks(4)
        assert store.polls == 1
        clock.advance(1.001)
        await ticks(4)
        assert store.polls == 2
        notifier.health(False)
        await ticks(4)
        assert store.polls == 3
        clock.advance(0.501)
        await ticks(4)
        assert store.polls == 4
        clock.advance(0.501)
        await ticks(4)
        assert store.polls == 5  # Degraded polling never waits on healthy backoff.
        notifier.health(True)
        await ticks(4)
        assert store.polls == 6
        clock.advance(29)
        await ticks(4)
        assert store.polls == 6
        clock.advance(1.001)
        await ticks(4)
        assert store.polls == 7
    finally:
        await owner.aclose()


async def test_healthy_status_every_fourth_idle_poll(clock):
    store, notifier, owner, run, follower = setup()
    try:
        await ticks(3)
        assert store.status_reads == 1
        for _ in range(3):
            clock.advance(30.001)
            await ticks(4)
        assert store.polls == 4
        assert store.status_reads == 2
    finally:
        await owner.aclose()


@pytest.mark.parametrize("factory", ["engine", "url"])
async def test_runstore_borrows_notifier_and_wires_feed(factory):
    from sqlalchemy.ext.asyncio import create_async_engine

    notifier = Notifier()
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        if factory == "engine":
            store = RunStore.from_engine(engine, notifier=notifier)
        else:
            store = RunStore.from_database_url("sqlite+aiosqlite://", notifier=notifier)
        store.follow("run", tenant_id=None)
        assert store._event_feed.notifier is notifier
        await store.close()
        assert notifier.started == notifier.stopped == 0
    finally:
        await engine.dispose()


async def test_runstore_owns_only_listen_url_notifier(monkeypatch):
    import dendrux.store as module

    notifier = Notifier()
    monkeypatch.setattr(module, "PostgresEventNotifier", lambda *a, **kw: notifier, raising=False)
    store = RunStore.from_database_url("sqlite+aiosqlite://", listen_url="unused")
    async with store:
        assert notifier.started == 1
    assert notifier.stopped == 1
    await store.close()
    assert notifier.stopped == 1


async def test_mutually_exclusive_notifier_options():
    with pytest.raises(ValueError):
        RunStore.from_database_url("sqlite+aiosqlite://", notifier=Notifier(), listen_url="unused")
