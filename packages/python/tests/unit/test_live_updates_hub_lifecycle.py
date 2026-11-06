"""Indexed wakers and safe cleanup after an owning loop has shut down."""

import asyncio
from collections import Counter

import pytest

from dendrux.store._follow import _RunWaker
from tests.unit.test_live_updates_base import Bus


async def test_only_matching_wakers_are_visited_and_gaps_reach_all():
    class Waker(_RunWaker):
        def on_hints(self, ids):
            calls[self.run_id] += 1
            super().on_hints(ids)

    calls = Counter()
    bus = Bus()
    wakers = [Waker(str(i)) for i in range(10_000)]
    handles = [bus._register(waker) for waker in wakers]
    bus._deliver('["7","9","7"]')
    assert calls == {"7": 1, "9": 1}
    assert sum(w.event.is_set() for w in wakers) == 2
    bus._unregister(handles[7])
    wakers[7].event.clear()
    bus._deliver('["7"]')
    assert calls == {"7": 1, "9": 1}
    bus._deliver(bus.GAP)
    assert sum(w.event.is_set() for w in wakers) == 9999
    await bus.close()
    assert not bus._consumers
    assert not bus._wakers_by_run


async def test_index_preserves_duplicates_and_general_consumers():
    bus = Bus()
    first, second = _RunWaker("same"), _RunWaker("same")
    first_handle = bus._register(first)
    bus._register(second)
    seen = []

    class General:
        def on_hints(self, ids):
            seen.append(ids)

        def on_gap(self):
            pass

    bus._register(General(), stream_events=False)
    bus._deliver('["same","other"]')
    assert first.event.is_set() and second.event.is_set()
    assert seen == [("same", "other")]
    bus._unregister(first_handle)
    second.event.clear()
    bus._deliver('["same"]')
    assert second.event.is_set()
    await bus.close()


def test_store_close_after_bound_loop_closed_aborts_without_awaiting():
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.store import PostgresStateStore

    class Cleanup(Bus):
        def __init__(self):
            super().__init__()
            self.listener_aborts = 0

        def abort_listener(self):
            self.listener_aborts += 1

        async def aclose_sender(self):
            raise AssertionError("cannot await the old loop")

    bus = Cleanup()
    store = PostgresStateStore(
        create_async_engine("postgresql+asyncpg://u@localhost/db"), live_updates=bus
    )

    async def first():
        bus.enqueue(["pending"])
        bus._register(_RunWaker("pending"))
        await bus._acquire_listener()

    asyncio.run(first())
    assert bus._loop.is_closed()
    asyncio.run(store.close())
    asyncio.run(store.close())
    assert bus.aborts == 1 and bus.listener_aborts == 1
    assert bus._state == "closed"
    assert bus.stats().pending == 0 and bus.stats().dropped_closed == 1
    assert not bus._leases and not bus._consumers


async def test_another_live_loop_still_refuses_close_without_mutation():
    bus = Bus()
    bus.enqueue(["pending"])

    async def other():
        with pytest.raises(RuntimeError, match="another event loop"):
            await bus.close()

    await asyncio.to_thread(asyncio.run, other())
    assert bus._state == "open" and bus.stats().pending == 1
    assert bus.aborts == 0
    await bus.close()
