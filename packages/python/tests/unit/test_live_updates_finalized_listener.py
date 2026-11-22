"""A listener coroutine finalized after its loop closed ends quietly."""

import asyncio
import gc
import sys

import pytest

from dendrux.db import _live_updates as live


class _FailingCleanupBus(live.LiveUpdates):
    """Its receive stream fails while closing, as a transport on a dead loop does."""

    async def send(self, messages):
        return None

    async def receive(self):
        yield None
        try:
            await asyncio.Future()
        except BaseException as error:
            raise RuntimeError("transport cleanup failed") from error


def _abandon_listener(bus: _FailingCleanupBus) -> None:
    loop = asyncio.new_event_loop()

    async def start() -> None:
        bus._listener_task = asyncio.create_task(bus._listen())
        async with asyncio.timeout(5):
            while not bus.healthy:
                await asyncio.sleep(0.01)

    loop.run_until_complete(start())
    loop.close()  # the listener task is still pending, its loop gone


def test_listener_finalized_without_a_loop_raises_nothing_unraisable(monkeypatch):
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda args: unraisable.append(args))
    bus = _FailingCleanupBus()
    _abandon_listener(bus)
    bus._listener_task = None
    gc.collect()
    assert unraisable == []


def test_listener_failure_without_a_loop_ends_the_coroutine_like_close():
    bus = _FailingCleanupBus()
    with pytest.raises(GeneratorExit):
        bus._listener_failure(RuntimeError("x"))
    assert not bus._listener_failed  # nothing is reported for a coroutine being finalized


def test_postgres_wait_finalized_after_its_loop_closed_raises_nothing_unraisable(monkeypatch):
    pytest.importorskip("asyncpg")
    from dendrux.db._postgres_updates import PostgresUpdates

    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda args: unraisable.append(args))
    bus = PostgresUpdates(url="postgresql://u@localhost/db")
    loop = asyncio.new_event_loop()
    bus._loop = loop

    async def wait_for_a_message() -> None:
        await bus._wait_message(bus._new_inbound_buffer(), asyncio.Event())

    task = loop.create_task(wait_for_a_message())
    loop.run_until_complete(asyncio.sleep(0.05))
    loop.close()  # the wait is still pending, its loop gone
    del task, loop
    bus._loop = None
    gc.collect()
    assert unraisable == []
