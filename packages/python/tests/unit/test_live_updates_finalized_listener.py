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


class _CleanupReplacesCancelBus(live.LiveUpdates):
    """Its transport turns a cancel into a RuntimeError, as a failing cleanup does."""

    async def send(self, messages):
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            raise RuntimeError("cleanup failed") from error

    async def receive(self):
        yield None
        await asyncio.Future()

    async def aclose_sender(self):
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            raise RuntimeError("cleanup failed") from error


async def _cancel_after_it_starts(operation) -> None:
    task = asyncio.create_task(operation)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelled_flush_stays_cancelled_when_cleanup_replaces_the_error():
    bus = _CleanupReplacesCancelBus(send_timeout_s=5, flush_deadline_s=5)
    bus.enqueue(["r"])
    await _cancel_after_it_starts(bus.flush())
    await bus.close()


async def test_cancelled_close_stays_cancelled_when_cleanup_replaces_the_error():
    bus = _CleanupReplacesCancelBus(close_timeout_s=5)
    await _cancel_after_it_starts(bus.close())


async def test_send_timeout_with_replaced_error_is_still_an_ordinary_failure():
    bus = _CleanupReplacesCancelBus(send_timeout_s=0.1, flush_deadline_s=5)
    bus.enqueue(["r"])
    await bus.flush()  # the timeout is not the caller's cancellation
    assert bus.stats().dropped_failed == 1


class _CleanupReplacesExitBus(_CleanupReplacesCancelBus):
    """A closing coroutine's GeneratorExit is replaced by a CancelledError."""

    async def send(self, messages):
        try:
            await asyncio.Future()
        except BaseException as error:
            raise asyncio.CancelledError from error

    async def aclose_sender(self):
        await self.send([])


@pytest.mark.parametrize("boundary", ["send", "close"])
def test_send_and_close_finalized_without_a_loop_raise_nothing_unraisable(monkeypatch, boundary):
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda args: unraisable.append(args))
    bus = _CleanupReplacesExitBus(send_timeout_s=30, flush_deadline_s=30, close_timeout_s=30)
    loop = asyncio.new_event_loop()

    async def run() -> None:
        bus.enqueue(["r"])
        operation = bus.flush() if boundary == "send" else bus.close()
        task = asyncio.create_task(operation)
        await asyncio.sleep(0.05)
        bus._abandoned = task

    loop.run_until_complete(run())
    loop.close()
    del loop
    bus.__dict__.pop("_abandoned", None)
    gc.collect()
    assert unraisable == []
