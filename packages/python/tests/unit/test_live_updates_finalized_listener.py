"""A listener coroutine finalized after its loop closed ends quietly."""

import asyncio
import gc
import sys
from typing import Any

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


def _abandon_listener(bus: _FailingCleanupBus) -> Any:
    """Strand the listener on a closed loop; returns its outermost coroutine."""
    loop = asyncio.new_event_loop()

    async def start() -> None:
        bus._listener_task = asyncio.create_task(bus._listen())
        async with asyncio.timeout(5):
            while not bus.healthy:
                await asyncio.sleep(0.01)

    loop.run_until_complete(start())
    loop.close()  # the listener task is still pending, its loop gone
    return bus._listener_task.get_coro()


def test_listener_finalized_without_a_loop_raises_nothing_unraisable(monkeypatch):
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda args: unraisable.append(args))
    bus = _FailingCleanupBus()
    coroutine = _abandon_listener(bus)
    bus._listener_task = None
    # Outermost first, as close() on a task's coroutine does; GC order of the inner
    # frames differs between interpreters (see the send/close test below).
    coroutine.close()
    del coroutine
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
    coroutines: list[Any] = []

    async def run() -> None:
        if boundary == "send":
            bus.enqueue(["r"])  # close needs nothing pending, so it reaches aclose_sender
        operation = bus.flush() if boundary == "send" else bus.close()
        task = asyncio.create_task(operation)
        coroutines.append(task.get_coro())
        await asyncio.sleep(0.05)

    loop.run_until_complete(run())
    loop.close()
    del loop
    # Finalize the outermost coroutine first, as close() on a task's coroutine does.
    # GC may instead close an inner frame (here _close_resources) on its own, and
    # this double replaces GeneratorExit with CancelledError there on any Python;
    # which frame goes first depends on the interpreter's collection order.
    coroutines[0].close()
    coroutines.clear()
    gc.collect()
    assert unraisable == []


def test_finalize_close_waiting_for_flush_with_live_listener(monkeypatch):
    bus = _CleanupReplacesCancelBus(send_timeout_s=30, flush_deadline_s=30, close_timeout_s=30)
    waiting_for_flush = asyncio.Event()
    wait_until_idle = bus._idle.wait

    async def wait_for_flush():
        waiting_for_flush.set()
        await wait_until_idle()

    monkeypatch.setattr(bus._idle, "wait", wait_for_flush)

    async def start(bus):
        lease = await bus._acquire_listener()
        bus.enqueue(["run"])
        flushing = asyncio.create_task(bus.flush())
        closing = asyncio.create_task(bus.close())
        async with asyncio.timeout(5):
            await waiting_for_flush.wait()
        assert bus._in_flight and bus.healthy and lease in bus._leases
        return closing, flushing, bus._listener_task

    loop = asyncio.new_event_loop()
    try:
        closing, flushing, listener = loop.run_until_complete(start(bus))
    finally:
        loop.close()
    try:
        closing.get_coro().close()
        assert listener.cancelling() == 0
    finally:
        # Finalize the flush after close has released its idle waiter.
        flushing.get_coro().close()
        listener.get_coro().close()
    del closing, flushing, listener, bus
    gc.collect()


class _CleanupTimesOutBus(_CleanupReplacesCancelBus):
    async def aclose_sender(self):
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            raise TimeoutError("cleanup timed out") from error


async def test_cancelled_close_stays_cancelled_when_cleanup_raises_timeout():
    await _cancel_after_it_starts(_CleanupTimesOutBus(close_timeout_s=5).close())


class _HangingConnection:
    def is_closed(self):
        return False

    async def close(self, timeout=None):
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            raise TimeoutError("cleanup timed out") from error

    def terminate(self):
        return None


async def test_cancelled_postgres_close_stays_cancelled_when_cleanup_raises_timeout():
    pytest.importorskip("asyncpg")
    from dendrux.db._postgres_updates import PostgresUpdates

    bus = PostgresUpdates(url="postgresql://u@localhost/db", close_timeout_s=5)
    bus._sender = _HangingConnection()
    await _cancel_after_it_starts(bus.close())


async def test_close_budget_expiry_with_timeout_cleanup_is_an_ordinary_close_timeout():
    bus = _CleanupTimesOutBus(close_timeout_s=0.1)
    await bus.close()
    assert bus._state == "closed"


class _FailingSendBus(live.LiveUpdates):
    def __init__(self, error=None, **limits):
        super().__init__(**limits)
        self.error = error or ConnectionError("send failed")

    async def send(self, messages):
        raise self.error

    async def receive(self):
        yield None
        await asyncio.Future()


async def _handle_one_cancel_and_keep_the_count() -> None:
    """The task handles a cancel without uncancel(), as ordinary code does."""
    task = asyncio.current_task()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(1)
    assert task.cancelling() == 1


async def test_handled_cancel_does_not_turn_a_send_failure_into_a_cancel():
    bus = _FailingSendBus()
    await _handle_one_cancel_and_keep_the_count()
    bus.enqueue(["r"])
    await bus.flush()  # an ordinary failure: counted, never a CancelledError
    assert bus.stats().dropped_failed == 1


async def test_handled_cancel_does_not_turn_a_module_timeout_into_a_cancel():
    bus = _CleanupReplacesCancelBus(send_timeout_s=0.1, flush_deadline_s=5)
    await _handle_one_cancel_and_keep_the_count()
    bus.enqueue(["r"])
    await bus.flush()
    assert bus.stats().dropped_failed == 1
    closing = _CleanupTimesOutBus(close_timeout_s=0.1)
    await closing.close()
    assert closing._state == "closed"


async def test_a_new_cancel_after_a_handled_one_is_still_a_cancel():
    async def flush_after_a_handled_cancel(bus):
        await _handle_one_cancel_and_keep_the_count()
        bus.enqueue(["r"])
        await bus.flush()

    bus = _CleanupReplacesCancelBus(send_timeout_s=5, flush_deadline_s=5)
    await _cancel_after_it_starts(flush_after_a_handled_cancel(bus))

    async def close_after_a_handled_cancel(bus):
        await _handle_one_cancel_and_keep_the_count()
        await bus.close()

    await _cancel_after_it_starts(
        close_after_a_handled_cancel(_CleanupReplacesCancelBus(close_timeout_s=5))
    )


async def test_failed_flush_after_a_committed_bind_does_not_cancel_the_bound_work():
    from types import SimpleNamespace

    from dendrux.runtime import durability

    store = SimpleNamespace(live_updates=_FailingSendBus())
    store.live_updates.enqueue(["r"])
    cancelled: list[object] = []

    async def operation():
        durability._mark_commit_issued()
        return "bound"

    async def cleanup(result):
        cancelled.append(result)

    await _handle_one_cancel_and_keep_the_count()
    result = await durability._ownership_handoff(store, operation, cleanup, run_id="run")
    assert result == "bound"
    assert cancelled == []


class _HangingBus(live.LiveUpdates):
    """send and aclose_sender hang; cleanup either lets the cancel through or replaces it."""

    def __init__(self, replace_with=None, **limits):
        super().__init__(**limits)
        self.replace_with = replace_with

    async def _hang(self):
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            if self.replace_with is None:
                raise
            raise self.replace_with("cleanup failed") from error

    async def send(self, messages):
        await self._hang()

    async def aclose_sender(self):
        await self._hang()

    async def receive(self):
        yield None
        await asyncio.Future()


@pytest.mark.parametrize("replace_with", [None, ConnectionError, TimeoutError])
@pytest.mark.parametrize("operation", ["flush", "close"])
async def test_a_cancel_pending_at_entry_is_not_consumed(operation, replace_with):
    bus = _HangingBus(replace_with, send_timeout_s=30, flush_deadline_s=30, close_timeout_s=30)

    async def cancelled_then_straight_into_the_bus():
        asyncio.current_task().cancel("caller gave up")
        if operation == "flush":
            bus.enqueue(["r"])
            await bus.flush()
        else:
            await bus.close()
        return "carried on"

    task = asyncio.create_task(cancelled_then_straight_into_the_bus())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_a_cancel_during_send_keeps_its_original_message():
    bus = _HangingBus(send_timeout_s=30, flush_deadline_s=30)
    bus.enqueue(["r"])
    task = asyncio.create_task(bus.flush())
    await asyncio.sleep(0.05)
    task.cancel("caller gave up")
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("caller gave up",)
