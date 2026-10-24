"""Bounded execution for synchronous tools.

Sync ``@tool`` functions run on one process-wide pool of **daemon** threads,
created lazily (size ``DENDRUX_TOOL_THREADS``, default 32), never on the event
loop's default executor, which DNS resolution and file reads also use. Each
Agent also has its own limiter (``Agent(tool_threads=8)``), so one Agent, or
one tenant's Agent, cannot occupy the whole pool.

A Python thread cannot be cancelled. When a sync call times out:

- if it is still queued (waiting for its Agent's limiter or a pool thread), it
  is cancelled and never runs: ``code="timeout_not_started"``, safe to retry;
- if it already started, it keeps running in the background:
  ``code="timeout_outcome_unknown"``. Its thread is a *zombie* until the
  function returns, and its Agent's limiter slot stays taken until then, so
  a tool that hangs can only exhaust its own Agent's slots.

:func:`tool_pool_stats` reports the live counts, including zombies. Because
pool threads are daemon threads, a hung sync tool never blocks interpreter
exit; its side effect may be cut off at exit. Sync tools should use their own
client-side timeouts (HTTP, database) so they return.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import logging
import os
import threading
import weakref
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux.types import Limits

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_TOOL_THREADS = 32
TOOL_THREADS_ENV = "DENDRUX_TOOL_THREADS"
# Queued (not yet running) calls the pool accepts beyond its threads. Every
# queued call is an awaiting tool call, already bounded by the per-Agent
# limiters; this is the process-wide backstop.
MAX_QUEUED_CALLS = 1024


class ToolPoolSaturatedError(RuntimeError):
    """The pool's queue is full; the call was not started."""


class ToolPoolStats(NamedTuple):
    """A snapshot of the sync-tool pool (see :func:`tool_pool_stats`)."""

    max_threads: int
    threads: int
    busy: int
    queued: int
    zombies: int


@dataclass(eq=False)
class _WorkItem:
    future: Future[Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    context: contextvars.Context
    abandoned: bool = False


class _ToolPool:
    """A fixed-size pool of daemon worker threads with a bounded queue."""

    def __init__(self, max_threads: int) -> None:
        self.max_threads = max_threads
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._queue: deque[_WorkItem] = deque()
        self._threads = 0
        self._idle = 0
        self._busy = 0
        self._zombies = 0
        self._shutdown = False

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> _WorkItem:
        """Queue ``fn`` to run in the caller's context copy; returns its work item."""
        item = _WorkItem(Future(), fn, args, kwargs, contextvars.copy_context())
        with self._lock:
            if self._shutdown:
                raise RuntimeError("The Dendrux tool pool is shut down.")
            if len(self._queue) >= MAX_QUEUED_CALLS:
                raise ToolPoolSaturatedError("The Dendrux tool pool queue is full.")
            self._queue.append(item)
            # Spawn while queued work outnumbers the threads waiting for it.
            if len(self._queue) > self._idle and self._threads < self.max_threads:
                self._threads += 1
                thread = threading.Thread(
                    target=self._worker,
                    name=f"dendrux-tool-{self._threads}",
                    daemon=True,
                )
                thread.start()
            self._work.notify()
        return item

    def abandon(self, item: _WorkItem) -> int:
        """Mark a started item whose caller gave up; returns the zombie count."""
        with self._lock:
            if not item.abandoned and not item.future.done():
                item.abandoned = True
                self._zombies += 1
            return self._zombies

    def stats(self) -> ToolPoolStats:
        with self._lock:
            return ToolPoolStats(
                max_threads=self.max_threads,
                threads=self._threads,
                busy=self._busy,
                queued=len(self._queue),
                zombies=self._zombies,
            )

    def shutdown(self) -> None:
        """Stop accepting work, cancel queued items and let idle workers exit."""
        with self._lock:
            self._shutdown = True
            queued = list(self._queue)
            self._queue.clear()
            self._work.notify_all()
        for item in queued:
            item.future.cancel()

    def _worker(self) -> None:
        while True:
            with self._lock:
                self._idle += 1
                while not self._queue and not self._shutdown:
                    self._work.wait()
                self._idle -= 1
                if not self._queue:
                    self._threads -= 1
                    return
                item = self._queue.popleft()
                if not item.future.set_running_or_notify_cancel():
                    continue
                self._busy += 1
            try:
                result = item.context.run(item.fn, *item.args, **item.kwargs)
            except BaseException as exc:
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)
            finally:
                with self._lock:
                    self._busy -= 1
                    finished_zombie = item.abandoned
                    if finished_zombie:
                        self._zombies -= 1
                    zombies = self._zombies
                del item
            if finished_zombie:
                logger.info(
                    "A timed-out sync tool call finished in the background "
                    "(code=tool_zombie_finished, zombies=%d)",
                    zombies,
                )


_pool: _ToolPool | None = None
_pool_lock = threading.Lock()


def _configured_threads() -> int:
    raw = os.environ.get(TOOL_THREADS_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_TOOL_THREADS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{TOOL_THREADS_ENV} must be a positive integer.")
    return value


def get_pool() -> _ToolPool:
    """The process-wide pool, created on first use."""
    global _pool
    pool = _pool
    if pool is not None:
        return pool
    with _pool_lock:
        if _pool is None:
            _pool = _ToolPool(_configured_threads())
        return _pool


def _shutdown_pool() -> None:
    pool = _pool
    if pool is not None:
        pool.shutdown()


# Every live limiter, so a forked child can reset them (see _reset_after_fork).
_limiters: weakref.WeakSet[ToolCallLimiter] = weakref.WeakSet()


def _reset_after_fork() -> None:
    # Worker threads do not survive fork; the child builds its own pool, and
    # slots held by the parent's in-flight calls would never come back.
    global _pool, _pool_lock
    _pool = None
    _pool_lock = threading.Lock()
    for limiter in list(_limiters):
        limiter._reset()


atexit.register(_shutdown_pool)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def tool_pool_stats() -> ToolPoolStats:
    """Live counts for the process-wide sync-tool pool.

    ``zombies`` counts sync tool calls that timed out (or whose run was
    cancelled) after they started and are still running. A steadily growing
    count means a tool hangs; once it nears ``max_threads`` every Agent's
    sync tools queue and time out as ``timeout_not_started``, and the
    process should be restarted.
    """
    pool = _pool
    if pool is None:
        return ToolPoolStats(_configured_threads(), 0, 0, 0, 0)
    return pool.stats()


class ToolCallLimiter:
    """Per-Agent cap on concurrently running sync tool calls.

    Thread-safe and not bound to an event loop, so one Agent can be used from
    several loops. A slot is released when the tool's thread finishes, not
    when its caller gives up, so hung calls keep holding their Agent's slots.
    """

    def __init__(self, slots: int) -> None:
        self.slots = slots
        # The owning Agent's caps; bound at its entry checks (see
        # runtime._limit_checks). Until then the default caps apply.
        self.limits = Limits()
        self._lock = threading.Lock()
        self._available = slots
        self._waiters: deque[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = deque()
        _limiters.add(self)

    def _reset(self) -> None:
        """Full capacity, no waiters (a forked child's fresh state)."""
        self._lock = threading.Lock()
        self._available = self.slots
        self._waiters = deque()

    @property
    def in_use(self) -> int:
        with self._lock:
            return self.slots - self._available

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._available > 0 and not self._waiters:
                self._available -= 1
                return
            waiter: asyncio.Future[None] = loop.create_future()
            self._waiters.append((loop, waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            # Already popped by release() → _grant settles the slot.
            with self._lock, contextlib.suppress(ValueError):
                self._waiters.remove((loop, waiter))
            if waiter.done() and not waiter.cancelled():
                # Granted, but the task was cancelled before it resumed.
                self.release()
            raise

    def release(self) -> None:
        """Return one slot; safe to call from any thread."""
        while True:
            with self._lock:
                if not self._waiters:
                    self._available += 1
                    return
                loop, waiter = self._waiters.popleft()
            try:
                loop.call_soon_threadsafe(self._grant, waiter)
                return
            except RuntimeError:
                # The waiter's loop is closed; hand the slot to the next one.
                continue

    def _grant(self, waiter: asyncio.Future[None]) -> None:
        if waiter.done():
            # Cancelled after it was popped: its cancel handler no longer
            # finds it in the queue, so return the slot here.
            self.release()
        else:
            waiter.set_result(None)


@dataclass(eq=False)
class CallTracker:
    """What happened to one tool call, for timeout and cancellation reporting.

    ``started``: the tool body began (an async tool was entered, or a sync
    tool's thread picked it up). ``abandoned``: a started sync call whose
    caller gave up while its thread kept running; ``zombies`` is the pool's
    zombie count at that moment. ``completed`` is the finished future of a
    sync call that ended exactly as its caller gave up.
    """

    started: bool = False
    abandoned: bool = False
    zombies: int = 0
    completed: Future[Any] | None = None
    # Set by the loop: the tool's result once it returned, and whether its
    # tool_calls row was written. A stopped sibling is never recorded twice.
    result: Any = None
    recorded: bool = False


async def run_sync_tool(
    fn: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    limiter: ToolCallLimiter | None,
    tracker: CallTracker,
) -> Any:
    """Run a sync tool on the pool under its Agent's limiter.

    Cancellation (a timeout or the run being cancelled) before the thread
    picks the call up cancels it for good. After that the thread keeps
    running; the call is marked abandoned and counted as a zombie until the
    function returns.
    """
    if limiter is not None:
        await limiter.acquire()
    try:
        item = get_pool().submit(fn, *args, **kwargs)
    except BaseException:
        if limiter is not None:
            limiter.release()
        raise
    future = item.future
    if limiter is not None:
        # Released when the thread finishes (or the queued call is cancelled),
        # never earlier: a hung call keeps holding its Agent's slot.
        future.add_done_callback(lambda _done: limiter.release())
    try:
        result = await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        if future.cancel():
            raise  # still queued: it will never run
        tracker.started = True
        if future.done():
            tracker.completed = future
        else:
            tracker.abandoned = True
            tracker.zombies = get_pool().abandon(item)
        raise
    tracker.started = True
    return result
