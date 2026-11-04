"""Bounded, best-effort live-update hints; durable reads remain the source of truth."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from time import monotonic
from typing import TYPE_CHECKING, Protocol, cast

from dendrux import _codes
from dendrux.errors import FollowCapacityError, LiveUpdatesUnavailableError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveUpdatesStats:
    """Payload-free counters and the age of stranded announcements; sent counts ids."""

    pending: int = 0
    oldest_pending_age_s: float = 0.0
    sent: int = 0
    dropped_overflow: int = 0
    dropped_invalid: int = 0
    dropped_oversize: int = 0
    dropped_inbound: int = 0
    dropped_failed: int = 0
    dropped_closed: int = 0
    dropped_wrong_loop: int = 0
    bad_messages: int = 0
    gaps: int = 0


def _positive(name: str, value: object, *, integer: bool = False) -> None:
    allowed = (int,) if integer else (int, float)
    if type(value) not in allowed:
        raise ValueError(f"{name} must be positive and finite")
    number = cast("int | float", value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _error_kind(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    return "connection" if isinstance(error, OSError) else "other"


def _caller_cancelled() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _quoted(run_id: str) -> str:
    return json.dumps(run_id, ensure_ascii=False, separators=(",", ":"))


class _Consumer(Protocol):
    def on_hints(self, run_ids: Sequence[str]) -> None: ...

    def on_gap(self) -> None: ...


class _InboundBuffer:
    def __init__(self, limit: int, dropped: Callable[[], None]) -> None:
        self._limit = limit
        self._dropped = dropped
        self._messages: deque[str] = deque()
        self._ready = asyncio.Event()
        self._gap = False

    def put(self, message: str) -> None:
        """Offer a transport message, dropping oldest on overflow with a sticky gap."""
        if len(self._messages) == self._limit:
            self._messages.popleft()
            self._gap = True
            self._dropped()
        self._messages.append(message)
        self._ready.set()

    async def get(self) -> str:
        """Return a gap before any retained messages after an overflow."""
        await self._ready.wait()
        if self._gap:
            self._gap = False
            return LiveUpdates.GAP
        message = self._messages.popleft()
        if not self._messages:
            self._ready.clear()
        return message

    def qsize(self) -> int:
        """Number of retained messages, excluding the coalesced gap marker."""
        return len(self._messages)


class LiveUpdates(ABC):
    """A lossy announcement bus with no sender task.

    ``send`` must cooperate with cancellation; abort hooks must be synchronous,
    prompt and idempotent. All limits count ids or UTF-8 wire bytes, never payloads.
    """

    GAP = "\x00dendrux-live-updates-gap"

    def __init__(
        self,
        *,
        max_pending: int = 10_000,
        max_batch: int = 500,
        max_message_bytes: int = 7_900,
        max_run_id_len: int = 255,
        send_timeout_s: float = 0.5,
        flush_deadline_s: float = 1.0,
        close_timeout_s: float = 3.0,
        max_inbound: int = 1_000,
        backoff_initial_s: float = 0.5,
        backoff_max_s: float = 30.0,
        health_check_s: float = 15.0,
    ) -> None:
        counts = dict(
            max_pending=max_pending,
            max_batch=max_batch,
            max_message_bytes=max_message_bytes,
            max_run_id_len=max_run_id_len,
            max_inbound=max_inbound,
        )
        times = dict(
            send_timeout_s=send_timeout_s,
            flush_deadline_s=flush_deadline_s,
            close_timeout_s=close_timeout_s,
            backoff_initial_s=backoff_initial_s,
            backoff_max_s=backoff_max_s,
            health_check_s=health_check_s,
        )
        for name, count in counts.items():
            _positive(name, count, integer=True)
        for name, seconds in times.items():
            _positive(name, seconds)
        if max_batch > max_pending:
            raise ValueError("max_batch must not exceed max_pending")
        if send_timeout_s > flush_deadline_s:
            raise ValueError("send_timeout_s must not exceed flush_deadline_s")
        if backoff_initial_s > backoff_max_s:
            raise ValueError("backoff_initial_s must not exceed backoff_max_s")
        self.max_pending = max_pending
        self.max_batch = max_batch
        self.max_message_bytes = max_message_bytes
        self.max_run_id_len = max_run_id_len
        self.max_inbound = max_inbound
        self.send_timeout_s = send_timeout_s
        self.flush_deadline_s = flush_deadline_s
        self.close_timeout_s = close_timeout_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.health_check_s = health_check_s
        self._init_sender()
        self._init_listener()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        hook = cls.__dict__.get("bind")
        if hook is None:
            return

        @wraps(hook)
        def guarded_bind(self: LiveUpdates, store: object) -> None:
            LiveUpdates.bind(self, store)
            hook(self, store)

        cls.bind = guarded_bind  # type: ignore[method-assign]

    def _init_sender(self) -> None:
        self._store: object | None = None
        self._bound = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state = "open"
        self._pending: OrderedDict[str, float] = OrderedDict()
        self._counts = dict(
            sent=0,
            dropped_overflow=0,
            dropped_invalid=0,
            dropped_oversize=0,
            dropped_inbound=0,
            dropped_failed=0,
            dropped_closed=0,
            dropped_wrong_loop=0,
            bad_messages=0,
            gaps=0,
        )
        self._in_flight = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._next_attempt_at = 0.0
        self._backoff = self.backoff_initial_s
        self._send_failed = False
        self._overflow_episode = False
        self._wrong_loop_logged = False
        self._bad_message_at = -math.inf

    @abstractmethod
    async def send(self, messages: Sequence[str]) -> None:
        """Send encoded messages in one transport round trip, including lazy setup."""

    @abstractmethod
    def receive(self) -> AsyncIterator[str | None]:
        """Yield None after subscription acknowledgement, then raw wire messages.

        Detect loss within twice ``health_check_s`` on the listening connection,
        bound undelivered messages to ``max_inbound``, and close in ``finally``.
        """

    def bind(self, store: object) -> None:
        """Associate this bus with one store; subclass hooks run after the ownership check."""
        if self._bound and self._store is not store:
            raise ValueError("LiveUpdates already belongs to another store")
        self._store = store
        self._bound = True

    def _check_loop(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._loop is None:
            self._loop = loop
            return True
        if self._loop is loop:
            return True
        if not self._wrong_loop_logged:
            self._wrong_loop_logged = True
            _codes.log_code(logger, _codes.LIVE_UPDATES_WRONG_LOOP)
        return False

    def enqueue(self, run_ids: Iterable[str]) -> None:
        """Retain bounded, ordered, deduplicated hints without raising."""
        try:
            self._enqueue(run_ids)
        except (Exception, asyncio.CancelledError):
            self._counts["dropped_invalid"] += 1

    def _enqueue(self, run_ids: Iterable[str]) -> None:
        valid_loop = self._check_loop()
        for run_id in run_ids:
            self._enqueue_one(run_id, valid_loop)

    def _enqueue_one(self, run_id: str, valid_loop: bool) -> None:
        if not valid_loop:
            self._counts["dropped_wrong_loop"] += 1
            return
        if self._state != "open":
            self._counts["dropped_closed"] += 1
            return
        if not self._valid_id(run_id):
            self._counts["dropped_invalid"] += 1
            return
        if len(_quoted(run_id).encode("utf-8")) + 2 > self.max_message_bytes:
            self._counts["dropped_oversize"] += 1
            return
        self._pending.setdefault(run_id, monotonic())
        self._trim_pending()

    def _valid_id(self, run_id: object) -> bool:
        if not isinstance(run_id, str) or not 0 < len(run_id) <= self.max_run_id_len:
            return False
        try:
            run_id.encode("utf-8")
        except UnicodeError:
            return False
        return True

    def _trim_pending(self) -> None:
        while len(self._pending) > self.max_pending:
            self._pending.popitem(last=False)
            self._counts["dropped_overflow"] += 1
            if not self._overflow_episode:
                self._overflow_episode = True
                _codes.log_code(
                    logger, _codes.LIVE_UPDATES_OVERFLOW, count=1, limit=self.max_pending
                )

    def _reset_overflow(self) -> None:
        if len(self._pending) < self.max_pending / 2:
            self._overflow_episode = False

    def stats(self) -> LiveUpdatesStats:
        """Return counters and age without ids, transport text or credentials."""
        oldest = next(iter(self._pending.values()), monotonic())
        age = max(0.0, monotonic() - oldest) if self._pending else 0.0
        return LiveUpdatesStats(
            pending=len(self._pending),
            oldest_pending_age_s=age,
            **self._counts,
        )

    def _encode(self, run_ids: Sequence[str]) -> list[str]:
        messages: list[str] = []
        parts: list[str] = []
        size = 2
        for run_id in run_ids:
            part = _quoted(run_id)
            part_size = len(part.encode("utf-8"))
            if parts and (
                size + part_size + 1 > self.max_message_bytes or len(parts) >= self.max_batch
            ):
                messages.append("[" + ",".join(parts) + "]")
                parts, size = [], 2
            size += part_size + bool(parts)
            parts.append(part)
        if parts:
            messages.append("[" + ",".join(parts) + "]")
        return messages

    def _decode(self, message: str) -> tuple[str, ...] | None:
        try:
            return self._parse_message(message)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            self._counts["bad_messages"] += 1
            now = monotonic()
            if now - self._bad_message_at >= 60:
                self._bad_message_at = now
                _codes.log_code(
                    logger, _codes.LIVE_UPDATES_BAD_MESSAGE, count=self._counts["bad_messages"]
                )
            return None

    def _parse_message(self, message: str) -> tuple[str, ...]:
        if not isinstance(message, str) or len(message) > self.max_message_bytes:
            raise ValueError("live_updates_bad_message")
        if len(message.encode("utf-8")) > self.max_message_bytes:
            raise ValueError("live_updates_bad_message")
        ids = json.loads(message)
        if not isinstance(ids, list) or len(ids) > self.max_batch:
            raise ValueError("live_updates_bad_message")
        if not all(self._valid_id(run_id) for run_id in ids):
            raise ValueError("live_updates_bad_message")
        return tuple(ids)

    async def flush(self) -> None:
        """Send pending hints within one deadline; propagate only caller cancellation."""
        if not self._check_loop() or self._state != "open":
            return
        await self._flush_until(monotonic() + self.flush_deadline_s)

    async def _flush_until(self, deadline: float) -> None:
        if self._in_flight or monotonic() < self._next_attempt_at:
            return
        self._in_flight = True
        self._idle.clear()
        try:
            await self._flush_batches(deadline)
        finally:
            self._in_flight = False
            self._idle.set()

    async def _flush_batches(self, deadline: float) -> None:
        while self._pending and self._state != "closed" and monotonic() < deadline:
            batch = [
                self._pending.popitem(last=False)
                for _ in range(min(self.max_batch, len(self._pending)))
            ]
            messages = self._encode([run_id for run_id, _ in batch])
            remaining = deadline - monotonic()
            if remaining <= 0:
                self._pending = OrderedDict([*batch, *self._pending.items()])
                return
            self._reset_overflow()
            if not await self._send_batch(messages, len(batch), remaining):
                return

    async def _send_batch(self, messages: Sequence[str], count: int, remaining: float) -> bool:
        try:
            async with asyncio.timeout(min(self.send_timeout_s, remaining)):
                await self.send(messages)
        except asyncio.CancelledError as error:
            self._drop_batch(count)
            if _caller_cancelled():
                raise
            self._send_failure(error, count)
            return False
        except Exception as error:
            self._drop_batch(count)
            self._send_failure(error, count)
            return False
        self._counts["sent"] += count
        self._send_success()
        return True

    def _drop_batch(self, count: int) -> None:
        self._counts["dropped_failed"] += count
        self._abort_sender_safe()

    def _abort_sender_safe(self) -> None:
        with suppress(Exception, asyncio.CancelledError):
            self.abort_sender()

    def _send_failure(self, error: BaseException, count: int) -> None:
        self._next_attempt_at = monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, self.backoff_max_s)
        if not self._send_failed:
            self._send_failed = True
            _codes.log_code(
                logger, _codes.LIVE_UPDATES_SEND_FAILED, error_kind=_error_kind(error), count=count
            )

    def _send_success(self) -> None:
        self._backoff = self.backoff_initial_s
        self._next_attempt_at = 0.0
        if self._send_failed:
            self._send_failed = False
            _codes.log_code(logger, _codes.LIVE_UPDATES_SEND_RECOVERED)

    def abort_sender(self) -> None:
        """Synchronously drop the sender connection, including after cancellation."""
        return None

    def abort_listener(self) -> None:
        """Synchronously drop the listener connection on forced shutdown."""
        return None

    async def aclose_sender(self) -> None:
        """Close the sender connection at instance shutdown."""
        return None

    def _init_listener(self) -> None:
        self._healthy = False
        self._generation = 0
        self._listener_failed = False
        self._listener_backoff = self.backoff_initial_s
        self._listener_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._listener_task: asyncio.Task[None] | None = None
        self._listener_stopping = False
        self._leases: set[object] = set()
        self._consumers: dict[object, tuple[_Consumer, bool]] = {}
        self._stream_consumers = 0

    @property
    def healthy(self) -> bool:
        """Whether the listener has confirmed a live subscription."""
        return self._healthy

    @property
    def generation(self) -> int:
        """Monotonic generation of health transitions and missed hints."""
        return self._generation

    def _new_inbound_buffer(self) -> _InboundBuffer:
        return _InboundBuffer(self.max_inbound, self._inbound_dropped)

    def _inbound_dropped(self) -> None:
        self._counts["dropped_inbound"] += 1

    # Subclasses may lower or raise this cap for their deployment.
    max_stream_consumers = 10_000

    def register(self, consumer: _Consumer, *, stream_events: bool = True) -> object:
        """Register callbacks; coordinators pass ``stream_events=False``.

        Return an opaque handle for idempotent ``unregister``. Stream consumers
        share the overridable ``max_stream_consumers`` class-level cap.
        """
        self._require_loop()
        self._require_open()
        if stream_events and self._stream_consumers >= self.max_stream_consumers:
            _codes.log_code(logger, _codes.FOLLOW_CAPACITY, limit=self.max_stream_consumers)
            raise FollowCapacityError(self.max_stream_consumers)
        handle = object()
        self._consumers[handle] = (consumer, stream_events)
        self._stream_consumers += int(stream_events)
        return handle

    def unregister(self, handle: object) -> None:
        """Idempotently remove a hub registration."""
        self._require_loop()
        registration = self._consumers.pop(handle, None)
        if registration is not None:
            self._stream_consumers -= int(registration[1])

    def _deliver(self, message: str) -> None:
        if message == self.GAP:
            self._gap()
            return
        ids = self._decode(message)
        if ids is not None:
            for consumer, _ in tuple(self._consumers.values()):
                with suppress(Exception, asyncio.CancelledError):
                    consumer.on_hints(ids)

    def _gap(self) -> None:
        self._generation += 1
        self._counts["gaps"] += 1
        for consumer, _ in tuple(self._consumers.values()):
            with suppress(Exception, asyncio.CancelledError):
                consumer.on_gap()

    def _health(self, healthy: bool) -> None:
        if self._healthy == healthy:
            return
        self._healthy = healthy
        self._gap()
        if healthy:
            self._listener_failed = False
            self._listener_backoff = self.backoff_initial_s
            _codes.log_code(logger, _codes.LIVE_UPDATES_RECOVERED)

    def _degraded(self, error: BaseException) -> None:
        self._health(False)
        if not self._listener_failed:
            self._listener_failed = True
            _codes.log_code(logger, _codes.LIVE_UPDATES_DEGRADED, error_kind=_error_kind(error))

    def _require_loop(self) -> None:
        if not self._check_loop():
            raise RuntimeError("LiveUpdates belongs to another event loop")

    def _require_open(self) -> None:
        if self._state != "open":
            raise LiveUpdatesUnavailableError("closed")

    async def acquire_listener(self) -> object:
        """Lease one listener task; readiness is exposed by ``healthy`` and hub gaps."""
        self._require_loop()
        async with self._listener_lock:
            self._require_open()
            await self._start_listener()
            handle = object()
            self._leases.add(handle)
            return handle

    async def _start_listener(self) -> None:
        if self._listener_stopping:
            await self._stop_listener(monotonic() + self.close_timeout_s)
        if self._listener_task is not None and not self._listener_task.done():
            if self._listener_stopping:
                raise LiveUpdatesUnavailableError("timeout")
            return
        self._require_open()
        self._listener_stopping = False
        self._listener_task = asyncio.create_task(self._listen(), name="dendrux-live-updates")

    async def release_listener(self, handle: object) -> None:
        """Idempotently release a lease, stopping the listener after the last one."""
        self._require_loop()
        async with self._listener_lock:
            if handle not in self._leases:
                return
            self._leases.remove(handle)
            if not self._leases:
                await self._stop_listener(monotonic() + self.close_timeout_s)

    async def _listen(self) -> None:
        try:
            while self._state == "open":
                await self._listen_attempt()
                delay = random.uniform(0, self._listener_backoff)
                self._listener_backoff = min(self._listener_backoff * 2, self.backoff_max_s)
                await asyncio.sleep(delay)
        finally:
            self._health(False)

    async def _listen_attempt(self) -> None:
        try:
            await self._receive_messages()
        except (Exception, asyncio.CancelledError) as error:
            self._listener_failure(error)
        else:
            self._degraded(ConnectionError())

    def _listener_failure(self, error: BaseException) -> None:
        if _caller_cancelled():
            raise asyncio.CancelledError from None
        self._degraded(error)

    async def _receive_messages(self) -> None:
        iterator = self.receive()
        try:
            async with asyncio.timeout(self.send_timeout_s * 4):
                first = await anext(iterator)
                if first is not None:
                    raise LiveUpdatesUnavailableError()
            self._health(True)
            async for message in iterator:
                if message is None:
                    raise LiveUpdatesUnavailableError()
                self._deliver(message)
        finally:
            self._health(False)
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _stop_listener(self, deadline: float) -> None:
        task = self._listener_task
        if task is None:
            return
        self._cancel_listener()
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0.0, deadline - monotonic()))
            if not done:
                self._abort_listener_safe()
                self._close_timeout()
        except asyncio.CancelledError:
            self._abort_listener_safe()
            raise
        finally:
            self._health(False)
        if task.done():
            self._listener_task = None
            self._listener_stopping = False

    def _cancel_listener(self) -> None:
        if self._listener_task is not None and not self._listener_stopping:
            self._listener_stopping = True
            self._listener_task.cancel()

    def _abort_listener_safe(self) -> None:
        with suppress(Exception, asyncio.CancelledError):
            self.abort_listener()

    def _abort_all(self) -> None:
        self._abort_sender_safe()
        self._cancel_listener()
        self._abort_listener_safe()
        self._health(False)

    def _close_timeout(self) -> None:
        _codes.log_code(logger, _codes.LIVE_UPDATES_CLOSE_TIMEOUT, timeout_s=self.close_timeout_s)

    async def close(self) -> None:
        """Drain and close within one total deadline; a cancelled closer may be retried."""
        self._require_loop()
        async with self._close_lock:
            if self._state == "closed":
                return
            self._state = "closing"
            await self._close_owned()
            self._state = "closed"
            self._counts["dropped_closed"] += len(self._pending)
            self._pending.clear()
            self._leases.clear()
            self._consumers.clear()
            self._stream_consumers = 0

    async def _close_owned(self) -> None:
        deadline = monotonic() + self.close_timeout_s
        try:
            async with asyncio.timeout(self.close_timeout_s):
                await self._close_resources(deadline)
        except asyncio.CancelledError:
            self._abort_all()
            if _caller_cancelled():
                raise
        except TimeoutError:
            self._abort_all()
            self._close_timeout()
        except Exception as error:
            self._abort_all()
            _codes.log_code(logger, _codes.LIVE_UPDATES_UNAVAILABLE, error_kind=_error_kind(error))

    async def _close_resources(self, deadline: float) -> None:
        await self._idle.wait()
        await self._flush_until(min(deadline, monotonic() + self.flush_deadline_s))
        async with self._listener_lock:
            self._leases.clear()
            await self._stop_listener(deadline)
        if monotonic() >= deadline:
            raise TimeoutError
        await self.aclose_sender()
