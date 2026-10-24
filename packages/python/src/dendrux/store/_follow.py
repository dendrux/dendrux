"""Live, tenant-scoped event feeds over ``RunStore`` (``RunStore.follow``).

Framework-agnostic: an async iterator of :class:`~dendrux.store.StoredEvent`
items plus :class:`FollowHeartbeat` and a final :class:`FollowEnd`. Serve it
as SSE, websockets or long-poll from your own routes; ``make_read_router``
is a thin wrapper over it.

Load model. One :class:`EventFeed` keeps **one poller per watched run** and
fans events out in memory to every follower of that run, so database load
grows with the number of watched runs, not with open connections. An idle
run is polled every 0.5 s, backing off to 2 s, plus a status read every 4th
idle poll. The poller is an ordinary asyncio task owned by the feed: it
starts with the first follower of a run and is cancelled, and awaited, when
the last one leaves. Nothing runs when nobody is following. Budget about one
query per second per watched run, and give a busy feed its own small engine
(``RunStore.from_database_url(...)``) so viewers cannot starve the agents'
pool. Push (Postgres LISTEN/NOTIFY) is planned for 0.6.

Never tie a run's lifetime to a client connection: follow the database log
(this module) rather than piping ``agent.stream()`` to a browser, where a
closed tab cancels the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

from dendrux.errors import RunNotFoundError
from dendrux.runtime.state import _check_scope

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from dendrux.runtime.state import TenantScope
    from dendrux.store import RunStore, StoredEvent

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"success", "error", "cancelled", "max_iterations"})
PAUSE_STATUSES = frozenset(
    {"waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"}
)
_TERMINAL_EVENT_TYPES = frozenset({"run.completed", "run.error", "run.cancelled"})

# Event-data keys that are payloads, hidden unless include_payloads=True.
_HIDDEN_EVENT_KEYS = frozenset(
    {"system_prompt", "provider_request", "provider_response", "pii_mapping"}
)

FollowEndReason = Literal["terminal", "max_duration", "lagging", "error"]


class FollowCapacityError(RuntimeError):
    """The :class:`EventFeed` already serves ``max_followers`` followers.

    Map it to HTTP 503 (retry shortly). ``limit`` is the cap.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"This event feed already serves its maximum of {limit} followers.")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (type(self), (self.limit,))


@dataclass(frozen=True, slots=True)
class FollowHeartbeat:
    """Nothing happened for ``heartbeat_s`` seconds. Send a keepalive to the client."""


@dataclass(frozen=True, slots=True)
class FollowEnd:
    """The last item of a follow: why it ended and the cursor to resume from.

    ``reason``: ``"terminal"`` (the run finished and every event was sent),
    ``"max_duration"``, ``"lagging"`` (this follower fell more than the
    buffer behind) or ``"error"``. Reconnect with ``after_sequence_index=cursor``
    unless the reason is ``"terminal"``.
    """

    reason: FollowEndReason
    cursor: int | None


FollowItem: TypeAlias = Union["StoredEvent", FollowHeartbeat, FollowEnd]  # noqa: UP007


# Pauses whose arguments are shown to a human as the model sent them.
_PLACEHOLDER_PAUSES = frozenset({"waiting_approval", "waiting_human_input"})


def to_placeholders(value: Any, pii_mapping: Mapping[str, str] | None) -> Any:
    """``value`` with every real value of ``pii_mapping`` put back as its placeholder.

    ``pii_mapping`` is the run's ``{placeholder: real value}`` map. The
    runtime de-anonymises tool arguments before an approval pause; this
    restores the form the model sent (``<<EMAIL_ADDRESS_1>>``). Longer values
    are replaced first. Dicts and lists are walked; other values are kept.
    """
    if not pii_mapping:
        return value
    pairs = sorted(
        ((real, ph) for ph, real in pii_mapping.items() if isinstance(real, str) and real),
        key=lambda p: len(p[0]),
        reverse=True,
    )

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            for real, ph in pairs:
                v = v.replace(real, ph)
            return v
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return walk(value)


def redact_event(
    event: StoredEvent,
    *,
    include_payloads: bool = False,
    active_pause: bool = False,
    pii_mapping: Mapping[str, str] | None = None,
) -> StoredEvent:
    """The event as a reader may see it (B2 and the approval-arguments rule).

    In every mode, the arguments of an **approval or input** ``run.paused``
    are shown in placeholder form, as the model sent them, using the run's
    ``pii_mapping`` (``RunStore.get_pii_mapping``): a human approving a call
    never sees de-anonymised values. Client-tool pauses keep the 0.4.1 form.

    With ``include_payloads=True`` nothing else changes. Otherwise:

    - the system prompt (``run.started``), provider request and response
      bodies and any ``pii_mapping`` are removed from ``data``;
    - in ``run.paused``, each pending call's ``params`` are removed **unless**
      ``active_pause`` is true. The active pause's arguments are an
      instruction to the client (run this tool, approve this call), so they
      stay visible; a past pause's arguments are history and are hidden.

    Tool results and server-tool arguments never appear in event data; read
    them with ``RunStore.get_tool_invocations`` (trusted readers only).
    """
    if not event.data:
        return event
    data = dict(event.data)
    if (
        event.event_type == "run.paused"
        and data.get("status") in _PLACEHOLDER_PAUSES
        and isinstance(data.get("pending_tool_calls"), list)
    ):
        data["pending_tool_calls"] = to_placeholders(data["pending_tool_calls"], pii_mapping)
    if include_payloads:
        return dataclasses.replace(event, data=data) if data != event.data else event
    data = {k: v for k, v in data.items() if k not in _HIDDEN_EVENT_KEYS}
    if (
        event.event_type == "run.paused"
        and not active_pause
        and isinstance(data.get("pending_tool_calls"), list)
    ):
        data["pending_tool_calls"] = [without_params(c) for c in data["pending_tool_calls"]]
    return dataclasses.replace(event, data=data)


def without_params(call: Any) -> Any:
    """A pending-call dict without its arguments."""
    if not isinstance(call, dict):
        return call
    return {k: v for k, v in call.items() if k not in ("params", "arguments")}


@dataclass(eq=False)
class _Follower:
    """One follower of a run feed: a bounded buffer plus a wake-up signal."""

    until_terminal: bool
    limit: int
    buffer: deque[StoredEvent] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    end_reason: FollowEndReason | None = None

    def push(self, event: StoredEvent) -> None:
        if len(self.buffer) >= self.limit:
            self.end("lagging")
            return
        self.buffer.append(event)
        self.wake.set()

    def end(self, reason: FollowEndReason) -> None:
        if self.end_reason is None:
            self.end_reason = reason
        self.wake.set()


class _RunFeed:
    """The single poller of one run, shared by every follower of it."""

    def __init__(self, owner: EventFeed, run_id: str, head: int | None) -> None:
        self.owner = owner
        self.run_id = run_id
        self.head = head
        self.followers: set[_Follower] = set()
        self.terminal = False
        self.task: asyncio.Task[None] | None = None

    def add(self, follower: _Follower) -> int | None:
        """Register ``follower``; return the head it must catch up to from the database."""
        self.followers.add(follower)
        if self.terminal and follower.until_terminal:
            follower.end("terminal")
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(
                self._poll(), name=f"dendrux-run-feed:{self.run_id}"
            )
        return self.head

    def remove(self, follower: _Follower) -> asyncio.Task[None] | None:
        """Unregister ``follower``; return the poller task to await when it was the last."""
        self.followers.discard(follower)
        if self.followers:
            return None
        self.owner._drop(self)
        task, self.task = self.task, None
        if task is not None and not task.done():
            task.cancel()
            return task
        return None

    def _publish(self, events: list[StoredEvent]) -> None:
        for event in events:
            for follower in list(self.followers):
                follower.push(event)
            self.head = event.sequence_index

    async def _fetch(self) -> list[StoredEvent]:
        return await self.owner.store.get_events(
            self.run_id, after_sequence_index=self.head, limit=self.owner.batch
        )

    async def _poll(self) -> None:
        owner = self.owner
        delay = owner.poll_interval_s
        idle_polls = 0
        check_status = True
        try:
            while self.followers:
                if self.terminal and all(f.until_terminal for f in self.followers):
                    return
                events = await self._fetch()
                self._publish(events)
                if len(events) == owner.batch:
                    continue  # more may be buffered; keep draining
                if events:
                    delay, idle_polls = owner.poll_interval_s, 0
                    check_status = check_status or any(
                        e.event_type in _TERMINAL_EVENT_TYPES for e in events
                    )
                else:
                    idle_polls += 1
                    check_status = check_status or idle_polls % owner.status_every == 0
                if not self.terminal and check_status:
                    check_status = False
                    run = await owner.store.get_run(self.run_id)
                    if run is None or run.status in TERMINAL_STATUSES:
                        # Events committed before the status flip: drain them first.
                        while True:
                            tail = await self._fetch()
                            self._publish(tail)
                            if len(tail) < owner.batch:
                                break
                        self.terminal = True
                        for follower in list(self.followers):
                            if follower.until_terminal:
                                follower.end("terminal")
                        continue
                await asyncio.sleep(delay)
                if not events:
                    delay = min(delay * 2, owner.poll_max_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "run feed failed: run_id=%s code=follow_feed_error",
                self.run_id,
                exc_info=True,
                extra={"run_id": self.run_id, "code": "follow_feed_error"},
            )
            for follower in list(self.followers):
                follower.end("error")


class EventFeed:
    """Shared live feeds for many followers of many runs, with a follower cap.

    Create one per process (or per router) and call :meth:`follow`.
    ``RunStore.follow`` uses one owned by the store. Not thread-safe: use it
    from one event loop.

    Args:
        store: The :class:`RunStore` to read from.
        max_followers: Most concurrent followers; :meth:`Follow.start`
            raises :class:`FollowCapacityError` beyond it.
        poll_interval_s / poll_max_s: idle polling starts at the first and
            doubles up to the second; a new event resets it.
        heartbeat_s: idle time before a :class:`FollowHeartbeat` is yielded.
        buffer: events buffered per follower before it ends as ``"lagging"``.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        max_followers: int = 500,
        poll_interval_s: float = 0.5,
        poll_max_s: float = 2.0,
        heartbeat_s: float = 15.0,
        buffer: int = 1000,
        status_every: int = 4,
        batch: int = 100,
        stop_timeout_s: float = 5.0,
    ) -> None:
        if isinstance(max_followers, bool) or not isinstance(max_followers, int):
            raise ValueError("max_followers must be a positive int.")
        if max_followers <= 0 or buffer <= 0 or status_every <= 0 or batch <= 0:
            raise ValueError("max_followers, buffer, status_every and batch must be positive.")
        if not 0 < poll_interval_s <= poll_max_s or heartbeat_s <= 0:
            raise ValueError("poll intervals and heartbeat_s must be positive (interval <= max).")
        self.store = store
        self.max_followers = max_followers
        self.poll_interval_s = poll_interval_s
        self.poll_max_s = poll_max_s
        self.heartbeat_s = heartbeat_s
        self.buffer = buffer
        self.status_every = status_every
        self.batch = batch
        self.stop_timeout_s = stop_timeout_s
        self._open = 0
        self._feeds: dict[str, _RunFeed] = {}

    @property
    def followers(self) -> int:
        """Followers currently holding a slot."""
        return self._open

    @property
    def watched_runs(self) -> int:
        """Runs with a live poller (one each, however many followers)."""
        return len(self._feeds)

    def follow(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        after_sequence_index: int | None = None,
        include_payloads: bool = False,
        until_terminal: bool = True,
        max_duration_s: float | None = 900.0,
    ) -> Follow:
        """A live feed of ``run_id``'s events after ``after_sequence_index``.

        ``tenant_id`` is required: a string or ``None`` scopes to that tenant
        (or to untenanted runs), ``UNSCOPED`` reads any run and must be
        passed explicitly. A run outside the scope raises
        ``RunNotFoundError`` exactly like a missing one.

        Iterate the result (``async for item in feed.follow(...)``) or call
        ``await follow.start()`` first to surface ``RunNotFoundError`` and
        :class:`FollowCapacityError` before streaming. Always close it
        (``async with``, or ``await follow.aclose()``) to free the slot.

        Items are :class:`StoredEvent` (passed through :func:`redact_event`),
        :class:`FollowHeartbeat`, and one final :class:`FollowEnd`. With
        ``until_terminal=True`` the follow ends once the run's status is
        terminal and every event was delivered; ``False`` keeps following.
        ``max_duration_s=None`` removes the time limit.
        """
        _check_scope(tenant_id)
        if max_duration_s is not None and not max_duration_s > 0:
            raise ValueError("max_duration_s must be positive or None.")
        return Follow(
            self,
            run_id,
            tenant_id=tenant_id,
            after=after_sequence_index,
            include_payloads=include_payloads,
            until_terminal=until_terminal,
            max_duration_s=max_duration_s,
        )

    def _acquire(self) -> None:
        if self._open >= self.max_followers:
            logger.warning(
                "follower limit reached: code=follow_capacity limit=%d",
                self.max_followers,
                extra={"code": "follow_capacity"},
            )
            raise FollowCapacityError(self.max_followers)
        self._open += 1

    def _release(self) -> None:
        self._open = max(0, self._open - 1)

    def _drop(self, feed: _RunFeed) -> None:
        if self._feeds.get(feed.run_id) is feed:
            del self._feeds[feed.run_id]

    async def _feed_for(self, run_id: str) -> _RunFeed:
        feed = self._feeds.get(run_id)
        if feed is not None:
            return feed
        head = await _current_head(self.store, run_id)
        # Another follower may have created the feed while we read the head.
        return self._feeds.get(run_id) or self._feeds.setdefault(
            run_id, _RunFeed(self, run_id, head)
        )


class Follow:
    """One follower's live feed (from :meth:`EventFeed.follow`). Async iterator."""

    def __init__(
        self,
        owner: EventFeed,
        run_id: str,
        *,
        tenant_id: TenantScope,
        after: int | None,
        include_payloads: bool,
        until_terminal: bool,
        max_duration_s: float | None,
    ) -> None:
        self._owner = owner
        self.run_id = run_id
        self._tenant_id = tenant_id
        self._cursor = after
        self._include_payloads = include_payloads
        self._until_terminal = until_terminal
        self._max_duration_s = max_duration_s
        self._started = False
        self._holds_slot = False
        self._closed = False
        self._active_pause_seq: int | None = None
        self._pii_mapping: dict[str, str] | None = None
        self._gen: AsyncIterator[FollowItem] | None = None
        self.drained_terminal = False
        """True after :meth:`start` when the run is terminal and nothing is left
        after the cursor (an HTTP route answers 204 instead of streaming)."""

    async def start(self) -> None:
        """Check the scope and take a follower slot; idempotent.

        Raises:
            RunNotFoundError: the run does not exist or is outside the tenant.
            FollowCapacityError: the feed is full.
        """
        if self._started:
            return
        if self._closed:
            raise RuntimeError("This follow is closed.")
        store = self._owner.store
        record = await store._scoped_run(self.run_id, self._tenant_id)
        if record is None:
            raise RunNotFoundError(self.run_id)
        if record.status in TERMINAL_STATUSES and self._until_terminal:
            pending = await store.get_events(
                self.run_id, after_sequence_index=self._cursor, limit=1
            )
            self.drained_terminal = not pending
        if record.status in PAUSE_STATUSES:
            self._active_pause_seq = await store._latest_pause_sequence(self.run_id)
        self._owner._acquire()
        self._holds_slot = True
        self._started = True

    def __aiter__(self) -> Follow:
        return self

    async def __anext__(self) -> FollowItem:
        await self.start()
        if self._gen is None:
            self._gen = self._run()
        try:
            return await self._gen.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Stop following and free the slot. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._gen is not None:
                await self._gen.aclose()  # type: ignore[attr-defined]
        finally:
            if self._holds_slot:
                self._holds_slot = False
                self._owner._release()

    async def __aenter__(self) -> Follow:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _view(self, event: StoredEvent, *, live: bool) -> StoredEvent:
        active = event.event_type == "run.paused" and (
            live or event.sequence_index == self._active_pause_seq
        )
        status = (event.data or {}).get("status")
        if event.event_type == "run.paused" and status in _PLACEHOLDER_PAUSES:
            # The mapping grows during a run: read it when an approval is shown.
            self._pii_mapping = await self._owner.store.get_pii_mapping(self.run_id) or None
        return redact_event(
            event,
            include_payloads=self._include_payloads,
            active_pause=active,
            pii_mapping=self._pii_mapping,
        )

    async def _run(self) -> AsyncIterator[FollowItem]:
        owner = self._owner
        store = owner.store
        if self.drained_terminal:
            yield FollowEnd("terminal", self._cursor)
            return
        feed = await owner._feed_for(self.run_id)
        follower = _Follower(until_terminal=self._until_terminal, limit=owner.buffer)
        loop = asyncio.get_running_loop()
        started = last_write = loop.time()
        cursor = self._cursor
        limit = self._max_duration_s if self._max_duration_s is not None else float("inf")
        try:
            head = feed.add(follower)
            # Catch up from the database to the feed's head, in batches.
            if head is not None and (cursor is None or cursor < head):
                passed_head = False
                while not passed_head:
                    batch = await store.get_events(
                        self.run_id, after_sequence_index=cursor, limit=owner.batch
                    )
                    for ev in batch:
                        if ev.sequence_index > head:
                            passed_head = True
                            break
                        yield await self._view(ev, live=False)
                        cursor = ev.sequence_index
                    if len(batch) < owner.batch or cursor is None or cursor >= head:
                        break
                last_write = loop.time()
            while True:
                while follower.buffer:
                    ev = follower.buffer.popleft()
                    if cursor is None or ev.sequence_index > cursor:
                        yield await self._view(ev, live=True)
                        cursor = ev.sequence_index
                        last_write = loop.time()
                if follower.end_reason is not None:
                    yield FollowEnd(follower.end_reason, cursor)
                    return
                now = loop.time()
                if now - started >= limit:
                    yield FollowEnd("max_duration", cursor)
                    return
                if now - last_write >= owner.heartbeat_s:
                    yield FollowHeartbeat()
                    last_write = loop.time()
                    continue
                timeout = min(owner.heartbeat_s - (now - last_write), limit - (now - started))
                follower.wake.clear()
                if follower.buffer or follower.end_reason is not None:
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(follower.wake.wait(), timeout=max(timeout, 0.0))
        finally:
            poller = feed.remove(follower)
            if poller is not None:
                # Let the cancelled poller finish (and return its connection):
                # no task outlives its last follower.
                await asyncio.wait({poller}, timeout=owner.stop_timeout_s)


async def _current_head(store: RunStore, run_id: str) -> int | None:
    """The run's highest event index now (None when it has none), from the database.

    A new feed starts here, never at a follower's cursor, so one follower's
    bogus cursor cannot hide events from the others.
    """
    state = getattr(store, "_state", None)
    next_index = getattr(state, "get_next_event_sequence", None)
    if next_index is not None:
        value = int(await next_index(run_id))
        return value - 1 if value > 0 else None
    head: int | None = None
    while True:  # custom store fallback: page to the end
        batch = await store.get_events(run_id, after_sequence_index=head, limit=100)
        if not batch:
            return head
        head = batch[-1].sequence_index
