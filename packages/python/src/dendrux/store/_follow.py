"""Live, tenant-scoped event feeds over ``RunStore`` (``RunStore.follow``).

Framework-agnostic: an async iterator of :class:`~dendrux.store.StoredEvent`
items plus :class:`FollowHeartbeat` and a final :class:`FollowEnd`. Serve it
as SSE, websockets or long-poll from your own routes.

Load model. One :class:`EventFeed` keeps **one poller per watched run** and
fans events out in memory to every follower of that run, so database load
grows with the number of watched runs, not with open connections. An idle
run is polled every 0.5 s, backing off to 2 s, plus a status read every 4th
idle poll. The poller is an ordinary asyncio task owned by the feed: it
starts with the first follower of a run and is cancelled, and awaited, when
the last one leaves. Slow cancellation remains tracked until the task finishes.
Budget about one
query per second per watched run, and give a busy feed its own small engine
(``RunStore.from_database_url(...)``) so viewers cannot starve the agents'
pool. An optional notifier reduces idle polling to reconciliation every
30 seconds while healthy, falling back to ``poll_interval_s`` on loss.

Never tie a run's lifetime to a client connection: follow the database log
(this module) rather than piping ``agent.stream()`` to a browser, where a
closed tab cancels the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

from dendrux import _codes
from dendrux._sentinel import _UnsetType
from dendrux.errors import RunNotFoundError
from dendrux.runtime._pending import approval_calls as approval_calls
from dendrux.runtime._pending import (
    check_read_bool,
    check_read_duration,
    check_read_int,
    check_read_scope,
    check_run_id,
)
from dendrux.store._redact import (
    PAUSE_CHANGING_EVENTS,
    PAUSE_VIEW_EVENTS,
    TOOL_EVIDENCE_EVENTS,
    PastParams,
    PauseView,
    ToolEvidence,
    _redact,
    needs_declared_tools,
    needs_tool_evidence,
    needs_unknown_calls,
    read_declared_tools,
    read_pause_view,
    read_tool_evidence,
    read_unknown_calls,
    view_with,
)
from dendrux.store._redact import (
    REVIEWED_EVENT_TYPES as REVIEWED_EVENT_TYPES,
)
from dendrux.store._redact import (
    redact_event as redact_event,
)
from dendrux.store._redact import (
    without_params as without_params,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dendrux.db._notify import EventNotifier, NotifierState
    from dendrux.runtime.state import TenantScope
    from dendrux.store import RunStore, StoredEvent

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"success", "error", "cancelled", "max_iterations"})
PAUSE_STATUSES = frozenset(
    {"waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"}
)
_TERMINAL_EVENT_TYPES = frozenset({"run.completed", "run.error", "run.cancelled"})

_DEFAULT_SHARE = _UnsetType()

FollowEndReason = Literal["terminal", "max_duration", "lagging", "error"]


class FollowCapacityError(RuntimeError):
    """The :class:`EventFeed` already serves ``max_followers`` followers, or
    this tenant already holds ``max_followers_per_tenant`` of them.

    Map it to HTTP 503 (retry shortly). ``limit`` is the cap that was hit;
    ``per_tenant`` is True when it was the per-tenant cap.
    """

    def __init__(self, limit: int, *, per_tenant: bool = False) -> None:
        self.limit = limit
        self.per_tenant = per_tenant
        scope = _CAPACITY_SCOPE[bool(per_tenant)]
        super().__init__(f"This event feed already serves its maximum of {limit} {scope}.")

    def __reduce__(self) -> Any:
        return (_capacity_error, (self.limit, self.per_tenant))


_CAPACITY_SCOPE = {False: "followers", True: "followers for one tenant scope"}


def _capacity_error(limit: int, per_tenant: bool) -> FollowCapacityError:
    return FollowCapacityError(limit, per_tenant=per_tenant)


def _scope_key(tenant_id: TenantScope) -> tuple[str, ...]:
    """The cap bucket of a scope: a tenant, all untenanted, or all unscoped followers."""
    if isinstance(tenant_id, str):
        return ("tenant", tenant_id)
    return ("untenanted",) if tenant_id is None else ("unscoped",)


def _tenant_cap(max_followers: int, value: object) -> int | None:
    """The per-tenant follower cap: a fifth of ``max_followers`` unless given."""
    if isinstance(value, _UnsetType):
        return max(1, max_followers // 5)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_followers_per_tenant must be a positive int or None.")
    return value


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


@dataclass(eq=False)
class _Follower:
    """One follower of a run feed: a bounded buffer plus a wake-up signal."""

    until_terminal: bool
    limit: int
    buffer: deque[StoredEvent] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    end_reason: FollowEndReason | None = None

    def push(self, event: StoredEvent) -> None:
        if self.end_reason is not None:
            return
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
        self.closing = False
        self.task: asyncio.Task[None] | None = None
        self.ready: asyncio.Future[None] | None = None
        self._missed_hint_logged = False
        self._poll_wake = asyncio.Event()
        self._view: PauseView | None = None
        self._view_generation = 0
        self._view_lock = asyncio.Lock()
        self._unknown: tuple[int, frozenset[str] | None] | None = None
        self._declared: tuple[int, frozenset[str] | None] | None = None
        self._tools: tuple[int, ToolEvidence] | None = None

    def _observe(self, event: StoredEvent) -> None:
        """Drop cached reads a newer event may have changed (the pause, tool records)."""
        tools = self._tools
        if (
            tools is not None
            and event.event_type in TOOL_EVIDENCE_EVENTS
            and event.sequence_index > tools[0]
        ):
            self._tools = None
        view = self._view
        if (
            view is not None
            and event.sequence_index > view.through
            and event.event_type in PAUSE_CHANGING_EVENTS
        ):
            self._view = None
            self._view_generation += 1

    async def pause_view(self, seq: int, tenant_id: TenantScope) -> PauseView:
        """The run's active pause for viewing event ``seq``, read once for every follower.

        A cached view answers for events up to its ``through`` index; a newer
        pause-changing event (published by the poller, or met by a follower
        ahead of it) triggers one fresh read. Every follower here passed the
        run's tenant scope, so one scoped read serves them all.
        """
        view = self._view
        if view is not None and seq <= view.through:
            return view
        async with self._view_lock:
            view = self._view
            if view is not None and seq <= view.through:
                return view
            generation = self._view_generation
            through = max(seq, self.head if self.head is not None else -1)
            fresh = await read_pause_view(self.owner.store, self.run_id, tenant_id, through)
            if generation == self._view_generation:
                self._view = fresh
            return fresh

    async def unknown_calls(self, seq: int, tenant_id: TenantScope) -> frozenset[str] | None:
        """The run's calls rejected as unknown tools, read once per newer failed call."""
        cached = self._unknown
        if cached is not None and seq <= cached[0]:
            return cached[1]
        through = max(seq, self.head if self.head is not None else -1)
        ids = await read_unknown_calls(self.owner.store, self.run_id, tenant_id)
        self._unknown = (through, ids)
        return ids

    async def tool_evidence(self, seq: int, tenant_id: TenantScope) -> ToolEvidence:
        """The run's tool records, read once and again only after a newer
        ``tools.declared`` / ``mcp.connected`` event (published before later events)."""
        cached = self._tools
        if cached is not None:
            return cached[1]
        through = max(seq, self.head if self.head is not None else -1)
        evidence = await read_tool_evidence(self.owner.store, self.run_id, tenant_id)
        self._tools = (through, evidence)
        return evidence

    async def declared_tools(self, seq: int, tenant_id: TenantScope) -> frozenset[str] | None:
        """Tool names the run recorded for declared calls, read once per newer event."""
        cached = self._declared
        if cached is not None and seq <= cached[0]:
            return cached[1]
        through = max(seq, self.head if self.head is not None else -1)
        names = await read_declared_tools(self.owner.store, self.run_id, tenant_id)
        self._declared = (through, names)
        return names

    def add(self, follower: _Follower) -> int | None:
        """Register ``follower``; return the head it must catch up to from the database."""
        self.followers.add(follower)
        if self.terminal and follower.until_terminal:
            follower.end("terminal")
        if self.task is None or self.task.done():
            self.ready = asyncio.get_running_loop().create_future()
            self.ready.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
            self.task = asyncio.get_running_loop().create_task(
                self._poll(self.ready), name=f"dendrux-run-feed:{self.run_id}"
            )
            self.task.add_done_callback(partial(self._done, self.ready))
        else:
            self._poll_wake.set()
        return self.head

    def remove(self, follower: _Follower) -> asyncio.Task[None] | None:
        """Unregister ``follower``; return the poller task to await when it was the last."""
        self.followers.discard(follower)
        if self.followers:
            return None
        if self.closing:
            return self.task
        self.closing = True
        if self.task is not None and not self.task.done():
            self.task.cancel()
            return self.task
        self.owner._drop(self)
        return None

    def _done(self, ready: asyncio.Future[None], task: asyncio.Task[None]) -> None:
        if not ready.done():
            ready.set_exception(RuntimeError("The run feed closed during startup."))
        if not task.cancelled():
            task.exception()  # observe failures, including shutdown failures
        if self.closing or not self.followers:
            self.owner._drop(self)

    def _publish(self, events: list[StoredEvent]) -> None:
        for event in events:
            self._observe(event)
            for follower in list(self.followers):
                follower.push(event)
            self.head = event.sequence_index

    async def _fetch(self) -> list[StoredEvent]:
        return await self.owner.store.get_events(
            self.run_id, after_sequence_index=self.head, limit=self.owner.batch
        )

    async def _poll(self, ready: asyncio.Future[None]) -> None:
        notifier = self.owner.notifier
        context = (
            notifier.subscribe(self.run_id)
            if notifier is not None
            else contextlib.nullcontext(None)
        )
        try:
            async with context as subscription:
                if subscription is not None:
                    self._poll_wake = subscription.event
                if not ready.done():
                    ready.set_result(None)
                await self._poll_loop(subscription.state if subscription else None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            if isinstance(exc, FollowCapacityError):
                return
            _codes.log_code(logger, _codes.FOLLOW_FEED_ERROR, run_id=self.run_id)
            for follower in list(self.followers):
                follower.end("error")

    async def _poll_loop(self, state: NotifierState | None) -> None:
        owner = self.owner
        delay = owner.poll_interval_s
        idle_polls = 0
        check_status = True
        generation = state.generation if state is not None else None
        woken = False
        reconciling = False
        try:
            while self.followers:
                if self.terminal and all(f.until_terminal for f in self.followers):
                    return
                if state is not None and state.generation != generation:
                    reconciling = False
                    delay = owner.poll_interval_s
                    jitter = random.uniform(0, owner.poll_interval_s)
                    if jitter:
                        await asyncio.sleep(jitter)
                    generation = state.generation
                # Clear before fetching so joins during the read stay pending.
                # Event.set coalesces a burst into one immediate poll.
                self._poll_wake.clear()
                events = await self._fetch()
                if (
                    events
                    and reconciling
                    and state is not None
                    and state.healthy
                    and state.generation == generation
                    and not self._poll_wake.is_set()
                    and not self._missed_hint_logged
                ):
                    _codes.log_code(
                        logger, _codes.NOTIFIER_MISSED_HINT, run_id=self.run_id, count=len(events)
                    )
                    self._missed_hint_logged = True
                reconciling = False
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
                    status = await owner.store._status(self.run_id)
                    if status is None or status in TERMINAL_STATUSES:
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
                healthy_wait = state is not None and state.healthy and not (woken and not events)
                if state is not None and state.healthy:
                    delay = owner.poll_healthy_s if healthy_wait else owner.poll_interval_s
                woken = False
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(delay):
                        await self._poll_wake.wait()
                        woken = True
                reconciling = healthy_wait and not woken
                if not events:
                    delay = min(delay * 2, owner.poll_max_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            _codes.log_code(logger, _codes.FOLLOW_FEED_ERROR, run_id=self.run_id)
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
        max_followers_per_tenant: Most concurrent followers for one tenant
            scope, so no scope can take every slot. The cap is per scope,
            not per user: each string ``tenant_id`` is one scope, all
            untenanted followers (``tenant_id=None``, every user of a
            single-tenant app) are one scope, and all ``UNSCOPED`` followers
            are one scope. Defaults to a fifth of ``max_followers`` (100 of
            500), at least 1; ``None`` turns it off. A single-tenant app that
            needs more followers passes ``max_followers_per_tenant=None`` (or
            a larger number) to its own ``EventFeed(store, ...)``.
        poll_interval_s / poll_max_s: idle polling starts at the first and
            doubles up to the second; a new event resets it.
        notifier: Borrowed wake-up hints, defaulting to the store's notifier.
            The caller owns its lifecycle.
        poll_healthy_s: Reconciliation interval while the notifier is healthy.
            Degraded notifiers start at ``poll_interval_s`` and back off while idle.
        heartbeat_s: idle time before a :class:`FollowHeartbeat` is yielded.
        buffer: events buffered per follower before it ends as ``"lagging"``.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        max_followers: int = 500,
        max_followers_per_tenant: int | None | _UnsetType = _DEFAULT_SHARE,
        poll_interval_s: float = 0.5,
        poll_max_s: float = 2.0,
        heartbeat_s: float = 15.0,
        buffer: int = 1000,
        status_every: int = 4,
        batch: int = 100,
        stop_timeout_s: float = 5.0,
        notifier: EventNotifier | None = None,
        poll_healthy_s: float = 30.0,
    ) -> None:
        if isinstance(max_followers, bool) or not isinstance(max_followers, int):
            raise ValueError("max_followers must be a positive int.")
        if max_followers <= 0 or buffer <= 0 or status_every <= 0 or batch <= 0:
            raise ValueError("max_followers, buffer, status_every and batch must be positive.")
        tenant_cap = _tenant_cap(max_followers, max_followers_per_tenant)
        if not 0 < poll_interval_s <= poll_max_s or heartbeat_s <= 0:
            raise ValueError("poll intervals and heartbeat_s must be positive (interval <= max).")
        if not math.isfinite(poll_healthy_s) or poll_healthy_s <= 0:
            raise ValueError("poll_healthy_s must be finite and positive.")
        self.store = store
        self.notifier = notifier if notifier is not None else getattr(store, "_notifier", None)
        self.poll_healthy_s = poll_healthy_s
        self.max_followers = max_followers
        self.max_followers_per_tenant = tenant_cap
        self.poll_interval_s = poll_interval_s
        self.poll_max_s = poll_max_s
        self.heartbeat_s = heartbeat_s
        self.buffer = buffer
        self.status_every = status_every
        self.batch = batch
        self.stop_timeout_s = stop_timeout_s
        self._closed = False
        self._open = 0
        self._by_scope: dict[tuple[str, ...], int] = {}
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
        (``async with``, or ``await follow.aclose()``) for deterministic cleanup.
        Abandoned iteration also releases resources through generator finalization.

        Items are :class:`StoredEvent` (passed through :func:`redact_event`),
        :class:`FollowHeartbeat`, and one final :class:`FollowEnd`. With
        ``until_terminal=True`` the follow ends once the run's status is
        terminal and every event was delivered; ``False`` keeps following.
        ``max_duration_s=None`` removes the time limit.
        """
        check_run_id(run_id)
        check_read_scope(tenant_id)
        check_read_int("after_sequence_index", after_sequence_index)
        check_read_bool("include_payloads", include_payloads)
        check_read_bool("until_terminal", until_terminal)
        check_read_duration("max_duration_s", max_duration_s)
        return Follow(
            self,
            run_id,
            tenant_id=tenant_id,
            after=after_sequence_index,
            include_payloads=include_payloads,
            until_terminal=until_terminal,
            max_duration_s=max_duration_s,
        )

    def _acquire(self, tenant_id: TenantScope) -> None:
        if self._closed:
            raise RuntimeError("This event feed is closed.")
        closing = sum(f.closing for f in self._feeds.values())
        if self._open + closing >= self.max_followers:
            _codes.log_code(logger, _codes.FOLLOW_CAPACITY, limit=self.max_followers)
            raise FollowCapacityError(self.max_followers)
        self._take_tenant_slot(tenant_id)
        self._open += 1

    def _take_tenant_slot(self, tenant_id: TenantScope) -> None:
        """Count the follower against its scope; refuse beyond the per-scope cap."""
        cap = self.max_followers_per_tenant
        if cap is None:
            return
        key = _scope_key(tenant_id)
        held = self._by_scope.get(key, 0)
        if held >= cap:
            # The tenant id is caller data: never logged.
            _codes.log_code(logger, _codes.FOLLOW_TENANT_CAPACITY, limit=cap)
            raise FollowCapacityError(cap, per_tenant=True)
        self._by_scope[key] = held + 1

    def _release(self, tenant_id: TenantScope) -> None:
        self._open = max(0, self._open - 1)
        self._give_tenant_slot(tenant_id)

    def _give_tenant_slot(self, tenant_id: TenantScope) -> None:
        key = _scope_key(tenant_id)
        held = self._by_scope.pop(key, 0)
        if held > 1:
            self._by_scope[key] = held - 1

    def _drop(self, feed: _RunFeed) -> None:
        if self._feeds.get(feed.run_id) is feed:
            del self._feeds[feed.run_id]

    async def _wait_for_closing(self, run_id: str) -> None:
        feed = self._feeds.get(run_id)
        if feed is None or not feed.closing:
            return
        task = feed.task
        if task is not None and not task.done():
            _, pending = await asyncio.wait({task}, timeout=self.stop_timeout_s)
            if pending:
                raise FollowCapacityError(self.max_followers)
        self._drop(feed)

    async def _feed_for(self, run_id: str) -> _RunFeed:
        while True:
            if self._closed:
                raise RuntimeError("This event feed is closed.")
            feed = self._feeds.get(run_id)
            if feed is None:
                head = await _current_head(self.store, run_id)
                if self._closed:
                    raise RuntimeError("This event feed is closed.")
                # Another follower may have created the feed while we read the head.
                feed = self._feeds.setdefault(run_id, _RunFeed(self, run_id, head))
            if not feed.closing:
                return feed
            await self._wait_for_closing(run_id)

    async def aclose(self) -> None:
        """Stop all pollers and await their actual completion before disposing the store.

        Cancellation of the caller never removes unfinished tasks from accounting.
        A later call can finish waiting for the same tasks.
        """
        self._closed = True
        tasks = []
        for feed in list(self._feeds.values()):
            for follower in feed.followers:
                follower.end("error")
            if not feed.closing:
                feed.closing = True
                if feed.task is not None and not feed.task.done():
                    feed.task.cancel()
            if feed.task is not None:
                tasks.append(feed.task)
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))


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
        self._past = PastParams()
        self._until_terminal = until_terminal
        self._max_duration_s = max_duration_s
        self._started = False
        self._holds_slot = False
        self._closed = False
        self._start_lock = asyncio.Lock()
        self._feed: _RunFeed | None = None
        self._follower: _Follower | None = None
        self._head_at_start: int | None = None
        self._deadline: float | None = None
        self._gen: AsyncIterator[FollowItem] | None = None
        self.drained_terminal = False
        """True after :meth:`start` when the run is terminal and nothing is left
        after the cursor (an HTTP route answers 204 instead of streaming)."""

    async def start(self) -> None:
        """Check the scope and take a follower slot; idempotent.

        Concurrent calls on one ``Follow`` take one slot. A :meth:`aclose`
        that lands while a start is waiting on the store wins: the start
        raises ``RuntimeError`` and takes no slot.

        Raises:
            RunNotFoundError: the run does not exist or is outside the tenant.
            FollowCapacityError: the feed is full.
        """
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            self._raise_if_closed()
            start_notifier = getattr(self._owner.store, "_start_owned_notifier", None)
            if start_notifier is not None:
                await start_notifier()
            await self._check_run()
            self._raise_if_closed()  # closed while waiting on the store: take no slot
            # No await from here to the slot: aclose() cannot interleave.
            self._owner._acquire(self._tenant_id)
            self._holds_slot = True
            try:
                if self._owner.notifier is not None:
                    feed = await self._owner._feed_for(self.run_id)
                    self._raise_if_closed()
                    follower = _Follower(
                        until_terminal=self._until_terminal, limit=self._owner.buffer
                    )
                    self._feed, self._follower = feed, follower
                    self._head_at_start = feed.add(follower)
                    assert feed.ready is not None
                    await asyncio.shield(feed.ready)
                    self._raise_if_closed()
            except BaseException:
                self._release()
                await self._detach()
                raise
            self._started = True
            if self._max_duration_s is not None:
                self._deadline = asyncio.get_running_loop().time() + self._max_duration_s

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("This follow is closed.")

    async def _check_run(self) -> None:
        """Scope check, terminal drain check and wait for a closing feed (all awaits)."""
        store = self._owner.store
        status = await store._status(self.run_id, self._tenant_id)
        if status is None:
            raise RunNotFoundError(self.run_id)
        if status in TERMINAL_STATUSES and self._until_terminal:
            pending = await store.get_events(
                self.run_id, after_sequence_index=self._cursor, limit=1
            )
            self.drained_terminal = not pending
        await self._owner._wait_for_closing(self.run_id)

    async def __aiter__(self) -> AsyncIterator[FollowItem]:
        # A separate generator gets finalized when an async-for is abandoned,
        # even if the caller retains the Follow/context-manager object.
        try:
            while True:
                try:
                    yield await self.__anext__()
                except StopAsyncIteration:
                    return
        finally:
            await self.aclose()

    def _release(self) -> None:
        if self._holds_slot:
            self._holds_slot = False
            self._owner._release(self._tenant_id)

    def __del__(self) -> None:
        self._release()
        if self._feed is not None and self._follower is not None:
            self._feed.remove(self._follower)

    async def __anext__(self) -> FollowItem:
        await self.start()
        if self._gen is None:
            self._gen = self._run()
        try:
            return await self._gen.__anext__()
        except BaseException:
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
            self._release()
            await self._detach()

    async def _detach(self) -> None:
        if self._feed is not None and self._follower is not None:
            poller = self._feed.remove(self._follower)
            self._feed = self._follower = None
            if poller is not None:
                _, pending = await asyncio.wait({poller}, timeout=self._owner.stop_timeout_s)
                if pending:
                    _codes.log_code(logger, _codes.FOLLOW_SHUTDOWN_PENDING, run_id=self.run_id)

    async def __aenter__(self) -> Follow:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _view(self, event: StoredEvent) -> StoredEvent:
        return await _feed_view(
            self._feed,
            event,
            tenant_id=self._tenant_id,
            include_payloads=self._include_payloads,
            past=self._past,
        )

    def _expired(self) -> bool:
        return self._deadline is not None and asyncio.get_running_loop().time() >= self._deadline

    async def _run(self) -> AsyncIterator[FollowItem]:
        owner = self._owner
        store = owner.store
        cursor = self._cursor
        loop = asyncio.get_running_loop()
        last_write = loop.time()
        try:
            if self.drained_terminal:
                yield FollowEnd("terminal", cursor)
                return
            if self._feed is None:
                async with asyncio.timeout_at(self._deadline):
                    feed = await owner._feed_for(self.run_id)
                follower = _Follower(until_terminal=self._until_terminal, limit=owner.buffer)
                self._feed, self._follower = feed, follower
                head = feed.add(follower)
            else:
                assert self._follower is not None
                feed, follower = self._feed, self._follower
                head = self._head_at_start
            if head is not None and (cursor is None or cursor < head):
                while cursor is None or cursor < head:
                    if self._expired():
                        yield FollowEnd("max_duration", cursor)
                        return
                    async with asyncio.timeout_at(self._deadline):
                        batch = await store.get_events(
                            self.run_id, after_sequence_index=cursor, limit=owner.batch
                        )
                    for ev in batch:
                        if ev.sequence_index > head:
                            break
                        if self._expired():
                            yield FollowEnd("max_duration", cursor)
                            return
                        async with asyncio.timeout_at(self._deadline):
                            view = await self._view(ev)
                        cursor = self._cursor = ev.sequence_index
                        yield view
                    if len(batch) < owner.batch or not batch or batch[-1].sequence_index >= head:
                        break
                last_write = loop.time()
            while True:
                if self._expired():
                    yield FollowEnd("max_duration", cursor)
                    return
                if follower.buffer:
                    ev = follower.buffer.popleft()
                    if cursor is None or ev.sequence_index > cursor:
                        async with asyncio.timeout_at(self._deadline):
                            view = await self._view(ev)
                        cursor = self._cursor = ev.sequence_index
                        yield view
                        last_write = loop.time()
                    continue
                if follower.end_reason is not None:
                    yield FollowEnd(follower.end_reason, cursor)
                    return
                now = loop.time()
                if now - last_write >= owner.heartbeat_s:
                    yield FollowHeartbeat()
                    last_write = loop.time()
                    continue
                timeout = owner.heartbeat_s - (now - last_write)
                if self._deadline is not None:
                    timeout = min(timeout, self._deadline - now)
                follower.wake.clear()
                if follower.buffer or follower.end_reason is not None:
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(follower.wake.wait(), timeout=max(timeout, 0.0))
        except TimeoutError:
            yield FollowEnd("max_duration", cursor)
        finally:
            self._release()
            await self._detach()


async def _feed_view(
    feed: _RunFeed | None,
    event: StoredEvent,
    *,
    tenant_id: TenantScope,
    include_payloads: bool,
    past: PastParams,
) -> StoredEvent:
    """A follower's view of ``event``, using the run feed's shared reads.

    Without a feed (never the case once events flow) nothing is known about
    the run, so pause arguments and failed calls' names stay hidden.
    """
    if feed is None:
        return _redact(
            event,
            include_payloads=include_payloads,
            active_pause=False,
            approval_params=None,
            unknown_calls=None,
        )
    unknown = declared = None
    tools: ToolEvidence | None = None
    if needs_unknown_calls(event, include_payloads):
        unknown = await feed.unknown_calls(event.sequence_index, tenant_id)
    if needs_declared_tools(event, include_payloads):
        declared = await feed.declared_tools(event.sequence_index, tenant_id)
    if needs_tool_evidence(event, include_payloads):
        tools = await feed.tool_evidence(event.sequence_index, tenant_id)
    view = PauseView(event.sequence_index, None, {})
    if event.event_type in PAUSE_VIEW_EVENTS:
        view = await feed.pause_view(event.sequence_index, tenant_id)
    return await view_with(
        feed.owner.store,
        feed.run_id,
        event,
        view,
        tenant_id=tenant_id,
        include_payloads=include_payloads,
        past=past,
        unknown_calls=unknown,
        declared_tools=declared,
        tools=tools,
    )


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
