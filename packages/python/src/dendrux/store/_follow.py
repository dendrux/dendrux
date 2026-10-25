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

from dendrux._sentinel import _UnsetType
from dendrux.errors import RunNotFoundError
from dendrux.runtime._pending import approval_calls as approval_calls
from dendrux.runtime.state import _check_scope
from dendrux.runtime.submit import DEFAULT_DENIAL_REASON

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
        scope = "followers for one tenant" if per_tenant else "followers"
        super().__init__(f"This event feed already serves its maximum of {limit} {scope}.")

    def __reduce__(self) -> Any:
        return (_capacity_error, (self.limit, self.per_tenant))


def _capacity_error(limit: int, per_tenant: bool) -> FollowCapacityError:
    return FollowCapacityError(limit, per_tenant=per_tenant)


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

# Top-level event-data keys holding free text (a human's or client's words,
# exception text) that a reader without payloads never sees.
_FREE_TEXT_KEYS: dict[str, frozenset[str]] = {
    "run.resumed": frozenset({"user_input"}),
    "run.error": frozenset({"error"}),
    "mcp.error": frozenset({"error"}),
}
# Dendrux's own fixed denial reasons; any other reason is a person's text.
_CONSTANT_DENIAL_REASONS = frozenset({"denied_by_policy", DEFAULT_DENIAL_REASON})


def redact_event(
    event: StoredEvent,
    *,
    include_payloads: bool = False,
    active_pause: bool = False,
    pii_mapping: Mapping[str, str] | None = None,
    approval_params: Mapping[str, dict[str, Any]] | None = None,
) -> StoredEvent:
    """The event as a reader may see it (B2 and the approval-arguments rule).

    In every mode, the arguments of an **approval or input** ``run.paused``
    use ``approval.requested.params``, exactly as the model sent them. Without
    those arguments, ``params`` is hidden. Client-tool pauses keep the 0.4.1 form.

    With ``include_payloads=True`` nothing else changes. Otherwise:

    - the system prompt (``run.started``), provider request and response
      bodies and any ``pii_mapping`` are removed from ``data``;
    - in ``run.paused``, each pending call's ``params`` and the requested
      pause's ``message`` are removed **unless** ``active_pause`` is true.
      Active client-tool arguments remain visible; active approval/input
      arguments require recorded model-sent params. A past pause's arguments
      are history and are hidden;
    - ``approval.requested.params`` is removed unless ``active_pause`` is true
      (the request belongs to the run's active pause);
    - free text is removed: ``run.resumed.user_input``, a denial ``reason``
      in ``approval.decided.decisions`` (Dendrux's constant reasons stay),
      ``error`` in ``run.error`` and ``mcp.error`` (``error_info`` and codes
      stay) and image descriptions (``image.state.described[].text``).

    Tool results and server-tool arguments never appear in event data; read
    them with ``RunStore.get_tool_invocations`` (trusted readers only).
    ``pii_mapping`` is accepted for compatibility and not used.
    """
    if not event.data:
        return event
    data = dict(event.data)
    if (
        event.event_type == "run.paused"
        and data.get("status") in _PLACEHOLDER_PAUSES
        and isinstance(data.get("pending_tool_calls"), list)
    ):
        data["pending_tool_calls"] = approval_calls(data["pending_tool_calls"], approval_params)
    if include_payloads:
        return dataclasses.replace(event, data=data) if data != event.data else event
    data = {k: v for k, v in data.items() if k not in _HIDDEN_EVENT_KEYS}
    if event.event_type == "run.paused" and not active_pause:
        if isinstance(data.get("pending_tool_calls"), list):
            data["pending_tool_calls"] = [without_params(c) for c in data["pending_tool_calls"]]
        data.pop("message", None)
    return dataclasses.replace(event, data=_without_free_text(event.event_type, data, active_pause))


def _without_free_text(event_type: str, data: dict[str, Any], active: bool) -> dict[str, Any]:
    """``data`` without the free text of its event type (one rule set for every reader)."""
    hidden = _FREE_TEXT_KEYS.get(event_type)
    if hidden:
        data = {k: v for k, v in data.items() if k not in hidden}
    if event_type == "approval.requested" and not active:
        data.pop("params", None)
    elif event_type == "approval.decided" and isinstance(data.get("decisions"), list):
        data["decisions"] = [_without_reason(d) for d in data["decisions"]]
    elif event_type == "image.state" and isinstance(data.get("described"), list):
        data["described"] = [_without_keys(d, ("text",)) for d in data["described"]]
    return data


def _without_reason(decision: Any) -> Any:
    if isinstance(decision, dict) and decision.get("reason") not in _CONSTANT_DENIAL_REASONS:
        return _without_keys(decision, ("reason",))
    return decision


def _without_keys(item: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(item, dict):
        return item
    return {k: v for k, v in item.items() if k not in keys}


# Event types whose view depends on the run's active pause.
_PAUSE_VIEW_EVENTS = frozenset({"run.paused", "approval.requested"})
# Event types after which a cached view of the active pause may be out of date.
_PAUSE_CHANGING_EVENTS = _PAUSE_VIEW_EVENTS | {
    "run.resumed",
    "run.completed",
    "run.error",
    "run.cancelled",
    "run.interrupted",
    "run.abandoned",
}
# Past pauses' recorded arguments kept per watched run (include_payloads readers).
_PAST_PARAMS_CACHE = 256


@dataclass(slots=True)
class _PauseView:
    """The run's active pause when it was read, valid for events up to ``through``.

    ``active`` is the index of the active ``run.paused`` event (None when the
    run is not paused); ``active_params`` its recorded model-sent arguments by
    call id (the ``approval.requested`` events of that pause).
    """

    through: int
    active: int | None
    active_params: dict[str, dict[str, Any]]
    past_params: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)


async def _read_pause_view(
    store: Any, run_id: str, tenant_id: TenantScope, through: int
) -> _PauseView:
    """Two bounded reads: the active pause's index, then its recorded arguments."""
    active = await store._active_pause_sequence(run_id, tenant_id)
    params = {} if active is None else await store._approval_params(run_id, active, tenant_id)
    return _PauseView(through, active, params)


async def _view_with(
    store: Any,
    run_id: str,
    event: StoredEvent,
    view: _PauseView,
    *,
    tenant_id: TenantScope,
    include_payloads: bool,
) -> StoredEvent:
    """``event`` as a reader sees it, given the run's active pause ``view``."""
    data = event.data or {}
    seq = event.sequence_index
    if event.event_type == "approval.requested":
        active = (
            view.active is not None
            and seq < view.active
            and isinstance(data.get("call_id"), str)
            and data["call_id"] in view.active_params
        )
        return redact_event(event, include_payloads=include_payloads, active_pause=active)
    if event.event_type != "run.paused":
        return redact_event(event, include_payloads=include_payloads)
    active = seq == view.active
    params = None
    if (include_payloads or active) and data.get("status") in _PLACEHOLDER_PAUSES:
        params = view.active_params if active else view.past_params.get(seq)
        if params is None:  # a past pause's recorded arguments never change
            params = await store._approval_params(run_id, seq, tenant_id)
            if len(view.past_params) < _PAST_PARAMS_CACHE:
                view.past_params[seq] = params
    return redact_event(
        event, include_payloads=include_payloads, active_pause=active, approval_params=params
    )


def without_params(call: Any) -> Any:
    """A pending-call dict without its arguments."""
    return _without_keys(call, ("params", "arguments"))


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
        self._view: _PauseView | None = None
        self._view_generation = 0
        self._view_lock = asyncio.Lock()

    def _observe(self, event: StoredEvent) -> None:
        """Drop the cached pause view when a newer event may have changed the pause."""
        view = self._view
        if (
            view is not None
            and event.sequence_index > view.through
            and event.event_type in _PAUSE_CHANGING_EVENTS
        ):
            self._view = None
            self._view_generation += 1

    async def pause_view(self, seq: int, tenant_id: TenantScope) -> _PauseView:
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
            fresh = await _read_pause_view(self.owner.store, self.run_id, tenant_id, through)
            if view is not None:
                fresh.past_params = view.past_params
            if generation == self._view_generation:
                self._view = fresh
            return fresh

    def add(self, follower: _Follower) -> int | None:
        """Register ``follower``; return the head it must catch up to from the database."""
        self.followers.add(follower)
        if self.terminal and follower.until_terminal:
            follower.end("terminal")
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(
                self._poll(), name=f"dendrux-run-feed:{self.run_id}"
            )
            self.task.add_done_callback(self._done)
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

    def _done(self, task: asyncio.Task[None]) -> None:
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
        max_followers_per_tenant: Most concurrent followers for one tenant
            (a string ``tenant_id``), so one tenant cannot take every slot.
            Defaults to a fifth of ``max_followers`` (100 of 500), at least 1;
            ``None`` turns it off. Untenanted (``None``) and ``UNSCOPED``
            followers are not one tenant and share only ``max_followers``.
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
        max_followers_per_tenant: int | None | _UnsetType = _DEFAULT_SHARE,
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
        if isinstance(max_followers_per_tenant, _UnsetType):
            max_followers_per_tenant = max(1, max_followers // 5)
        elif max_followers_per_tenant is not None and (
            isinstance(max_followers_per_tenant, bool)
            or not isinstance(max_followers_per_tenant, int)
            or max_followers_per_tenant <= 0
        ):
            raise ValueError("max_followers_per_tenant must be a positive int or None.")
        if not 0 < poll_interval_s <= poll_max_s or heartbeat_s <= 0:
            raise ValueError("poll intervals and heartbeat_s must be positive (interval <= max).")
        self.store = store
        self.max_followers = max_followers
        self.max_followers_per_tenant: int | None = max_followers_per_tenant
        self.poll_interval_s = poll_interval_s
        self.poll_max_s = poll_max_s
        self.heartbeat_s = heartbeat_s
        self.buffer = buffer
        self.status_every = status_every
        self.batch = batch
        self.stop_timeout_s = stop_timeout_s
        self._closed = False
        self._open = 0
        self._by_tenant: dict[str, int] = {}
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

    def _acquire(self, tenant_id: TenantScope) -> None:
        if self._closed:
            raise RuntimeError("This event feed is closed.")
        closing = sum(f.closing for f in self._feeds.values())
        if self._open + closing >= self.max_followers:
            logger.warning(
                "follower limit reached: code=follow_capacity limit=%d",
                self.max_followers,
                extra={"code": "follow_capacity"},
            )
            raise FollowCapacityError(self.max_followers)
        cap = self.max_followers_per_tenant
        if isinstance(tenant_id, str) and cap is not None:
            if self._by_tenant.get(tenant_id, 0) >= cap:
                # The tenant id is caller data: never logged.
                logger.warning(
                    "per-tenant follower limit reached: code=follow_tenant_capacity limit=%d",
                    cap,
                    extra={"code": "follow_tenant_capacity"},
                )
                raise FollowCapacityError(cap, per_tenant=True)
            self._by_tenant[tenant_id] = self._by_tenant.get(tenant_id, 0) + 1
        self._open += 1

    def _release(self, tenant_id: TenantScope) -> None:
        self._open = max(0, self._open - 1)
        if isinstance(tenant_id, str) and tenant_id in self._by_tenant:
            held = self._by_tenant[tenant_id] - 1
            if held > 0:
                self._by_tenant[tenant_id] = held
            else:
                del self._by_tenant[tenant_id]

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
        self._until_terminal = until_terminal
        self._max_duration_s = max_duration_s
        self._started = False
        self._holds_slot = False
        self._closed = False
        self._feed: _RunFeed | None = None
        self._follower: _Follower | None = None
        self._deadline: float | None = None
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
        status = await store._status(self.run_id, self._tenant_id)
        if status is None:
            raise RunNotFoundError(self.run_id)
        if status in TERMINAL_STATUSES and self._until_terminal:
            pending = await store.get_events(
                self.run_id, after_sequence_index=self._cursor, limit=1
            )
            self.drained_terminal = not pending
        await self._owner._wait_for_closing(self.run_id)
        self._owner._acquire(self._tenant_id)
        self._holds_slot = True
        self._started = True
        if self._max_duration_s is not None:
            self._deadline = asyncio.get_running_loop().time() + self._max_duration_s

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

    async def __aenter__(self) -> Follow:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _view(self, event: StoredEvent) -> StoredEvent:
        if event.event_type not in _PAUSE_VIEW_EVENTS:
            return redact_event(event, include_payloads=self._include_payloads)
        feed = self._feed
        assert feed is not None  # set before the first event is delivered
        view = await feed.pause_view(event.sequence_index, self._tenant_id)
        return await _view_with(
            self._owner.store,
            self.run_id,
            event,
            view,
            tenant_id=self._tenant_id,
            include_payloads=self._include_payloads,
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
            async with asyncio.timeout_at(self._deadline):
                feed = await owner._feed_for(self.run_id)
            follower = _Follower(until_terminal=self._until_terminal, limit=owner.buffer)
            self._feed, self._follower = feed, follower
            head = feed.add(follower)
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
            if self._feed is not None and self._follower is not None:
                poller = self._feed.remove(self._follower)
                self._feed = self._follower = None
                if poller is not None:
                    # Timeout/caller cancellation leaves the task tracked in EventFeed.
                    _, pending = await asyncio.wait({poller}, timeout=owner.stop_timeout_s)
                    if pending:
                        logger.warning(
                            "poller shutdown pending: run_id=%s code=follow_shutdown_pending",
                            self.run_id,
                            extra={"run_id": self.run_id, "code": "follow_shutdown_pending"},
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
