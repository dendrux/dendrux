"""Live, tenant-scoped event feeds over ``RunStore`` (``RunStore.follow``).

Framework-agnostic: an async iterator of :class:`~dendrux.store.StoredEvent`
items plus :class:`FollowHeartbeat` and a final :class:`FollowEnd`. Serve it
as SSE, websockets or long-poll from your own routes.

Load model. Every follower on one ``RunStore`` (per event loop) is served by
**one coordinator task** that reads all watched runs in batches: per chunk of
up to 200 due runs, one events query and one status query (a custom store
without the batched reads gets one read per run under the same budget). The
coordinator spends at most 20 queries per second, oldest-due run first, and
fans events out in memory, so database load grows with watched runs, not with
open connections. Without live updates an idle run is read every 0.5 s,
backing off to 2 s, with a status read every 4th read. With healthy live
updates (``RunStore(store)`` on a store with ``live_updates``) a hint makes a
run due at once and a safety poll runs every 10 s (±20 %). The coordinator
starts with the first follower and ends with ``RunStore.close()`` or after
60 s with nothing to watch. Give a busy reader its own small engine so viewers
cannot starve the agents' pool.

Never tie a run's lifetime to a client connection: follow the database log
(this module) rather than piping ``agent.stream()`` to a browser, where a
closed tab cancels the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUSES as PAUSE_STATUSES
from dendrux._lifecycle import TERMINAL_EVENT_TYPES as _TERMINAL_EVENT_TYPES
from dendrux._lifecycle import TERMINAL_STATUSES as TERMINAL_STATUSES
from dendrux._sentinel import _UnsetType
from dendrux.db._live_updates import _error_kind
from dendrux.errors import FollowCapacityError as FollowCapacityError
from dendrux.errors import LiveUpdatesUnavailableError, RunNotFoundError
from dendrux.errors import _capacity_error as _capacity_error
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

DEFAULT_MAX_FOLLOWERS = 500
DEFAULT_POLL_INTERVAL_S = 0.5
DEFAULT_POLL_MAX_S = 2.0
DEFAULT_HEARTBEAT_S = 15.0
DEFAULT_BUFFER = 1000
DEFAULT_STATUS_EVERY = 4
DEFAULT_BATCH = 100
DEFAULT_STOP_TIMEOUT_S = 5.0
DEFAULT_POLL_HEALTHY_S = 10.0
DEFAULT_MAX_DURATION_S = 900.0
_DEFAULT_TENANT_DIVISOR = 5

#: Coordinator limits (spec §4.2, §4.3); module constants so tests and docs share them.
POLL_BATCH_RUNS = 200
MAX_ROWS_PER_READ = 2_000
MAX_LIMIT_PER_RUN = 100
MAX_READS_PER_S = 20.0
STATUS_MAX_AGE_S = 30.0
IDLE_EXIT_S = 60.0
MISSED_HINT_GRACE_S = 5.0
HEALTHY_JITTER = 0.2
MAX_RESTARTS = 3

_NEVER = math.inf
_ULID = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from dendrux.db._live_updates import LiveUpdates
    from dendrux.runtime.state import TenantScope
    from dendrux.store import RunStore, StoredEvent

logger = logging.getLogger(__name__)


_DEFAULT_SHARE = _UnsetType()

FollowEndReason = Literal["terminal", "max_duration", "lagging", "error"]


def _scope_key(tenant_id: TenantScope) -> tuple[str, ...]:
    """The cap bucket of a scope: a tenant, all untenanted, or all unscoped followers."""
    if isinstance(tenant_id, str):
        return ("tenant", tenant_id)
    return ("untenanted",) if tenant_id is None else ("unscoped",)


def _tenant_cap(max_followers: int, value: object) -> int | None:
    """The per-tenant follower cap: a fifth of ``max_followers`` unless given."""
    if isinstance(value, _UnsetType):
        return max(1, max_followers // _DEFAULT_TENANT_DIVISOR)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_followers_per_tenant must be a positive int or None.")
    return value


def _log_run_id(run_id: str) -> str | None:
    """Run ids reach logs only in ULID form; anything else is caller data."""
    return run_id if _ULID.fullmatch(run_id) else None


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


@dataclass(frozen=True, slots=True)
class _Params:
    """Polling settings of one ``EventFeed``; a run watched by several uses the tightest."""

    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    poll_max_s: float = DEFAULT_POLL_MAX_S
    poll_healthy_s: float = DEFAULT_POLL_HEALTHY_S
    status_every: int = DEFAULT_STATUS_EVERY
    batch: int = DEFAULT_BATCH


_DEFAULT_PARAMS = _Params()


def _tightest(params: Iterable[_Params]) -> _Params:
    chosen = list(params)
    if not chosen:
        return _DEFAULT_PARAMS
    return _Params(
        poll_interval_s=min(p.poll_interval_s for p in chosen),
        poll_max_s=min(p.poll_max_s for p in chosen),
        poll_healthy_s=min(p.poll_healthy_s for p in chosen),
        status_every=min(p.status_every for p in chosen),
        batch=min(p.batch for p in chosen),
    )


@dataclass(eq=False)
class _Follower:
    """One follower of a run: a bounded buffer plus a wake-up signal."""

    until_terminal: bool
    limit: int
    buffer: deque[StoredEvent] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    end_reason: FollowEndReason | None = None
    params: _Params = _DEFAULT_PARAMS

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


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """What a run looked like when a chunk read started (for after-read decisions)."""

    dirty: int
    hints: int
    reason: str
    status_terminal: bool
    first: bool
    healthy: bool
    generation: int


class _WatchedRun:
    """One watched run: shared head, followers, schedule and shared redaction reads.

    The head is the highest sequence read from the database, never a caller's
    cursor. Every follower here passed the run's tenant check, so one scoped
    redaction read serves them all.
    """

    def __init__(self, coordinator: _Coordinator, run_id: str, head: int | None, order: int):
        self.coordinator = coordinator
        self.store = coordinator.store
        self.run_id = run_id
        self.head = head
        self.order = order
        self.followers: set[_Follower] = set()
        self.params = _DEFAULT_PARAMS
        self.terminal = self.status_terminal = self.drain = False
        self.events_due = self.status_due = self.requested = _NEVER
        self.status_with_next = False
        self.reason = "join"
        self.dirty = self.hints = self.reads = 0
        self.delay = self.params.poll_interval_s
        self.events_fail_delay = self.status_fail_delay = 0.0
        self.missed_logged = False
        self._view: PauseView | None = None
        self._view_generation = 0
        self._view_lock = asyncio.Lock()
        self._unknown: tuple[int, frozenset[str] | None] | None = None
        self._declared: tuple[int, frozenset[str] | None] | None = None
        self._tools: tuple[int, ToolEvidence] | None = None

    @property
    def active(self) -> bool:
        """Still read: it has followers and is not finished for all of them."""
        if not self.followers:
            return False
        return not (self.terminal and all(f.until_terminal for f in self.followers))

    def due_at(self) -> float:
        return min(self.events_due, self.status_due) if self.active else _NEVER

    def add(self, follower: _Follower) -> int | None:
        """Register ``follower``; return the head it must catch up to from the database."""
        self.followers.add(follower)
        self.params = _tightest(f.params for f in self.followers)
        if self.terminal and follower.until_terminal:
            follower.end("terminal")
        self.coordinator.request(self, "join")
        return self.head

    def remove(self, follower: _Follower) -> None:
        self.followers.discard(follower)
        if self.followers:
            self.params = _tightest(f.params for f in self.followers)
        else:
            self.coordinator.unwatch(self)

    def publish(self, events: Sequence[StoredEvent]) -> None:
        for event in events:
            self._observe(event)
            for follower in list(self.followers):
                follower.push(event)
            self.head = event.sequence_index

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

    def _through(self, seq: int) -> int:
        return max(seq, self.head if self.head is not None else -1)

    async def pause_view(self, seq: int, tenant_id: TenantScope) -> PauseView:
        """The run's active pause for viewing event ``seq``, read once for every follower.

        A cached view answers for events up to its ``through`` index; a newer
        pause-changing event (published by the coordinator, or met by a follower
        ahead of it) triggers one fresh read.
        """
        view = self._view
        if view is not None and seq <= view.through:
            return view
        async with self._view_lock:
            view = self._view
            if view is not None and seq <= view.through:
                return view
            generation = self._view_generation
            fresh = await read_pause_view(self.store, self.run_id, tenant_id, self._through(seq))
            if generation == self._view_generation:
                self._view = fresh
            return fresh

    async def unknown_calls(self, seq: int, tenant_id: TenantScope) -> frozenset[str] | None:
        """The run's calls rejected as unknown tools, read once per newer failed call."""
        cached = self._unknown
        if cached is not None and seq <= cached[0]:
            return cached[1]
        through = self._through(seq)
        ids = await read_unknown_calls(self.store, self.run_id, tenant_id)
        self._unknown = (through, ids)
        return ids

    async def tool_evidence(self, seq: int, tenant_id: TenantScope) -> ToolEvidence:
        """The run's tool records, read once and again only after a newer
        ``tools.declared`` / ``mcp.connected`` event (published before later events)."""
        cached = self._tools
        if cached is not None:
            return cached[1]
        through = self._through(seq)
        evidence = await read_tool_evidence(self.store, self.run_id, tenant_id)
        self._tools = (through, evidence)
        return evidence

    async def declared_tools(self, seq: int, tenant_id: TenantScope) -> frozenset[str] | None:
        """Tool names the run recorded for declared calls, read once per newer event."""
        cached = self._declared
        if cached is not None and seq <= cached[0]:
            return cached[1]
        through = self._through(seq)
        names = await read_declared_tools(self.store, self.run_id, tenant_id)
        self._declared = (through, names)
        return names


class _Coordinator:
    """One reader-owned poll task per ``RunStore`` and event loop (spec §4.2).

    Also the live-updates hub consumer of that ``RunStore``: ``on_hints`` and
    ``on_gap`` only mark runs due and wake the task; a hint never adds a run.
    """

    def __init__(self, store: RunStore, live: LiveUpdates | None) -> None:
        self.store = store
        self.live = live
        self.runs: dict[str, _WatchedRun] = {}
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._crashed = False
        self._closed = False
        self._restarts: deque[float] = deque()
        self._order = itertools.count()
        self._tokens = MAX_READS_PER_S
        self._tokens_at = self._now()
        self._candidates: dict[str, tuple[float, int, _WatchedRun]] = {}
        self._registration: object | None = None
        if live is not None:
            try:
                self._registration = live.register(self, stream_events=False)
            except (LiveUpdatesUnavailableError, RuntimeError):
                self.live = None  # closed or another loop: poll

    def _now(self) -> float:
        return self._loop.time()

    # ------------------------------------------------------------ hub consumer

    def on_hints(self, run_ids: Sequence[str]) -> None:
        for run_id in run_ids:
            self._candidates.pop(run_id, None)  # announced after all: not a missed hint
            run = self.runs.get(run_id)
            if run is not None:
                run.hints += 1
                self.request(run, "hint")

    def on_gap(self) -> None:
        now = self._now()
        self._candidates.clear()
        for run in self.runs.values():
            self.request(run, "gap", now + random.uniform(0, run.params.poll_interval_s))

    def _healthy(self) -> bool:
        return self.live is not None and self.live.healthy

    def _generation(self) -> int:
        return self.live.generation if self.live is not None else 0

    # ------------------------------------------------------------ watched runs

    async def watch(self, run_id: str) -> _WatchedRun:
        """The watched run for ``run_id``; a new one starts at the database head."""
        self._raise_if_closed()
        run = self.runs.get(run_id)
        if run is None:
            head = await _current_head(self.store, run_id)
            self._raise_if_closed()
            run = self.runs.get(run_id)
            if run is None:
                run = _WatchedRun(self, run_id, head, next(self._order))
                self.runs[run_id] = run
        return run

    def unwatch(self, run: _WatchedRun) -> None:
        if self.runs.get(run.run_id) is run:
            del self.runs[run.run_id]
            self._wake.set()

    def request(self, run: _WatchedRun, reason: str, at: float | None = None) -> None:
        """Make ``run`` due (events and status) no later than ``at`` (default now)."""
        when = self._now() if at is None else at
        run.events_due = min(run.events_due, when)
        if not run.status_terminal:
            run.status_due = min(run.status_due, when)
        run.requested = min(run.requested, when)
        run.dirty += 1
        run.reason = reason
        self._wake.set()
        if reason == "join" and not self._closed:
            self.revive()  # hints and gaps never start a task (e.g. during loop shutdown)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("This RunStore is closed.")

    # ------------------------------------------------------------ task lifecycle

    def revive(self) -> None:
        """Start the task if it is not running; a crashed task counts as a restart.

        Called by joins, hints and every follower before it waits, so an
        existing follower notices a dead coordinator at its next wait.
        """
        task = self._task
        if self._closed or (task is not None and not task.done()):
            return
        if self._crashed:
            if not self._restart_allowed():
                self._fail_all()
                return
            _codes.log_code(logger, _codes.FOLLOW_COORDINATOR_RESTART, attempt=len(self._restarts))
            now = self._now()
            for run in self.runs.values():
                self.request_due(run, now)
        if not self.runs:
            return
        self._crashed = False
        self._task = self._loop.create_task(self._main(), name="dendrux-follow-coordinator")
        self._task.add_done_callback(self._on_exit)

    @staticmethod
    def request_due(run: _WatchedRun, now: float) -> None:
        run.events_due = min(run.events_due, now)
        if not run.status_terminal:
            run.status_due = min(run.status_due, now)

    def _restart_allowed(self) -> bool:
        now = self._now()
        window = max((r.params.poll_max_s for r in self.runs.values()), default=DEFAULT_POLL_MAX_S)
        while self._restarts and now - self._restarts[0] > window * 10:
            self._restarts.popleft()
        if len(self._restarts) >= MAX_RESTARTS:
            return False
        self._restarts.append(now)
        return True

    def _fail_all(self) -> None:
        _codes.log_code(logger, _codes.FOLLOW_FEED_ERROR)
        self._crashed = False
        self._restarts.clear()
        for run in list(self.runs.values()):
            for follower in list(run.followers):
                follower.end("error")
        self.runs.clear()
        self._candidates.clear()

    def _on_exit(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or task.exception() is None:
            return
        self._crashed = True
        for run in self.runs.values():
            for follower in run.followers:
                follower.wake.set()

    async def aclose(self, timeout: float) -> None:
        """End every follower, stop the task (bounded) and leave the hub."""
        self._closed = True
        for run in list(self.runs.values()):
            for follower in list(run.followers):
                follower.end("error")
        self.runs.clear()
        self._candidates.clear()
        self._leave_hub()
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        _, pending = await asyncio.wait({task}, timeout=timeout)
        if pending:
            _codes.log_code(logger, _codes.FOLLOW_SHUTDOWN_PENDING)

    def _leave_hub(self) -> None:
        live, handle = self.live, self._registration
        self._registration = None
        if live is not None and handle is not None:
            with contextlib.suppress(RuntimeError, LiveUpdatesUnavailableError):
                live.unregister(handle)

    # ------------------------------------------------------------ main loop

    async def _main(self) -> None:
        while True:
            self._wake.clear()
            now = self._now()
            self._check_missed(now)
            if not self.runs and not self._candidates:
                if await self._sleep(IDLE_EXIT_S) or self.runs:
                    continue
                return
            chunk = self._due_chunk(now)
            if chunk:
                await self._cycle(chunk, now)
            else:
                await self._sleep(self._next_wake(now) - now)

    async def _sleep(self, seconds: float) -> bool:
        """Wait for a wake or ``seconds``; True when woken."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(max(seconds, 0.0)):
                await self._wake.wait()
                return True
        return False

    def _due_chunk(self, now: float) -> list[_WatchedRun]:
        due = [run for run in self.runs.values() if run.due_at() <= now]
        due.sort(key=lambda run: (run.due_at(), run.order))
        return due[:POLL_BATCH_RUNS]

    def _next_wake(self, now: float) -> float:
        times = [run.due_at() for run in self.runs.values()]
        times.extend(at + MISSED_HINT_GRACE_S for at, _, _ in self._candidates.values())
        return min([*times, now + IDLE_EXIT_S])

    async def _token(self) -> None:
        """Take one read from the shared bucket; wait (never spin) when it is empty."""
        while True:
            now = self._now()
            rate = MAX_READS_PER_S
            self._tokens = min(rate, self._tokens + (now - self._tokens_at) * rate)
            self._tokens_at = now
            if self._tokens >= 1:
                self._tokens -= 1
                return
            await asyncio.sleep((1 - self._tokens) / rate)

    def _snapshot(self, run: _WatchedRun) -> _Snapshot:
        return _Snapshot(
            dirty=run.dirty,
            hints=run.hints,
            reason=run.reason,
            status_terminal=run.status_terminal,
            first=run.reads == 0,
            healthy=self._healthy(),
            generation=self._generation(),
        )

    async def _cycle(self, chunk: list[_WatchedRun], now: float) -> None:
        snapshots = {run: self._snapshot(run) for run in chunk}
        events = [run for run in chunk if run.events_due <= now]
        statuses = [run for run in chunk if self._status_due(run, now, events)]
        for run in chunk:
            run.requested = _NEVER
        if events:
            await self._read_events(events, snapshots)
        if statuses:
            await self._read_statuses(statuses)
        for run in chunk:
            self._requeue_if_dirty(run, snapshots[run])

    @staticmethod
    def _status_due(run: _WatchedRun, now: float, events: list[_WatchedRun]) -> bool:
        if run.status_terminal or (run.status_fail_delay and run.status_due > now):
            return False  # terminal, or backing off after a failed status read
        due = run.status_due <= now or (run.status_with_next and run in events)
        if due:
            run.status_with_next = False
        return due

    def _requeue_if_dirty(self, run: _WatchedRun, snapshot: _Snapshot) -> None:
        """A hint or join that arrived during the read makes the run due again."""
        if run.dirty == snapshot.dirty or run.requested == _NEVER:
            return
        run.events_due = min(run.events_due, run.requested)
        if not run.status_terminal:
            run.status_due = min(run.status_due, run.requested)
        run.requested = _NEVER

    # ------------------------------------------------------------ events

    async def _read_events(
        self, runs: list[_WatchedRun], snapshots: dict[_WatchedRun, _Snapshot]
    ) -> None:
        limit = max(1, min(min(r.params.batch for r in runs), MAX_LIMIT_PER_RUN))
        limit = max(1, min(limit, MAX_ROWS_PER_READ // len(runs)))
        rows, failed = await self._fetch_events(runs, limit)
        if failed is not None:
            self._failed(failed[0], failed[1])
        for run in runs:
            if run.run_id in rows:
                self._after_events(run, rows[run.run_id], limit, snapshots[run])

    async def _fetch_events(
        self, runs: list[_WatchedRun], limit: int
    ) -> tuple[dict[str, list[StoredEvent]], tuple[list[_WatchedRun], BaseException] | None]:
        """Rows by run id for the runs read; failed runs (with the error) are absent."""
        batched = self.store._batched_events()
        if batched is None:
            return await self._per_run(runs, lambda run: self._run_events(run, limit))
        heads = {run.run_id: run.head if run.head is not None else -1 for run in runs}
        await self._token()
        try:
            found = await batched(heads, limit)
        except Exception as error:
            return {}, (runs, error)
        return {run.run_id: found.get(run.run_id, []) for run in runs}, None

    async def _run_events(self, run: _WatchedRun, limit: int) -> list[StoredEvent]:
        return await self.store.get_events(run.run_id, after_sequence_index=run.head, limit=limit)

    async def _per_run(
        self, runs: list[_WatchedRun], read: Any
    ) -> tuple[dict[str, Any], tuple[list[_WatchedRun], BaseException] | None]:
        """The custom-store fallback: one read per run, each charged to the budget."""
        found: dict[str, Any] = {}
        failed: list[_WatchedRun] = []
        error: BaseException | None = None
        for run in runs:
            await self._token()
            try:
                found[run.run_id] = await read(run)
            except Exception as exc:
                failed.append(run)
                error = exc
        return found, ((failed, error) if error is not None else None)

    def _after_events(
        self, run: _WatchedRun, events: list[StoredEvent], limit: int, snapshot: _Snapshot
    ) -> None:
        if self.runs.get(run.run_id) is not run:
            return
        run.reads += 1
        run.events_fail_delay = 0.0
        fresh = [e for e in events if run.head is None or e.sequence_index > run.head]
        self._diagnose(run, fresh, snapshot)
        run.publish(fresh)
        now = self._now()
        if len(events) >= limit:
            self._again(run, now, "full")
            return
        if any(e.event_type in _TERMINAL_EVENT_TYPES for e in fresh):
            run.drain = True
            self.request_due(run, now)
        if run.drain and self._drained(run, events, snapshot, now):
            return
        self._schedule(run, bool(fresh), now)

    def _drained(
        self, run: _WatchedRun, events: list[StoredEvent], snapshot: _Snapshot, now: float
    ) -> bool:
        """Keep draining until an empty page; finish once terminal status was seen first."""
        if events:
            self._again(run, now, "drain")
            return True
        run.drain = False
        if snapshot.status_terminal:
            self._finish(run, now)
            return True
        return False

    def _again(self, run: _WatchedRun, now: float, reason: str) -> None:
        """Due again now, behind runs already waiting (a hot run re-enters at the back)."""
        run.events_due = now
        run.order = next(self._order)
        run.reason = reason

    def _finish(self, run: _WatchedRun, now: float) -> None:
        run.terminal = True
        for follower in list(run.followers):
            if follower.until_terminal:
                follower.end("terminal")
        self._schedule(run, False, now)

    def _schedule(self, run: _WatchedRun, got_new: bool, now: float) -> None:
        params = run.params
        if self._healthy():
            run.delay = params.poll_interval_s
            jitter = random.uniform(1 - HEALTHY_JITTER, 1 + HEALTHY_JITTER)
            run.events_due = now + params.poll_healthy_s * jitter
            run.reason = "safety"
            run.status_with_next = True
            return
        if got_new:
            run.delay = params.poll_interval_s
        run.events_due = now + run.delay
        run.reason = "poll"
        if not got_new:
            run.delay = min(run.delay * 2, params.poll_max_s)
        run.status_with_next = run.reads % params.status_every == 0

    # ------------------------------------------------------------ statuses

    async def _read_statuses(self, runs: list[_WatchedRun]) -> None:
        found, failed = await self._fetch_statuses(runs)
        if failed is not None:
            self._failed(failed[0], failed[1], status=True)
        now = self._now()
        for run in runs:
            if failed is None or run not in failed[0]:
                self._after_status(run, found.get(run.run_id), now)

    async def _fetch_statuses(
        self, runs: list[_WatchedRun]
    ) -> tuple[dict[str, Any], tuple[list[_WatchedRun], BaseException] | None]:
        batched = self.store._batched_statuses()
        if batched is None:
            return await self._per_run(runs, lambda run: self.store._status(run.run_id))
        await self._token()
        try:
            return dict(await batched([run.run_id for run in runs])), None
        except Exception as error:
            return {}, (runs, error)

    def _after_status(self, run: _WatchedRun, status: str | None, now: float) -> None:
        """A missing run (deleted) counts as terminal; a failed read never does."""
        if self.runs.get(run.run_id) is not run:
            return
        run.status_fail_delay = 0.0
        if status is not None and status not in TERMINAL_STATUSES:
            run.status_due = now + STATUS_MAX_AGE_S
            return
        run.status_due = _NEVER
        if not run.status_terminal:
            run.status_terminal = run.drain = True
            run.events_due = min(run.events_due, now)

    # ------------------------------------------------------------ failures

    def _failed(
        self, runs: list[_WatchedRun], error: BaseException, *, status: bool = False
    ) -> None:
        """Isolate a failed chunk: its runs retry after ``poll_interval_s`` with backoff."""
        _codes.log_code(
            logger, _codes.FOLLOW_READ_FAILED, error_kind=_error_kind(error), count=len(runs)
        )
        now = self._now()
        for run in runs:
            if status:
                run.status_fail_delay = _backoff(run.status_fail_delay, run.params)
                run.status_due = now + run.status_fail_delay
            else:
                run.events_fail_delay = _backoff(run.events_fail_delay, run.params)
                run.events_due = now + run.events_fail_delay

    # ------------------------------------------------------------ missed hints

    def _diagnose(self, run: _WatchedRun, fresh: list[StoredEvent], snapshot: _Snapshot) -> None:
        """Record a healthy safety poll that found events no hint announced (a heuristic)."""
        if not fresh or snapshot.reason != "safety" or snapshot.first or not snapshot.healthy:
            return
        if run.missed_logged or run.run_id in self._candidates:
            return
        if self._generation() != snapshot.generation or run.hints != snapshot.hints:
            return
        self._candidates[run.run_id] = (self._now(), snapshot.generation, run)

    def _check_missed(self, now: float) -> None:
        """Log a candidate no hint followed within the grace period, once per watched run."""
        for run_id, (seen_at, generation, run) in list(self._candidates.items()):
            if self._generation() != generation:
                del self._candidates[run_id]
            elif now >= seen_at + MISSED_HINT_GRACE_S:
                del self._candidates[run_id]
                run.missed_logged = True
                _codes.log_code(
                    logger,
                    _codes.LIVE_UPDATES_MISSED_HINT,
                    run_id=_log_run_id(run.run_id),
                    count=1,
                )


def _backoff(delay: float, params: _Params) -> float:
    """The next failure delay: ``poll_interval_s``, doubling up to ``poll_max_s``."""
    return min(max(delay * 2, params.poll_interval_s), params.poll_max_s)


class _RunWaker:
    """A ``stream_events`` hub consumer: one coalescing event for one run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.event = asyncio.Event()

    def on_hints(self, run_ids: Sequence[str]) -> None:
        if self.run_id in run_ids:
            self.event.set()

    def on_gap(self) -> None:
        self.event.set()


class EventFeed:
    """Followers of many runs with a follower cap, over one ``RunStore``.

    Create one per process (or per router) and call :meth:`follow`.
    ``RunStore.follow`` uses one owned by the store. Every ``EventFeed`` on the
    same ``RunStore`` shares that store's coordinator (one read schedule per
    store and event loop). Not thread-safe: use it from one event loop.

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
        poll_interval_s / poll_max_s: without healthy live updates, an idle
            run is read every ``poll_interval_s``, doubling up to ``poll_max_s``;
            a new event resets it.
        poll_healthy_s: the safety poll while live updates are healthy
            (±20 % jitter).
        status_every: without healthy live updates, a status read rides with
            every Nth events read.
        batch: most events per run in one read (the coordinator may read
            fewer to keep one query under 2,000 rows).
        stop_timeout_s: how long ``RunStore.close()`` waits for the coordinator.
        heartbeat_s: idle time before a :class:`FollowHeartbeat` is yielded.
        buffer: events buffered per follower before it ends as ``"lagging"``.

    A run followed through several feeds uses the tightest of their settings.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        max_followers: int = DEFAULT_MAX_FOLLOWERS,
        max_followers_per_tenant: int | None | _UnsetType = _DEFAULT_SHARE,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        poll_max_s: float = DEFAULT_POLL_MAX_S,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
        buffer: int = DEFAULT_BUFFER,
        status_every: int = DEFAULT_STATUS_EVERY,
        batch: int = DEFAULT_BATCH,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        poll_healthy_s: float = DEFAULT_POLL_HEALTHY_S,
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
        self._params = _Params(poll_interval_s, poll_max_s, poll_healthy_s, status_every, batch)
        self._closed = False
        self._open = 0
        self._by_scope: dict[tuple[str, ...], int] = {}
        self._attached: dict[_Follower, _WatchedRun] = {}
        self._watching: dict[_WatchedRun, int] = {}  # followers per watched run

    @property
    def followers(self) -> int:
        """Followers currently holding a slot."""
        return self._open

    @property
    def watched_runs(self) -> int:
        """Runs this feed's followers watch (each read once, however many followers)."""
        return len(self._watching)

    def follow(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        after_sequence_index: int | None = None,
        include_payloads: bool = False,
        until_terminal: bool = True,
        max_duration_s: float | None = DEFAULT_MAX_DURATION_S,
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
        if self._open >= self.max_followers:
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

    def _coordinator(self) -> _Coordinator:
        return self.store._follow_coordinator()

    def _attach(self, run: _WatchedRun, follower: _Follower) -> int | None:
        self._attached[follower] = run
        self._watching[run] = self._watching.get(run, 0) + 1
        return run.add(follower)

    def _detach(self, follower: _Follower) -> None:
        run = self._attached.pop(follower, None)
        if run is None:
            return
        left = self._watching.pop(run) - 1
        if left:
            self._watching[run] = left
        run.remove(follower)

    async def aclose(self) -> None:
        """End this feed's followers and stop watching their runs. Idempotent.

        The shared coordinator belongs to the ``RunStore``; ``RunStore.close()``
        stops it.
        """
        self._closed = True
        for follower in list(self._attached):
            follower.end("error")
            self._detach(follower)


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
        self._feed: _WatchedRun | None = None
        self._follower: _Follower | None = None
        self._head_at_start: int | None = None
        self._deadline: float | None = None
        self._gen: AsyncIterator[FollowItem] | None = None
        self.drained_terminal = False
        """True after :meth:`start` when the run is terminal and nothing is left
        after the cursor (an HTTP route answers 204 instead of streaming)."""

    async def start(self) -> None:
        """Check the scope, take a follower slot and join the run's watch; idempotent.

        Concurrent calls on one ``Follow`` take one slot. A :meth:`aclose`
        that lands while a start is waiting on the store wins: the start
        raises ``RuntimeError`` and takes no slot. Live updates that cannot
        be leased mean polling, never a failed start.

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
            coordinator = self._owner._coordinator()
            await self._owner.store._lease_live_updates()
            await self._check_run()
            self._raise_if_closed()  # closed while waiting on the store: take no slot
            # No await from here to the slot: aclose() cannot interleave.
            self._owner._acquire(self._tenant_id)
            self._holds_slot = True
            try:
                if not self.drained_terminal:
                    await self._join(coordinator)
            except BaseException:
                self._release()
                self._detach()
                raise
            self._started = True
            if self._max_duration_s is not None:
                self._deadline = asyncio.get_running_loop().time() + self._max_duration_s

    async def _join(self, coordinator: _Coordinator) -> None:
        run = await coordinator.watch(self.run_id)
        if self._closed:
            if not run.followers:
                coordinator.unwatch(run)  # never leave a run nobody follows
            self._raise_if_closed()
        follower = _Follower(
            until_terminal=self._until_terminal,
            limit=self._owner.buffer,
            params=self._owner._params,
        )
        self._feed, self._follower = run, follower
        self._head_at_start = self._owner._attach(run, follower)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("This follow is closed.")

    async def _check_run(self) -> None:
        """Scope check and terminal drain check (start checks are outside the budget)."""
        store = self._owner.store
        status = await store._status(self.run_id, self._tenant_id)
        if status is None:
            raise RunNotFoundError(self.run_id)
        if status in TERMINAL_STATUSES and self._until_terminal:
            pending = await store.get_events(
                self.run_id, after_sequence_index=self._cursor, limit=1
            )
            self.drained_terminal = not pending

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

    def _detach(self) -> None:
        follower, self._feed, self._follower = self._follower, None, None
        if follower is not None:
            self._owner._detach(follower)

    def __del__(self) -> None:
        self._release()
        with contextlib.suppress(RuntimeError):  # collected after its event loop closed
            self._detach()

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
            self._detach()

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
        try:
            if self._owner._closed:
                raise RuntimeError("This event feed is closed.")
            if self.drained_terminal:
                yield FollowEnd("terminal", self._cursor)
                return
            async for item in self._catch_up():
                yield item
                if isinstance(item, FollowEnd):
                    return
            async for item in self._live():
                yield item
        except TimeoutError:
            yield FollowEnd("max_duration", self._cursor)
        finally:
            self._release()
            self._detach()

    async def _catch_up(self) -> AsyncIterator[FollowItem]:
        """Events up to the head at join, read from the database (outside the budget)."""
        head = self._head_at_start
        while head is not None and (self._cursor is None or self._cursor < head):
            async with asyncio.timeout_at(self._deadline):
                batch = await self._owner.store.get_events(
                    self.run_id, after_sequence_index=self._cursor, limit=self._owner.batch
                )
            for event in batch:
                if event.sequence_index > head:
                    return
                item = await self._deliver(event)
                yield item
                if isinstance(item, FollowEnd):
                    return
            if len(batch) < self._owner.batch or not batch or batch[-1].sequence_index >= head:
                return

    async def _deliver(self, event: StoredEvent) -> FollowItem:
        if self._expired():
            return FollowEnd("max_duration", self._cursor)
        async with asyncio.timeout_at(self._deadline):
            view = await self._view(event)
        self._cursor = event.sequence_index
        return view

    async def _live(self) -> AsyncIterator[FollowItem]:
        assert self._follower is not None and self._feed is not None
        follower, coordinator = self._follower, self._feed.coordinator
        loop = asyncio.get_running_loop()
        last_write = loop.time()
        while True:
            item = await self._next_live(follower, coordinator, last_write)
            if item is None:
                continue
            yield item
            if isinstance(item, FollowEnd):
                return
            last_write = loop.time()

    async def _next_live(
        self, follower: _Follower, coordinator: _Coordinator, last_write: float
    ) -> FollowItem | None:
        """The next item for this follower, or None after a wait or a duplicate."""
        if self._expired():
            return FollowEnd("max_duration", self._cursor)
        if follower.buffer:
            event = follower.buffer.popleft()
            if self._cursor is None or event.sequence_index > self._cursor:
                return await self._deliver(event)
            return None
        if follower.end_reason is not None:
            return FollowEnd(follower.end_reason, self._cursor)
        if asyncio.get_running_loop().time() - last_write >= self._owner.heartbeat_s:
            return FollowHeartbeat()
        coordinator.revive()  # a follower notices a dead coordinator at its next wait
        await self._wait(follower, last_write)
        return None

    async def _wait(self, follower: _Follower, last_write: float) -> None:
        now = asyncio.get_running_loop().time()
        timeout = self._owner.heartbeat_s - (now - last_write)
        if self._deadline is not None:
            timeout = min(timeout, self._deadline - now)
        follower.wake.clear()
        if follower.buffer or follower.end_reason is not None:
            return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(max(timeout, 0.0)):
                await follower.wake.wait()


async def _feed_view(
    feed: _WatchedRun | None,
    event: StoredEvent,
    *,
    tenant_id: TenantScope,
    include_payloads: bool,
    past: PastParams,
) -> StoredEvent:
    """A follower's view of ``event``, using the watched run's shared reads.

    Without a watched run (never the case once events flow) nothing is known
    about the run, so pause arguments and failed calls' names stay hidden.
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
        feed.store,
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

    A new watch starts here, never at a follower's cursor, so one follower's
    bogus cursor cannot hide events from the others.
    """
    state = getattr(store, "_state", None)
    next_index = getattr(state, "get_next_event_sequence", None)
    if next_index is not None:
        value = int(await next_index(run_id))
        return value - 1 if value > 0 else None
    head: int | None = None
    while True:  # custom store fallback: page to the end
        batch = await store.get_events(run_id, after_sequence_index=head, limit=DEFAULT_BATCH)
        if not batch:
            return head
        head = batch[-1].sequence_index
