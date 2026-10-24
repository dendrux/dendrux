"""``make_read_router`` — read-only public HTTP surface over ``RunStore``.

The factory returns a FastAPI ``APIRouter`` the developer mounts into
their own app. The caller-supplied ``authorize`` dependency runs on
every route except ``/health`` (intentionally unauth'd so Kubernetes
liveness/readiness probes can reach it without secrets). Auth failures
raise ``HTTPException`` from that dependency; this module does not
interpret the dependency's return value.

Every authenticated route is scoped by ``tenant_resolver``: a run outside
the caller's tenant reads exactly like a missing one (the same 404 body).

Endpoints:

    GET /health
    GET /runs
    GET /runs/{run_id}
    GET /runs/{run_id}/events
    GET /runs/{run_id}/events/stream   (Server-Sent Events)
    GET /runs/{run_id}/llm-calls
    GET /runs/{run_id}/tool-calls
    GET /runs/{run_id}/traces
    GET /runs/{run_id}/pauses

SSE and database load
---------------------
Clients watching the same run share one polling query per router (per
process) and fan out in memory, so database load grows with the number of
*watched runs*, not open connections. An idle run is polled every 0.5 s,
backing off to 2 s; budget roughly one query per second per watched run,
plus a periodic status read. When many viewers are served, give the router
its own small engine (``RunStore.from_database_url(...)`` with a modest
pool) so viewers cannot starve the agents' connection pool. Push
(Postgres LISTEN/NOTIFY) is planned for 0.6.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime  # noqa: TC003 — FastAPI resolves query-param types at runtime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from dendrux.runtime.state import MAX_LIST_OFFSET, UNSCOPED, _Unscoped

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from starlette.types import Receive, Scope, Send

    from dendrux.runtime.state import TenantScope
    from dendrux.store import RunStore, StoredEvent

logger = logging.getLogger(__name__)

_DRILL_DOWN_LIMIT_DEFAULT = 100
_DRILL_DOWN_LIMIT_MAX = 1000
_LLM_CALLS_LIMIT_DEFAULT = 50
_LLM_CALLS_LIMIT_MAX = 200

# How often the SSE endpoint emits a keepalive comment when no events
# arrive. Overridable in tests via monkeypatch. Nginx's default
# `proxy_read_timeout` is 60s, so anything well under that works.
_SSE_HEARTBEAT_INTERVAL_S = 15.0

# Idle polling of a watched run starts at _SSE_POLL_INTERVAL_S and doubles up
# to _SSE_POLL_MAX_S; any new event resets it. Overridable in tests.
_SSE_POLL_INTERVAL_S = 0.5
_SSE_POLL_MAX_S = 2.0
# While idle, the run's status is read on every Nth poll (and right after a
# terminal-looking event), so a run finalized without an event still ends.
_SSE_STATUS_EVERY = 4
_SSE_BATCH = 100
# Events buffered for one slow client before it is closed with a cursor
# (it reconnects with Last-Event-ID and catches up from the database).
_SSE_CLIENT_BUFFER = 1000
# Longest a closing stream waits for its cancelled poller to finish.
_SSE_POLLER_STOP_S = 5.0

_TERMINAL_STATUSES = frozenset({"success", "error", "cancelled", "max_iterations"})
_TERMINAL_EVENT_TYPES = frozenset({"run.completed", "run.error", "run.cancelled"})

# Event-data keys hidden unless include_payloads=True.
_HIDDEN_EVENT_KEYS = frozenset(
    {"system_prompt", "provider_request", "provider_response", "pii_mapping"}
)


@dataclass(frozen=True)
class _ReadRouterConfig:
    """Validated ``make_read_router`` options shared by the route handlers."""

    tenant_dependency: Callable[..., Any]
    include_payloads: bool
    sse_max_connections: int
    sse_max_duration_s: float


def _unscoped_tenant() -> _Unscoped:
    return UNSCOPED


def _tenant_dependency(tenant_resolver: Callable[..., Any] | _Unscoped) -> Callable[..., Any]:
    """The FastAPI dependency resolving the caller's tenant scope.

    The ``UNSCOPED`` sentinel itself stands for a constant all-tenant resolver.
    A generator dependency (``yield``) is rejected: the resolved value must be
    a plain return value.
    """
    if isinstance(tenant_resolver, _Unscoped):
        return _unscoped_tenant
    if not callable(tenant_resolver):
        raise TypeError(
            "tenant_resolver must be a FastAPI dependency callable or UNSCOPED, "
            f"got {type(tenant_resolver).__name__}."
        )
    target = getattr(tenant_resolver, "__call__", tenant_resolver)  # noqa: B004
    if any(
        inspect.isgeneratorfunction(f) or inspect.isasyncgenfunction(f)
        for f in (tenant_resolver, target)
    ):
        raise TypeError(
            "tenant_resolver must return the tenant (a str, None or UNSCOPED); "
            "a dependency that yields is not supported."
        )
    return tenant_resolver


def _validated_scope(value: Any) -> TenantScope:
    """The resolver's value as a tenant scope; anything else fails closed with 500."""
    if value is None or isinstance(value, _Unscoped):
        return value
    if isinstance(value, str) and value:
        return value
    logger.error(
        "tenant_resolver returned an invalid value (type %s): code=invalid_tenant_scope",
        type(value).__name__,
        extra={"code": "invalid_tenant_scope"},
    )
    raise HTTPException(status_code=500, detail="Tenant scope could not be resolved.")


def _require_aware(name: str, value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must include a timezone offset (RFC 3339), e.g. 2026-01-01T00:00:00Z.",
        )
    return value


def _not_found(run_id: str) -> HTTPException:
    # Identical for a missing run and a run outside the caller's tenant.
    return HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")


def make_read_router(
    *,
    store: RunStore,
    authorize: Callable[..., Any],
    tenant_resolver: Callable[..., Any] | _Unscoped,
    include_payloads: bool = False,
    sse_max_connections: int = 500,
    sse_max_duration_s: float = 900,
) -> APIRouter:
    """Build a mountable read-only router backed by ``store``.

    Args:
        store: The :class:`dendrux.store.RunStore` instance this router reads from.
        authorize: A FastAPI dependency callable. Runs on every authenticated
            route. Raise ``HTTPException`` inside to deny; return any value
            (ignored) to allow. ``/health`` bypasses this dependency.
        tenant_resolver: Required. A FastAPI dependency (it may take
            ``Request`` or other dependencies) that **returns** the caller's
            tenant: a non-empty ``str`` (that tenant's runs), ``None``
            (untenanted runs only) or ``UNSCOPED`` (every tenant: an
            all-tenant reader, for admins only). Pass the ``UNSCOPED``
            sentinel itself for single-tenant apps. It must return a plain
            value, not ``yield``. Any other value fails the request with 500.
            ``?tenant_id=`` on ``/runs`` narrows the list only for an
            ``UNSCOPED`` caller; for a scoped caller a different tenant
            matches nothing.
        include_payloads: Include raw payloads in responses: provider
            requests and responses, tool arguments and results, trace
            content, the system prompt in ``run.started``, tool arguments in
            ``run.paused`` and pause listings. Off by default; turn it on only
            for trusted internal viewers.
        sse_max_connections: Most concurrent SSE connections this router
            serves (per process). Further requests get 503.
        sse_max_duration_s: Longest one SSE connection stays open, in
            seconds. It then ends with an ``end`` frame carrying the cursor;
            the client reconnects (and is re-authorized) with ``Last-Event-ID``.

    Returns:
        An ``APIRouter`` suitable for ``app.include_router(...)``. The
        caller owns routing prefixes, CORS, TLS, and deployment.
    """
    if not isinstance(include_payloads, bool):
        raise TypeError("include_payloads must be a bool.")
    if (
        isinstance(sse_max_connections, bool)
        or not isinstance(sse_max_connections, int)
        or sse_max_connections <= 0
    ):
        raise ValueError("sse_max_connections must be a positive int.")
    if (
        isinstance(sse_max_duration_s, bool)
        or not isinstance(sse_max_duration_s, (int, float))
        or not 0 < sse_max_duration_s < float("inf")
    ):
        raise ValueError("sse_max_duration_s must be a positive number of seconds.")
    config = _ReadRouterConfig(
        tenant_dependency=_tenant_dependency(tenant_resolver),
        include_payloads=include_payloads,
        sse_max_connections=sse_max_connections,
        sse_max_duration_s=float(sse_max_duration_s),
    )
    hub = _SseHub(store, config)
    resolver = config.tenant_dependency
    payloads = config.include_payloads

    async def resolve_scope(value: Any = Depends(resolver)) -> Any:  # noqa: B008
        return _validated_scope(value)

    router = APIRouter()

    # /health bypasses auth so external probes (Kubernetes, load balancers)
    # can reach it without credentials.
    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # All other routes share the auth dependency at router level — keeps
    # endpoint signatures clean and guarantees no route accidentally skips it.
    auth_router = APIRouter(dependencies=[Depends(authorize)])

    @auth_router.get("/runs")
    async def list_runs(
        status: Annotated[list[str] | None, Query()] = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        tenant_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        offset: Annotated[int, Query(ge=0, le=MAX_LIST_OFFSET)] = 0,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> dict[str, Any]:
        _require_aware("started_after", started_after)
        _require_aware("started_before", started_before)
        if tenant_id == "":
            raise HTTPException(status_code=422, detail="tenant_id must not be empty.")
        effective: TenantScope = scope
        if isinstance(scope, _Unscoped):
            if tenant_id is not None:
                effective = tenant_id
        elif tenant_id is not None and tenant_id != scope:
            # A scoped caller asked for another tenant: nothing matches.
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
        filters: dict[str, Any] = {
            "status": status,
            "agent_name": agent_name,
            "parent_run_id": parent_run_id,
            "tenant_id": effective,
            "started_after": started_after,
            "started_before": started_before,
        }
        records = await store.list_runs(**filters, limit=limit, offset=offset)
        total = await store.count_runs(**filters)
        return {
            "items": records,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @auth_router.get("/runs/{run_id}")
    async def get_run(run_id: str, scope: Any = Depends(resolve_scope)) -> Any:  # noqa: B008
        detail = await store.get_run(run_id, tenant_id=scope)
        if detail is None:
            raise _not_found(run_id)
        return detail

    @auth_router.get("/runs/{run_id}/events")
    async def get_events(
        run_id: str,
        after: int | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> dict[str, Any]:
        # 404 fast on missing run rather than silently returning an empty page.
        if await store.get_run(run_id, tenant_id=scope) is None:
            raise _not_found(run_id)
        events = await store.get_events(
            run_id, after_sequence_index=after, limit=limit, tenant_id=scope
        )
        # Echo the client's cursor when the batch is empty — an empty
        # response means "nothing new yet," not "stream is over." A poller
        # that reset to null after an empty batch would miss future events.
        next_cursor = events[-1].sequence_index if events else after
        return {"items": [_event_view(e, payloads) for e in events], "next_cursor": next_cursor}

    @auth_router.get("/runs/{run_id}/events/stream")
    async def stream_events(
        run_id: str,
        request: Request,
        after: int | None = None,
        follow: bool = False,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> Response:
        """Server-Sent Events stream of new run events.

        Cursor resolution (first match wins):
          1. ``Last-Event-ID`` header (SSE reconnect convention)
          2. ``?after=N`` query param
          3. ``None`` (stream from the beginning of the log)

        The stream ends with an ``event: end`` frame once the run's status
        is terminal and every event has been sent; reconnecting to such a
        run gets HTTP 204. ``?follow=true`` keeps the stream open after
        that (the 0.4 behaviour). It also ends after ``sse_max_duration_s``
        or when this client falls too far behind, with the cursor in the
        ``end`` frame. A ``: keepalive`` comment is sent every
        ``_SSE_HEARTBEAT_INTERVAL_S`` seconds of idleness.
        """
        run = await store.get_run(run_id, tenant_id=scope)
        if run is None:
            raise _not_found(run_id)

        cursor = _resolve_sse_cursor(request.headers.get("last-event-id"), after)
        if not follow and run.status in _TERMINAL_STATUSES:
            pending = await store.get_events(
                run_id, after_sequence_index=cursor, limit=1, tenant_id=scope
            )
            if not pending:
                return Response(status_code=204)

        if not hub.try_acquire():
            logger.warning(
                "SSE connection limit reached: run_id=%s code=sse_connection_limit limit=%d",
                run_id,
                config.sse_max_connections,
                extra={"run_id": run_id, "code": "sse_connection_limit"},
            )
            raise HTTPException(
                status_code=503,
                detail="Too many open event streams. Retry shortly.",
                headers={"Retry-After": "5"},
            )
        return _SseResponse(
            hub.stream(run_id, cursor, follow=follow),
            release=hub.release,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    @auth_router.get("/runs/{run_id}/llm-calls")
    async def get_llm_calls(
        run_id: str,
        iteration: int | None = None,
        limit: Annotated[int, Query(ge=1, le=_LLM_CALLS_LIMIT_MAX)] = _LLM_CALLS_LIMIT_DEFAULT,
        offset: Annotated[int, Query(ge=0, le=MAX_LIST_OFFSET)] = 0,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> dict[str, Any]:
        if await store.get_run(run_id, tenant_id=scope) is None:
            raise _not_found(run_id)
        items = await store.get_llm_calls(
            run_id, iteration=iteration, limit=limit, offset=offset, tenant_id=scope
        )
        if not payloads:
            items = [
                dataclasses.replace(i, provider_request=None, provider_response=None) for i in items
            ]
        return {"items": items, "limit": limit, "offset": offset}

    @auth_router.get("/runs/{run_id}/tool-calls")
    async def get_tool_calls(
        run_id: str,
        iteration: int | None = None,
        limit: Annotated[int, Query(ge=1, le=_DRILL_DOWN_LIMIT_MAX)] = _DRILL_DOWN_LIMIT_DEFAULT,
        offset: Annotated[int, Query(ge=0, le=MAX_LIST_OFFSET)] = 0,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> dict[str, Any]:
        if await store.get_run(run_id, tenant_id=scope) is None:
            raise _not_found(run_id)
        items = await store.get_tool_invocations(
            run_id, iteration=iteration, limit=limit, offset=offset, tenant_id=scope
        )
        if not payloads:
            items = [dataclasses.replace(i, params=None, result=None) for i in items]
        return {"items": items, "limit": limit, "offset": offset}

    @auth_router.get("/runs/{run_id}/traces")
    async def get_traces(
        run_id: str,
        limit: Annotated[int, Query(ge=1, le=_DRILL_DOWN_LIMIT_MAX)] = _DRILL_DOWN_LIMIT_DEFAULT,
        offset: Annotated[int, Query(ge=0, le=MAX_LIST_OFFSET)] = 0,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> dict[str, Any]:
        if await store.get_run(run_id, tenant_id=scope) is None:
            raise _not_found(run_id)
        items: list[Any] = await store.get_traces(
            run_id, limit=limit, offset=offset, tenant_id=scope
        )
        if not payloads:
            # Raw conversation content is a payload: keep the shape, drop the text.
            items = [{**dataclasses.asdict(i), "content": None, "meta": None} for i in items]
        return {"items": items, "limit": limit, "offset": offset}

    @auth_router.get("/runs/{run_id}/pauses")
    async def get_pauses(run_id: str, scope: Any = Depends(resolve_scope)) -> dict[str, Any]:  # noqa: B008
        if await store.get_run(run_id, tenant_id=scope) is None:
            raise _not_found(run_id)
        items = await store.get_pauses(run_id, tenant_id=scope)
        if not payloads:
            items = [
                dataclasses.replace(
                    p, pending_tool_calls=[_without_params(c) for c in p.pending_tool_calls]
                )
                for p in items
            ]
        return {"items": items}

    router.include_router(auth_router)
    return router


# ---------------------------------------------------------------------------
# Payload redaction (include_payloads=False)
# ---------------------------------------------------------------------------


def _without_params(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    return {k: v for k, v in call.items() if k not in ("params", "arguments")}


def _redact_event_data(event_type: str, data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Event data without raw payloads: the system prompt, provider bodies, tool arguments."""
    if not data:
        return data
    out = {k: v for k, v in data.items() if k not in _HIDDEN_EVENT_KEYS}
    if event_type == "run.paused" and isinstance(out.get("pending_tool_calls"), list):
        out["pending_tool_calls"] = [_without_params(c) for c in out["pending_tool_calls"]]
    return out


def _event_view(event: StoredEvent, include_payloads: bool) -> StoredEvent:
    if include_payloads:
        return event
    return dataclasses.replace(event, data=_redact_event_data(event.event_type, event.data))


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


def _resolve_sse_cursor(last_event_id: str | None, after: int | None) -> int | None:
    """Resolve an SSE cursor from the ``Last-Event-ID`` header and
    ``?after=N`` query param.

    Precedence: a valid integer in the header wins. A malformed header
    (non-integer) falls back to ``after`` silently — SSE clients and
    reverse proxies occasionally inject junk values on reconnect, and
    500ing on them would break reconnection flows.
    """
    if last_event_id:
        try:
            return int(last_event_id)
        except ValueError:
            pass
    return after


class _SseResponse(StreamingResponse):
    """A streaming response that releases its connection slot however it ends."""

    def __init__(self, content: AsyncIterator[str], *, release: Callable[[], None], **kw: Any):
        super().__init__(content, **kw)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


@dataclass(eq=False)
class _Subscriber:
    """One SSE client of a run feed: a bounded buffer plus a wake-up signal."""

    follow: bool
    buffer: deque[StoredEvent] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    end_reason: str | None = None

    def push(self, event: StoredEvent) -> bool:
        if len(self.buffer) >= _SSE_CLIENT_BUFFER:
            self.end("lagging")
            return False
        self.buffer.append(event)
        self.wake.set()
        return True

    def end(self, reason: str) -> None:
        if self.end_reason is None:
            self.end_reason = reason
        self.wake.set()


class _RunFeed:
    """The single poller of one run, shared by every SSE client watching it.

    It polls events after ``head`` (the last index it published) and pushes
    them to each subscriber. It stops when its last subscriber leaves, or
    once the run is terminal and drained if no subscriber follows.
    """

    def __init__(self, hub: _SseHub, run_id: str, head: int | None) -> None:
        self.hub = hub
        self.run_id = run_id
        self.head = head
        self.subscribers: set[_Subscriber] = set()
        self.terminal = False
        self.task: asyncio.Task[None] | None = None

    def add(self, sub: _Subscriber) -> int | None:
        """Register ``sub``; return the head it must catch up to from the database."""
        self.subscribers.add(sub)
        if self.terminal and not sub.follow:
            sub.end("terminal")
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(
                self._poll(), name=f"dendrux-sse-feed:{self.run_id}"
            )
        return self.head

    def remove(self, sub: _Subscriber) -> asyncio.Task[None] | None:
        """Unregister ``sub``; return the poller task to await when it was the last client."""
        self.subscribers.discard(sub)
        if self.subscribers:
            return None
        self.hub.drop(self)
        task, self.task = self.task, None
        if task is not None and not task.done():
            task.cancel()
            return task
        return None

    def _publish(self, events: list[StoredEvent]) -> None:
        for event in events:
            for sub in list(self.subscribers):
                sub.push(event)
            self.head = event.sequence_index

    def _end_non_followers(self) -> None:
        for sub in list(self.subscribers):
            if not sub.follow:
                sub.end("terminal")

    async def _fetch(self) -> list[StoredEvent]:
        return await self.hub.store.get_events(
            self.run_id, after_sequence_index=self.head, limit=_SSE_BATCH
        )

    async def _poll(self) -> None:
        delay = _SSE_POLL_INTERVAL_S
        idle_polls = 0
        check_status = True
        try:
            while self.subscribers:
                if self.terminal and not any(s.follow for s in self.subscribers):
                    return
                events = await self._fetch()
                self._publish(events)
                if len(events) == _SSE_BATCH:
                    continue  # more may be buffered; keep draining
                if events:
                    delay, idle_polls = _SSE_POLL_INTERVAL_S, 0
                    check_status = check_status or any(
                        e.event_type in _TERMINAL_EVENT_TYPES for e in events
                    )
                else:
                    idle_polls += 1
                    check_status = check_status or idle_polls % _SSE_STATUS_EVERY == 0
                if not self.terminal and check_status:
                    check_status = False
                    run = await self.hub.store.get_run(self.run_id)
                    if run is None or run.status in _TERMINAL_STATUSES:
                        # Events committed before the status flip: drain them first.
                        while True:
                            tail = await self._fetch()
                            self._publish(tail)
                            if len(tail) < _SSE_BATCH:
                                break
                        self.terminal = True
                        self._end_non_followers()
                        continue
                await asyncio.sleep(delay)
                if not events:
                    delay = min(delay * 2, _SSE_POLL_MAX_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "SSE feed failed: run_id=%s code=sse_feed_error",
                self.run_id,
                exc_info=True,
                extra={"run_id": self.run_id, "code": "sse_feed_error"},
            )
            for sub in list(self.subscribers):
                sub.end("error")


class _SseHub:
    """Per-router SSE state: the connection cap and one feed per watched run."""

    def __init__(self, store: RunStore, config: _ReadRouterConfig) -> None:
        self.store = store
        self.config = config
        self.open = 0
        self.feeds: dict[str, _RunFeed] = {}

    def try_acquire(self) -> bool:
        if self.open >= self.config.sse_max_connections:
            return False
        self.open += 1
        return True

    def release(self) -> None:
        self.open = max(0, self.open - 1)

    def drop(self, feed: _RunFeed) -> None:
        if self.feeds.get(feed.run_id) is feed:
            del self.feeds[feed.run_id]

    def stream(self, run_id: str, cursor: int | None, *, follow: bool) -> AsyncIterator[str]:
        return _sse_event_generator(
            self.store, run_id, cursor, config=self.config, hub=self, follow=follow
        )


async def _sse_event_generator(
    store: RunStore,
    run_id: str,
    cursor: int | None,
    *,
    config: _ReadRouterConfig | None = None,
    hub: _SseHub | None = None,
    follow: bool = False,
) -> AsyncIterator[str]:
    """Yield SSE frames for one client until the stream ends or the client leaves.

    The client registers with the run's shared feed, then reads the events
    between its cursor and the feed's head straight from the database (in
    batches), then takes live events from the feed. No transaction is held
    across a wait. Ends with an ``end`` frame (terminal and drained,
    ``max_duration``, ``lagging`` or ``error``).
    """
    if config is None:
        config = _ReadRouterConfig(
            tenant_dependency=_unscoped_tenant,
            include_payloads=True,
            sse_max_connections=1,
            sse_max_duration_s=float("inf"),
        )
    if hub is None:
        hub = _SseHub(store, config)
    feed = hub.feeds.get(run_id)
    if feed is None:
        head = await _current_head(store, run_id)
        # Another client may have created the feed while we read the head.
        feed = hub.feeds.get(run_id) or hub.feeds.setdefault(run_id, _RunFeed(hub, run_id, head))
    sub = _Subscriber(follow=follow)
    loop = asyncio.get_running_loop()
    started = last_write = loop.time()
    try:
        head = feed.add(sub)
        if head is not None and (cursor is None or cursor < head):
            while True:
                batch = await store.get_events(
                    run_id, after_sequence_index=cursor, limit=_SSE_BATCH
                )
                for ev in batch:
                    if ev.sequence_index > head:
                        break
                    yield _format_sse_frame(ev, config.include_payloads)
                    cursor = ev.sequence_index
                if len(batch) < _SSE_BATCH or cursor is None or cursor >= head:
                    break
            last_write = loop.time()

        while True:
            while sub.buffer:
                ev = sub.buffer.popleft()
                if cursor is None or ev.sequence_index > cursor:
                    yield _format_sse_frame(ev, config.include_payloads)
                    cursor = ev.sequence_index
                    last_write = loop.time()
            if sub.end_reason is not None:
                yield _format_end_frame(sub.end_reason, cursor)
                return
            now = loop.time()
            if now - started >= config.sse_max_duration_s:
                yield _format_end_frame("max_duration", cursor)
                return
            if now - last_write >= _SSE_HEARTBEAT_INTERVAL_S:
                yield ": keepalive\n\n"
                last_write = loop.time()
                continue
            timeout = min(
                _SSE_HEARTBEAT_INTERVAL_S - (now - last_write),
                config.sse_max_duration_s - (now - started),
            )
            sub.wake.clear()
            if sub.buffer or sub.end_reason is not None:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(sub.wake.wait(), timeout=max(timeout, 0.0))
    finally:
        poller = feed.remove(sub)
        if poller is not None:
            # Let the cancelled poller finish (and return its connection) before
            # the stream is reported closed: no task outlives its last client.
            await asyncio.wait({poller}, timeout=_SSE_POLLER_STOP_S)


async def _current_head(store: RunStore, run_id: str) -> int | None:
    """The run's highest event index now (None when it has none), from the database.

    A new feed starts here, never at a client-supplied cursor, so one
    client's bogus ``Last-Event-ID`` cannot hide events from the others.
    """
    state = getattr(store, "_state", None)
    next_index = getattr(state, "get_next_event_sequence", None)
    if next_index is not None:
        value = int(await next_index(run_id))
        return value - 1 if value > 0 else None
    head: int | None = None
    while True:  # custom store fallback: page to the end
        batch = await store.get_events(run_id, after_sequence_index=head, limit=_SSE_BATCH)
        if not batch:
            return head
        head = batch[-1].sequence_index


def _rfc3339(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _format_sse_frame(event: StoredEvent, include_payloads: bool = True) -> str:
    """Encode a StoredEvent as one SSE frame.

    Uses ``event: message`` (the SSE default) with ``event_type`` inside
    the JSON payload so generic clients that dispatch on the SSE event
    name don't have to pre-enroll every Dendrux event type. ``timestamp``
    is RFC 3339 in UTC (``...Z``).
    """
    data = event.data if include_payloads else _redact_event_data(event.event_type, event.data)
    payload = {
        "sequence_index": event.sequence_index,
        "iteration_index": event.iteration_index,
        "event_type": event.event_type,
        "correlation_id": event.correlation_id,
        "timestamp": _rfc3339(event.created_at),
        "data": data,
    }
    return (
        f"id: {event.sequence_index}\nevent: message\ndata: {json.dumps(payload, default=str)}\n\n"
    )


def _format_end_frame(reason: str, cursor: int | None) -> str:
    """The last frame of a stream: why it ended and the cursor to resume from."""
    return f"event: end\ndata: {json.dumps({'reason': reason, 'cursor': cursor})}\n\n"
