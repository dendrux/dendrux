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
import dataclasses
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime  # noqa: TC003 — FastAPI resolves query-param types at runtime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from dendrux.errors import RunNotFoundError
from dendrux.runtime.state import MAX_LIST_OFFSET, UNSCOPED, _Unscoped
from dendrux.store._follow import (
    EventFeed,
    Follow,
    FollowCapacityError,
    FollowEnd,
    FollowHeartbeat,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

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

# EventFeed tuning for the router's feed (read when the router's feed is
# first used). Overridable in tests. See ``dendrux.store._follow``.
_SSE_POLL_INTERVAL_S = 0.5
_SSE_POLL_MAX_S = 2.0
_SSE_STATUS_EVERY = 4
_SSE_CLIENT_BUFFER = 1000


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
    feed_holder: list[EventFeed] = []

    def feed() -> EventFeed:
        # Built on first use so the module tuning knobs can be patched in tests.
        if not feed_holder:
            feed_holder.append(_make_feed(store, config.sse_max_connections))
        return feed_holder[0]

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
        active = (
            await store._active_pause_sequence(run_id, scope)
            if any(e.event_type == "run.paused" for e in events)
            else None
        )
        items = [
            await store._event_view(
                run_id,
                e,
                tenant_id=scope,
                include_payloads=payloads,
                active_pause=e.sequence_index == active,
            )
            for e in events
        ]
        return {"items": items, "next_cursor": next_cursor}

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
        cursor = _resolve_sse_cursor(request.headers.get("last-event-id"), after)
        stream = feed().follow(
            run_id,
            tenant_id=scope,
            after_sequence_index=cursor,
            include_payloads=payloads,
            until_terminal=not follow,
            max_duration_s=config.sse_max_duration_s,
        )
        try:
            await stream.start()
        except RunNotFoundError:
            raise _not_found(run_id) from None
        except FollowCapacityError:
            raise HTTPException(
                status_code=503,
                detail="Too many open event streams. Retry shortly.",
                headers={"Retry-After": "5"},
            ) from None
        if stream.drained_terminal:
            await stream.aclose()
            return Response(status_code=204)
        return _SseResponse(
            _sse_frames(stream),
            close=stream.aclose,
            deadline=stream._deadline,
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
    async def get_pauses(
        run_id: str,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        scope: Any = Depends(resolve_scope),  # noqa: B008
    ) -> dict[str, Any]:
        if await store._status(run_id, scope) is None:
            raise _not_found(run_id)
        items, truncated = await store.get_pause_page(
            run_id, tenant_id=scope, limit=limit, include_payloads=payloads
        )
        return {"items": items, "limit": limit, "truncated": truncated}

    router.include_router(auth_router)
    return router


# ---------------------------------------------------------------------------
# Helpers over the store primitives
# ---------------------------------------------------------------------------


def _make_feed(store: RunStore, max_followers: int) -> EventFeed:
    return EventFeed(
        store,
        max_followers=max_followers,
        poll_interval_s=_SSE_POLL_INTERVAL_S,
        poll_max_s=_SSE_POLL_MAX_S,
        heartbeat_s=_SSE_HEARTBEAT_INTERVAL_S,
        buffer=_SSE_CLIENT_BUFFER,
        status_every=_SSE_STATUS_EVERY,
    )


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
    """A streaming response that closes its follow (freeing the slot) however it ends."""

    def __init__(
        self,
        content: AsyncIterator[str],
        *,
        close: Callable[[], Awaitable[None]],
        deadline: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(content, **kw)
        self._close = close
        self._deadline = deadline

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def bounded_send(message: Any) -> None:
            # A stalled transport cannot hold the connection beyond its deadline.
            # Give the end frame a small, bounded flush interval.
            timeout = None
            if self._deadline is not None:
                body = message.get("body", b"")
                end_frame = message["type"] == "http.response.body" and (
                    not body or body.startswith(b"event: end\n")
                )
                timeout = max(
                    0.05 if end_frame else 0.0,
                    self._deadline - asyncio.get_running_loop().time(),
                )
            await asyncio.wait_for(send(message), timeout)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            await self._close()


async def _sse_frames(stream: Follow) -> AsyncIterator[str]:
    """SSE frames for a follow: events, ``: keepalive`` comments, then ``event: end``."""
    async for item in stream:
        if isinstance(item, FollowEnd):
            yield _format_end_frame(item.reason, item.cursor)
            return
        if isinstance(item, FollowHeartbeat):
            yield ": keepalive\n\n"
            continue
        yield _format_sse_frame(item)


async def _sse_event_generator(
    store: RunStore,
    run_id: str,
    cursor: int | None,
    *,
    follow: bool = False,
    include_payloads: bool = True,
    max_duration_s: float | None = None,
) -> AsyncIterator[str]:
    """SSE frames for one standalone follower (its own feed). Kept for tests and tooling."""
    stream = _make_feed(store, 1).follow(
        run_id,
        tenant_id=UNSCOPED,
        after_sequence_index=cursor,
        include_payloads=include_payloads,
        until_terminal=not follow,
        max_duration_s=max_duration_s,
    )
    try:
        async for frame in _sse_frames(stream):
            yield frame
    finally:
        await stream.aclose()


def _rfc3339(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _format_sse_frame(event: StoredEvent) -> str:
    """Encode a StoredEvent as one SSE frame.

    Uses ``event: message`` (the SSE default) with ``event_type`` inside
    the JSON payload so generic clients that dispatch on the SSE event
    name don't have to pre-enroll every Dendrux event type. ``timestamp``
    is RFC 3339 in UTC (``...Z``).
    """
    payload = {
        "sequence_index": event.sequence_index,
        "iteration_index": event.iteration_index,
        "event_type": event.event_type,
        "correlation_id": event.correlation_id,
        "timestamp": _rfc3339(event.created_at),
        "data": event.data,
    }
    return (
        f"id: {event.sequence_index}\nevent: message\ndata: {json.dumps(payload, default=str)}\n\n"
    )


def _format_end_frame(reason: str, cursor: int | None) -> str:
    """The last frame of a stream: why it ended and the cursor to resume from."""
    return f"event: end\ndata: {json.dumps({'reason': reason, 'cursor': cursor})}\n\n"
