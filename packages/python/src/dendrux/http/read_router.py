"""``make_read_router`` — read-only public HTTP surface over ``RunStore``.

The factory returns a FastAPI ``APIRouter`` the developer mounts into
their own app. The caller-supplied ``authorize`` dependency runs on
every route except ``/health`` (intentionally unauth'd so Kubernetes
liveness/readiness probes can reach it without secrets). Auth failures
raise ``HTTPException`` from that dependency; this module does not
interpret the dependency's return value.

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
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from datetime import datetime

    from dendrux.store import RunStore, StoredEvent


_DRILL_DOWN_LIMIT_DEFAULT = 100
_DRILL_DOWN_LIMIT_MAX = 1000

# How often the SSE endpoint emits a keepalive comment when no events
# arrive. Overridable in tests via monkeypatch. Nginx's default
# `proxy_read_timeout` is 60s, so anything well under that works.
_SSE_HEARTBEAT_INTERVAL_S = 15.0

# How often the SSE endpoint polls the DB for new events when the stream
# is idle. Overridable in tests. Per concurrent SSE connection, this
# produces ~(1 / _SSE_POLL_INTERVAL_S) DB queries per second. At default
# 0.5s that's 2 QPS per connection. A planned Postgres LISTEN/NOTIFY
# backend will replace polling with push notifications — until then,
# dev should size their connection pool for expected SSE fanout.
_SSE_POLL_INTERVAL_S = 0.5


def make_read_router(
    *,
    store: RunStore,
    authorize: Callable[..., Any],
) -> APIRouter:
    """Build a mountable read-only router backed by ``store``.

    Args:
        store: The :class:`dendrux.store.RunStore` instance this router reads from.
        authorize: A FastAPI dependency callable. Runs on every authenticated
            route. Raise ``HTTPException`` inside to deny; return any value
            (ignored) to allow. ``/health`` bypasses this dependency.

    Returns:
        An ``APIRouter`` suitable for ``app.include_router(...)``. The
        caller owns routing prefixes, CORS, TLS, and deployment.
    """
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
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        records = await store.list_runs(
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            tenant_id=tenant_id,
            started_after=started_after,
            started_before=started_before,
            limit=limit,
            offset=offset,
        )
        total = await store.count_runs(
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            tenant_id=tenant_id,
            started_after=started_after,
            started_before=started_before,
        )
        return {
            "items": records,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @auth_router.get("/runs/{run_id}")
    async def get_run(run_id: str) -> Any:
        detail = await store.get_run(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        return detail

    @auth_router.get("/runs/{run_id}/events")
    async def get_events(
        run_id: str,
        after: int | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, Any]:
        # 404 fast on missing run rather than silently returning an empty page.
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        events = await store.get_events(run_id, after_sequence_index=after, limit=limit)
        # Echo the client's cursor when the batch is empty — an empty
        # response means "nothing new yet," not "stream is over." A poller
        # that reset to null after an empty batch would miss future events.
        next_cursor = events[-1].sequence_index if events else after
        return {"items": events, "next_cursor": next_cursor}

    @auth_router.get("/runs/{run_id}/events/stream")
    async def stream_events(
        run_id: str,
        request: Request,
        after: int | None = None,
    ) -> StreamingResponse:
        """Server-Sent Events stream of new run events.

        Reads from ``run_events`` via ``RunStore.get_events`` in a per-
        connection polling loop. No shared queue, no fan-out manager —
        each connection is an independent async iterator.

        Cursor resolution (first match wins):
          1. ``Last-Event-ID`` header (SSE reconnect convention)
          2. ``?after=N`` query param
          3. ``None`` (stream from the beginning of the log)

        Emits a ``: keepalive`` comment line every
        ``_SSE_HEARTBEAT_INTERVAL_S`` seconds of idleness so reverse
        proxies don't close the connection.

        The stream stays open until the client disconnects; terminal
        run events (``run.succeeded``/``.failed``/``.cancelled``) are
        emitted as normal frames and the client decides when to leave.
        """
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        cursor = _resolve_sse_cursor(request.headers.get("last-event-id"), after)

        return StreamingResponse(
            _sse_event_generator(store, run_id, cursor),
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
        limit: Annotated[int, Query(ge=1, le=_DRILL_DOWN_LIMIT_MAX)] = _DRILL_DOWN_LIMIT_DEFAULT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        items = await store.get_llm_calls(run_id, iteration=iteration, limit=limit, offset=offset)
        return {"items": items, "limit": limit, "offset": offset}

    @auth_router.get("/runs/{run_id}/tool-calls")
    async def get_tool_calls(
        run_id: str,
        iteration: int | None = None,
        limit: Annotated[int, Query(ge=1, le=_DRILL_DOWN_LIMIT_MAX)] = _DRILL_DOWN_LIMIT_DEFAULT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        items = await store.get_tool_invocations(
            run_id, iteration=iteration, limit=limit, offset=offset
        )
        return {"items": items, "limit": limit, "offset": offset}

    @auth_router.get("/runs/{run_id}/traces")
    async def get_traces(
        run_id: str,
        limit: Annotated[int, Query(ge=1, le=_DRILL_DOWN_LIMIT_MAX)] = _DRILL_DOWN_LIMIT_DEFAULT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        items = await store.get_traces(run_id, limit=limit, offset=offset)
        return {"items": items, "limit": limit, "offset": offset}

    @auth_router.get("/runs/{run_id}/pauses")
    async def get_pauses(run_id: str) -> dict[str, Any]:
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        items = await store.get_pauses(run_id)
        return {"items": items}

    router.include_router(auth_router)
    return router


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


async def _sse_event_generator(
    store: RunStore,
    run_id: str,
    cursor: int | None,
) -> AsyncIterator[str]:
    """Yield SSE frames (events + heartbeat comments) until disconnect.

    Uses ``store.get_events`` in a polling loop rather than
    ``store.stream_events`` so heartbeat timing can be fused with poll
    timing in one coroutine (no concurrent-task juggling). Each iteration
    opens and closes its own DB session — no transaction is held across
    ``asyncio.sleep``.
    """
    loop = asyncio.get_running_loop()
    last_write = loop.time()
    while True:
        events = await store.get_events(run_id, after_sequence_index=cursor, limit=100)
        if events:
            for ev in events:
                yield _format_sse_frame(ev)
                cursor = ev.sequence_index
            last_write = loop.time()
            # Keep draining when the batch is full — more may be buffered.
            if len(events) == 100:
                continue

        # Idle — emit heartbeat if due, then sleep before next poll.
        if loop.time() - last_write >= _SSE_HEARTBEAT_INTERVAL_S:
            yield ": keepalive\n\n"
            last_write = loop.time()

        await asyncio.sleep(_SSE_POLL_INTERVAL_S)


def _format_sse_frame(event: StoredEvent) -> str:
    """Encode a StoredEvent as one SSE frame.

    Uses ``event: message`` (the SSE default) with ``event_type`` inside
    the JSON payload so generic clients that dispatch on the SSE event
    name don't have to pre-enroll every Dendrux event type.
    """
    payload = {
        "sequence_index": event.sequence_index,
        "iteration_index": event.iteration_index,
        "event_type": event.event_type,
        "correlation_id": event.correlation_id,
        "timestamp": event.created_at.isoformat() + "Z" if event.created_at is not None else None,
        "data": event.data,
    }
    return (
        f"id: {event.sequence_index}\nevent: message\ndata: {json.dumps(payload, default=str)}\n\n"
    )
