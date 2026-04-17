"""``make_read_router`` — read-only public HTTP surface over ``RunStore``.

The factory returns a FastAPI ``APIRouter`` the developer mounts into
their own app. The caller-supplied ``authorize`` dependency runs on
every route except ``/health`` (intentionally unauth'd so Kubernetes
liveness/readiness probes can reach it without secrets). Auth failures
raise ``HTTPException`` from that dependency; this module does not
interpret the dependency's return value.

Endpoints in this release (PR 2):

    GET /health
    GET /runs
    GET /runs/{run_id}
    GET /runs/{run_id}/events
    GET /runs/{run_id}/llm-calls
    GET /runs/{run_id}/tool-calls
    GET /runs/{run_id}/traces
    GET /runs/{run_id}/pauses

SSE streaming (``/runs/{run_id}/events/stream``) lands in PR 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from dendrux.store import RunStore


_DRILL_DOWN_LIMIT_DEFAULT = 100
_DRILL_DOWN_LIMIT_MAX = 1000


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
