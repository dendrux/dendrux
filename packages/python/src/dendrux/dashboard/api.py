"""Read-only dashboard API — serves normalized timeline data.

Mounted by the CLI's `dendrux dashboard` command. All endpoints
are read-only. Trace content is served raw (the DB is ground truth);
when a PII guardrail is active, ``agent_runs.pii_mapping`` carries
the placeholder→real bijection the LLM saw and can be applied at
render time to produce an LLM-eye view.

Never exposes ``pause_data`` — that column is execution state,
cleared on finalize, and out of scope for public readers.

Closed by default (see ``dendrux.dashboard.security``): Host check, no CORS
unless origins are listed, bearer token required beyond loopback. Every list
endpoint is paged with a hard maximum.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dendrux.dashboard.normalizer import normalize_timeline, timeline_to_dict
from dendrux.dashboard.security import (
    ANY_HOST,
    DashboardGuard,
    is_loopback_host,
    normalize_origins,
)

_STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("dendrux.dashboard")

# Page-size bounds. Every list endpoint returns at most ``*_MAX`` rows; a
# caller asking for more is clamped, never refused.
RUNS_MAX_LIMIT = 200
RUNS_MAX_OFFSET = 10_000
DRILLDOWN_DEFAULT_LIMIT = 200
DRILLDOWN_MAX_LIMIT = 500  # the store clamps to 1000; keep room for the has_more probe
LLM_CALLS_MAX_LIMIT = 200  # rows carry raw provider payloads

# Delegation block on the run-detail page. A run with more direct children
# than this, or whose full traversal takes longer than the time budget, gets a
# bounded view flagged ``truncated`` instead of walking the whole tree.
MAX_DELEGATION_CHILDREN = 500
MAX_DELEGATION_ANCESTRY = 25
DELEGATION_TIME_BUDGET_S = 5.0

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from dendrux.runtime.state import DelegationInfo, StateStore


async def _count_pauses_per_run(state_store: StateStore, run_ids: list[str]) -> dict[str, int]:
    """Count ``run.paused`` events per run_id.

    Uses ``StateStore.count_pauses_per_run`` when the backend exposes it
    (the default SQLAlchemy implementation does — single GROUP BY). Falls
    back to a per-run ``get_run_events(event_types=["run.paused"])`` loop
    for custom backends that don't implement the batch method.
    """
    if not run_ids:
        return {}
    batch = getattr(state_store, "count_pauses_per_run", None)
    if batch is not None:
        counts: dict[str, int] = await batch(run_ids)
        return counts
    result: dict[str, int] = {}
    for rid in run_ids:
        events = await state_store.get_run_events(
            rid, event_types=["run.paused"], limit=DRILLDOWN_MAX_LIMIT
        )
        count = len(events)
        if count:
            result[rid] = count
    return result


def _continued_from(meta: dict[str, Any] | None) -> list[str]:
    """Run ids a run replayed via ``runs=`` (empty when it replayed none)."""
    from dendrux.runtime.replay import REPLAY_META_KEY

    replay = (meta or {}).get(REPLAY_META_KEY)
    if not isinstance(replay, dict):
        return []
    return [r for r in replay.get("runs") or [] if isinstance(r, str)]


def _utc_iso(ts: datetime | None) -> str | None:
    """Format a naive UTC datetime as ISO 8601 with Z suffix.

    SQLite CURRENT_TIMESTAMP stores UTC but as a naive datetime.
    Appending 'Z' tells the browser the timestamp is UTC so relative
    time calculations (e.g. "5m ago") are correct in any timezone.
    """
    if ts is None:
        return None
    return ts.isoformat() + "Z"


def _delegation_to_dict(info: DelegationInfo) -> dict[str, Any]:
    """Serialize DelegationInfo to a JSON-safe dict."""
    parent_dict: dict[str, Any] | None = None
    if info.parent is not None:
        parent_dict = {
            "run_id": info.parent.run_id,
            "resolved": info.parent.resolved,
            "agent_name": info.parent.agent_name,
            "status": info.parent.status,
            "delegation_level": info.parent.delegation_level,
        }

    return {
        "parent": parent_dict,
        "children": [
            {
                "run_id": c.run_id,
                "agent_name": c.agent_name,
                "status": c.status,
                "delegation_level": c.delegation_level,
            }
            for c in info.children
        ],
        "ancestry": [
            {
                "run_id": a.run_id,
                "agent_name": a.agent_name,
                "status": a.status,
                "delegation_level": a.delegation_level,
            }
            for a in info.ancestry
        ],
        "subtree_summary": {
            "direct_child_count": info.subtree_summary.direct_child_count,
            "descendant_count": info.subtree_summary.descendant_count,
            "max_depth": info.subtree_summary.max_depth,
            "subtree_input_tokens": info.subtree_summary.subtree_input_tokens,
            "subtree_output_tokens": info.subtree_summary.subtree_output_tokens,
            "subtree_cost_usd": info.subtree_summary.subtree_cost_usd,
            "unknown_cost_count": info.subtree_summary.unknown_cost_count,
            "status_counts": info.subtree_summary.status_counts,
        },
        "ancestry_complete": info.ancestry_complete,
    }


def _page_bounds(limit: int, offset: int, *, max_limit: int) -> tuple[int, int]:
    """Clamp ``limit`` into 1..max_limit; ``offset`` is validated by the endpoint."""
    return min(max(1, limit), max_limit), offset


def _page_info(limit: int, offset: int, has_more: bool) -> dict[str, Any]:
    """Paging fields. An end that stops at the offset cap is explicit, never repeated.

    ``next_offset`` is ``None`` once the next page would start past
    ``RUNS_MAX_OFFSET``; ``capped`` then tells the caller that rows remain but
    this endpoint cannot page to them (narrow the query instead).
    """
    nxt = offset + limit
    reachable = has_more and nxt <= RUNS_MAX_OFFSET
    return {
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "next_offset": nxt if reachable else None,
        "capped": has_more and not reachable,
    }


def _brief(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "agent_name": run.agent_name,
        "status": run.status,
        "delegation_level": run.delegation_level,
    }


async def _bounded_delegation(
    state_store: StateStore, run_id: str, child_count: int
) -> dict[str, Any] | None:
    """Delegation block built from bounded reads: capped ancestry and children."""
    run = await state_store.get_run(run_id)
    if run is None:
        return None
    ancestry: list[Any] = []
    complete = True
    seen = {run_id}
    current = run.parent_run_id
    while current:
        if current in seen or len(ancestry) >= MAX_DELEGATION_ANCESTRY:
            complete = False
            break
        seen.add(current)
        row = await state_store.get_run(current)
        if row is None:
            complete = False
            break
        ancestry.append(row)
        current = row.parent_run_id
    children = await state_store.list_runs(parent_run_id=run_id, limit=MAX_DELEGATION_CHILDREN)
    status_counts: dict[str, int] = {run.status: 1}
    for child in children:
        status_counts[child.status] = status_counts.get(child.status, 0) + 1
    parent: dict[str, Any] | None = None
    if run.parent_run_id:
        row = ancestry[0] if ancestry else None
        parent = {
            "run_id": run.parent_run_id,
            "resolved": row is not None and row.id == run.parent_run_id,
            "agent_name": row.agent_name if row is not None else None,
            "status": row.status if row is not None else None,
            "delegation_level": row.delegation_level if row is not None else None,
        }
    return {
        "parent": parent,
        "children": [_brief(c) for c in children],
        "ancestry": [_brief(a) for a in reversed(ancestry)],
        "subtree_summary": {
            "direct_child_count": child_count,
            "descendant_count": child_count,
            "max_depth": 1,
            "subtree_input_tokens": run.total_input_tokens,
            "subtree_output_tokens": run.total_output_tokens,
            "subtree_cost_usd": None,
            "unknown_cost_count": 0,
            "status_counts": status_counts,
        },
        "ancestry_complete": complete,
        "truncated": True,
    }


async def _delegation_block(state_store: StateStore, run_id: str) -> dict[str, Any] | None:
    """Full delegation info when the tree is small; a bounded, flagged view otherwise."""
    child_count = await state_store.count_runs(parent_run_id=run_id)
    if child_count <= MAX_DELEGATION_CHILDREN:
        try:
            info = await asyncio.wait_for(
                state_store.get_delegation_info(run_id), DELEGATION_TIME_BUDGET_S
            )
        except TimeoutError:
            info = None
            logger.warning("dashboard.delegation_budget_exceeded run_id=%s", run_id)
        else:
            if info is None:
                return None
            return {**_delegation_to_dict(info), "truncated": False}
    return await _bounded_delegation(state_store, run_id, child_count)


def create_dashboard_api(
    state_store: StateStore,
    auth_token: str | None = None,
    *,
    bind_host: str = "127.0.0.1",
    allowed_hosts: Sequence[str] = (),
    allow_origins: Sequence[str] = (),
) -> FastAPI:
    """Create the read-only dashboard API.

    Args:
        state_store: The same StateStore used by the runtime.
            Reads from the same DB — no separate connection needed.
        auth_token: Bearer token for API access. When set, every ``/api/``
            request must carry ``Authorization: Bearer <token>``. Static
            files (HTML/CSS/JS) are served without auth. Required when
            ``bind_host`` is not loopback.
        bind_host: The interface the server listens on. A non-loopback value
            without a token raises ``ValueError``; the host is also added to
            the accepted ``Host`` names. The guard cannot see the socket, so an
            embedder that serves this app beyond loopback with its own server
            must pass the real ``bind_host`` (and a token) here.
        allowed_hosts: Extra ``Host`` names to accept besides loopback names,
            IP literals and ``bind_host`` (``["*"]`` turns the check off, for
            embedding behind a gateway that validates it).
        allow_origins: Browser origins allowed to call the API cross-origin
            (``https://host[:port]``). Empty by default: no CORS headers.
    """
    if not is_loopback_host(bind_host) and not auth_token:
        raise ValueError(
            "The dashboard serves run payloads; set a token "
            "(DENDRUX_DASHBOARD_TOKEN) when binding to a non-loopback address."
        )
    origins = normalize_origins(allow_origins)
    hosts = [*allowed_hosts]
    if bind_host and bind_host not in {"0.0.0.0", "::", ANY_HOST}:
        hosts.append(bind_host)

    # Interactive docs and the schema are not needed by the React app.
    app = FastAPI(
        title="Dendrux Dashboard API",
        version="0.1.0a1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Added innermost first: token check, then CORS (so a 401 still carries the
    # CORS headers the browser needs to read it), then the Host check.
    app.add_middleware(DashboardGuard, allowed_hosts=[ANY_HOST], token=auth_token or None)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET"],
            allow_headers=["Authorization"],
        )
    app.add_middleware(DashboardGuard, allowed_hosts=hosts, token=None)

    @app.get("/api/runs")
    async def list_runs(
        limit: int = 50,
        offset: int = Query(0, ge=0, le=RUNS_MAX_OFFSET),
        status: str | None = None,
        agent: str | None = None,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """List runs with filters.

        Uses SQL-side filtering through StateStore — the same query
        surface that powers the public ``RunStore``. Pause counts are
        rolled up in a single aggregation query (no N+1).
        """
        capped_limit, clamped_offset = _page_bounds(limit, offset, max_limit=RUNS_MAX_LIMIT)

        # No ``?tenant=`` means every tenant: leave the argument out (the
        # store's default is the all-tenants scope) rather than passing None.
        scope: dict[str, Any] = {} if tenant is None else {"tenant_id": tenant}
        runs = await state_store.list_runs(
            limit=capped_limit,
            offset=clamped_offset,
            status=status,
            agent_name=agent,
            **scope,
        )
        total_filtered = await state_store.count_runs(
            status=status,
            agent_name=agent,
            **scope,
        )

        # Batch aggregation on the default SQLAlchemy backend; falls back
        # to a per-run query on custom StateStore implementations that
        # don't expose count_pauses_per_run.
        pause_counts = await _count_pauses_per_run(state_store, [r.id for r in runs])

        items = [
            {
                "run_id": r.id,
                "agent_name": r.agent_name,
                "status": r.status,
                "iteration_count": r.iteration_count,
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "total_cost_usd": r.total_cost_usd,
                "total_cache_read_tokens": r.total_cache_read_tokens,
                "total_cache_creation_tokens": r.total_cache_creation_tokens,
                "model": r.model,
                "parent_run_id": r.parent_run_id,
                "delegation_level": r.delegation_level,
                "pause_count": pause_counts.get(r.id, 0),
                "continued_from": _continued_from(r.meta),
                "created_at": _utc_iso(r.created_at),
                "updated_at": _utc_iso(r.updated_at),
            }
            for r in runs
        ]

        return {"runs": items, "total": total_filtered}

    @app.get("/api/runs/{run_id}")
    async def get_run_detail(run_id: str) -> dict[str, Any]:
        """Get the full normalized timeline for a single run.

        This is the "money endpoint" — everything the run detail
        page needs in one request, including the delegation block.
        """
        timeline = await normalize_timeline(run_id, state_store)
        if timeline is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        result = timeline_to_dict(timeline)

        # Delegation block — always present since timeline check
        # already verified the run exists. The else is defensive.
        result["delegation"] = await _delegation_block(state_store, run_id)

        return result

    @app.get("/api/runs/{run_id}/events")
    async def get_run_events(
        run_id: str,
        limit: int = DRILLDOWN_DEFAULT_LIMIT,
        after: int | None = None,
    ) -> dict[str, Any]:
        """Get raw run events (for debugging), one bounded page at a time.

        ``after`` is a ``sequence_index`` cursor: pass the previous
        ``next_after`` to read the next page.
        """
        page = min(max(1, limit), DRILLDOWN_MAX_LIMIT)
        rows = await state_store.get_run_events(run_id, after_sequence_index=after, limit=page + 1)
        if not rows:
            run = await state_store.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        has_more = len(rows) > page
        events = rows[:page]

        return {
            "events": [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "sequence_index": e.sequence_index,
                    "iteration_index": e.iteration_index,
                    "correlation_id": e.correlation_id,
                    "data": e.data,
                    "created_at": _utc_iso(e.created_at),
                }
                for e in events
            ],
            "limit": page,
            "has_more": has_more,
            "next_after": events[-1].sequence_index if has_more else None,
        }

    @app.get("/api/runs/{run_id}/traces")
    async def get_run_traces(
        run_id: str,
        limit: int = DRILLDOWN_DEFAULT_LIMIT,
        offset: int = Query(0, ge=0, le=RUNS_MAX_OFFSET),
    ) -> dict[str, Any]:
        """Get conversation traces for payload inspection (one bounded page)."""
        page, skip = _page_bounds(limit, offset, max_limit=DRILLDOWN_MAX_LIMIT)
        run = await state_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        rows = await state_store.get_traces(run_id, limit=page + 1, offset=skip)
        traces = rows[:page]
        return {
            "traces": [
                {
                    "id": t.id,
                    "role": t.role,
                    "content": t.content,
                    "order_index": t.order_index,
                    "meta": t.meta,
                    "created_at": _utc_iso(t.created_at),
                }
                for t in traces
            ],
            **_page_info(page, skip, len(rows) > page),
        }

    @app.get("/api/runs/{run_id}/tool-calls")
    async def get_run_tool_calls(
        run_id: str,
        limit: int = DRILLDOWN_DEFAULT_LIMIT,
        offset: int = Query(0, ge=0, le=RUNS_MAX_OFFSET),
    ) -> dict[str, Any]:
        """Get tool call records for a run (one bounded page)."""
        page, skip = _page_bounds(limit, offset, max_limit=DRILLDOWN_MAX_LIMIT)
        run = await state_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        rows = await state_store.get_tool_calls(run_id, limit=page + 1, offset=skip)
        tool_calls = rows[:page]
        return {
            "tool_calls": [
                {
                    "id": tc.id,
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "target": tc.target,
                    "params": tc.params,
                    "result": tc.result,
                    "success": tc.success,
                    "duration_ms": tc.duration_ms,
                    "iteration_index": tc.iteration_index,
                    "error_message": tc.error_message,
                    "created_at": _utc_iso(tc.created_at),
                }
                for tc in tool_calls
            ],
            **_page_info(page, skip, len(rows) > page),
        }

    @app.get("/api/runs/{run_id}/llm-calls")
    async def get_run_llm_calls(
        run_id: str,
        limit: int = LLM_CALLS_MAX_LIMIT,
        offset: int = Query(0, ge=0, le=RUNS_MAX_OFFSET),
    ) -> dict[str, Any]:
        """Get LLM interaction records for payload inspection (one bounded page)."""
        page, skip = _page_bounds(limit, offset, max_limit=LLM_CALLS_MAX_LIMIT)
        run = await state_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        rows = await state_store.get_llm_interactions(run_id, limit=page + 1, offset=skip)
        interactions = rows[:page]
        return {
            "llm_calls": [
                {
                    "id": i.id,
                    "iteration_index": i.iteration_index,
                    "model": i.model,
                    "provider": i.provider,
                    "semantic_request": i.semantic_request,
                    "semantic_response": i.semantic_response,
                    "provider_request": i.provider_request,
                    "provider_response": i.provider_response,
                    "input_tokens": i.input_tokens,
                    "output_tokens": i.output_tokens,
                    "cost_usd": i.cost_usd,
                    "duration_ms": i.duration_ms,
                    "cache_read_input_tokens": i.cache_read_input_tokens,
                    "cache_creation_input_tokens": i.cache_creation_input_tokens,
                    "created_at": _utc_iso(i.created_at),
                }
                for i in interactions
            ],
            **_page_info(page, skip, len(rows) > page),
        }

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Serve pre-built React dashboard (if static assets exist).
    # In development, Vite dev server handles this via proxy.
    # Check for index.html, not just the directory — .gitkeep keeps
    # the dir in git but doesn't mean assets are built.
    if (_STATIC_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="dashboard")

    return app
