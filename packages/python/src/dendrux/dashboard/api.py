"""Read-only dashboard API — serves normalized timeline data.

Mounted by the CLI's `dendrux dashboard` command. All endpoints
are read-only and return only observable, redacted data.

Never exposes pause_data or unredacted content.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dendrux.dashboard.normalizer import normalize_timeline, timeline_to_dict

_STATIC_DIR = Path(__file__).parent / "static"

if TYPE_CHECKING:
    from datetime import datetime

    from dendrux.runtime.state import DelegationInfo, StateStore


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


def create_dashboard_api(
    state_store: StateStore,
    auth_token: str | None = None,
) -> FastAPI:
    """Create the read-only dashboard API.

    Args:
        state_store: The same StateStore used by the runtime.
            Reads from the same DB — no separate connection needed.
        auth_token: Optional Bearer token for API access. When set,
            all API requests must include ``Authorization: Bearer <token>``.
            Static files (HTML/CSS/JS) are served without auth.
    """
    app = FastAPI(title="Dendrux Dashboard API", version="0.1.0a1")

    # Allow dashboard frontend (served on same or different port) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*", "Authorization"],
    )

    if auth_token is not None:
        import hmac

        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        _expected = f"Bearer {auth_token}"

        class _AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
                # Static files and CORS preflight served without auth
                if not request.url.path.startswith("/api/") or request.method == "OPTIONS":
                    return await call_next(request)
                header = request.headers.get("authorization", "")
                if not hmac.compare_digest(header, _expected):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing auth token."},
                    )
                return await call_next(request)

        app.add_middleware(_AuthMiddleware)

    @app.get("/api/runs")
    async def list_runs(
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        agent: str | None = None,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """List runs with filters.

        When the agent filter is used, we fetch up to 1000 rows from the
        DB, filter in Python, then paginate the filtered results. Both
        results and total are approximate beyond that window.

        This is correct for a local dev tool. At scale, agent filtering
        should move to a DB query parameter on StateStore.list_runs().
        """
        if agent:
            # Fetch a larger window to filter from, then paginate
            all_runs = await state_store.list_runs(
                limit=1000,
                offset=0,
                status=status,
                tenant_id=tenant,
            )
            filtered = [r for r in all_runs if r.agent_name == agent]
            total_filtered = len(filtered)
            clamped_offset = max(0, offset)
            runs = filtered[clamped_offset : clamped_offset + min(limit, 200)]
        else:
            runs = await state_store.list_runs(
                limit=min(limit, 200),
                offset=max(0, offset),
                status=status,
                tenant_id=tenant,
            )
            # Total is approximate without a COUNT query — acceptable for dev tool
            total_filtered = len(runs)

        items = []
        for r in runs:
            # Count pause events for the "Pauses" column
            events = await state_store.get_run_events(r.id)
            pause_count = sum(1 for e in events if e.event_type == "run.paused")

            items.append(
                {
                    "run_id": r.id,
                    "agent_name": r.agent_name,
                    "status": r.status,
                    "iteration_count": r.iteration_count,
                    "total_input_tokens": r.total_input_tokens,
                    "total_output_tokens": r.total_output_tokens,
                    "total_cost_usd": r.total_cost_usd,
                    "model": r.model,
                    "parent_run_id": r.parent_run_id,
                    "delegation_level": r.delegation_level,
                    "pause_count": pause_count,
                    "created_at": _utc_iso(r.created_at),
                    "updated_at": _utc_iso(r.updated_at),
                }
            )

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
        delegation_info = await state_store.get_delegation_info(run_id)
        result["delegation"] = _delegation_to_dict(delegation_info) if delegation_info else None

        return result

    @app.get("/api/runs/{run_id}/events")
    async def get_run_events(run_id: str) -> dict[str, Any]:
        """Get raw run events (for debugging)."""
        events = await state_store.get_run_events(run_id)
        if not events:
            run = await state_store.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

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
            ]
        }

    @app.get("/api/runs/{run_id}/traces")
    async def get_run_traces(run_id: str) -> dict[str, Any]:
        """Get conversation traces for payload inspection."""
        run = await state_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        traces = await state_store.get_traces(run_id)
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
            ]
        }

    @app.get("/api/runs/{run_id}/tool-calls")
    async def get_run_tool_calls(run_id: str) -> dict[str, Any]:
        """Get tool call records for a run."""
        run = await state_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        tool_calls = await state_store.get_tool_calls(run_id)
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
            ]
        }

    @app.get("/api/runs/{run_id}/llm-calls")
    async def get_run_llm_calls(run_id: str) -> dict[str, Any]:
        """Get LLM interaction records for payload inspection."""
        run = await state_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        interactions = await state_store.get_llm_interactions(run_id)
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
                    "created_at": _utc_iso(i.created_at),
                }
                for i in interactions
            ]
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
