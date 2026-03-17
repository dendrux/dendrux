"""Read-only dashboard API — serves normalized timeline data.

Mounted by the CLI's `dendrite dashboard` command. All endpoints
are read-only and return only observable, redacted data.

Never exposes pause_data or unredacted content.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dendrite.dashboard.normalizer import normalize_timeline, timeline_to_dict

_STATIC_DIR = Path(__file__).parent / "static"

if TYPE_CHECKING:
    from dendrite.runtime.state import StateStore


def create_dashboard_api(state_store: StateStore) -> FastAPI:
    """Create the read-only dashboard API.

    Args:
        state_store: The same StateStore used by the runtime.
            Reads from the same DB — no separate connection needed.
    """
    app = FastAPI(title="Dendrite Dashboard API", version="0.1.0a1")

    # Allow dashboard frontend (served on same or different port) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Local dev tool — no restriction needed
        allow_methods=["GET"],
        allow_headers=["*"],
    )

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
                    "pause_count": pause_count,
                    "created_at": str(r.created_at) if r.created_at else None,
                    "updated_at": str(r.updated_at) if r.updated_at else None,
                }
            )

        return {"runs": items, "total": total_filtered}

    @app.get("/api/runs/{run_id}")
    async def get_run_detail(run_id: str) -> dict[str, Any]:
        """Get the full normalized timeline for a single run.

        This is the "money endpoint" — everything the run detail
        page needs in one request.
        """
        timeline = await normalize_timeline(run_id, state_store)
        if timeline is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        return timeline_to_dict(timeline)

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
                    "created_at": str(e.created_at) if e.created_at else None,
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
                    "created_at": str(t.created_at) if t.created_at else None,
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
                    "created_at": str(tc.created_at) if tc.created_at else None,
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
                    "created_at": str(i.created_at) if i.created_at else None,
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
