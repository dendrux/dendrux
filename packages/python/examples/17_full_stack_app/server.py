"""Full-stack Dendrux demo — read router, dashboard, governance, all in one app.

What's in this app:
  - ``make_read_router``      mounted at /api/dendrux   (Dendrux)
  - ``create_dashboard_api``  mounted at /dashboard     (Dendrux)
  - dev-written write routes  /chat, /runs/{id}/..      (this file)
  - all 4 governance layers   (see ``agent_setup.py``)

What the dev owns:
  - routing, auth, request/response schemas, UI
  - ownership checks + rate limiting (see TODO(production) markers)

Two providers, one agent definition per provider. The browser picks
anthropic or openai at session start; same routes, same governance.

Streaming is intentionally out of scope here: guardrails require the
full LLM response to scan, so they're incompatible with the current
``agent.stream()`` path. See example 09 for the streaming client-tool
pattern without governance.

Run:
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
        python examples/17_full_stack_app/server.py

Open http://localhost:8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.dashboard.api import create_dashboard_api
from dendrux.db.models import Base
from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    PersistenceNotConfiguredError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.http import make_read_router
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import RunStore
from dendrux.types import ToolResult

from agent_setup import build_anthropic_agent, build_openai_agent  # type: ignore[import-not-found]  # isort: skip


# ---------------------------------------------------------------------------
# Request / response schemas (module-scope — FastAPI needs resolvable types)
# ---------------------------------------------------------------------------

Provider = Literal["anthropic", "openai"]


class ChatIn(BaseModel):
    input: str
    provider: Provider = "anthropic"


class ToolResultItem(BaseModel):
    tool_call_id: str
    tool_name: str
    payload: str  # JSON-encoded string


class ToolResultsIn(BaseModel):
    results: list[ToolResultItem]
    provider: Provider = "anthropic"


class ApprovalIn(BaseModel):
    approved: bool
    rejection_reason: str | None = None
    provider: Provider = "anthropic"


# ---------------------------------------------------------------------------
# Auth dep — shared across read router, dashboard, and write routes
# ---------------------------------------------------------------------------

_DEMO_TOKEN = os.environ.get("DENDRUX_DEMO_TOKEN", "dev-secret-abc")


async def require_caller(
    request: Request,
    x_app_token: Annotated[str | None, Header()] = None,
) -> str:
    """Verify the caller's shared-secret token.

    Accepts the token from either the ``X-App-Token`` header (normal
    requests) or a ``?token=`` query param (EventSource / SSE, which
    can't send custom headers).

    TODO(production): replace with your real auth (JWT, signed cookies,
    API key lookup). For SSE, prefer signed cookies on the same origin
    over a URL query param — query params leak into access logs.
    """
    token = x_app_token or request.query_params.get("token")
    if token != _DEMO_TOKEN:
        raise HTTPException(status_code=401, detail="missing or invalid X-App-Token")
    return "demo-user"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _run_result_body(result: Any) -> dict[str, Any]:
    """Shape a RunResult for the HTTP response.

    We ship ``answer`` verbatim (the LLM's output with redaction placeholders
    intact) AND the ``pii_mapping`` so the UI can deanonymize for display.
    This keeps the teaching split visible: LLM saw placeholders, user sees
    real values, audit trail is explicit.
    """
    meta = result.meta or {}
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "answer": result.answer,
        "pii_mapping": meta.get("pii_mapping") or {},
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_demo_app() -> FastAPI:
    load_dotenv(Path(__file__).resolve().parents[4] / ".env")
    db_path = Path.home() / ".dendrux" / "dendrux_example17.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(db_url)
    state_store = SQLAlchemyStateStore(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # noqa: ARG001 — FastAPI contract
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield
        await engine.dispose()

    app = FastAPI(title="Dendrux — Full Stack Demo", lifespan=lifespan)

    # --- Translate Dendrux exceptions to HTTP status codes -----------------
    # Dendrux never imposes an HTTP contract; the dev maps each exception to
    # the status code their app uses. These are the suggested mappings from
    # ``dendrux.errors``. Adjust per your app's conventions if you'd like.
    @app.exception_handler(RunNotFoundError)
    async def _h_not_found(_req: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(RunAlreadyTerminalError)
    async def _h_terminal(_req: Request, exc: RunAlreadyTerminalError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "status": exc.current_status.value},
        )

    @app.exception_handler(RunNotPausedError)
    async def _h_not_paused(_req: Request, exc: RunNotPausedError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "status": exc.current_status.value},
        )

    @app.exception_handler(PauseStatusMismatchError)
    async def _h_pause_mismatch(_req: Request, exc: PauseStatusMismatchError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(RunAlreadyClaimedError)
    async def _h_claimed(_req: Request, exc: RunAlreadyClaimedError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidToolResultError)
    async def _h_invalid(_req: Request, exc: InvalidToolResultError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(PersistenceNotConfiguredError)
    async def _h_no_db(_req: Request, exc: PersistenceNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    agents = {
        "anthropic": build_anthropic_agent(database_url=db_url),
        "openai": build_openai_agent(database_url=db_url),
    }
    # Reuse the shared state_store so agents, routes, and dashboard all
    # hit one connection pool.
    for a in agents.values():
        a._state_store = state_store  # noqa: SLF001 — demo wiring

    # --- Dendrux-provided surfaces --------------------------------------
    store = RunStore.from_engine(engine)
    app.include_router(
        make_read_router(store=store, authorize=require_caller),
        prefix="/api/dendrux",
    )
    app.mount("/dashboard", create_dashboard_api(state_store))

    # --- Static UI -------------------------------------------------------
    client_html = Path(__file__).parent / "client.html"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(client_html, media_type="text/html")

    # --- Write routes (dev-owned) ---------------------------------------
    # Every write route takes ``Depends(require_caller)``. Production apps
    # additionally fetch the run and verify ownership before mutating.

    @app.post("/chat")
    async def start_chat(body: ChatIn, _: Annotated[str, Depends(require_caller)]) -> JSONResponse:
        # TODO(production): stamp identity onto the run via metadata= so
        # later write routes can verify ownership.
        result = await agents[body.provider].run(body.input)
        return JSONResponse(_run_result_body(result))

    @app.post("/runs/{run_id}/tool-results")
    async def submit_tool_results(
        run_id: str,
        body: ToolResultsIn,
        _: Annotated[str, Depends(require_caller)],
    ) -> JSONResponse:
        # TODO(production): fetch run, compare owner_id to authenticated identity.
        results = [
            ToolResult(name=r.tool_name, call_id=r.tool_call_id, payload=r.payload)
            for r in body.results
        ]
        result = await agents[body.provider].submit_tool_results(run_id, results)
        return JSONResponse(_run_result_body(result))

    @app.post("/runs/{run_id}/approvals")
    async def submit_approval(
        run_id: str,
        body: ApprovalIn,
        _: Annotated[str, Depends(require_caller)],
    ) -> JSONResponse:
        # TODO(production): ownership check as above.
        result = await agents[body.provider].submit_approval(
            run_id,
            approved=body.approved,
            rejection_reason=body.rejection_reason,
        )
        return JSONResponse(_run_result_body(result))

    @app.delete("/runs/{run_id}")
    async def cancel_run(
        run_id: str,
        request: Request,
        _: Annotated[str, Depends(require_caller)],
    ) -> JSONResponse:
        # TODO(production): ownership check as above.
        provider: Provider = request.query_params.get("provider", "anthropic")  # type: ignore[assignment]
        result = await agents[provider].cancel_run(run_id)
        return JSONResponse(_run_result_body(result))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="0.0.0.0", port=8000)
