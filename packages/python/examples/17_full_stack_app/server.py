"""Developer-owned HTTP routes over Agent and RunStore.

This single-user demo authenticates every API call with X-App-Token.
The one demo actor owns the dedicated demo tenant. Production apps must
resolve identity, tenant, run ownership and approval authority separately.
"""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db.models import Base
from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.store import RunStore
from dendrux.types import ToolResult

Provider = Literal["anthropic", "openai"]
TENANT = "demo-user"


class ChatIn(BaseModel):
    input: str = Field(min_length=1, max_length=10_000)
    provider: Provider = "anthropic"


class ToolResultItem(BaseModel):
    tool_call_id: str
    tool_name: str
    payload: str = Field(max_length=100_000)


class ToolResultsIn(BaseModel):
    results: list[ToolResultItem] = Field(min_length=1, max_length=100)


class ApprovalIn(BaseModel):
    approved: bool
    expected_pause_id: str | None = Field(default=None, max_length=26)
    rejection_reason: str | None = Field(default=None, max_length=2000)


def browser_result(result: Any) -> dict[str, Any]:
    """Allowlist the response; answer is intended for this authenticated actor."""
    return {"run_id": result.run_id, "status": result.status.value, "answer": result.answer}


def build_agents(database_url: str) -> dict[str, Agent]:
    """Use the same governance configuration with either provider."""
    from agent_setup import build_anthropic_agent, build_openai_agent

    return {
        "anthropic": build_anthropic_agent(database_url=database_url),
        "openai": build_openai_agent(database_url=database_url),
    }


def create_demo_app(*, database_url: str | None = None) -> FastAPI:
    """Build routes owned by this app, with a dedicated demo tenant."""
    if database_url is None:
        output = Path(__file__).parent / "output"
        output.mkdir(exist_ok=True)
        database_url = f"sqlite+aiosqlite:///{output / 'demo.db'}"
    engine = create_async_engine(database_url)
    store = RunStore.from_engine(engine)
    agents = build_agents(database_url)
    token = os.environ.get("DENDRUX_DEMO_TOKEN", "dev-secret-abc").encode()
    # App-owned identity/provider association. Persist it in your app DB for
    # recovery across server restarts; Dendrux does not own this association.
    owners: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            yield
        finally:
            for agent in agents.values():
                await agent.close()
            await store.close()
            await engine.dispose()

    app = FastAPI(title="Dendrux developer routes demo", lifespan=lifespan)

    async def caller(
        x_app_token: Annotated[str | None, Header()] = None,
    ) -> str:
        if not token or not hmac.compare_digest((x_app_token or "").encode(), token):
            raise HTTPException(401, "missing or invalid X-App-Token")
        return TENANT

    async def owned_run(run_id: str, tenant: Annotated[str, Depends(caller)]) -> str:
        if run_id not in owners or await store.get_run(run_id, tenant_id=tenant) is None:
            raise HTTPException(404, "run not found")
        return owners[run_id]

    async def conflict(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": type(exc).__name__})

    for error in (
        RunNotPausedError,
        PauseStatusMismatchError,
        RunAlreadyClaimedError,
        RunAlreadyTerminalError,
    ):
        app.add_exception_handler(error, conflict)

    @app.exception_handler(RunNotFoundError)
    async def not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "run not found"})

    @app.exception_handler(InvalidToolResultError)
    async def invalid(_: Request, exc: InvalidToolResultError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "invalid tool results"})

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "client.html")

    @app.post("/chat")
    async def start(body: ChatIn, tenant: Annotated[str, Depends(caller)]) -> dict[str, Any]:
        if body.provider not in agents:
            raise HTTPException(400, "provider unavailable")
        result = await agents[body.provider].run(body.input, tenant_id=tenant)
        owners[result.run_id] = body.provider
        return browser_result(result)

    @app.get("/runs/{run_id}/pending")
    async def pending(
        run_id: str,
        tenant: Annotated[str, Depends(caller)],
        _: Annotated[str, Depends(owned_run)],
    ) -> Any:
        pause = await store.get_pending(run_id, tenant_id=tenant)
        return pause.to_dict() if pause is not None else None

    @app.post("/runs/{run_id}/tool-results")
    async def tool_results(
        run_id: str,
        body: ToolResultsIn,
        tenant: Annotated[str, Depends(caller)],
        provider: Annotated[str, Depends(owned_run)],
    ) -> dict[str, Any]:
        results = [
            ToolResult(name=r.tool_name, call_id=r.tool_call_id, payload=r.payload)
            for r in body.results
        ]
        result = await agents[provider].submit_tool_results(run_id, results, tenant_id=tenant)
        return browser_result(result)

    @app.post("/runs/{run_id}/approvals")
    async def approval(
        run_id: str,
        body: ApprovalIn,
        tenant: Annotated[str, Depends(caller)],
        provider: Annotated[str, Depends(owned_run)],
    ) -> dict[str, Any]:
        # The single demo actor is also the approver. Production must check
        # the actor's approval role and record the decision in app audit data.
        pause = await store.get_pending(run_id, tenant_id=tenant)
        if pause is None or pause.kind != "approval":
            raise HTTPException(409, "not waiting for approval")
        if any(call.args_hidden for call in pause.tool_calls):
            raise HTTPException(409, "approval arguments unavailable; resolve on the server")
        # Bind the decision to the pause generation reviewed by the client.
        result = await agents[provider].submit_approval(
            run_id,
            approved=body.approved,
            expected_pause_id=body.expected_pause_id,
            rejection_reason=body.rejection_reason,
            tenant_id=tenant,
        )
        return browser_result(result)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="127.0.0.1", port=8000)
