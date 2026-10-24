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

from dendrux import Agent, tool
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
    rejection_reason: str | None = Field(default=None, max_length=2000)


def browser_result(result: Any) -> dict[str, Any]:
    """Allowlist the response; answer is intended for this authenticated actor."""
    return {"run_id": result.run_id, "status": result.status.value, "answer": result.answer}


# -- Server tool: executes on the server, no pause --
@tool()
async def lookup_price(ticker: str) -> str:
    """Look up the current stock price for a ticker symbol."""
    prices = {"AAPL": 227.50, "GOOGL": 178.30, "MSFT": 445.20, "TSLA": 312.80}
    price = prices.get(ticker.upper())
    if price is None:
        return f"Unknown ticker: {ticker}"
    return f"{ticker.upper()}: ${price:.2f}"


# -- Client tool: the agent pauses and waits for the client to execute --
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read a range of cells from the user's Excel spreadsheet.

    This tool runs on the client side. The server will pause and wait
    for the client to provide the result.
    """
    return ""


def build_agents(database_url: str) -> dict[str, Agent]:
    """Configure the spreadsheet agent with server and client tools."""
    from dendrux.llm.anthropic import AnthropicProvider

    return {
        "anthropic": Agent(
            name="SpreadsheetAnalyst",
            provider=AnthropicProvider(model="claude-sonnet-4-6"),
            database_url=database_url,
            tools=[lookup_price, read_excel_range],
            prompt="Use lookup_price for prices and read_excel_range for spreadsheet data.",
        )
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
        # TODO: pass expected_pause_id when the run-integrity contract lands.
        # Current Agent has no such argument; this demo has no stale-submit guarantee.
        result = await agents[provider].submit_tool_results(run_id, results, tenant_id=tenant)
        return browser_result(result)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="127.0.0.1", port=8000)
