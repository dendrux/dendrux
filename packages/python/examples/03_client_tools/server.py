"""Client-side tool pause/resume — agent with server + client tools.

Demonstrates Dendrux's core differentiator: an agent that pauses when
it needs a client-side tool, waits for the browser to execute it, then
resumes reasoning.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/03_client_tools/server.py

Then open http://localhost:8000

For the observability dashboard, run in a separate terminal:
    dendrux dashboard --open

Dendrux provides:
  - Pause/resume orchestration (``@tool(target="client")`` + ``agent.run``)
  - DB-backed events + SSE (``make_read_router``)
  - Race-safe resume methods (``agent.submit_tool_results``, ``agent.cancel_run``)

The developer owns:
  - HTTP routes for start/tool-results/cancel (below)
  - Auth — this example uses a no-op dep; real apps plug in their user auth
  - Request/response schemas (Pydantic models below)
  - The HTML/JS client (``client.html``)

Note: request-body Pydantic models live at module scope (not inside
``create_demo_app``) so FastAPI can resolve their type annotations —
with ``from __future__ import annotations`` enabled, nested-class
annotations would otherwise stay as strings and FastAPI would treat
them as query params.
"""

from pydantic import BaseModel

from dendrux import Agent, tool


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


# -- Dev-owned request-body schemas (module scope so FastAPI can resolve them) --
class StartChatIn(BaseModel):
    input: str


class ToolResultItem(BaseModel):
    tool_call_id: str
    tool_name: str
    payload: str  # JSON-encoded string


class SubmitToolResultsIn(BaseModel):
    results: list[ToolResultItem]


def create_demo_app():  # type: ignore[no-untyped-def]
    """Build the demo FastAPI application.

    Wires ``make_read_router`` for reads + SSE and three hand-written
    routes for ``/chat``, ``/tool-results``, and ``/cancel`` that delegate
    to the agent's public methods.
    """
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse

    from dendrux.http import make_read_router
    from dendrux.llm.anthropic import AnthropicProvider
    from dendrux.store import RunStore
    from dendrux.types import ToolResult

    load_dotenv(Path(__file__).resolve().parents[4] / ".env")

    db_path = Path.home() / ".dendrux" / "dendrux.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    agent = Agent(
        name="SpreadsheetAnalyst",
        provider=AnthropicProvider(
            model="claude-sonnet-4-6",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        ),
        database_url=db_url,
        prompt=(
            "You are a spreadsheet analyst. You can look up stock prices with "
            "lookup_price (runs instantly on the server) and read data from the "
            "user's Excel file with read_excel_range (runs on the client — you "
            "must wait for the user to provide the data). "
            "Always use the tools when the user asks about prices or spreadsheet data."
        ),
        tools=[lookup_price, read_excel_range],
    )

    store = RunStore.from_database_url(db_url)

    app = FastAPI(title="Dendrux Client Tools Demo")

    # Dev-mode auth: no-op. Real apps plug in a user-session dependency
    # here — the same dep would also be applied to the write routes below
    # (e.g. ``Depends(require_user)``) so every route that touches a run
    # enforces identity + ownership consistently.
    async def allow_all() -> None:  # pragma: no cover — trivial demo dep
        return None

    # Mount Dendrux's read router — it owns run listing, run detail, events,
    # drill-downs, and the SSE stream (the hard transport piece).
    app.include_router(
        make_read_router(store=store, authorize=allow_all),
        prefix="/api",
    )

    # -- Static UI -----------------------------------------------------
    client_html = Path(__file__).parent / "client.html"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(client_html, media_type="text/html")

    # -- Dev-owned write routes ----------------------------------------
    # Production apps add auth + ownership guards on every write route,
    # e.g. ``user = Depends(require_user)`` in the signature plus an
    # ownership check against the run's owner_id before calling the agent
    # method. The demo below keeps the bodies minimal; the TODO(auth)
    # markers show where that plumbing goes.

    @app.post("/chat")
    async def start_run(body: StartChatIn) -> JSONResponse:
        # TODO(auth): add ``user = Depends(require_user)`` and stamp the
        # authenticated identity onto the run (e.g. metadata={"owner_id": user.id}).
        if not body.input:
            return JSONResponse(status_code=400, content={"error": "input is required"})
        result = await agent.run(body.input)
        return JSONResponse(
            content={
                "run_id": result.run_id,
                "status": result.status.value,
                "answer": result.answer,
            }
        )

    @app.post("/runs/{run_id}/tool-results")
    async def submit_tool_results(run_id: str, body: SubmitToolResultsIn) -> JSONResponse:
        # TODO(auth): fetch run via store, verify ownership against the
        # authenticated identity, return 404 on mismatch.
        results = [
            ToolResult(name=r.tool_name, call_id=r.tool_call_id, payload=r.payload)
            for r in body.results
        ]
        result = await agent.submit_tool_results(run_id, results)
        return JSONResponse(
            content={
                "run_id": result.run_id,
                "status": result.status.value,
                "answer": result.answer,
            }
        )

    @app.delete("/runs/{run_id}")
    async def cancel(run_id: str) -> JSONResponse:
        # TODO(auth): same ownership check as /tool-results above.
        result = await agent.cancel_run(run_id)
        return JSONResponse(content={"run_id": result.run_id, "status": result.status.value})

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="0.0.0.0", port=8000)
