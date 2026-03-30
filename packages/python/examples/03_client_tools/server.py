"""Client Tool Bridge — agent with server + client tools.

Demonstrates Dendrite's core differentiator: an agent that pauses when
it needs a client-side tool (Excel, browser, mobile), waits for the
client to execute it, then resumes reasoning.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/03_client_tools/server.py

Then open http://localhost:8000

The developer owns run creation (POST /chat). The bridge handles
everything after the first pause: tool result submission, SSE
streaming, polling, and cancellation.
"""

from __future__ import annotations

from dendrite import Agent, bridge, tool


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


def create_demo_app() -> FastAPI:  # noqa: F821
    """Create the demo FastAPI application.

    Uses the new Agent API: provider on agent, bridge() for transport.
    Developer owns run creation via POST /chat. Bridge handles paused-run
    interaction via mounted sub-app.
    """
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse

    from dendrite.llm.anthropic import AnthropicProvider

    load_dotenv(Path(__file__).resolve().parents[4] / ".env")

    db_path = Path.home() / ".dendrite" / "dendrite.db"
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

    app = FastAPI(title="Dendrite Client Tools Demo")

    # Serve the HTML client at /
    client_html = Path(__file__).parent / "client.html"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(client_html, media_type="text/html")

    # Developer-owned run creation endpoint.
    # The bridge is a paused-run interaction layer — it doesn't start runs.
    @app.post("/chat")
    async def start_run(request: dict) -> JSONResponse:  # type: ignore[type-arg]
        user_input = request.get("input", "")
        if not user_input:
            return JSONResponse(
                status_code=400,
                content={"error": "input is required"},
            )
        result = await agent.run(user_input)
        return JSONResponse(
            content={
                "run_id": result.run_id,
                "status": result.status.value,
                "answer": result.answer,
            }
        )

    # Mount the bridge — handles paused-run interaction (tool results, SSE, cancel)
    app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="0.0.0.0", port=8000)
