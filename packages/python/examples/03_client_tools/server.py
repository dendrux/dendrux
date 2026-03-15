"""Client Tool Bridge — hosted agent with server + client tools.

Demonstrates Dendrite's core differentiator: an agent that pauses when
it needs a client-side tool (Excel, browser, mobile), waits for the
client to execute it, then resumes reasoning.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/03_client_tools/server.py

Then open http://localhost:8000 in your browser.

Try a prompt like:
    "Look up AAPL price, then read cell A1 from the user's spreadsheet."

The agent will call lookup_price (server-side, instant) and then pause
for read_excel_range (client-side, you provide the result in the UI).
"""

from __future__ import annotations

from dendrite import Agent, tool


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
    # Never executed server-side — the loop pauses before calling this.
    return ""


agent = Agent(
    name="SpreadsheetAnalyst",
    model="claude-sonnet-4-6",
    prompt=(
        "You are a spreadsheet analyst. You can look up stock prices with "
        "lookup_price (runs instantly on the server) and read data from the "
        "user's Excel file with read_excel_range (runs on the client — you "
        "must wait for the user to provide the data). "
        "Always use the tools when the user asks about prices or spreadsheet data."
    ),
    tools=[lookup_price, read_excel_range],
)


def create_demo_app(
    *,
    state_store: object | None = None,
    registry: object | None = None,
) -> FastAPI:  # noqa: F821
    """Create the demo FastAPI application.

    Import-safe: provider construction and env lookup happen inside
    factories, not at import time. No side effects on import.

    Args:
        state_store: Override the state store (for testing with MockServerStore).
            When None, a lazy proxy defers to lifespan-initialized SQLAlchemy store.
        registry: Override the agent registry (for testing with MockLLM).
            When None, uses AnthropicProvider with ANTHROPIC_API_KEY from env.
    """
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    from dendrite.server import create_app

    if registry is None:
        registry = _build_default_registry()

    # When no store override is provided, use a lazy proxy so the app
    # can be assembled eagerly (routes available immediately) while
    # deferring async DB init to lifespan.
    store_holder: dict[str, object] = {}
    using_lazy_store = state_store is None

    if using_lazy_store:

        class _LazyStore:
            """Proxy that delegates to the real store after lifespan init."""

            def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
                return getattr(store_holder["store"], name)

        state_store = _LazyStore()

    # Mount Dendrite eagerly — routes are registered now, not at startup
    dendrite_app = create_app(
        state_store=state_store,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        allow_insecure_dev_mode=True,
    )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        if using_lazy_store:
            from dendrite.db.session import get_engine
            from dendrite.runtime.state import SQLAlchemyStateStore

            engine = await get_engine()
            store_holder["store"] = SQLAlchemyStateStore(engine)
        yield

    app = FastAPI(title="Dendrite Client Tools Demo", lifespan=lifespan)

    # Serve the HTML client at /
    client_html = Path(__file__).parent / "client.html"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(client_html, media_type="text/html")

    app.mount("/dendrite", dendrite_app)

    return app


def _build_default_registry() -> AgentRegistry:  # noqa: F821
    """Build the default registry with AnthropicProvider. Defers env lookup to factory."""
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from dendrite.llm.anthropic import AnthropicProvider
    from dendrite.server import AgentRegistry, HostedAgentConfig

    # .env at repo root (5 levels up from this file)
    load_dotenv(Path(__file__).resolve().parents[4] / ".env")

    registry = AgentRegistry()
    registry.register(
        HostedAgentConfig(
            agent=agent,
            provider_factory=lambda: AnthropicProvider(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                model=agent.model,
            ),
        )
    )
    return registry


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="0.0.0.0", port=8000)
