"""Persistent Agent — run with SQLite state persistence.

Demonstrates Dendrite's built-in persistence layer: agent runs, traces,
tool calls, and token usage are all stored in a local SQLite database.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/02_persistent_agent.py

After running, inspect the data with:
    dendrite runs
    dendrite traces <run_id> --tools
"""

from dendrite import Agent, tool


@tool()
async def lookup_price(ticker: str) -> str:
    """Look up the current stock price for a ticker symbol."""
    # Simulated prices for demo purposes
    prices = {"AAPL": 227.50, "GOOGL": 178.30, "MSFT": 445.20, "TSLA": 312.80}
    price = prices.get(ticker.upper())
    if price is None:
        return f"Unknown ticker: {ticker}"
    return f"{ticker.upper()}: ${price:.2f}"


@tool()
async def calculate_portfolio_value(holdings: str) -> str:
    """Calculate total portfolio value from a comma-separated list of 'TICKER:SHARES' pairs."""
    prices = {"AAPL": 227.50, "GOOGL": 178.30, "MSFT": 445.20, "TSLA": 312.80}
    total = 0.0
    breakdown = []
    for item in holdings.split(","):
        ticker, shares = item.strip().split(":")
        price = prices.get(ticker.upper(), 0)
        value = price * int(shares)
        total += value
        breakdown.append(f"  {ticker.upper()}: {shares} × ${price:.2f} = ${value:,.2f}")
    return "Portfolio breakdown:\n" + "\n".join(breakdown) + f"\n  Total: ${total:,.2f}"


agent = Agent(
    name="StockAnalyst",
    model="claude-sonnet-4-6",
    prompt=(
        "You are a stock analyst assistant. Use the lookup_price tool to check "
        "stock prices and calculate_portfolio_value to compute portfolio totals. "
        "Always look up prices before making calculations."
    ),
    tools=[lookup_price, calculate_portfolio_value],
)


if __name__ == "__main__":
    import asyncio
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel

    from dendrite import run
    from dendrite.db.session import get_engine
    from dendrite.llm.anthropic import AnthropicProvider
    from dendrite.runtime.state import SQLAlchemyStateStore

    # Load .env from repo root
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    console = Console()

    async def main() -> None:
        provider = AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=agent.model,
        )

        # Initialize the database engine (auto-creates dendrite.db with tables)
        engine = await get_engine()
        store = SQLAlchemyStateStore(engine)

        console.print(Panel("[bold]Persistent Agent Demo[/bold]", border_style="cyan"))
        console.print()

        # Run with persistence enabled — all traces, tool calls, and usage
        # are automatically stored in the local SQLite database.
        result = await run(
            agent,
            provider=provider,
            user_input="What's my portfolio worth? I have 10 AAPL, 5 GOOGL, and 20 MSFT.",
            state_store=store,
        )

        # Print the result
        console.print(f"\n[bold green]Answer:[/bold green] {result.answer}")
        console.print(
            f"\n[dim]Completed in {result.iteration_count} iterations, "
            f"{result.usage.total_tokens} tokens[/dim]"
        )
        console.print(f"[dim]Run ID: {result.run_id}[/dim]")
        console.print(
            "\n[bold]Inspect persisted data with:[/bold]"
            f"\n  dendrite runs"
            f"\n  dendrite traces {result.run_id} --tools"
            "\n"
            "\n[bold]Or view the full timeline in the dashboard:[/bold]"
            "\n  dendrite dashboard --open"
        )

    asyncio.run(main())
