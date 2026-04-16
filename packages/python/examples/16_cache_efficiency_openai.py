"""Cache efficiency demo for OpenAI — Chat Completions and Responses APIs.

OpenAI prompt caching works differently from Anthropic:
  - It is automatic. No cache_control markers needed.
  - The minimum cacheable prefix is 1,024 tokens (smaller than Anthropic).
  - Only cache READS show up in usage. There is no cache_creation field
    because OpenAI caches in the background — you get a discounted read
    on subsequent calls without an explicit write step.
  - Dendrux passes a stable `prompt_cache_key` derived from
    `agent_name:model` so all iterations of the same run (and all runs
    of the same agent + model) route to the same cache pool.

This example runs the same portfolio agent twice — once via Chat
Completions, once via the Responses API — so you can see cache hits
on both endpoints.

Run with:
    OPENAI_API_KEY=sk-... python examples/16_cache_efficiency_openai.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from dendrux import Agent, tool
from dendrux.llm.openai import OpenAIProvider
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.notifiers.console import ConsoleNotifier

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

console = Console()


# ---------------------------------------------------------------------------
# Verbose system prompt — needs to be ~1024+ tokens so OpenAI caches it.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a meticulous portfolio analyst working for a long-term value
investing firm. Your job is to look up market data, compute portfolio
metrics, and explain results in plain language to a non-technical
investor.

# Operating principles

You always work step by step. You never skip a tool call to save effort.
You never invent prices, volumes, market caps, or any other quantitative
fact. If a tool call fails or returns no data, you say so explicitly and
stop instead of guessing. You prefer fewer correct answers over more
detailed wrong ones.

You carry forward all numerical results between steps verbatim — you do
not round, abbreviate, or restate figures unless explicitly asked. When
the user asks for a derived metric (a ratio, a delta, a percentage), you
compute it from the source numbers you already have, not from memory.

# Tool usage discipline

The tools below are the only source of truth for prices, ratios, and
holdings data. Always use them in this order when answering portfolio
questions:

1. Look up each ticker's current price with `lookup_price`.
2. Look up each ticker's fundamentals (P/E ratio, dividend yield, market
   cap) with `lookup_fundamentals` if the question touches on valuation,
   income, or company size.
3. Compute portfolio totals with `calculate_portfolio_value` only after
   you have all relevant prices. Pass holdings as a comma-separated list
   of `TICKER:SHARES` pairs.
4. Compute weighted-average ratios with `weighted_average_pe` when the
   user asks about portfolio-level valuation.

You do not call the same tool twice with the same arguments. If you need
the same data again, refer to the result you already have in your earlier
working notes.

# Output format

Your final answer must include:

- One short sentence stating the headline result.
- A bulleted list of the underlying numbers, each labeled with the
  source ticker and the unit ($, %, x, etc.).
- One sentence of plain-language interpretation suitable for someone who
  does not read 10-K filings for fun.
- If any data was missing or any tool returned an error, a final line
  starting with `Caveat:` describing what was incomplete.

You never use Markdown headings, never use bold or italic emphasis, and
never use emoji. Bullets are fine. Tables are not.

# Style guardrails

You write in calm, declarative sentences. You avoid hedging language
("might", "could", "perhaps") when stating facts that came from a tool
call — those are facts, not guesses. You use hedging language correctly
when interpreting facts (drawing investment conclusions from data is
inherently uncertain, and you should signal that).

You do not recommend specific buy/sell actions. You explain what the
numbers mean and let the investor decide. If asked directly for a
recommendation, you respond that you provide analysis, not advice, and
you suggest the investor speak to a licensed advisor.

# Error handling

If a ticker is unknown, you stop and report which ticker failed. You do
not silently substitute a similar ticker. You do not retry the same
lookup more than once.

If `calculate_portfolio_value` is called before all prices have been
looked up, the prices it falls back to may be stale — explicitly call
`lookup_price` for each ticker first.

# Examples of correct behavior

User: "What's my AAPL position worth at 50 shares?"
You: lookup_price("AAPL") → $227.50, then state "Your AAPL position is
worth $11,375.00 at the current price." with a bullet showing the math.

User: "Compare GOOGL and MSFT on valuation."
You: lookup_fundamentals("GOOGL") and lookup_fundamentals("MSFT") in
order, then present both companies' P/E ratios side by side and explain
in one sentence which is cheaper on a price-to-earnings basis and what
that means for a long-term investor.

# Examples of incorrect behavior to avoid

Do not skip the price lookup and use a "typical" price from training
data. Do not estimate market cap from share price alone. Do not call
calculate_portfolio_value before looking up the constituent tickers'
prices. Do not present results in a table. Do not use Markdown headings
in your final answer."""


_PRICES = {
    "AAPL": 227.50,
    "GOOGL": 178.30,
    "MSFT": 445.20,
    "TSLA": 312.80,
    "NVDA": 1280.40,
}

_FUNDAMENTALS = {
    "AAPL": {"pe": 33.4, "dividend_yield": 0.45, "market_cap_b": 3450},
    "GOOGL": {"pe": 26.1, "dividend_yield": 0.00, "market_cap_b": 2210},
    "MSFT": {"pe": 36.7, "dividend_yield": 0.71, "market_cap_b": 3310},
    "TSLA": {"pe": 78.5, "dividend_yield": 0.00, "market_cap_b": 998},
    "NVDA": {"pe": 65.3, "dividend_yield": 0.03, "market_cap_b": 3140},
}


@tool()
async def lookup_price(ticker: str) -> str:
    """Look up the current stock price for a ticker symbol."""
    price = _PRICES.get(ticker.upper())
    if price is None:
        return f"Unknown ticker: {ticker}"
    return f"{ticker.upper()}: ${price:.2f}"


@tool()
async def lookup_fundamentals(ticker: str) -> str:
    """Look up P/E ratio, dividend yield, and market cap for a ticker."""
    f = _FUNDAMENTALS.get(ticker.upper())
    if f is None:
        return f"Unknown ticker: {ticker}"
    return (
        f"{ticker.upper()} — P/E {f['pe']:.1f}x, "
        f"dividend yield {f['dividend_yield']:.2f}%, "
        f"market cap ${f['market_cap_b']}B"
    )


@tool()
async def calculate_portfolio_value(holdings: str) -> str:
    """Total portfolio value from a comma-separated 'TICKER:SHARES' list."""
    total = 0.0
    breakdown = []
    for item in holdings.split(","):
        ticker, shares = item.strip().split(":")
        price = _PRICES.get(ticker.upper(), 0)
        value = price * int(shares)
        total += value
        breakdown.append(f"  {ticker.upper()}: {shares} × ${price:.2f} = ${value:,.2f}")
    return "Portfolio breakdown:\n" + "\n".join(breakdown) + f"\n  Total: ${total:,.2f}"


@tool()
async def weighted_average_pe(holdings: str) -> str:
    """Weighted average P/E across a portfolio (weighted by position value)."""
    total_value = 0.0
    weighted_pe = 0.0
    for item in holdings.split(","):
        ticker, shares = item.strip().split(":")
        ticker = ticker.upper()
        price = _PRICES.get(ticker)
        f = _FUNDAMENTALS.get(ticker)
        if price is None or f is None:
            return f"Unknown ticker in holdings: {ticker}"
        value = price * int(shares)
        total_value += value
        weighted_pe += f["pe"] * value
    if total_value == 0:
        return "Cannot compute weighted P/E for an empty portfolio."
    avg = weighted_pe / total_value
    return f"Weighted-average portfolio P/E: {avg:.2f}x (across ${total_value:,.2f})"


_QUESTION = (
    "I hold 50 AAPL, 20 GOOGL, 30 MSFT, and 10 NVDA. "
    "What's the portfolio worth and how does its weighted P/E compare "
    "to MSFT's standalone P/E?"
)


async def _run_with(provider, label: str) -> None:
    notifier = ConsoleNotifier(show_params=True)

    console.print()
    console.print(Panel(f"[bold]{label}[/bold]", border_style="cyan", width=60))

    async with Agent(
        name="PortfolioAnalyst",
        provider=provider,
        prompt=SYSTEM_PROMPT,
        tools=[
            lookup_price,
            lookup_fundamentals,
            calculate_portfolio_value,
            weighted_average_pe,
        ],
        max_iterations=10,
    ) as agent:
        result = await agent.run(_QUESTION, notifier=notifier)
        notifier.print_summary(result)


async def main() -> None:
    # Chat Completions API — broadly compatible with OpenAI-compatible
    # backends (Groq, Together, vLLM, Ollama). Cache routing kwargs
    # are skipped on non-default base_urls.
    await _run_with(
        OpenAIProvider(model="gpt-4o"),
        "OpenAI Chat Completions (gpt-4o)",
    )

    # Responses API — newer endpoint, supports built-in tools and
    # reasoning models. Same prompt_cache_key routing.
    await _run_with(
        OpenAIResponsesProvider(model="gpt-4o"),
        "OpenAI Responses API (gpt-4o)",
    )


if __name__ == "__main__":
    asyncio.run(main())
