"""Governance: Advisory Budget — token spend visibility.

Budget fires governance events as usage crosses thresholds but
does NOT pause or stop the run. Developers observe events and
take action in their own integration.

Uses a low max_tokens cap to trigger warnings quickly in a demo.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/governance/03_budget.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import Budget

load_dotenv(Path(__file__).resolve().parents[4] / ".env")


@tool()
async def search(query: str) -> str:
    """Search for information."""
    return f"Found 3 results for '{query}': result_1, result_2, result_3"


@tool()
async def summarize(text: str) -> str:
    """Summarize a piece of text."""
    return f"Summary of '{text[:50]}...': This is a concise summary."


async def main() -> None:
    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            "You are a research assistant. "
            "When asked to research a topic, first search for it, "
            "then summarize what you found. Always use both tools."
        ),
        tools=[search, summarize],
        budget=Budget(max_tokens=2000),  # Low cap — triggers warnings fast
    ) as agent:
        notifier = ConsoleNotifier()
        result = await agent.run(
            "Research the history of artificial intelligence and summarize your findings",
            notifier=notifier,
        )
        notifier.print_summary(result)

        print(f"\nAnswer: {result.answer[:200]}...")
        print(f"\nTotal tokens used: {result.usage.total_tokens}")
        print("Budget cap: 2,000 (advisory - run completed normally)")


if __name__ == "__main__":
    asyncio.run(main())
