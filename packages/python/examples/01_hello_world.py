"""Hello World — minimal Dendrux agent with a tool.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/01_hello_world.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider

# Load .env from repo root (three levels up: examples/ → python/ → packages/ → dendrux/)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


async def main() -> None:
    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a helpful calculator. Use the add tool when asked to add numbers.",
        tools=[add],
    ) as agent:
        result = await agent.run("What is 15 + 27?")
        print(f"Answer: {result.answer}")
        print(f"Steps: {result.iteration_count}, Tokens: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
