"""Streaming text — watch the LLM respond token by token.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/05_streaming_text.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import RunEventType

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


async def main() -> None:
    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a concise assistant. Keep answers under 100 words.",
        tools=[],
    ) as agent:
        stream = agent.stream("Explain what a black hole is.")
        print(f"Run ID: {stream.run_id}\n")

        async with stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    print(event.text, end="", flush=True)

                elif event.type == RunEventType.RUN_COMPLETED:
                    r = event.run_result
                    print(f"\n\n--- {r.usage.total_tokens} tokens ---")


if __name__ == "__main__":
    asyncio.run(main())
