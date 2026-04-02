"""SingleCall — one LLM call, no tools, no iteration.

Demonstrates the SingleCall loop for agents that don't need tools:
classification, summarization, extraction, one-turn Q&A.

Two modes shown:
  1. Batch  — agent.run() returns the full result
  2. Stream — agent.stream() yields text token by token

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/10_single_call.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, SingleCall
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import RunEventType

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

CLASSIFIER_PROMPT = """\
Classify the sentiment of the input as exactly one word: positive, negative, or neutral.
Respond with only that word, nothing else."""

SUMMARIZER_PROMPT = """\
Summarize the input in exactly one sentence. Be concise."""


async def main() -> None:
    # --- Batch: classification ---
    print("=== Batch: Sentiment Classification ===\n")

    async with Agent(
        name="Classifier",
        provider=AnthropicProvider(model="claude-haiku-4-5"),
        loop=SingleCall(),
        prompt=CLASSIFIER_PROMPT,
    ) as classifier:
        for text in [
            "I absolutely love this framework!",
            "The documentation is confusing and incomplete.",
            "The package weighs 2.3 kilograms.",
        ]:
            result = await classifier.run(text)
            print(f"  {text[:50]:50s} → {result.answer}")

    # --- Stream: summarization ---
    print("\n=== Stream: Summarization ===\n")

    async with Agent(
        name="Summarizer",
        provider=AnthropicProvider(model="claude-haiku-4-5"),
        loop=SingleCall(),
        prompt=SUMMARIZER_PROMPT,
    ) as summarizer:
        text = (
            "The Apollo 11 mission launched on July 16, 1969, from Kennedy Space Center. "
            "Neil Armstrong and Buzz Aldrin landed on the Moon on July 20, while Michael "
            "Collins orbited above. Armstrong became the first human to walk on the lunar "
            "surface, famously declaring 'That's one small step for man, one giant leap "
            "for mankind.' The crew returned safely to Earth on July 24."
        )
        print(f"  Input: {text[:80]}...\n")
        print("  Summary: ", end="", flush=True)

        async with summarizer.stream(text) as stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    print(event.text, end="", flush=True)
                elif event.type == RunEventType.RUN_COMPLETED:
                    r = event.run_result
                    print(f"\n\n  ({r.usage.total_tokens} tokens)")


if __name__ == "__main__":
    asyncio.run(main())
