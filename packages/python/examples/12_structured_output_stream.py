"""Structured output + streaming — current v1 behavior.

Demonstrates that:
  1. SingleCall streaming works normally (no output_type)
  2. Structured output via batch run() works alongside streaming agents
  3. Streaming with output_type is explicitly rejected (NotImplementedError)

The pattern for v1: use agent.run() for structured output,
use agent.stream() for text streaming. Same agent class, different calls.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/12_structured_output_stream.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from dendrux import Agent, SingleCall
from dendrux.types import RunEventType

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class Analysis(BaseModel):
    summary: str
    key_points: list[str]
    word_count: int


ANALYSIS_PROMPT = """\
Analyze the input text. Provide a brief summary, extract the key points
as a list, and count the approximate number of words in the input."""

STREAMING_PROMPT = """\
Summarize the input in 2-3 sentences. Be concise and factual."""

INPUT_TEXT = (
    "Artificial intelligence has transformed multiple industries in recent years. "
    "In healthcare, AI assists with medical imaging, drug discovery, and patient "
    "diagnosis. The financial sector uses AI for fraud detection, algorithmic trading, "
    "and credit scoring. Transportation has seen advances in autonomous vehicles and "
    "route optimization. Education is being reshaped by personalized learning platforms "
    "and automated grading systems. Despite these advances, concerns persist about "
    "job displacement, algorithmic bias, and the environmental cost of training "
    "large language models."
)


async def demo_batch_structured(provider: object) -> None:
    """Batch structured output — typed Pydantic response."""
    print("=== 1. Batch Structured Output (agent.run + output_type) ===\n")

    agent = Agent(
        name="Analyzer",
        provider=provider,  # type: ignore[arg-type]
        loop=SingleCall(),
        prompt=ANALYSIS_PROMPT,
        output_type=Analysis,
    )
    result = await agent.run(INPUT_TEXT)

    a = result.output
    print(f"  summary: {a.summary}")
    print(f"  key_points: {a.key_points}")
    print(f"  word_count: {a.word_count}")
    print(f"  type: {type(a).__name__}")
    print(f"  tokens: {result.usage.total_tokens}")
    assert isinstance(a, Analysis)
    assert isinstance(a.key_points, list)
    print("  type check: passed\n")


async def demo_stream_text(provider: object) -> None:
    """Regular streaming — no output_type, token-by-token text."""
    print("=== 2. Streaming Text (agent.stream, no output_type) ===\n")

    agent = Agent(
        name="Summarizer",
        provider=provider,  # type: ignore[arg-type]
        loop=SingleCall(),
        prompt=STREAMING_PROMPT,
    )
    print("  ", end="", flush=True)
    async with agent.stream(INPUT_TEXT) as stream:
        async for event in stream:
            if event.type == RunEventType.TEXT_DELTA:
                print(event.text, end="", flush=True)
            elif event.type == RunEventType.RUN_COMPLETED:
                r = event.run_result
                print(f"\n\n  tokens: {r.usage.total_tokens}")
                assert r.output is None  # no output_type = no typed output
                print("  output is None (no output_type): correct\n")


async def demo_stream_guard() -> None:
    """Show that streaming with output_type is cleanly rejected."""
    print("=== 3. Streaming + output_type Guard ===\n")

    provider_cls = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from dendrux.llm.anthropic import AnthropicProvider

        provider_cls = AnthropicProvider
    elif os.environ.get("OPENAI_API_KEY"):
        from dendrux.llm.openai import OpenAIProvider

        provider_cls = OpenAIProvider

    if provider_cls is None:
        print("  skipped (no API key)")
        return

    try:
        agent = Agent(
            name="GuardTest",
            provider=provider_cls(
                model="gpt-4o-mini" if "OpenAI" in provider_cls.__name__ else "claude-haiku-4-5",
            ),
            loop=SingleCall(),
            prompt="Test",
            output_type=Analysis,
        )
        agent.stream("test")
        print("  ERROR: should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  caught NotImplementedError: {e}")
        print("  guard working correctly\n")
    finally:
        await agent.close()


async def main() -> None:
    # Pick a provider
    provider = None
    provider_name = ""

    if os.environ.get("ANTHROPIC_API_KEY"):
        from dendrux.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(model="claude-haiku-4-5")
        provider_name = "Anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        from dendrux.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini")
        provider_name = "OpenAI"
    else:
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run this example.")
        return

    print(f"Using provider: {provider_name}\n")

    await demo_batch_structured(provider)
    await demo_stream_text(provider)
    await demo_stream_guard()

    await provider.close()
    print("All checks passed!")


if __name__ == "__main__":
    asyncio.run(main())
