"""Structured output — typed Pydantic responses from LLMs.

Demonstrates output_type on SingleCall: the agent returns a validated
Pydantic model in result.output instead of raw text.

Tests all three providers:
  1. Anthropic (tool-use trick internally)
  2. OpenAI Chat Completions (response_format)
  3. OpenAI Responses API (text.format)

Run with:
    ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... python examples/11_structured_output.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from dendrux import Agent, SingleCall

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


# --- Pydantic models ---


class Sentiment(BaseModel):
    label: str
    confidence: float
    reasoning: str


class MovieReview(BaseModel):
    title: str
    rating: float
    pros: list[str]
    cons: list[str]
    recommendation: str


SENTIMENT_PROMPT = """\
Classify the sentiment of the input text.
Return a label (positive, negative, or neutral), a confidence score (0 to 1),
and a brief reasoning explaining your classification."""

REVIEW_PROMPT = """\
Analyze the input as a movie review. Extract the movie title, assign a rating
out of 10, list the pros and cons mentioned, and give a one-sentence recommendation."""


async def run_sentiment(name: str, provider: object) -> None:
    """Test sentiment classification with a provider."""
    print(f"\n  [{name}] Sentiment Classification")

    agent = Agent(
        name=f"Sentiment-{name}",
        provider=provider,  # type: ignore[arg-type]
        loop=SingleCall(),
        prompt=SENTIMENT_PROMPT,
        output_type=Sentiment,
    )
    texts = [
        "I absolutely love this framework! It makes everything so easy.",
        "The documentation is confusing, the API keeps breaking, and support is nonexistent.",
        "The package weighs 2.3 kilograms and ships in a brown box.",
    ]
    for text in texts:
        result = await agent.run(text)
        s = result.output
        print(f"    {text[:55]:55s}")
        print(f"      label={s.label}, confidence={s.confidence:.2f}")
        print(f"      reasoning: {s.reasoning[:80]}")
        print(f"      tokens: {result.usage.total_tokens}")
        print()


async def run_review(name: str, provider: object) -> None:
    """Test movie review extraction with a provider."""
    print(f"\n  [{name}] Movie Review Extraction")

    agent = Agent(
        name=f"Review-{name}",
        provider=provider,  # type: ignore[arg-type]
        loop=SingleCall(),
        prompt=REVIEW_PROMPT,
        output_type=MovieReview,
    )
    review_text = (
        "Just saw Oppenheimer and wow. Cillian Murphy's performance is haunting, "
        "the cinematography is breathtaking, and Nolan's direction is masterful. "
        "The 3-hour runtime does drag a bit in the middle, and the timeline "
        "jumping can be confusing. But overall, an incredible film. Must watch."
    )
    result = await agent.run(review_text)
    r = result.output
    print(f"    title: {r.title}")
    print(f"    rating: {r.rating}/10")
    print(f"    pros: {r.pros}")
    print(f"    cons: {r.cons}")
    print(f"    recommendation: {r.recommendation}")
    print(f"    tokens: {result.usage.total_tokens}")

    # Verify types are correct
    assert isinstance(r, MovieReview)
    assert isinstance(r.rating, float)
    assert isinstance(r.pros, list)
    print("    type check: passed")


async def run_per_call_override(name: str, provider: object) -> None:
    """Test output_type override on run()."""
    print(f"\n  [{name}] Per-call output_type override")

    # Agent has no output_type, but run() passes one
    agent = Agent(
        name=f"Override-{name}",
        provider=provider,  # type: ignore[arg-type]
        loop=SingleCall(),
        prompt=SENTIMENT_PROMPT,
    )
    result = await agent.run(
        "This is absolutely wonderful!",
        output_type=Sentiment,
    )
    print(f"    output_type on run(): {result.output}")
    assert isinstance(result.output, Sentiment)

    # Without output_type, result.output is None
    result2 = await agent.run("This is wonderful!")
    print(f"    no output_type: output={result2.output}, answer={result2.answer[:50]}...")
    assert result2.output is None
    print("    type check: passed")


async def main() -> None:
    providers: list[tuple[str, object]] = []

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        from dendrux.llm.anthropic import AnthropicProvider

        providers.append(("Anthropic", AnthropicProvider(model="claude-haiku-4-5")))
    else:
        print("  ANTHROPIC_API_KEY not set, skipping Anthropic")

    # OpenAI Chat Completions
    if os.environ.get("OPENAI_API_KEY"):
        from dendrux.llm.openai import OpenAIProvider

        providers.append(("OpenAI-Chat", OpenAIProvider(model="gpt-4o-mini")))
    else:
        print("  OPENAI_API_KEY not set, skipping OpenAI Chat")

    # OpenAI Responses API
    if os.environ.get("OPENAI_API_KEY"):
        from dendrux.llm.openai_responses import OpenAIResponsesProvider

        providers.append(("OpenAI-Responses", OpenAIResponsesProvider(model="gpt-4o-mini")))

    if not providers:
        print("No API keys set. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.")
        return

    for name, provider in providers:
        print(f"\n{'=' * 60}")
        print(f"  Provider: {name}")
        print(f"{'=' * 60}")

        await run_sentiment(name, provider)
        await run_review(name, provider)
        await run_per_call_override(name, provider)

    # Cleanup
    for _, provider in providers:
        await provider.close()  # type: ignore[union-attr]

    print(f"\n{'=' * 60}")
    print("  All structured output tests passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
