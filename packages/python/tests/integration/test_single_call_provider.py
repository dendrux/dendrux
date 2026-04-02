"""Integration test: SingleCall with a real Anthropic provider.

Skipped when ANTHROPIC_API_KEY is not set — opt-in for CI and local dev.
"""

from __future__ import annotations

import os

import pytest

from dendrux import Agent, SingleCall

requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@requires_anthropic_key
class TestSingleCallAnthropic:
    async def test_classification_batch(self) -> None:
        """SingleCall agent classifies sentiment via real Claude call."""
        from dendrux.llm.anthropic import AnthropicProvider

        agent = Agent(
            provider=AnthropicProvider(model="claude-haiku-4-5"),
            loop=SingleCall(),
            prompt=(
                "Classify the sentiment of the input as exactly one word: "
                "positive, negative, or neutral. Respond with only that word."
            ),
        )

        async with agent:
            result = await agent.run("I absolutely love this framework!")

        assert result.status.value == "success"
        assert result.answer is not None
        assert "positive" in result.answer.lower()
        assert result.iteration_count == 1
        assert result.usage.total_tokens > 0

    async def test_classification_stream(self) -> None:
        """SingleCall streaming via real Claude — text deltas arrive."""
        from dendrux.llm.anthropic import AnthropicProvider
        from dendrux.types import RunEventType

        agent = Agent(
            provider=AnthropicProvider(model="claude-haiku-4-5"),
            loop=SingleCall(),
            prompt=(
                "Classify the sentiment of the input as exactly one word: "
                "positive, negative, or neutral. Respond with only that word."
            ),
        )

        events = []
        async with agent:
            async for event in agent.stream("This is terrible and broken."):
                events.append(event)

        # Should have at least one TEXT_DELTA and one RUN_COMPLETED
        event_types = [e.type for e in events]
        assert RunEventType.RUN_STARTED in event_types
        assert RunEventType.TEXT_DELTA in event_types
        assert RunEventType.RUN_COMPLETED in event_types

        completed = events[-1]
        assert completed.run_result is not None
        assert completed.run_result.status.value == "success"
        assert completed.run_result.answer is not None
        assert "negative" in completed.run_result.answer.lower()
