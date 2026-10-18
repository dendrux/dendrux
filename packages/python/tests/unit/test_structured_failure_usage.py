"""A structured call whose output fails validation is still recorded and priced (#26)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from dendrux import Agent, ModelPricing, PriceTable, SingleCall
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    StructuredOutputValidationError,
    UsageStats,
)


def _usage() -> UsageStats:
    return UsageStats(input_tokens=100, output_tokens=10, total_tokens=110)


class Out(BaseModel):
    label: str


def _structured(*responses: LLMResponse) -> MockLLM:
    llm = MockLLM(list(responses))
    llm.capabilities = ProviderCapabilities(supports_structured_output=True)
    return llm


class _Calls(BaseNotifier):
    def __init__(self) -> None:
        self.completed: list[LLMResponse] = []
        self.failed: list[BaseException] = []

    async def on_llm_call_completed(self, run_id: str, response: Any, iteration: int, **_: Any):
        self.completed.append(response)

    async def on_llm_call_failed(self, run_id: str, iteration: int, error: Any, **_: Any):
        self.failed.append(error)


class TestFailedStructuredCall:
    async def test_error_carries_priced_usage_and_model(self) -> None:
        llm = _structured(LLMResponse(text="nope", usage=_usage()))
        notifier = _Calls()
        agent = Agent(
            provider=llm,
            prompt="t",
            loop=SingleCall(),
            output_type=Out,
            pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}),
        )
        with pytest.raises(StructuredOutputValidationError) as info:
            await agent.run("hi", notifier=notifier)
        err = info.value
        assert err.model == "mock"
        assert err.usage is not None
        assert err.usage.output_tokens == 10
        assert err.usage.cost_usd == pytest.approx(120 / 1_000_000)
        assert err.usage.cost_source == "table"
        # Recorded as a completed (priced) call, not a failed one.
        assert len(notifier.completed) == 1
        assert notifier.completed[0].usage.cost_usd == pytest.approx(120 / 1_000_000)
        assert notifier.completed[0].text == "nope"
        assert notifier.failed == []

    async def test_structured_complete_attaches_unpriced_usage(self) -> None:
        from dendrux.llm.structured import structured_complete
        from dendrux.types import Message, Role

        llm = _structured(LLMResponse(text="nope", usage=_usage()))
        with pytest.raises(StructuredOutputValidationError) as info:
            await structured_complete(llm, [Message(role=Role.USER, content="x")], Out)
        assert info.value.usage == _usage()
        assert info.value.model == "mock"
