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

    async def test_error_exposes_stop_reason(self) -> None:
        from dendrux.llm.structured import structured_complete
        from dendrux.types import Message, Role

        llm = _structured(LLMResponse(text='{"lab', usage=_usage(), stop_reason="max_tokens"))
        with pytest.raises(StructuredOutputValidationError) as info:
            await structured_complete(llm, [Message(role=Role.USER, content="x")], Out)
        assert info.value.stop_reason == "max_tokens"

    def test_stop_reason_is_keyword_only_and_optional(self) -> None:
        assert StructuredOutputValidationError("x", "Out", "bad").stop_reason is None
        err = StructuredOutputValidationError("x", "Out", "bad", stop_reason="max_tokens")
        assert err.stop_reason == "max_tokens"


def _rejected(
    text: str = "secret sk-live-123 and alice@example.com",
) -> StructuredOutputValidationError:
    response = LLMResponse(
        text=text,
        usage=_usage(),
        model="mock",
        stop_reason="max_tokens",
        raw_stop_reason="length",
        provider_request={"messages": []},
        provider_response={"choices": [{"message": {"content": text}}]},
    )
    err = StructuredOutputValidationError(
        text, "Out", "bad", usage=response.usage, model="mock", stop_reason="max_tokens"
    )
    err._response = response
    return err


async def _record(agent: Agent) -> tuple[LLMResponse, StructuredOutputValidationError]:
    from dendrux.loops.single import _record_rejected_call

    notifier = _Calls()
    err = _rejected()
    await _record_rejected_call(
        err,
        agent=agent,
        provider=agent.provider,
        run_id="run1",
        recorder=None,
        notifier=notifier,
        warnings=[],
        semantic_messages=None,
        duration_ms=1,
    )
    assert len(notifier.completed) == 1
    return notifier.completed[0], err


class TestRejectedCallGuardrails:
    """With output guardrails, the rejected output is never recorded raw."""

    async def test_scrubbed_when_guardrails_configured(self) -> None:
        from dendrux.guardrails import PII

        agent = Agent(
            provider=MockLLM([]),
            prompt="t",
            loop=SingleCall(),
            guardrails=[PII()],
            pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}),
        )
        recorded, err = await _record(agent)
        assert recorded.text == "[structured output rejected]"
        assert recorded.provider_response is None
        assert recorded.tool_calls is None
        # Accounting survives the scrub.
        assert recorded.usage.output_tokens == 10
        assert recorded.usage.cost_usd == pytest.approx(120 / 1_000_000)
        assert recorded.model == "mock"
        assert recorded.stop_reason == "max_tokens"
        assert recorded.raw_stop_reason == "length"
        assert err.usage is not None and err.usage.cost_usd == pytest.approx(120 / 1_000_000)

    async def test_raw_without_guardrails(self) -> None:
        agent = Agent(
            provider=MockLLM([]),
            prompt="t",
            loop=SingleCall(),
            pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}),
        )
        recorded, _ = await _record(agent)
        assert recorded.text == "secret sk-live-123 and alice@example.com"
        assert recorded.provider_response is not None
        assert recorded.stop_reason == "max_tokens"
        assert recorded.usage.cost_usd == pytest.approx(120 / 1_000_000)
