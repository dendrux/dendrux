"""Loop-level stop reason and truncation flag (#42)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from dendrux import Agent, SingleCall, tool
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    RunEventType,
    RunStatus,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    import pytest


def _usage() -> UsageStats:
    return UsageStats(input_tokens=100, output_tokens=10, total_tokens=110)


def _cut(text: str = "The answer is") -> LLMResponse:
    return LLMResponse(text=text, usage=_usage(), stop_reason="max_tokens")


@tool()
async def ping() -> str:
    """Ping."""
    return "pong"


class TestReActStopReason:
    async def test_truncated_final_answer_is_success_and_flagged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = Agent(provider=MockLLM([_cut()]), prompt="t")
        with caplog.at_level(logging.WARNING):
            result = await agent.run("hi")
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "The answer is"
        assert result.stop_reason == "max_tokens"
        assert result.truncated is True
        warnings = [r for r in caplog.records if "cut off" in r.getMessage()]
        assert len(warnings) == 1
        assert result.run_id in warnings[0].getMessage()
        assert "mock" in warnings[0].getMessage()

    async def test_end_turn_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = LLMResponse(text="done", usage=_usage(), stop_reason="end_turn")
        with caplog.at_level(logging.WARNING):
            result = await Agent(provider=MockLLM([resp]), prompt="t").run("hi")
        assert result.stop_reason == "end_turn"
        assert result.truncated is False
        assert not [r for r in caplog.records if "cut off" in r.getMessage()]

    async def test_reports_the_final_call_not_earlier_ones(self) -> None:
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("ping", {})], stop_reason="tool_use"),
                LLMResponse(text="pong", stop_reason="end_turn"),
            ]
        )
        result = await Agent(provider=llm, prompt="t", tools=[ping]).run("hi")
        assert result.stop_reason == "end_turn"

    async def test_unreported_stop_reason_is_none(self) -> None:
        result = await Agent(provider=MockLLM([LLMResponse(text="x")]), prompt="t").run("hi")
        assert result.stop_reason is None
        assert result.truncated is False

    async def test_stream(self) -> None:
        agent = Agent(provider=MockLLM([_cut()]), prompt="t")
        async with agent.stream("hi") as stream:
            events = [e async for e in stream]
        final = events[-1]
        assert final.type == RunEventType.RUN_COMPLETED
        assert final.truncated is True
        assert final.run_result is not None
        assert final.run_result.stop_reason == "max_tokens"

    async def test_non_success_results_carry_no_stop_reason(self) -> None:
        llm = MockLLM([LLMResponse(tool_calls=[ToolCall("ping", {})], stop_reason="max_tokens")])
        result = await Agent(provider=llm, prompt="t", tools=[ping], max_iterations=1).run("hi")
        assert result.status == RunStatus.MAX_ITERATIONS
        assert result.stop_reason is None


class Out(BaseModel):
    label: str


def _structured(*responses: LLMResponse) -> MockLLM:
    llm = MockLLM(list(responses))
    llm.capabilities = ProviderCapabilities(supports_structured_output=True)
    return llm


class TestSingleCallStopReason:
    async def test_plain_text(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = Agent(provider=MockLLM([_cut()]), prompt="t", loop=SingleCall())
        with caplog.at_level(logging.WARNING):
            result = await agent.run("hi")
        assert result.status == RunStatus.SUCCESS
        assert result.truncated is True
        assert [r for r in caplog.records if "cut off" in r.getMessage()]

    async def test_structured(self) -> None:
        llm = _structured(LLMResponse(text='{"label": "ok"}', stop_reason="end_turn"))
        agent = Agent(provider=llm, prompt="t", loop=SingleCall(), output_type=Out)
        result = await agent.run("hi")
        assert result.output == Out(label="ok")
        assert result.stop_reason == "end_turn"

    async def test_stream(self) -> None:
        agent = Agent(provider=MockLLM([_cut()]), prompt="t", loop=SingleCall())
        async with agent.stream("hi") as stream:
            events = [e async for e in stream]
        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert events[-1].stop_reason == "max_tokens"
        assert events[-1].truncated is True


async def test_context_window_cut_answer_is_truncated(caplog: pytest.LogCaptureFixture) -> None:
    """Anthropic ``model_context_window_exceeded`` flags the run truncated and warns."""
    import logging

    from dendrux.llm._stop_reason import anthropic_stop_reason

    reason, raw = anthropic_stop_reason("model_context_window_exceeded")
    resp = LLMResponse(text="cut", usage=_usage(), stop_reason=reason, raw_stop_reason=raw)
    with caplog.at_level(logging.WARNING):
        result = await Agent(provider=MockLLM([resp]), prompt="t").run("hi")
    assert result.status == RunStatus.SUCCESS
    assert result.truncated is True
    assert [r for r in caplog.records if "cut off" in r.getMessage()]
