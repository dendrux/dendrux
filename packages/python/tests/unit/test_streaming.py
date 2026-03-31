"""Tests for run-level streaming — agent.stream() and the run_stream pipeline.

Covers:
  - RunStream lifecycle (single-use, early break, idempotent cleanup)
  - Event flow (RUN_STARTED, TEXT_DELTA, TOOL_USE_END, TOOL_RESULT, RUN_COMPLETED)
  - Error semantics (yield RUN_ERROR, no re-raise)
  - kwargs forwarding (agent → runner → loop → provider)
  - .text() convenience filter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dendrux import Agent, run, tool
from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    RunEventType,
    RunStatus,
    RunStream,
    StreamEvent,
    StreamEventType,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class StreamingMockLLM(LLMProvider):
    """Mock that yields real streaming events (token-by-token text deltas).

    Unlike the base class fallback (which yields one big TEXT_DELTA),
    this splits text into individual characters for realistic streaming tests.
    """

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_streaming=True,
    )

    def __init__(self, responses: list[LLMResponse], *, model: str = "mock-stream") -> None:
        self._responses = list(responses)
        self._model = model
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            raise IndexError("StreamingMockLLM exhausted")
        self.call_history.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    async def complete_stream(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._call_count >= len(self._responses):
            raise IndexError("StreamingMockLLM exhausted")
        self.call_history.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        resp = self._responses[self._call_count]
        self._call_count += 1

        # Yield text as individual character deltas
        if resp.text:
            for char in resp.text:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=char)

        # Yield tool calls as TOOL_USE_START + TOOL_USE_END
        if resp.tool_calls:
            for tc in resp.tool_calls:
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_name=tc.name,
                    tool_call_id=tc.provider_tool_call_id or tc.id,
                )
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name=tc.name,
                    tool_call_id=tc.provider_tool_call_id or tc.id,
                )

        yield StreamEvent(type=StreamEventType.DONE, raw=resp)


class ErrorLLM(LLMProvider):
    """LLM that raises on every call. Uses the base class complete_stream()
    fallback which calls complete() — so the error propagates through both paths."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def model(self) -> str:
        return "error-mock"

    async def complete(
        self, messages: list[Any], tools: list[Any] | None = None, **kw: Any,
    ) -> LLMResponse:
        raise self._error


# ------------------------------------------------------------------
# RunStream unit tests
# ------------------------------------------------------------------


class TestRunStreamLifecycle:
    async def test_single_use_guard(self) -> None:
        """Second iteration raises RuntimeError."""
        from dendrux.types import RunEvent, RunResult

        async def _gen() -> AsyncGenerator[RunEvent, None]:
            yield RunEvent(
                type=RunEventType.RUN_COMPLETED,
                run_result=RunResult(run_id="test", status=RunStatus.SUCCESS),
            )

        stream = RunStream(run_id="test", generator=_gen(), cleanup=_noop_cleanup)
        async for _ in stream:
            pass

        with pytest.raises(RuntimeError, match="single-use"):
            async for _ in stream:
                pass

    async def test_run_id_available_before_iteration(self) -> None:
        """run_id is accessible immediately on the RunStream object."""
        from dendrux.types import RunEvent, RunResult

        async def _gen() -> AsyncGenerator[RunEvent, None]:
            yield RunEvent(
                type=RunEventType.RUN_COMPLETED,
                run_result=RunResult(run_id="test", status=RunStatus.SUCCESS),
            )

        stream = RunStream(run_id="my-run-123", generator=_gen(), cleanup=_noop_cleanup)
        assert stream.run_id == "my-run-123"

    async def test_cleanup_fires_on_explicit_aclose(self) -> None:
        """Explicit aclose() after break triggers cleanup."""
        from dendrux.types import RunEvent

        cleanup_called = False

        async def _cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        async def _gen() -> AsyncGenerator[RunEvent, None]:
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="hi")
            yield RunEvent(type=RunEventType.TEXT_DELTA, text=" world")

        stream = RunStream(run_id="test", generator=_gen(), cleanup=_cleanup)
        async for _event in stream:
            break  # Consumer abandons early

        await stream.aclose()
        assert cleanup_called

    async def test_cleanup_fires_with_async_with(self) -> None:
        """async with guarantees immediate deterministic cleanup on break."""
        from dendrux.types import RunEvent

        cleanup_called = False

        async def _cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        async def _gen() -> AsyncGenerator[RunEvent, None]:
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="hi")
            yield RunEvent(type=RunEventType.TEXT_DELTA, text=" world")

        stream = RunStream(run_id="test", generator=_gen(), cleanup=_cleanup)
        async with stream:
            async for _event in stream:
                break  # __aexit__ will call aclose()

        assert cleanup_called

    async def test_cleanup_skipped_after_terminal_event(self) -> None:
        """If stream reaches a terminal event, cleanup does NOT cancel the run."""
        from dendrux.types import RunEvent, RunResult

        cleanup_called = False

        async def _cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        async def _gen() -> AsyncGenerator[RunEvent, None]:
            yield RunEvent(
                type=RunEventType.RUN_COMPLETED,
                run_result=RunResult(run_id="test", status=RunStatus.SUCCESS, answer="done"),
            )

        stream = RunStream(run_id="test", generator=_gen(), cleanup=_cleanup)
        async for _ in stream:
            pass

        await stream.aclose()
        assert not cleanup_called

    async def test_cleanup_idempotent(self) -> None:
        """Multiple aclose() calls don't re-run cleanup."""
        from dendrux.types import RunEvent

        call_count = 0

        async def _cleanup() -> None:
            nonlocal call_count
            call_count += 1

        async def _gen() -> AsyncGenerator[RunEvent, None]:
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="hi")
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="more")

        stream = RunStream(run_id="test", generator=_gen(), cleanup=_cleanup)
        async for _ in stream:
            break

        await stream.aclose()
        await stream.aclose()
        await stream.aclose()
        assert call_count == 1


async def _noop_cleanup() -> None:
    pass


# ------------------------------------------------------------------
# kwargs forwarding
# ------------------------------------------------------------------


class TestKwargsForwarding:
    async def test_run_forwards_kwargs_to_provider(self) -> None:
        """agent.run() forwards temperature/max_tokens to provider.complete()."""
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        await agent.run("Hi", temperature=0.5, max_tokens=100)

        assert llm.call_history[0]["kwargs"]["temperature"] == 0.5
        assert llm.call_history[0]["kwargs"]["max_tokens"] == 100

    async def test_run_stream_forwards_kwargs_to_provider(self) -> None:
        """run_stream() forwards kwargs to provider.complete_stream()."""
        llm = StreamingMockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("Hi", temperature=0.7)
        async for _ in stream:
            pass

        assert llm.call_history[0]["kwargs"]["temperature"] == 0.7

    async def test_runner_run_forwards_kwargs(self) -> None:
        """runner.run() forwards kwargs through the loop to the provider."""
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", temperature=0.3)

        assert llm.call_history[0]["kwargs"]["temperature"] == 0.3


# ------------------------------------------------------------------
# Stream event flow
# ------------------------------------------------------------------


class TestStreamEventFlow:
    async def test_text_only_stream(self) -> None:
        """Text response → RUN_STARTED + TEXT_DELTAs + RUN_COMPLETED."""
        llm = StreamingMockLLM([LLMResponse(text="Hi!")])
        agent = Agent(prompt="Be friendly.", tools=[add], provider=llm)

        stream = agent.stream("Hello")
        events = [e async for e in stream]

        types = [e.type for e in events]
        assert types[0] == RunEventType.RUN_STARTED
        assert types[-1] == RunEventType.RUN_COMPLETED

        text_events = [e for e in events if e.type == RunEventType.TEXT_DELTA]
        assert len(text_events) == 3  # H, i, !
        assert "".join(e.text for e in text_events) == "Hi!"

    async def test_run_started_carries_run_id(self) -> None:
        """First event is RUN_STARTED with run_id matching stream.run_id."""
        llm = StreamingMockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("Hi")
        events = [e async for e in stream]

        first = events[0]
        assert first.type == RunEventType.RUN_STARTED
        assert first.run_id is not None
        assert first.run_id == stream.run_id

    async def test_tool_call_stream(self) -> None:
        """Tool call → TOOL_USE_START + TOOL_USE_END + TOOL_RESULT + next turn."""
        tc = ToolCall(name="add", params={"a": 3, "b": 4}, provider_tool_call_id="t1")
        llm = StreamingMockLLM([
            LLMResponse(tool_calls=[tc]),
            LLMResponse(text="7"),
        ])
        agent = Agent(prompt="Calculate.", tools=[add], provider=llm)

        stream = agent.stream("3+4?")
        events = [e async for e in stream]

        types = [e.type for e in events]
        assert RunEventType.TOOL_USE_START in types
        assert RunEventType.TOOL_USE_END in types
        assert RunEventType.TOOL_RESULT in types
        assert types[-1] == RunEventType.RUN_COMPLETED

        # Verify tool result
        tool_results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert len(tool_results) == 1
        assert tool_results[0].tool_result is not None
        assert tool_results[0].tool_result.success is True

    async def test_run_completed_carries_run_result(self) -> None:
        """RUN_COMPLETED event carries RunResult with status, answer, and usage."""
        llm = StreamingMockLLM([
            LLMResponse(
                text="Hello!",
                usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("Hi")
        events = [e async for e in stream]

        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
        result = completed[0].run_result
        assert result is not None
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Hello!"
        assert result.usage.input_tokens == 10

    async def test_max_iterations_stream(self) -> None:
        """Agent exceeding max_iterations yields RUN_COMPLETED(MAX_ITERATIONS)."""
        tc = ToolCall(name="add", params={"a": 1, "b": 1}, provider_tool_call_id="t1")
        # Keep calling tools forever
        llm = StreamingMockLLM([
            LLMResponse(tool_calls=[tc]),
            LLMResponse(tool_calls=[tc]),
            LLMResponse(tool_calls=[tc]),
        ])
        agent = Agent(prompt="Loop.", tools=[add], max_iterations=2, provider=llm)

        stream = agent.stream("Go")
        events = [e async for e in stream]

        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
        assert completed[0].run_result.status == RunStatus.MAX_ITERATIONS


# ------------------------------------------------------------------
# Error semantics
# ------------------------------------------------------------------


class TestStreamErrorSemantics:
    async def test_error_yields_run_error_no_exception(self) -> None:
        """Exception in LLM → RUN_ERROR event, no re-raise to consumer."""
        llm = ErrorLLM(RuntimeError("LLM crashed"))
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("Hi")

        # Should NOT raise — error is yielded as an event
        events = [e async for e in stream]

        types = [e.type for e in events]
        assert RunEventType.RUN_STARTED in types
        assert RunEventType.RUN_ERROR in types

        error_event = [e for e in events if e.type == RunEventType.RUN_ERROR][0]
        assert error_event.error is not None
        assert "LLM crashed" in error_event.error
        assert error_event.run_result is not None
        assert error_event.run_result.status == RunStatus.ERROR

    async def test_setup_failure_yields_run_error(self) -> None:
        """Exception during setup (before first event) → RUN_ERROR, not raw exception."""
        from dendrux.runtime.runner import run_stream

        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        async def _bad_resolver() -> None:
            raise ConnectionError("DB unreachable")

        stream = run_stream(
            agent,
            provider=llm,
            user_input="Hi",
            state_store_resolver=_bad_resolver,
        )

        # Should NOT raise — setup error is yielded as an event
        events = [e async for e in stream]

        types = [e.type for e in events]
        assert RunEventType.RUN_ERROR in types
        error_event = [e for e in events if e.type == RunEventType.RUN_ERROR][0]
        assert "DB unreachable" in error_event.error


# ------------------------------------------------------------------
# .text() convenience
# ------------------------------------------------------------------


class TestTextConvenience:
    async def test_text_yields_only_strings(self) -> None:
        """stream.text() yields only text delta strings."""
        llm = StreamingMockLLM([LLMResponse(text="Hi!")])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("Hello")
        chunks = [chunk async for chunk in stream.text()]

        assert chunks == ["H", "i", "!"]

    async def test_text_skips_tool_events(self) -> None:
        """stream.text() ignores tool calls — only text from both turns."""
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = StreamingMockLLM([
            LLMResponse(tool_calls=[tc]),
            LLMResponse(text="3"),
        ])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("1+2?")
        chunks = [chunk async for chunk in stream.text()]

        assert chunks == ["3"]


# ------------------------------------------------------------------
# MockLLM base fallback streaming
# ------------------------------------------------------------------


class TestBaseFallbackStreaming:
    """Verify that MockLLM (using the base class complete_stream() fallback)
    also works correctly with run_stream — important for existing tests."""

    async def test_base_fallback_produces_valid_stream(self) -> None:
        """MockLLM (no streaming override) still produces correct events."""
        llm = MockLLM([LLMResponse(text="Hello from fallback")])
        agent = Agent(prompt="Test.", tools=[add], provider=llm)

        stream = agent.stream("Hi")
        events = [e async for e in stream]

        types = [e.type for e in events]
        assert types[0] == RunEventType.RUN_STARTED
        assert RunEventType.TEXT_DELTA in types
        assert types[-1] == RunEventType.RUN_COMPLETED

        # Base fallback yields one TEXT_DELTA with the full text
        text_events = [e for e in events if e.type == RunEventType.TEXT_DELTA]
        combined = "".join(e.text for e in text_events)
        assert combined == "Hello from fallback"
