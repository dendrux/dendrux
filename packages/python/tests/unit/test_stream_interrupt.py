"""Mid-stream interruption of streamed runs (loop level, no DB).

Covers:
  - ``interruptible()``: passthrough without a signal, interrupt while the
    provider is blocked, interrupt already set, event-vs-interrupt tie
    prefers the event, provider errors propagate, deterministic close.
  - ReActLoop.run_stream / SingleCall.run_stream with ``interrupt=``:
    RUN_CANCELLED carries the partial answer, the provider stream is
    closed, and ``on_llm_call_failed`` pairs with ``on_llm_call_started``.
  - Completion wins once DONE has been received.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from dendrux import Agent, tool
from dendrux.loops._helpers import StreamInterruptedError, interruptible
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    LLMResponse,
    RunEvent,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from tests._helpers.gated_provider import GatedStreamLLM

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class RecordingRecorder:
    """Minimal LoopRecorder capturing hook names in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.on_llm_completed_hook: Any = None

    async def on_run_started(self, run_id: str, **kwargs: Any) -> None:
        self.calls.append(("on_run_started", (run_id,)))

    async def on_run_finished(self, run_id: str, result: Any) -> None:
        self.calls.append(("on_run_finished", (run_id,)))

    async def on_run_failed(self, run_id: str, error: BaseException, **kwargs: Any) -> None:
        self.calls.append(("on_run_failed", (run_id, error)))

    async def on_message_appended(self, run_id: str, message: Any, iteration: int) -> None:
        self.calls.append(("on_message_appended", (iteration, message.role.value)))

    async def on_llm_call_started(self, run_id: str, iteration: int, **kwargs: Any) -> None:
        self.calls.append(("on_llm_call_started", (iteration,)))

    async def on_llm_call_completed(
        self, run_id: str, response: Any, iteration: int, **kwargs: Any
    ) -> None:
        self.calls.append(("on_llm_call_completed", (iteration,)))
        if self.on_llm_completed_hook is not None:
            self.on_llm_completed_hook()

    async def on_llm_call_failed(
        self, run_id: str, iteration: int, error: BaseException, **kwargs: Any
    ) -> None:
        self.calls.append(("on_llm_call_failed", (iteration, error)))

    async def on_tool_started(self, run_id: str, tool_call: Any, iteration: int) -> None:
        self.calls.append(("on_tool_started", (tool_call.name, iteration)))

    async def on_tool_completed(
        self, run_id: str, tool_call: Any, tool_result: Any, iteration: int
    ) -> None:
        self.calls.append(("on_tool_completed", (tool_call.name, iteration)))

    async def on_governance_event(
        self, run_id: str, event_type: str, iteration: int, data: Any, **kwargs: Any
    ) -> None:
        self.calls.append(("on_governance_event", (event_type,)))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


async def _drive(
    gen: AsyncGenerator[RunEvent, None],
    provider: GatedStreamLLM,
    interrupt: asyncio.Event,
) -> list[RunEvent]:
    """Consume ``gen``; fire ``interrupt`` once the provider is blocked."""
    events: list[RunEvent] = []

    async def _fire() -> None:
        await asyncio.wait_for(provider.first_sent.wait(), timeout=5)
        interrupt.set()

    firer = asyncio.create_task(_fire())
    try:
        async for event in gen:
            events.append(event)
    finally:
        firer.cancel()
    return events


# ------------------------------------------------------------------
# interruptible()
# ------------------------------------------------------------------


async def _three_events() -> AsyncGenerator[StreamEvent, None]:
    for text in ("a", "b", "c"):
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=text)


class TestInterruptible:
    async def test_passthrough_without_signal(self) -> None:
        texts = [ev.text async for ev in interruptible(_three_events(), None)]
        assert texts == ["a", "b", "c"]

    async def test_passthrough_when_signal_never_fires(self) -> None:
        texts = [ev.text async for ev in interruptible(_three_events(), asyncio.Event())]
        assert texts == ["a", "b", "c"]

    async def test_interrupt_while_provider_blocked_cancels_pending_pull(self) -> None:
        provider = GatedStreamLLM()
        interrupt = asyncio.Event()
        stream = provider.complete_stream([])
        seen: list[str] = []

        async def _fire() -> None:
            await provider.first_sent.wait()
            interrupt.set()

        firer = asyncio.create_task(_fire())
        with pytest.raises(StreamInterruptedError):
            async for ev in interruptible(stream, interrupt):
                seen.append(ev.text or "")
        await firer

        assert seen == [provider.first]
        assert provider.cancelled is True
        assert provider.closed is True

    async def test_interrupt_already_set_stops_at_first_block(self) -> None:
        """A pre-set signal still lets non-blocking output through; the
        provider is abandoned at its first real wait."""
        provider = GatedStreamLLM()
        interrupt = asyncio.Event()
        interrupt.set()
        seen: list[str] = []
        with pytest.raises(StreamInterruptedError):
            async for ev in interruptible(provider.complete_stream([]), interrupt):
                seen.append(ev.text or "")
        assert seen == [provider.first]
        assert provider.cancelled is True

    async def test_output_available_without_blocking_is_delivered_after_signal(self) -> None:
        """Completion wins once data is in hand: a generator that never
        blocks runs to exhaustion even though the signal fired mid-way."""
        interrupt = asyncio.Event()

        async def _gen() -> AsyncGenerator[StreamEvent, None]:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            interrupt.set()
            yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="x"))

        seen = [ev.type async for ev in interruptible(_gen(), interrupt)]
        assert seen == [StreamEventType.TEXT_DELTA, StreamEventType.DONE]

    async def test_provider_error_propagates(self) -> None:
        async def _boom() -> AsyncGenerator[StreamEvent, None]:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            raise RuntimeError("provider exploded")

        with pytest.raises(RuntimeError, match="provider exploded"):
            async for _ in interruptible(_boom(), asyncio.Event()):
                pass

    async def test_no_dangling_tasks_after_exhaustion(self) -> None:
        before = len(asyncio.all_tasks())
        async for _ in interruptible(_three_events(), asyncio.Event()):
            pass
        await asyncio.sleep(0)
        assert len(asyncio.all_tasks()) <= before


# ------------------------------------------------------------------
# ReActLoop.run_stream
# ------------------------------------------------------------------


class TestReActStreamInterrupt:
    async def test_interrupt_mid_stream_yields_cancelled_with_partial_answer(self) -> None:
        provider = GatedStreamLLM()
        recorder = RecordingRecorder()
        interrupt = asyncio.Event()
        agent = Agent(prompt="Test.", tools=[add], provider=provider)

        events = await _drive(
            ReActLoop().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                recorder=recorder,
                interrupt=interrupt,
            ),
            provider,
            interrupt,
        )

        assert [e.type for e in events] == [RunEventType.TEXT_DELTA, RunEventType.RUN_CANCELLED]
        result = events[-1].run_result
        assert result is not None
        assert result.status == RunStatus.CANCELLED
        assert result.answer == provider.first
        assert result.meta["interrupted"] is True
        assert result.meta["partial_output"] is True
        assert result.iteration_count == 1
        assert provider.closed is True
        assert provider.gate.is_set() is False

        names = recorder.names()
        assert "on_llm_call_started" in names
        assert "on_llm_call_failed" in names
        assert "on_llm_call_completed" not in names
        failed = next(c for c in recorder.calls if c[0] == "on_llm_call_failed")
        assert isinstance(failed[1][1], StreamInterruptedError)

    async def test_interrupt_set_before_stream_starts(self) -> None:
        provider = GatedStreamLLM()
        interrupt = asyncio.Event()
        interrupt.set()
        agent = Agent(prompt="Test.", tools=[add], provider=provider)

        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                interrupt=interrupt,
            )
        ]

        assert [e.type for e in events] == [RunEventType.RUN_CANCELLED]
        result = events[-1].run_result
        assert result is not None
        assert result.answer is None
        assert result.iteration_count == 0
        # The provider call was never started.
        assert provider.call_count == 0

    async def test_completion_wins_once_done_received(self) -> None:
        """Interrupt set after DONE arrived (during llm.completed) → SUCCESS."""
        provider = GatedStreamLLM()
        provider.gate.set()
        recorder = RecordingRecorder()
        interrupt = asyncio.Event()
        recorder.on_llm_completed_hook = interrupt.set
        agent = Agent(prompt="Test.", tools=[add], provider=provider)

        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                recorder=recorder,
                interrupt=interrupt,
            )
        ]

        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert events[-1].run_result is not None
        assert events[-1].run_result.status == RunStatus.SUCCESS
        assert events[-1].run_result.answer == provider.first + provider.second

    async def test_interrupt_in_second_iteration_keeps_first_iteration_steps(self) -> None:
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="ptc_1")
        provider = GatedStreamLLM(before=[LLMResponse(text="Adding.", tool_calls=[tc])])
        recorder = RecordingRecorder()
        interrupt = asyncio.Event()
        agent = Agent(prompt="Test.", tools=[add], provider=provider)

        events = await _drive(
            ReActLoop().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                recorder=recorder,
                interrupt=interrupt,
            ),
            provider,
            interrupt,
        )

        assert events[-1].type == RunEventType.RUN_CANCELLED
        result = events[-1].run_result
        assert result is not None
        assert result.answer == provider.first
        assert result.iteration_count == 2
        assert len(result.steps) == 1
        assert ("on_tool_completed", ("add", 1)) in recorder.calls
        # Usage from the completed iteration is kept; the interrupted call
        # reported none.
        assert result.usage.total_tokens == 0

    async def test_no_interrupt_kwarg_is_backwards_compatible(self) -> None:
        provider = GatedStreamLLM()
        provider.gate.set()
        agent = Agent(prompt="Test.", tools=[add], provider=provider)
        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
            )
        ]
        assert events[-1].type == RunEventType.RUN_COMPLETED


# ------------------------------------------------------------------
# SingleCall.run_stream
# ------------------------------------------------------------------


class TestSingleCallStreamInterrupt:
    async def test_interrupt_set_before_call_skips_provider(self) -> None:
        provider = GatedStreamLLM()
        interrupt = asyncio.Event()
        interrupt.set()
        agent = Agent(prompt="Test.", provider=provider, loop=SingleCall())
        events = [
            e
            async for e in SingleCall().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                interrupt=interrupt,
            )
        ]
        assert [e.type for e in events] == [RunEventType.RUN_CANCELLED]
        assert events[-1].run_result is not None
        assert events[-1].run_result.answer is None
        assert provider.call_count == 0

    async def test_interrupt_mid_stream_yields_cancelled_with_partial_answer(self) -> None:
        provider = GatedStreamLLM()
        recorder = RecordingRecorder()
        interrupt = asyncio.Event()
        agent = Agent(prompt="Test.", provider=provider, loop=SingleCall())

        events = await _drive(
            SingleCall().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                recorder=recorder,
                interrupt=interrupt,
            ),
            provider,
            interrupt,
        )

        assert [e.type for e in events] == [RunEventType.TEXT_DELTA, RunEventType.RUN_CANCELLED]
        result = events[-1].run_result
        assert result is not None
        assert result.status == RunStatus.CANCELLED
        assert result.answer == provider.first
        assert result.meta["interrupted"] is True
        assert result.iteration_count == 1
        assert provider.closed is True
        assert "on_llm_call_failed" in recorder.names()
        assert "on_message_appended" not in [
            n for n, args in recorder.calls if args and args[-1] == "assistant"
        ]

    async def test_completion_wins_once_done_received(self) -> None:
        provider = GatedStreamLLM()
        provider.gate.set()
        recorder = RecordingRecorder()
        interrupt = asyncio.Event()
        recorder.on_llm_completed_hook = interrupt.set
        agent = Agent(prompt="Test.", provider=provider, loop=SingleCall())

        events = [
            e
            async for e in SingleCall().run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="go",
                run_id="r1",
                recorder=recorder,
                interrupt=interrupt,
            )
        ]
        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert events[-1].run_result is not None
        assert events[-1].run_result.status == RunStatus.SUCCESS
