"""Commentary vs final-answer tagging across RunResult, traces, and streams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent, tool
from dendrux.db.models import Base
from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import RunStore, StoredEvent, TraceEntry
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    UsageStats,
)
from tests._helpers.gated_provider import GatedStreamLLM

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@tool()
async def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


def _two_step() -> list[LLMResponse]:
    return [
        LLMResponse(
            text="Let me look that up.",
            tool_calls=[ToolCall("lookup", {"q": "a"})],
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
        ),
        LLMResponse(
            text="",
            tool_calls=[ToolCall("lookup", {"q": "b"})],
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
        ),
        LLMResponse(
            text="Checking one more source.",
            tool_calls=[ToolCall("lookup", {"q": "c"})],
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
        ),
        LLMResponse(
            text="The answer is 42.",
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
        ),
    ]


class ScriptedStreamLLM(LLMProvider):
    """Replays explicit StreamEvent scripts — one list per provider call."""

    capabilities = ProviderCapabilities(supports_native_tools=True, supports_streaming=True)

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self._scripts = list(scripts)

    @property
    def model(self) -> str:
        return "scripted"

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        raise AssertionError("streaming only")

    async def complete_stream(
        self, messages: Any, tools: Any = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        for event in self._scripts.pop(0):
            yield event


def _done(text: str, tool_calls: list[ToolCall] | None = None) -> StreamEvent:
    return StreamEvent(
        type=StreamEventType.DONE,
        raw=LLMResponse(text=text, tool_calls=tool_calls, usage=UsageStats()),
    )


@pytest.fixture
async def engine(tmp_path):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'phase.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def state_store(engine):
    return SQLAlchemyStateStore(engine)


# ------------------------------------------------------------------
# RunResult.commentary
# ------------------------------------------------------------------


async def test_commentary_lists_tool_step_text_and_answer_is_final_turn():
    agent = Agent(provider=MockLLM(_two_step()), prompt="p", tools=[lookup])
    result = await agent.run("hi")
    assert result.status == RunStatus.SUCCESS
    assert result.answer == "The answer is 42."
    assert result.commentary == ["Let me look that up.", "Checking one more source."]


async def test_commentary_empty_without_narration():
    provider = MockLLM(
        [
            LLMResponse(text="  ", tool_calls=[ToolCall("lookup", {"q": "a"})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(provider=provider, prompt="p", tools=[lookup])
    result = await agent.run("hi")
    assert result.commentary == []
    assert result.answer == "done"


async def test_commentary_empty_for_single_call():
    agent = Agent(provider=MockLLM([LLMResponse(text="hello")]), prompt="p", loop=SingleCall())
    result = await agent.run("hi")
    assert result.answer == "hello"
    assert result.commentary == []


async def test_single_call_stream_tags_final_answer():
    agent = Agent(provider=MockLLM([LLMResponse(text="hello")]), prompt="p", loop=SingleCall())
    events = [e async for e in agent.stream("hi")]
    text = [e for e in events if e.type == RunEventType.TEXT_DELTA]
    assert [(e.iteration, e.phase) for e in text] == [(1, "final_answer")]
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert events[-1].phase == "final_answer"
    assert events[-1].run_result.commentary == []


async def test_commentary_persists_and_answer_is_final_with_store(state_store):
    agent = Agent(
        provider=MockLLM(_two_step()), prompt="p", tools=[lookup], state_store=state_store
    )
    result = await agent.run("hi")
    assert result.commentary == ["Let me look that up.", "Checking one more source."]
    assert (await state_store.get_run(result.run_id)).answer == "The answer is 42."


async def test_max_iterations_commentary_holds_all_tool_steps_answer_none():
    responses = [
        LLMResponse(text=f"step {i}", tool_calls=[ToolCall("lookup", {"q": str(i)})])
        for i in range(2)
    ]
    agent = Agent(provider=MockLLM(responses), prompt="p", tools=[lookup], max_iterations=2)
    result = await agent.run("hi")
    assert result.status == RunStatus.MAX_ITERATIONS
    assert result.answer is None
    assert result.commentary == ["step 0", "step 1"]


async def test_max_iterations_stream_has_no_final_answer_phase():
    responses = [LLMResponse(text="step", tool_calls=[ToolCall("lookup", {"q": "x"})])]
    agent = Agent(provider=MockLLM(responses), prompt="p", tools=[lookup], max_iterations=1)
    events = [e async for e in agent.stream("hi")]
    terminal = events[-1]
    assert terminal.type == RunEventType.RUN_COMPLETED
    assert terminal.run_result.status == RunStatus.MAX_ITERATIONS
    assert terminal.phase is None
    assert terminal.run_result.commentary == ["step"]


async def test_mid_stream_interrupt_keeps_partial_in_answer_and_completed_steps_in_commentary(
    state_store,
):
    provider = GatedStreamLLM(
        first="Partial final ",
        second="never sent",
        before=[LLMResponse(text="Narration.", tool_calls=[ToolCall("lookup", {"q": "a"})])],
    )
    agent = Agent(provider=provider, prompt="p", tools=[lookup], state_store=state_store)
    stream = agent.stream("hi")
    terminal = None
    async with stream:
        async for event in stream:
            if event.type == RunEventType.TEXT_DELTA and event.text == "Partial final ":
                await agent.cancel_run(stream.run_id)
            if event.type == RunEventType.RUN_CANCELLED:
                terminal = event
    assert terminal is not None
    result = terminal.run_result
    assert result.status == RunStatus.CANCELLED
    assert result.answer == "Partial final "
    assert result.commentary == ["Narration."]


async def test_cooperative_cancel_before_next_iteration_keeps_commentary(state_store):
    agent = Agent(
        provider=MockLLM(_two_step()), prompt="p", tools=[lookup], state_store=state_store
    )
    stream = agent.stream("hi")
    terminal = None
    async with stream:
        async for event in stream:
            if event.type == RunEventType.TOOL_RESULT and terminal is None:
                await agent.cancel_run(stream.run_id)
            if event.type == RunEventType.RUN_CANCELLED:
                terminal = event
    assert terminal is not None
    assert terminal.run_result.answer is None
    assert terminal.run_result.commentary == ["Let me look that up."]


# ------------------------------------------------------------------
# Persisted phase on traces + SSE event log
# ------------------------------------------------------------------


async def test_assistant_traces_carry_phase(engine, state_store):
    agent = Agent(
        provider=MockLLM(_two_step()), prompt="p", tools=[lookup], state_store=state_store
    )
    result = await agent.run("hi")
    traces = await RunStore.from_engine(engine).get_traces(result.run_id)
    assistant = [t for t in traces if t.role == "assistant"]
    assert [t.phase for t in assistant] == [
        "commentary",
        "commentary",
        "commentary",
        "final_answer",
    ]
    assert [t.meta["phase"] for t in assistant] == [t.phase for t in assistant]
    assert all(t.phase is None for t in traces if t.role != "assistant")


async def test_single_call_trace_is_final_answer(engine, state_store):
    agent = Agent(
        provider=MockLLM([LLMResponse(text="hello")]),
        prompt="p",
        loop=SingleCall(),
        state_store=state_store,
    )
    result = await agent.run("hi")
    traces = await RunStore.from_engine(engine).get_traces(result.run_id)
    assert [t.phase for t in traces if t.role == "assistant"] == ["final_answer"]


def test_trace_entry_phase_defaults_none_for_old_rows():
    entry = TraceEntry(role="assistant", content="x", order_index=0, meta={}, created_at=None)
    assert entry.phase is None


async def test_sse_event_log_carries_iteration_and_phase(engine, state_store):
    agent = Agent(
        provider=MockLLM(_two_step()), prompt="p", tools=[lookup], state_store=state_store
    )
    result = await agent.run("hi")
    store = RunStore.from_engine(engine)

    frames = []
    async with store.follow(result.run_id, tenant_id=None) as feed:
        async for item in feed:
            if isinstance(item, StoredEvent):
                frames.append(
                    {
                        "iteration_index": item.iteration_index,
                        "data": item.data,
                        "event_type": item.event_type,
                    }
                )

    llm = [
        (f["iteration_index"], f["data"]["phase"])
        for f in frames
        if f["event_type"] == "llm.completed"
    ]
    assert llm == [
        (1, "commentary"),
        (2, "commentary"),
        (3, "commentary"),
        (4, "final_answer"),
    ]


# ------------------------------------------------------------------
# Stream: iteration + phase on RunEvents
# ------------------------------------------------------------------


async def test_stream_events_carry_iteration_and_phase():
    agent = Agent(provider=MockLLM(_two_step()), prompt="p", tools=[lookup])
    events = [e async for e in agent.stream("hi")]

    lifecycle = {RunEventType.RUN_STARTED, RunEventType.RUN_COMPLETED}
    for e in events:
        if e.type in lifecycle:
            assert e.iteration is None
        else:
            assert e.iteration is not None

    tool_events = [
        e
        for e in events
        if e.type
        in (RunEventType.TOOL_USE_START, RunEventType.TOOL_USE_END, RunEventType.TOOL_RESULT)
    ]
    assert {e.phase for e in tool_events} == {"commentary"}
    assert [e.iteration for e in tool_events if e.type == RunEventType.TOOL_RESULT] == [1, 2, 3]

    terminal = events[-1]
    assert terminal.type == RunEventType.RUN_COMPLETED
    assert terminal.phase == "final_answer"

    buckets: dict[int, str] = {}
    commentary_iters: set[int] = set()
    for e in events:
        if e.type == RunEventType.TEXT_DELTA:
            buckets[e.iteration] = buckets.get(e.iteration, "") + (e.text or "")
        elif e.phase == "commentary" and e.iteration is not None:
            commentary_iters.add(e.iteration)
    commentary = [buckets[i] for i in sorted(buckets) if i in commentary_iters]
    answer = [buckets[i] for i in sorted(buckets) if i not in commentary_iters]
    assert commentary == terminal.run_result.commentary
    assert answer == [terminal.run_result.answer]


async def test_text_after_tool_start_in_same_iteration_is_tagged_commentary():
    tc = ToolCall("lookup", {"q": "a"}, id="t1")
    provider = ScriptedStreamLLM(
        [
            [
                StreamEvent(type=StreamEventType.TEXT_DELTA, text="Before. "),
                StreamEvent(
                    type=StreamEventType.TOOL_USE_START, tool_name="lookup", tool_call_id="t1"
                ),
                StreamEvent(type=StreamEventType.TEXT_DELTA, text="After."),
                StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name="lookup",
                    tool_call_id="t1",
                ),
                _done("Before. After.", [tc]),
            ],
            [
                StreamEvent(type=StreamEventType.TEXT_DELTA, text="Done."),
                _done("Done."),
            ],
        ]
    )
    agent = Agent(provider=provider, prompt="p", tools=[lookup])
    events = [e async for e in agent.stream("hi")]
    text = [(e.text, e.iteration, e.phase) for e in events if e.type == RunEventType.TEXT_DELTA]
    assert text == [("Before. ", 1, None), ("After.", 1, "commentary"), ("Done.", 2, None)]
    assert events[-1].run_result.commentary == ["Before. After."]


async def test_provider_native_phase_passes_through_on_text_delta():
    provider = ScriptedStreamLLM(
        [
            [
                StreamEvent(type=StreamEventType.TEXT_DELTA, text="Answer.", phase="final_answer"),
                _done("Answer."),
            ]
        ]
    )
    agent = Agent(provider=provider, prompt="p", tools=[lookup])
    events = [e async for e in agent.stream("hi")]
    text = [(e.iteration, e.phase) for e in events if e.type == RunEventType.TEXT_DELTA]
    assert text == [(1, "final_answer")]


async def test_reasoning_delta_carries_iteration():
    provider = ScriptedStreamLLM(
        [
            [
                StreamEvent(type=StreamEventType.REASONING_DELTA, text="thinking"),
                StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok"),
                _done("ok"),
            ]
        ]
    )
    agent = Agent(provider=provider, prompt="p", tools=[lookup])
    events = [e async for e in agent.stream("hi")]
    reasoning = [e for e in events if e.type == RunEventType.REASONING_DELTA]
    assert [(e.iteration, e.phase) for e in reasoning] == [(1, None)]


async def test_approval_resume_stream_tags_tool_results_and_keeps_commentary(state_store):
    agent = Agent(
        provider=MockLLM(
            [
                LLMResponse(text="Needs sign-off.", tool_calls=[ToolCall("lookup", {"q": "a"})]),
                LLMResponse(text="Approved and done."),
            ]
        ),
        prompt="p",
        tools=[lookup],
        require_approval=["lookup"],
        state_store=state_store,
    )
    paused = await agent.run("hi")
    assert paused.status == RunStatus.WAITING_APPROVAL
    assert paused.commentary == ["Needs sign-off."]

    events = [
        e async for e in agent.resume_stream(paused.run_id, expected_pause_id=paused.pause_id)
    ]
    results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
    assert [(e.iteration, e.phase) for e in results] == [(1, "commentary")]
    terminal = events[-1]
    assert terminal.type == RunEventType.RUN_COMPLETED
    assert terminal.phase == "final_answer"
    assert terminal.run_result.answer == "Approved and done."
    assert terminal.run_result.commentary == ["Needs sign-off."]
