"""Mid-stream interruption through the public surface, DB-backed.

Covers:
  - ``agent.cancel_run`` while ``agent.stream()`` is blocked inside the
    provider: stream ends with RUN_CANCELLED, row is ``cancelled`` with the
    partial answer, ``run.cancelled`` carries ``interrupted``, flag cleared.
  - ``cancel_run`` return value distinguishes requested from completed.
  - ``RunStream.aclose()`` mid-stream persists the partial answer and
    closes the provider stream deterministically.
  - ``agent.resume_stream`` honours the same interrupt.
  - ``RunStore.get_run`` exposes ``cancel_requested``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sqlalchemy import update

from dendrux.agent import Agent
from dendrux.db.models import AgentRun
from dendrux.loops.single import SingleCall
from dendrux.store import RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall, ToolResult
from tests._helpers.gated_provider import GatedStreamLLM

if TYPE_CHECKING:
    from dendrux.types import RunEvent, RunStream


@tool(target="client")
async def client_read(sheet: str) -> str:
    return ""


async def _consume(stream: RunStream) -> list[RunEvent]:
    events: list[RunEvent] = []
    async with stream as s:
        async for event in s:
            events.append(event)
    return events


async def _wait_first_sent(provider: GatedStreamLLM) -> None:
    await asyncio.wait_for(provider.first_sent.wait(), timeout=5)


class TestCancelRunInterruptsStream:
    async def test_cancel_run_mid_stream_persists_partial_answer(self, store) -> None:
        provider = GatedStreamLLM()
        agent = Agent(provider=provider, prompt="Test.", state_store=store)
        stream = agent.stream("go")
        consumer = asyncio.create_task(_consume(stream))
        await _wait_first_sent(provider)

        requested = await agent.cancel_run(stream.run_id)
        # Deterministic "requested, not yet stopped": the durable flag is
        # recorded and read back before the in-process signal fires.
        assert requested.status == RunStatus.RUNNING
        assert requested.meta["cancel_requested"] is True

        events = await asyncio.wait_for(consumer, timeout=5)
        assert events[-1].type == RunEventType.RUN_CANCELLED
        assert events[-1].run_result is not None
        assert events[-1].run_result.answer == provider.first
        assert provider.closed is True
        assert provider.gate.is_set() is False

        record = await store.get_run(stream.run_id)
        assert record is not None
        assert record.status == RunStatus.CANCELLED.value
        assert record.answer == provider.first
        assert record.cancel_requested is False

        event_types = [e.event_type for e in await store.get_run_events(stream.run_id)]
        assert event_types.count("run.cancelled") == 1
        assert "run.completed" not in event_types
        cancelled = next(
            e for e in await store.get_run_events(stream.run_id) if e.event_type == "run.cancelled"
        )
        assert cancelled.data["reason"] == "cancel_requested"
        assert cancelled.data["interrupted"] is True
        assert cancelled.data["partial_output"] is True

    async def test_cancel_run_mid_stream_single_call(self, store) -> None:
        provider = GatedStreamLLM()
        agent = Agent(provider=provider, prompt="Test.", state_store=store, loop=SingleCall())
        stream = agent.stream("go")
        consumer = asyncio.create_task(_consume(stream))
        await _wait_first_sent(provider)

        await agent.cancel_run(stream.run_id)
        events = await asyncio.wait_for(consumer, timeout=5)

        assert events[-1].type == RunEventType.RUN_CANCELLED
        record = await store.get_run(stream.run_id)
        assert record is not None
        assert record.status == RunStatus.CANCELLED.value
        assert record.answer == provider.first

    async def test_cancel_after_completion_is_terminal_noop(self, store) -> None:
        provider = GatedStreamLLM()
        provider.gate.set()
        agent = Agent(provider=provider, prompt="Test.", state_store=store)
        stream = agent.stream("go")
        events = await _consume(stream)
        assert events[-1].type == RunEventType.RUN_COMPLETED

        result = await agent.cancel_run(stream.run_id)
        assert result.status == RunStatus.SUCCESS
        assert result.answer == provider.first + provider.second

    async def test_stream_from_another_agent_instance_is_not_preempted(self, store) -> None:
        """The in-process signal lives on the Agent that owns the stream.
        A different Agent instance only sets the durable flag; the run
        still finishes its current call and completes."""
        provider = GatedStreamLLM()
        owner = Agent(provider=provider, prompt="Test.", state_store=store)
        other = Agent(provider=provider, prompt="Test.", state_store=store)
        stream = owner.stream("go")
        consumer = asyncio.create_task(_consume(stream))
        await _wait_first_sent(provider)

        await other.cancel_run(stream.run_id)
        record = await store.get_run(stream.run_id)
        assert record is not None
        assert record.cancel_requested is True

        provider.gate.set()
        events = await asyncio.wait_for(consumer, timeout=5)
        assert events[-1].type == RunEventType.RUN_COMPLETED


class TestAcloseMidStream:
    async def test_aclose_persists_partial_answer_and_closes_provider(self, store) -> None:
        provider = GatedStreamLLM()
        agent = Agent(provider=provider, prompt="Test.", state_store=store)
        stream = agent.stream("go")

        async with stream as s:
            async for event in s:
                if event.type == RunEventType.TEXT_DELTA:
                    break

        assert provider.closed is True

        record = await store.get_run(stream.run_id)
        assert record is not None
        assert record.status == RunStatus.CANCELLED.value
        assert record.answer == provider.first

        event_types = [e.event_type for e in await store.get_run_events(stream.run_id)]
        assert event_types.count("run.cancelled") == 1

    async def test_aclose_before_any_text_persists_no_answer(self, store) -> None:
        provider = GatedStreamLLM()
        agent = Agent(provider=provider, prompt="Test.", state_store=store)
        stream = agent.stream("go")

        async with stream as s:
            async for _event in s:
                break  # RUN_STARTED only

        record = await store.get_run(stream.run_id)
        assert record is not None
        assert record.status == RunStatus.CANCELLED.value
        assert record.answer is None


class TestResumeStreamInterrupt:
    async def test_cancel_run_during_resume_stream(self, store) -> None:
        tc = ToolCall(name="client_read", params={"sheet": "S1"}, provider_tool_call_id="ptc_1")
        provider = GatedStreamLLM(before=[LLMResponse(text="Reading.", tool_calls=[tc])])
        agent = Agent(provider=provider, prompt="Test.", tools=[client_read], state_store=store)

        paused = await agent.run("go")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL

        pause = await store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]
        stream = agent.resume_stream(
            paused.run_id,
            tool_results=[ToolResult(name="client_read", call_id=call_id, payload='"cells"')],
        )
        consumer = asyncio.create_task(_consume(stream))
        await _wait_first_sent(provider)

        await agent.cancel_run(paused.run_id)
        events = await asyncio.wait_for(consumer, timeout=5)

        assert events[-1].type == RunEventType.RUN_CANCELLED
        record = await store.get_run(paused.run_id)
        assert record is not None
        assert record.status == RunStatus.CANCELLED.value
        assert record.answer == provider.first
        assert provider.closed is True


class TestRunDetailCancelRequested:
    async def test_run_store_exposes_cancel_requested(self, store) -> None:
        await store.create_run("r1", "agent")
        async with store._session_factory() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == "r1").values(status="running")
            )
            await session.commit()

        detail = await RunStore(store).get_run("r1")
        assert detail is not None
        assert detail.cancel_requested is False

        await store.request_cancel("r1")
        detail = await RunStore(store).get_run("r1")
        assert detail is not None
        assert detail.cancel_requested is True
