"""Stop reason persistence (#42) and failed structured-call cost (#26)."""

from __future__ import annotations

import pytest

from dendrux import Agent, ModelPricing, PriceTable, SingleCall
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import STOP_REASON_META_KEY
from dendrux.types import (
    LLMResponse,
    RunEventType,
    RunStatus,
    UsageStats,
)

_PRICING = PriceTable({"mock": ModelPricing(input=1.0, output=2.0)})


def _usage() -> UsageStats:
    return UsageStats(input_tokens=100, output_tokens=10, total_tokens=110)


def _cut() -> LLMResponse:
    return LLMResponse(
        text="The answer is", usage=_usage(), stop_reason="max_tokens", raw_stop_reason="length"
    )


def _agent(store, llm: MockLLM, **kwargs) -> Agent:
    return Agent(provider=llm, prompt="test", state_store=store, **kwargs)


async def _completed_event(store, run_id: str) -> dict:
    events = await store.get_run_events(run_id, event_types=["run.completed"])
    assert len(events) == 1
    return events[0].data or {}


@pytest.mark.parametrize("single", [False, True], ids=["react", "single"])
async def test_truncated_run_persists_stop_reason(store, single: bool) -> None:
    kwargs = {"loop": SingleCall()} if single else {}
    result = await _agent(store, MockLLM([_cut()]), **kwargs).run("hi")

    assert result.status == RunStatus.SUCCESS
    assert result.stop_reason == "max_tokens"
    assert result.truncated is True

    record = await store.get_run(result.run_id)
    assert record.meta[STOP_REASON_META_KEY] == "max_tokens"
    assert await _completed_event(store, result.run_id) == {
        "status": "success",
        "stop_reason": "max_tokens",
        "truncated": True,
    }
    [call] = await store.get_llm_interactions(result.run_id)
    assert call.semantic_response["stop_reason"] == "max_tokens"


@pytest.mark.parametrize("single", [False, True], ids=["react", "single"])
async def test_streamed_truncated_run(store, single: bool) -> None:
    kwargs = {"loop": SingleCall()} if single else {}
    agent = _agent(store, MockLLM([_cut()]), **kwargs)
    async with agent.stream("hi") as stream:
        events = [e async for e in stream]

    final = events[-1]
    assert final.type == RunEventType.RUN_COMPLETED
    assert final.stop_reason == "max_tokens"
    assert final.truncated is True
    assert final.run_result is not None
    assert final.run_result.stop_reason == "max_tokens"
    assert final.run_result.truncated is True

    run_id = final.run_result.run_id
    record = await store.get_run(run_id)
    assert record.meta[STOP_REASON_META_KEY] == "max_tokens"
    data = await _completed_event(store, run_id)
    assert data["stop_reason"] == "max_tokens"
    assert data["truncated"] is True


async def test_normal_run_records_end_turn(store) -> None:
    resp = LLMResponse(text="done", usage=_usage(), stop_reason="end_turn")
    result = await _agent(store, MockLLM([resp])).run("hi")
    assert result.stop_reason == "end_turn"
    assert result.truncated is False
    record = await store.get_run(result.run_id)
    assert record.meta[STOP_REASON_META_KEY] == "end_turn"
    data = await _completed_event(store, result.run_id)
    assert data["stop_reason"] == "end_turn"
    assert data["truncated"] is False


async def test_unknown_stop_reason_is_not_persisted(store) -> None:
    """A provider that reports nothing leaves meta untouched (never a guess)."""
    result = await _agent(store, MockLLM([LLMResponse(text="done")])).run(
        "hi", metadata={"thread_id": "t1"}
    )
    assert result.stop_reason is None
    record = await store.get_run(result.run_id)
    assert STOP_REASON_META_KEY not in record.meta
    assert record.meta["thread_id"] == "t1"
    data = await _completed_event(store, result.run_id)
    assert data == {"status": "success", "stop_reason": None, "truncated": False}


async def test_stop_reason_does_not_clobber_developer_metadata(store) -> None:
    result = await _agent(store, MockLLM([_cut()])).run(
        "hi", metadata={"thread_id": "t1", "stop_reason": "mine"}
    )
    record = await store.get_run(result.run_id)
    assert record.meta["thread_id"] == "t1"
    assert record.meta["stop_reason"] == "mine"
    assert record.meta[STOP_REASON_META_KEY] == "max_tokens"


async def test_max_iterations_has_no_stop_reason(store) -> None:
    from dendrux import tool
    from dendrux.types import ToolCall

    @tool()
    async def ping() -> str:
        """Ping."""
        return "pong"

    llm = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("ping", {})], stop_reason="tool_use", usage=_usage())]
    )
    result = await _agent(store, llm, tools=[ping], max_iterations=1).run("hi")
    assert result.status == RunStatus.MAX_ITERATIONS
    assert result.stop_reason is None
    record = await store.get_run(result.run_id)
    assert STOP_REASON_META_KEY not in (record.meta or {})
    data = await _completed_event(store, result.run_id)
    assert data["stop_reason"] is None
    assert data["truncated"] is False


async def test_idempotent_cached_result_restores_stop_reason(store) -> None:
    llm = MockLLM([_cut()])
    agent = _agent(store, llm)
    first = await agent.run("hi", idempotency_key="k1")
    cached = await agent.run("hi", idempotency_key="k1")
    assert llm.calls_made == 1
    assert cached.run_id == first.run_id
    assert cached.stop_reason == "max_tokens"
    assert cached.truncated is True


async def test_idempotent_cached_stream_restores_stop_reason(store) -> None:
    agent = _agent(store, MockLLM([_cut()]))
    await agent.run("hi", idempotency_key="k2")
    async with agent.stream("hi", idempotency_key="k2") as stream:
        events = [e async for e in stream]
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert events[-1].truncated is True


async def test_cached_result_before_041_has_no_stop_reason(store) -> None:
    from dendrux.runtime.runner import _build_cached_result

    await store.create_run("old", "test")
    await store.finalize_run("old", status="success", answer="x")
    cached = await _build_cached_result(store, "old")
    assert cached.stop_reason is None


async def test_cancel_run_on_finished_run_reports_stop_reason(store) -> None:
    agent = _agent(store, MockLLM([_cut()]))
    first = await agent.run("hi")
    again = await agent.cancel_run(first.run_id)
    assert again.status == RunStatus.SUCCESS
    assert again.stop_reason == "max_tokens"
    assert again.truncated is True
