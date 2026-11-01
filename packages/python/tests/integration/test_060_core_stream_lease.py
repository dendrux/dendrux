"""Streaming renewal is bounded by time, never by chunk count (§12.29)."""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import event

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops import ReActLoop, SingleCall
from dendrux.types import LLMResponse, RunStatus, StreamEvent, StreamEventType
from tests._fixtures.storage import pin_storage_format


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("single", [False, True])
async def test_500_chunks_use_one_ticker_and_at_most_two_progress_writes(
    store, engine, monkeypatch, storage_format, single
):
    await pin_storage_format(engine, storage_format)
    writes, tickers = [], []
    create_task = asyncio.create_task

    def task(coroutine, **kwargs):
        if coroutine.__qualname__.endswith("lease_scope.<locals>.tick"):
            tickers.append(coroutine)
        return create_task(coroutine, **kwargs)

    def capture(conn, cursor, statement, parameters, context, executemany):
        if (
            statement.startswith("UPDATE agent_runs SET last_progress_at=")
            and "AND agent_runs.status =" in statement
        ):
            writes.append(statement)

    class Model(MockLLM):
        async def complete_stream(self, *args, **kwargs):
            for _ in range(500):
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="x" * 500))

    monkeypatch.setattr(asyncio, "create_task", task)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        a = Agent(
            prompt="p",
            provider=Model([]),
            loop=SingleCall() if single else ReActLoop(),
            state_store=store,
            strict_tenancy=False,
        )
        async with a.stream("go") as stream:
            events = [e async for e in stream]
        assert events[-1].run_result.status == RunStatus.SUCCESS
        assert len(writes) <= 2, len(writes)
        assert len(tickers) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.parametrize("single", [False, True])
async def test_twelve_second_provider_stall_keeps_five_second_lease(store, engine, single):
    writes = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if (
            statement.startswith("UPDATE agent_runs SET last_progress_at=")
            and "AND agent_runs.status =" in statement
        ):
            writes.append(statement)

    class Model(MockLLM):
        async def complete_stream(self, *args, **kwargs):
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            before = len(writes)
            await asyncio.sleep(12)
            assert len(writes) - before >= 2
            yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="x"))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        a = Agent(
            prompt="p",
            provider=Model([]),
            loop=SingleCall() if single else ReActLoop(),
            lease_interval=timedelta(seconds=5),
            state_store=store,
            strict_tenancy=False,
        )
        async with a.stream("go") as stream:
            events = [e async for e in stream]
        assert events[-1].run_result.status == RunStatus.SUCCESS
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
