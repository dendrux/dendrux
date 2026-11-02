"""Cancellation counts belong to the recorder's original task only."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, SingleCall
from dendrux.runtime import _cancel_guard
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import LLMResponse, ToolCall, generate_ulid
from tests._helpers.scripted_provider import ScriptedLLM

RUN_ID = generate_ulid()


@pytest.fixture(autouse=True)
def enable_guard(monkeypatch):
    monkeypatch.setattr(_cancel_guard, "_ENABLED", True)


async def lose_cancel():
    asyncio.current_task().cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.sleep(0)


def recorder():
    return PersistenceRecorder(SimpleNamespace(touch_progress=AsyncMock()), RUN_ID)


@pytest.mark.parametrize("hook", ["model", "tool"])
async def test_legacy_dispatch_recovers_a_lost_cancel(hook):
    async def run():
        persistence = recorder()
        await lose_cancel()
        with pytest.raises(asyncio.CancelledError):
            if hook == "model":
                await persistence.on_llm_call_started(RUN_ID, 1)
            else:
                await persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1)
        assert asyncio.current_task().cancelling() == 1

    await asyncio.create_task(run())


async def test_nonzero_entry_count_and_uncancel_are_respected():
    async def run():
        await lose_cancel()
        guard = _cancel_guard.capture()
        await guard.check(RUN_ID)
        assert asyncio.current_task().uncancel() == 0
        await guard.check(RUN_ID)
        await lose_cancel()
        with pytest.raises(asyncio.CancelledError):
            await guard.check(RUN_ID)
        assert asyncio.current_task().cancelling() == 1

    await asyncio.create_task(run())


async def test_pending_native_cancel_is_delivered_once(caplog):
    async def run():
        guard = _cancel_guard.capture()
        asyncio.current_task().cancel("native")
        with pytest.raises(asyncio.CancelledError) as caught:
            await guard.check(RUN_ID)
        assert caught.value.args == ("native",)
        assert asyncio.current_task().cancelling() == 1

    await asyncio.create_task(run())
    assert not any(getattr(r, "code", None) == "run_cancel_recovered" for r in caplog.records)


async def test_count_is_rechecked_after_yield():
    async def run():
        guard = _cancel_guard.capture()
        await lose_cancel()
        asyncio.get_running_loop().call_soon(asyncio.current_task().uncancel)
        await guard.check(RUN_ID)
        assert asyncio.current_task().cancelling() == 0

    await asyncio.create_task(run())


async def test_other_task_does_not_use_recorders_baseline():
    persistence = recorder()

    async def run():
        await lose_cancel()
        await persistence.on_llm_call_started(RUN_ID, 1)
        await persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1)
        assert asyncio.current_task().cancelling() == 1

    await asyncio.create_task(run())


async def test_gate_off_has_no_guard_or_extra_suspension(monkeypatch):
    monkeypatch.setattr(_cancel_guard, "_ENABLED", False)
    persistence = recorder()
    assert persistence._cancel_guard is None
    for operation in (
        persistence.on_llm_call_started(RUN_ID, 1),
        persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1),
    ):
        with pytest.raises(StopIteration):
            operation.send(None)


def test_capture_without_running_loop():
    assert _cancel_guard.capture() is None


async def test_capture_without_current_task(monkeypatch):
    monkeypatch.setattr(_cancel_guard.asyncio, "current_task", lambda: None)
    assert _cancel_guard.capture() is None


@pytest.mark.parametrize("stream", [False, True])
async def test_no_store_does_not_capture_guard(monkeypatch, stream):
    def unexpected():
        raise AssertionError("non-persisted run captured a cancellation guard")

    monkeypatch.setattr("dendrux.runtime.persistence.capture", unexpected)
    provider = ScriptedLLM([LLMResponse(text="done")])
    async with Agent(prompt="test", provider=provider, loop=SingleCall()) as agent:
        async with asyncio.timeout(5):
            if stream:
                async with agent.stream("go") as events:
                    async for _ in events:
                        pass
                result = events.result
            else:
                result = await agent.run("go")
        assert result.answer == "done"
