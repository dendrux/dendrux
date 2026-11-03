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


async def lose_cancel(*args, **kwargs):
    asyncio.current_task().cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.sleep(0)


def recorder():
    return PersistenceRecorder(SimpleNamespace(touch_progress=AsyncMock()), RUN_ID)


@pytest.mark.parametrize("hook", ["model", "tool"])
async def test_legacy_dispatch_recovers_a_lost_cancel(hook):
    async def run():
        persistence = recorder()
        persistence._store.touch_progress.side_effect = lose_cancel
        await persistence._store_calls.touch_progress(RUN_ID)
        persistence._store.touch_progress.side_effect = None
        with pytest.raises(asyncio.CancelledError):
            if hook == "model":
                await persistence.on_llm_call_started(RUN_ID, 1)
            else:
                await persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1)
        assert asyncio.current_task().cancelling() == 1
        await persistence._cancel_guard.check(RUN_ID)

    await asyncio.create_task(run())


async def test_nonzero_entry_count_and_uncancel_are_respected():
    async def run():
        await lose_cancel()
        guard = _cancel_guard.capture()
        await guard.check(RUN_ID)
        assert asyncio.current_task().uncancel() == 0
        await guard.check(RUN_ID)
        await guard.watch(lose_cancel())
        with pytest.raises(asyncio.CancelledError):
            await guard.check(RUN_ID)
        assert asyncio.current_task().cancelling() == 1

    await asyncio.create_task(run())


async def test_pending_native_cancel_is_delivered_once(caplog):
    async def run():
        guard = _cancel_guard.capture()

        async def store_call():
            asyncio.current_task().cancel("native")

        await guard.watch(store_call())
        with pytest.raises(asyncio.CancelledError) as caught:
            await guard.check(RUN_ID)
        assert caught.value.args == ("native",)
        assert asyncio.current_task().cancelling() == 1
        await guard.check(RUN_ID)

    await asyncio.create_task(run())
    assert not any(getattr(r, "code", None) == "run_cancel_recovered" for r in caplog.records)


async def test_count_is_rechecked_after_yield():
    async def run():
        guard = _cancel_guard.capture()
        await guard.watch(lose_cancel())
        asyncio.get_running_loop().call_soon(asyncio.current_task().uncancel)
        await guard.check(RUN_ID)
        assert asyncio.current_task().cancelling() == 0

    await asyncio.create_task(run())


@pytest.mark.parametrize("delivered", [False, True])
async def test_balanced_or_delivered_cancel_clears_store_loss(delivered):
    async def run():
        guard = _cancel_guard.capture()
        await guard.watch(lose_cancel())
        if delivered:

            async def store_call():
                asyncio.current_task().cancel()
                await asyncio.sleep(0)

            with pytest.raises(asyncio.CancelledError):
                await guard.watch(store_call())
        else:
            asyncio.current_task().uncancel()
        await guard.check(RUN_ID)

    await asyncio.create_task(run())


async def test_store_error_after_lost_cancel_preserves_evidence():
    async def run():
        guard = _cancel_guard.capture()

        async def failed():
            await lose_cancel()
            raise ValueError("retryable store failure")

        with pytest.raises(ValueError):
            await guard.watch(failed())
        # A subsequent successful await must not erase the first loss.
        await guard.watch(asyncio.sleep(0))
        with pytest.raises(asyncio.CancelledError):
            await guard.check(RUN_ID)
        await guard.check(RUN_ID)

    await asyncio.create_task(run())


async def test_new_recorder_ignores_inherited_count_but_detects_its_own_loss():
    async def run():
        await lose_cancel()
        persistence = recorder()
        await persistence.on_llm_call_started(RUN_ID, 1)
        persistence._store.touch_progress.side_effect = lose_cancel
        with pytest.raises(asyncio.CancelledError):
            await persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1)
        assert asyncio.current_task().cancelling() == 2

    await asyncio.create_task(run())


async def test_other_task_does_not_use_recorders_baseline():
    persistence = recorder()

    async def run():
        persistence._store.touch_progress.side_effect = lose_cancel
        await persistence._store_calls.touch_progress(RUN_ID)
        persistence._store.touch_progress.side_effect = None
        await persistence.on_llm_call_started(RUN_ID, 1)
        await persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1)
        assert asyncio.current_task().cancelling() == 1

    await asyncio.create_task(run())
    await persistence._cancel_guard.check(RUN_ID)


async def test_gate_off_has_no_guard_or_extra_suspension(monkeypatch):
    monkeypatch.setattr(_cancel_guard, "_ENABLED", False)
    persistence = recorder()
    await assert_inactive(persistence._cancel_guard)
    assert persistence._store_calls is persistence._store
    for operation in (
        persistence.on_llm_call_started(RUN_ID, 1),
        persistence.on_tool_started(RUN_ID, ToolCall("effect"), 1),
    ):
        with pytest.raises(StopIteration):
            operation.send(None)


async def assert_inactive(guard):
    store = object()
    assert not isinstance(guard, _cancel_guard._CancelGuard)
    assert guard.store_calls(store) is store
    assert await guard.check("run") is None


def test_capture_without_running_loop():
    asyncio.run(assert_inactive(_cancel_guard.capture()))


async def test_capture_without_current_task(monkeypatch):
    monkeypatch.setattr(_cancel_guard.asyncio, "current_task", lambda: None)
    await assert_inactive(_cancel_guard.capture())


async def test_capture_with_gate_off_is_inactive(monkeypatch):
    monkeypatch.setattr(_cancel_guard, "_ENABLED", False)
    await assert_inactive(_cancel_guard.capture())


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
