"""Cancellation, stream idempotency, and answer contracts reported by Oreo."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, tool
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock
from dendrux.llm.mock import MockLLM
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall
from tests._helpers.gated_provider import GatedStreamLLM


@tool()
async def echo(text: str) -> str:
    """Echo the supplied text."""
    return text


async def test_run_timeout_finalizes_and_reraises(store, monkeypatch):
    provider = MockLLM([])
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(provider, "complete", blocked)
    notifier = AsyncMock(on_run_started=AsyncMock(), on_run_finished=AsyncMock())
    agent = Agent(provider=provider, prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(entered.wait(), 5)
    run_id = notifier.on_run_started.call_args.args[0]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, 0.01)

    assert task.cancelled()
    record = await store.get_run(run_id)
    assert record.status == "cancelled"
    events = await store.get_run_events(run_id)
    assert [e.event_type for e in events].count("run.cancelled") == 1
    assert "run.error" not in [e.event_type for e in events]
    assert notifier.on_run_finished.call_args.args[1].status == RunStatus.CANCELLED
    assert get_delegation_context() is None


@pytest.mark.parametrize("existing_status", [None, "success", "waiting_client_tool"])
async def test_cancellation_during_startup_preserves_existing_outcomes(store, existing_status):
    entered = asyncio.Event()
    run_ids = []

    async def started(run_id, **kwargs):
        run_ids.append(run_id)
        entered.set()
        await asyncio.Event().wait()

    notifier = AsyncMock(on_run_started=started, on_run_finished=AsyncMock())
    provider = MockLLM([])
    agent = Agent(provider=provider, prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(entered.wait(), 5)
    run_id = run_ids[0]
    if existing_status == "success":
        await store.finalize_run(run_id, status="success", answer="winner")
    elif existing_status:
        await store.pause_run(run_id, status=existing_status, pause_data={})
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await store.get_run(run_id)).status == (existing_status or "cancelled")
    events = await store.get_run_events(run_id)
    assert [e.event_type for e in events].count("run.cancelled") == (existing_status is None)
    assert provider.calls_made == 0


async def test_cancellation_before_row_exists_is_quiet(store, monkeypatch, caplog):
    async def cancelled_create(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(store, "create_run", cancelled_create)
    agent = Agent(provider=MockLLM([]), prompt="test", state_store=store)
    with caplog.at_level(logging.ERROR, logger="dendrux"), pytest.raises(asyncio.CancelledError):
        await agent.run("hello")
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_cancellation_cleanup_failure_does_not_mask_cancellation(store, monkeypatch):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    provider = MockLLM([])
    monkeypatch.setattr(provider, "complete", cancelled)
    monkeypatch.setattr(store, "finalize_run", AsyncMock(side_effect=RuntimeError("db down")))
    agent = Agent(provider=provider, prompt="test", state_store=store)
    with pytest.raises(asyncio.CancelledError):
        await agent.run("hello")


async def test_cancellation_without_persistence_closes_lifecycle(monkeypatch):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    provider = MockLLM([])
    monkeypatch.setattr(provider, "complete", cancelled)
    notifier = AsyncMock(on_run_finished=AsyncMock())
    agent = Agent(provider=provider, prompt="test")
    monkeypatch.setattr(agent, "_resolve_state_store", AsyncMock(return_value=None))
    with pytest.raises(asyncio.CancelledError):
        await agent.run("hello", notifier=notifier)
    assert notifier.on_run_finished.call_args.args[1].status == RunStatus.CANCELLED


async def test_timeout_during_error_finalization_cancels_run(store, monkeypatch):
    entered = asyncio.Event()
    finalize = store.finalize_run
    run_ids = []

    async def blocked_error(run_id, **kwargs):
        run_ids.append(run_id)
        if kwargs["status"] == "error":
            entered.set()
            await asyncio.Event().wait()
        return await finalize(run_id, **kwargs)

    monkeypatch.setattr(store, "finalize_run", blocked_error)
    notifier = AsyncMock()
    agent = Agent(provider=MockLLM([]), prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(entered.wait(), 5)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, 0.01)
    assert (await store.get_run(run_ids[0])).status == "cancelled"
    events = [e.event_type for e in await store.get_run_events(run_ids[0])]
    assert events.count("run.cancelled") == 1
    assert "run.error" not in events
    notifier.on_run_failed.assert_not_awaited()
    assert notifier.on_run_finished.call_args.args[1].status == RunStatus.CANCELLED


@pytest.mark.parametrize("blocked_write", ["finalize", "event"])
async def test_repeated_cancellation_waits_for_durable_cleanup(store, monkeypatch, blocked_write):
    model_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    run_ids = []
    provider = MockLLM([])

    async def blocked_model(*args, **kwargs):
        model_entered.set()
        await asyncio.Event().wait()

    finalize = store.finalize_run
    save_event = store.save_run_event

    async def gated_finalize(run_id, **kwargs):
        run_ids.append(run_id)
        if blocked_write == "finalize":
            cleanup_entered.set()
            await release_cleanup.wait()
        return await finalize(run_id, **kwargs)

    async def gated_event(run_id, **kwargs):
        if kwargs["event_type"] == "run.cancelled" and blocked_write == "event":
            cleanup_entered.set()
            await release_cleanup.wait()
        return await save_event(run_id, **kwargs)

    monkeypatch.setattr(provider, "complete", blocked_model)
    monkeypatch.setattr(store, "finalize_run", gated_finalize)
    monkeypatch.setattr(store, "save_run_event", gated_event)
    notifier = AsyncMock()
    agent = Agent(provider=provider, prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(model_entered.wait(), 5)
    task.cancel("initial cancellation")
    await asyncio.wait_for(cleanup_entered.wait(), 5)
    for _ in range(2):
        task.cancel("later cancellation")
        await asyncio.sleep(0)
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError, match="initial cancellation"):
        await task
    record = await store.get_run(run_ids[0])
    assert record.status == "cancelled"
    events = [e.event_type for e in await store.get_run_events(run_ids[0])]
    assert events.count("run.cancelled") == 1
    notifier.on_run_finished.assert_awaited_once()


@pytest.mark.parametrize("hook", ["recorder", "notifier"])
async def test_cancel_during_finish_hook_does_not_repeat_it(store, monkeypatch, hook):
    from dendrux.runtime.persistence import PersistenceRecorder

    entered = asyncio.Event()
    recorder_results = []
    notifier_results = []

    async def recorded(self, run_id, result):
        recorder_results.append(result)
        if hook == "recorder" and len(recorder_results) == 1:
            entered.set()
            await asyncio.Event().wait()

    async def notified(run_id, result):
        notifier_results.append(result)
        if hook == "notifier" and len(notifier_results) == 1:
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(PersistenceRecorder, "on_run_finished", recorded)
    notifier = AsyncMock(on_run_finished=notified)
    agent = Agent(provider=MockLLM([LLMResponse(text="done")]), prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(recorder_results) == len(notifier_results) == 1
    assert recorder_results[0].answer == notifier_results[0].answer == "done"
    assert notifier_results[0].status == RunStatus.SUCCESS
    record = await store.get_run(notifier_results[0].run_id)
    assert record.status == "success"
    assert record.answer == "done"
    events = [e.event_type for e in await store.get_run_events(record.id)]
    assert events.count("run.completed") == 1
    assert "run.cancelled" not in events


async def test_cancel_during_failure_hook_does_not_finish_lifecycle_again(store):
    entered = asyncio.Event()

    async def failed(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    notifier = AsyncMock(on_run_failed=failed)
    agent = Agent(provider=MockLLM([]), prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    notifier.on_run_finished.assert_not_awaited()
    run_id = notifier.on_run_started.call_args.args[0]
    assert (await store.get_run(run_id)).status == "error"


async def test_cancellation_keeps_lifecycle_hooks_in_run_task(store, monkeypatch):
    context = ContextVar("notifier_context", default=None)
    token = None
    finished = False
    entered = asyncio.Event()
    provider = MockLLM([])

    async def started(*args, **kwargs):
        nonlocal token
        token = context.set("active")

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def finish(*args, **kwargs):
        nonlocal finished
        assert asyncio.current_task() is task
        assert context.get() == "active"
        context.reset(token)
        finished = True

    monkeypatch.setattr(provider, "complete", blocked)
    notifier = AsyncMock(on_run_started=started, on_run_finished=finish)
    agent = Agent(provider=provider, prompt="test", state_store=store)
    task = asyncio.create_task(agent.run("hello", notifier=notifier))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished
    assert context.get() is None


@pytest.mark.parametrize("first_stream", [False, True])
async def test_stream_idempotency_reuses_run_result(store, first_stream):
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(provider=provider, prompt="test", state_store=store)
    if first_stream:
        original = agent.stream("hello", idempotency_key="request")
        async with original:
            _ = [event async for event in original]
        result = original.result
    else:
        result = await agent.run("hello", idempotency_key="request")
    before = await store.get_run_events(result.run_id)

    stream = agent.stream("hello", idempotency_key="request")
    async with stream:
        events = [event async for event in stream]
    assert [e.type for e in events] == [RunEventType.RUN_COMPLETED]
    assert stream.run_id == result.run_id
    assert events[0].run_id == result.run_id
    assert stream.result.answer == "done"
    assert provider.calls_made == 1
    assert "idempotency_key" not in provider.call_history[0]["kwargs"]
    assert await store.get_run_events(result.run_id) == before
    assert (await agent.run("hello", idempotency_key="request")).run_id == result.run_id


@pytest.mark.parametrize(
    "status,event_type",
    [
        ("error", RunEventType.RUN_ERROR),
        ("cancelled", RunEventType.RUN_CANCELLED),
        ("max_iterations", RunEventType.RUN_COMPLETED),
    ],
)
async def test_cached_stream_preserves_terminal_status(store, status, event_type):
    from dendrux.types import compute_idempotency_fingerprint

    provider = MockLLM([])
    agent = Agent(provider=provider, prompt="test", state_store=store)
    await store.create_run(
        "cached",
        agent.name,
        idempotency_key="request",
        idempotency_fingerprint=compute_idempotency_fingerprint(agent.name, "hi"),
    )
    await store.finalize_run("cached", status=status, answer="partial", error="original error")
    stream = agent.stream("hi", idempotency_key="request")
    events = [event async for event in stream]
    assert len(events) == 1
    assert events[0].type == event_type
    assert stream.result.status.value == status
    assert stream.result.answer == "partial"
    assert stream.result.error == "original error"
    assert provider.calls_made == 0


@pytest.mark.parametrize("change", ["input", "history", "context"])
async def test_stream_idempotency_conflicts(store, change):
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(provider=provider, prompt="test", state_store=store)
    result = await agent.run("hello", idempotency_key="request")
    kwargs = {}
    if change == "history":
        kwargs["history"] = [ChatMessage.user("hi"), ChatMessage.assistant("hello")]
    elif change == "context":
        kwargs["context"] = [ContextBlock(content="new context")]
    notifier = AsyncMock()
    stream = agent.stream(
        "changed" if change == "input" else "hello",
        idempotency_key="request",
        notifier=notifier,
        **kwargs,
    )
    events = [event async for event in stream]
    assert events[-1].type == RunEventType.RUN_ERROR
    assert stream.result.meta["error_type"] == "IdempotencyConflictError"
    notifier.on_run_started.assert_not_awaited()
    notifier.on_run_failed.assert_not_awaited()
    notifier.on_run_finished.assert_not_awaited()
    assert (await store.get_run(result.run_id)).status == "success"
    assert provider.calls_made == 1


async def test_active_duplicate_stream_cannot_cancel_owner(store):
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(provider=provider, prompt="test", state_store=store)
    notifier = AsyncMock()
    owner = agent.stream("hello", idempotency_key="request", notifier=notifier)
    async with owner:
        iterator = owner.__aiter__()
        assert (await anext(iterator)).type == RunEventType.RUN_STARTED
        before = await store.get_run_events(owner.run_id)
        duplicate = agent.stream("hello", idempotency_key="request", notifier=notifier)
        async with duplicate:
            events = [event async for event in duplicate]
        assert events[-1].type == RunEventType.RUN_ERROR
        assert duplicate.result.meta["error_type"] == "RunAlreadyActiveError"
        notifier.on_run_failed.assert_not_awaited()
        notifier.on_run_finished.assert_not_awaited()
        assert (await store.get_run(owner.run_id)).status == "running"
        assert await store.get_run_events(owner.run_id) == before
        remaining = [event async for event in iterator]
        assert remaining[-1].type == RunEventType.RUN_COMPLETED
    assert provider.calls_made == 1


async def test_stream_idempotency_requires_persistence(monkeypatch):
    provider = MockLLM([])
    agent = Agent(provider=provider, prompt="test")
    monkeypatch.setattr(agent, "_resolve_state_store", AsyncMock(return_value=None))
    stream = agent.stream("hello", idempotency_key="request")
    events = [event async for event in stream]
    assert events[-1].type == RunEventType.RUN_ERROR
    assert "idempotency_key requires persistence" in stream.result.error
    assert provider.calls_made == 0


async def test_duplicate_stream_preserves_owners_interrupt(store):
    provider = GatedStreamLLM()
    agent = Agent(provider=provider, prompt="test", state_store=store)
    owner = agent.stream("hello", idempotency_key="request")

    async def consume():
        async with owner:
            return [event async for event in owner]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(provider.first_sent.wait(), 5)
        duplicate = agent.stream("hello", idempotency_key="request")
        events = [event async for event in duplicate]
        assert events[-1].type == RunEventType.RUN_ERROR
        await agent.cancel_run(owner.run_id)
        events = await asyncio.wait_for(task, 5)
        assert events[-1].type == RunEventType.RUN_CANCELLED
        assert events[-1].run_result.answer == provider.first
        assert provider.call_count == 1
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


async def test_cancelled_cached_lookup_never_finalizes_existing_run(store, monkeypatch):
    from dendrux.runtime import runner

    agent = Agent(provider=MockLLM([LLMResponse(text="done")]), prompt="test", state_store=store)
    result = await agent.run("hello", idempotency_key="request")
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_build_cached_result", blocked)
    finalize = AsyncMock(wraps=store.finalize_run)
    monkeypatch.setattr(store, "finalize_run", finalize)
    stream = agent.stream("hello", idempotency_key="request")

    async def consume():
        async with stream:
            return [event async for event in stream]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finalize.assert_not_awaited()
    assert stream.run_id == result.run_id
    assert (await store.get_run(result.run_id)).status == "success"


async def test_unstarted_idempotent_stream_has_no_side_effects(store):
    provider = MockLLM([])
    agent = Agent(provider=provider, prompt="test", state_store=store)
    stream = agent.stream("hello", idempotency_key="request")
    await stream.aclose()
    assert await store.get_run(stream.run_id) is None
    assert provider.calls_made == 0


async def test_run_answer_contains_only_final_turn(store):
    provider = MockLLM(
        [
            LLMResponse(text="before tool", tool_calls=[ToolCall("echo", {"text": "ok"})]),
            LLMResponse(text="final turn"),
        ]
    )
    agent = Agent(provider=provider, prompt="test", tools=[echo], state_store=store)
    result = await agent.run("hello")
    assert result.answer == "final turn"
    assert (await store.get_run(result.run_id)).answer == "final turn"


@pytest.mark.parametrize("stop_at", [None, RunEventType.TOOL_RESULT, RunEventType.TEXT_DELTA])
async def test_answer_is_final_turn_and_cancel_buffer_resets(store, stop_at):
    provider = MockLLM(
        [
            LLMResponse(
                text="before tool", tool_calls=[ToolCall(name="echo", params={"text": "ok"})]
            ),
            LLMResponse(text="final turn"),
        ]
    )
    agent = Agent(provider=provider, prompt="test", tools=[echo], state_store=store)
    stream = agent.stream("hello")
    text = []
    tool_finished = False
    async with stream:
        async for event in stream:
            if event.type == RunEventType.TEXT_DELTA:
                text.append(event.text)
            if event.type == RunEventType.TOOL_RESULT:
                tool_finished = True
            if tool_finished and event.type == stop_at:
                break
    record = await store.get_run(stream.run_id)
    assert record.answer == (None if stop_at == RunEventType.TOOL_RESULT else "final turn")
    assert text[0] == "before tool"
    assert record.status == ("success" if stop_at is None else "cancelled")
