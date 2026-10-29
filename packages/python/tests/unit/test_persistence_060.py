"""Lost-ack retries must retain the logical recorder write's identity."""

from functools import wraps

import pytest
import pytest_asyncio
from sqlalchemy import select

from dendrux.db.models import LLMInteraction, ReactTrace, RunEvent, ToolCallRecord
from dendrux.errors import PersistenceError, RunFencedError
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import LLMResponse, Message, Role, ToolCall, ToolResult, generate_ulid
from tests._helpers.v030_oracle import create_sqlite_engine


@pytest_asyncio.fixture
async def store():
    engine = await create_sqlite_engine()
    try:
        yield SQLAlchemyStateStore(engine)
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "kind,model",
    [
        ("save_trace", ReactTrace),
        ("save_tool_call", ToolCallRecord),
        ("save_run_event", RunEvent),
        ("save_llm_interaction", LLMInteraction),
    ],
)
async def test_a21_recorder_retries_the_same_identity(store, monkeypatch, kind, model):
    run_id = generate_ulid()
    await store.create_run(run_id, "retry")
    original = getattr(store, kind)
    calls = 0

    @wraps(original)
    async def lose_ack(*args, **kwargs):
        nonlocal calls
        result = await original(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise PersistenceError(retryable=True)
        return result

    monkeypatch.setattr(store, kind, lose_ack)
    recorder = PersistenceRecorder(store, run_id)
    if kind == "save_trace":
        await recorder.on_message_appended(run_id, Message(role=Role.USER, content="hello"), 0)
    elif kind == "save_tool_call":
        call = ToolCall(name="search", params={})
        await recorder.on_tool_completed(
            run_id, call, ToolResult(call_id=call.id, name="search", payload="1", success=True), 0
        )
    elif kind == "save_run_event":
        await recorder._emit_event("llm.completed", 0)
    else:
        await recorder.on_llm_call_completed(run_id, LLMResponse(text="ok"), 0)
    async with store._engine.connect() as conn:
        ids = (
            (await conn.execute(select(model.id).where(model.agent_run_id == run_id)))
            .scalars()
            .all()
        )
    assert calls == 2 and len(ids) == 1


@pytest.mark.parametrize("method", ["save_usage", "save_llm_interaction", "touch_progress"])
async def test_a3_best_effort_recorder_never_swallows_fence(store, monkeypatch, method):
    run_id = generate_ulid()
    await store.create_run(run_id, "fence")

    @wraps(getattr(store, method))
    async def fenced(*args, **kwargs):
        raise RunFencedError(run_id, claim_epoch=0)

    monkeypatch.setattr(store, method, fenced)
    with pytest.raises(RunFencedError):
        await PersistenceRecorder(store, run_id).on_llm_call_completed(
            run_id, LLMResponse(text="ok"), 0
        )


async def test_format1_tool_start_cancels_the_progress_write(store, monkeypatch):
    import asyncio

    entered = asyncio.Event()
    released = asyncio.Event()
    cancelled = asyncio.Event()
    completed = []

    async def stalled_write(run_id):
        entered.set()
        try:
            await released.wait()
            completed.append(run_id)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(store, "touch_progress", stalled_write)
    recorder = PersistenceRecorder(store, generate_ulid())
    assert not recorder._format2
    task = asyncio.create_task(recorder.on_tool_started("run", ToolCall("effect"), 1))
    await entered.wait()
    task.cancel()
    try:
        # The write must observe cancellation without waiting for release.
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not completed
    finally:
        released.set()
        await asyncio.gather(task, return_exceptions=True)
