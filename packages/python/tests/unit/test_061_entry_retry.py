"""Boundary cases for successful retry history trimming."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import _retry_history
from dendrux.runtime.state import RunRecord, StoreCapabilities, TraceRecord


def _trace(index: int, role: str, content: str, **meta: object) -> TraceRecord:
    return TraceRecord(id=str(index), order_index=index, role=role, content=content, meta=meta)


@pytest.mark.parametrize("status", ["success", "error", "cancelled", "max_iterations"])
@pytest.mark.parametrize("tail", ["answer", "tool_call", "user"])
async def test_retry_trims_only_successful_final_answer(status: str, tail: str) -> None:
    traces = [
        _trace(0, "user", "Earlier question"),
        _trace(1, "assistant", "Earlier answer"),
        _trace(2, "user", "Current question"),
    ]
    if tail == "tool_call":
        traces.append(
            _trace(3, "assistant", "Calling lookup", tool_calls=[{"id": "c", "name": "lookup"}])
        )
    elif tail == "answer":
        traces.append(_trace(3, "assistant", "Final answer"))
    store = AsyncMock()
    store.get_traces.return_value = traces
    provider = MockLLM([])
    original = RunRecord(id="source", agent_name="a", status=status)
    _, history = await _retry_history(
        store, original, Agent(provider=provider, prompt="Answer."), provider, {}
    )
    expected = [t.content for t in traces]
    if status == "success" and tail == "answer":
        expected.pop()
    assert [m.content for m in history] == expected
    assert len(traces) == (3 if tail == "user" else 4)


@pytest.mark.parametrize(
    "storage_format,checkpointing", [(1, False), (1, True), (2, False), (2, True)]
)
async def test_retry_selects_conversation_reader_and_filters_system(
    monkeypatch, storage_format, checkpointing
):
    from dendrux.runtime import replay

    traces = [_trace(0, "system", "Old prompt"), _trace(1, "user", "Question")]
    store = SimpleNamespace(
        capabilities=StoreCapabilities(checkpointing=checkpointing),
        get_traces=AsyncMock(return_value=traces),
    )
    reader = AsyncMock(return_value=traces)
    monkeypatch.setattr(replay, "conversation_records", reader)
    provider = MockLLM([])
    agent = Agent(provider=provider, prompt="New prompt")
    source = RunRecord(id="source", agent_name="a", status="error", storage_format=storage_format)
    found, history = await _retry_history(store, source, agent, provider, {})
    assert found is traces
    assert [m.content for m in history] == (
        ["Question"] if storage_format == 2 else ["Old prompt", "Question"]
    )
    if storage_format == 2 and checkpointing:
        reader.assert_awaited_once_with(store, source.id, limits=agent._limits)
        store.get_traces.assert_not_awaited()
    else:
        store.get_traces.assert_awaited_once_with(source.id)
        reader.assert_not_awaited()


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("roles", [[], ["assistant"], ["system"]])
async def test_retry_refuses_history_empty_after_trimming_or_filtering(storage_format, roles):
    store = AsyncMock()
    store.get_traces.return_value = [_trace(i, role, "text") for i, role in enumerate(roles)]
    provider = MockLLM([])
    source = RunRecord(id="source", agent_name="a", status="success", storage_format=storage_format)
    if storage_format == 1 and roles == ["system"]:
        _, history = await _retry_history(
            store, source, Agent(provider=provider, prompt="p"), provider, {}
        )
        assert [m.role.value for m in history] == ["system"]
    else:
        with pytest.raises(ValueError, match="has no persisted traces"):
            await _retry_history(store, source, Agent(provider=provider, prompt="p"), provider, {})
