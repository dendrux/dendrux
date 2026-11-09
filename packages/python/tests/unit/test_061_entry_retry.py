"""Boundary cases for successful retry history trimming."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import _retry_history
from dendrux.runtime.state import RunRecord, TraceRecord


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
