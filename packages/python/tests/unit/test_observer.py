"""Tests for PersistenceRecorder — unit tests with a mock StateStore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolResult,
    UsageStats,
)

# ------------------------------------------------------------------
# Mock StateStore — records calls for assertions
# ------------------------------------------------------------------


@dataclass
class MockStateStore:
    """Fake StateStore that records all calls."""

    traces: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def save_trace(
        self, run_id: str, role: str, content: str, *, order_index: int, meta: Any = None
    ) -> None:
        self.traces.append(
            {
                "run_id": run_id,
                "role": role,
                "content": content,
                "order_index": order_index,
                "meta": meta,
            }
        )

    async def save_tool_call(self, run_id: str, **kwargs: Any) -> None:
        self.tool_calls.append({"run_id": run_id, **kwargs})

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        self.usages.append({"run_id": run_id, **kwargs})

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self._events.setdefault(run_id, []).append(kwargs)

    async def get_run_events(self, run_id: str) -> list[Any]:
        @dataclass
        class _Event:
            sequence_index: int
            event_type: str = ""
            iteration_index: int = 0
            correlation_id: str | None = None
            data: dict[str, Any] | None = None

        return [
            _Event(
                sequence_index=e.get("sequence_index", 0),
                event_type=e.get("event_type", ""),
                iteration_index=e.get("iteration_index", 0),
                correlation_id=e.get("correlation_id"),
                data=e.get("data"),
            )
            for e in self._events.get(run_id, [])
        ]


# ------------------------------------------------------------------
# on_message_appended
# ------------------------------------------------------------------


class TestOnMessageAppended:
    async def test_persists_user_message(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        msg = Message(role=Role.USER, content="hello")
        await obs.on_message_appended(msg, iteration=0)

        assert len(store.traces) == 1
        trace = store.traces[0]
        assert trace["run_id"] == "run_1"
        assert trace["role"] == "user"
        assert trace["content"] == "hello"
        assert trace["order_index"] == 0
        assert trace["meta"]["iteration"] == 0

    async def test_increments_order_index(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        await obs.on_message_appended(Message(role=Role.USER, content="a"), iteration=0)
        await obs.on_message_appended(Message(role=Role.ASSISTANT, content="b"), iteration=1)
        tool_msg = Message(role=Role.TOOL, content="c", call_id="c1", name="t")
        await obs.on_message_appended(tool_msg, iteration=1)

        assert [t["order_index"] for t in store.traces] == [0, 1, 2]

    async def test_assistant_with_tool_calls_stores_meta(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="p1")
        msg = Message(role=Role.ASSISTANT, content="calling", tool_calls=[tc])
        await obs.on_message_appended(msg, iteration=1)

        meta = store.traces[0]["meta"]
        assert "tool_calls" in meta
        assert len(meta["tool_calls"]) == 1
        assert meta["tool_calls"][0]["name"] == "add"
        assert meta["tool_calls"][0]["provider_tool_call_id"] == "p1"

    async def test_tool_message_stores_call_id_and_name(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        msg = Message(role=Role.TOOL, content='{"result": 42}', call_id="tc_1", name="add")
        await obs.on_message_appended(msg, iteration=1)

        meta = store.traces[0]["meta"]
        assert meta["call_id"] == "tc_1"
        assert meta["tool_name"] == "add"

    async def test_preserves_existing_message_meta(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        msg = Message(role=Role.USER, content="hi", meta={"custom_key": "value"})
        await obs.on_message_appended(msg, iteration=0)

        meta = store.traces[0]["meta"]
        assert meta["custom_key"] == "value"
        assert meta["iteration"] == 0


# ------------------------------------------------------------------
# on_llm_call_completed
# ------------------------------------------------------------------


class TestOnLLMCallCompleted:
    async def test_persists_usage(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1", model="claude-sonnet", provider_name="Anthropic")

        response = LLMResponse(
            text="hi",
            usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        await obs.on_llm_call_completed(response, iteration=1)

        assert len(store.usages) == 1
        usage = store.usages[0]
        assert usage["run_id"] == "run_1"
        assert usage["iteration_index"] == 1
        assert usage["usage"].input_tokens == 100
        assert usage["usage"].output_tokens == 50
        assert usage["model"] == "claude-sonnet"
        assert usage["provider"] == "Anthropic"

    async def test_no_model_or_provider(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        response = LLMResponse(text="hi")
        await obs.on_llm_call_completed(response, iteration=0)

        usage = store.usages[0]
        assert usage["model"] is None
        assert usage["provider"] is None


# ------------------------------------------------------------------
# on_tool_completed
# ------------------------------------------------------------------


class TestOnToolCompleted:
    async def test_persists_tool_call(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="p1")
        result = ToolResult(
            name="add",
            call_id=tc.id,
            payload='{"result": 3}',
            success=True,
            duration_ms=42,
        )
        await obs.on_tool_completed(tc, result, iteration=1)

        assert len(store.tool_calls) == 1
        record = store.tool_calls[0]
        assert record["run_id"] == "run_1"
        assert record["tool_call_id"] == tc.id
        assert record["provider_tool_call_id"] == "p1"
        assert record["tool_name"] == "add"
        assert record["target"] == "server"
        assert record["params"] == {"a": 1, "b": 2}
        assert record["result_payload"] == '{"result": 3}'
        assert record["success"] is True
        assert record["duration_ms"] == 42
        assert record["iteration_index"] == 1

    async def test_failed_tool_call(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        tc = ToolCall(name="bad_tool", params={})
        result = ToolResult(
            name="bad_tool",
            call_id=tc.id,
            payload='{"error": "boom"}',
            success=False,
            error="boom",
            duration_ms=5,
        )
        await obs.on_tool_completed(tc, result, iteration=2)

        record = store.tool_calls[0]
        assert record["success"] is False
        assert record["error_message"] == "boom"

    async def test_empty_params_become_none(self) -> None:
        store = MockStateStore()
        obs = PersistenceRecorder(store, "run_1")

        tc = ToolCall(name="no_args", params={})
        result = ToolResult(name="no_args", call_id=tc.id, payload="{}", success=True)
        await obs.on_tool_completed(tc, result, iteration=1)

        assert store.tool_calls[0]["params"] is None
