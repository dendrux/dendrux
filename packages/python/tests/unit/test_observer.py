"""Tests for PersistenceObserver — unit tests with a mock StateStore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendrux.runtime.observer import PersistenceObserver
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
        obs = PersistenceObserver(store, "run_1")

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
        obs = PersistenceObserver(store, "run_1")

        await obs.on_message_appended(Message(role=Role.USER, content="a"), iteration=0)
        await obs.on_message_appended(Message(role=Role.ASSISTANT, content="b"), iteration=1)
        tool_msg = Message(role=Role.TOOL, content="c", call_id="c1", name="t")
        await obs.on_message_appended(tool_msg, iteration=1)

        assert [t["order_index"] for t in store.traces] == [0, 1, 2]

    async def test_assistant_with_tool_calls_stores_meta(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

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
        obs = PersistenceObserver(store, "run_1")

        msg = Message(role=Role.TOOL, content='{"result": 42}', call_id="tc_1", name="add")
        await obs.on_message_appended(msg, iteration=1)

        meta = store.traces[0]["meta"]
        assert meta["call_id"] == "tc_1"
        assert meta["tool_name"] == "add"

    async def test_preserves_existing_message_meta(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

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
        obs = PersistenceObserver(store, "run_1", model="claude-sonnet", provider_name="Anthropic")

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
        obs = PersistenceObserver(store, "run_1")

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
        obs = PersistenceObserver(store, "run_1")

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
        obs = PersistenceObserver(store, "run_1")

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
        obs = PersistenceObserver(store, "run_1")

        tc = ToolCall(name="no_args", params={})
        result = ToolResult(name="no_args", call_id=tc.id, payload="{}", success=True)
        await obs.on_tool_completed(tc, result, iteration=1)

        assert store.tool_calls[0]["params"] is None


# ------------------------------------------------------------------
# Redaction
# ------------------------------------------------------------------


class TestRedaction:
    async def test_redact_scrubs_trace_content(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda s: s.replace("secret", "***"))

        msg = Message(role=Role.USER, content="my secret password")
        await obs.on_message_appended(msg, iteration=0)

        assert store.traces[0]["content"] == "my *** password"

    async def test_redact_scrubs_tool_params(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda s: s.replace("token123", "***"))

        tc = ToolCall(name="auth", params={"key": "token123", "count": 5})
        msg = Message(role=Role.ASSISTANT, content="calling", tool_calls=[tc])
        await obs.on_message_appended(msg, iteration=1)

        saved_params = store.traces[0]["meta"]["tool_calls"][0]["params"]
        assert saved_params["key"] == "***"
        assert saved_params["count"] == 5  # non-string values preserved

    async def test_redact_scrubs_tool_result_payload(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda s: s.replace("secret", "***"))

        tc = ToolCall(name="fetch", params={"url": "x"})
        result = ToolResult(name="fetch", call_id=tc.id, payload='{"data": "secret"}', success=True)
        await obs.on_tool_completed(tc, result, iteration=1)

        assert "secret" not in store.tool_calls[0]["result_payload"]
        assert "***" in store.tool_calls[0]["result_payload"]

    async def test_redact_scrubs_error_message(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda s: s.replace("secret", "***"))

        tc = ToolCall(name="bad", params={})
        result = ToolResult(
            name="bad", call_id=tc.id, payload="{}", success=False, error="secret leaked"
        )
        await obs.on_tool_completed(tc, result, iteration=1)

        assert store.tool_calls[0]["error_message"] == "*** leaked"

    async def test_aggressive_redactor_does_not_crash(self) -> None:
        """A redactor that returns '***' for everything should not raise."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda _: "***")

        msg = Message(role=Role.USER, content="hello")
        await obs.on_message_appended(msg, iteration=0)
        assert store.traces[0]["content"] == "***"

        tc = ToolCall(name="fetch", params={"url": "http://example.com"})
        result = ToolResult(name="fetch", call_id=tc.id, payload='{"ok": true}', success=True)
        await obs.on_tool_completed(tc, result, iteration=1)
        assert store.tool_calls[0]["result_payload"] == "***"
        assert store.tool_calls[0]["params"] == {"url": "***"}

    async def test_no_redact_is_identity(self) -> None:
        """Without redact, content passes through unchanged."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

        msg = Message(role=Role.USER, content="secret data")
        await obs.on_message_appended(msg, iteration=0)
        assert store.traces[0]["content"] == "secret data"

    async def test_redact_handles_nested_params(self) -> None:
        """Redaction should recurse into nested dicts and lists."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda s: s.replace("secret", "***"))

        tc = ToolCall(
            name="deploy",
            params={
                "name": "my-app",
                "config": {"api_key": "secret-key", "port": 8080},
                "tags": ["secret-tag", "public"],
            },
        )
        msg = Message(role=Role.ASSISTANT, content="deploying", tool_calls=[tc])
        await obs.on_message_appended(msg, iteration=1)

        saved = store.traces[0]["meta"]["tool_calls"][0]["params"]
        assert saved["name"] == "my-app"
        assert saved["config"]["api_key"] == "***-key"
        assert saved["config"]["port"] == 8080
        assert saved["tags"] == ["***-tag", "public"]

    async def test_redact_nested_in_tool_completed(self) -> None:
        """on_tool_completed also recurses into nested params."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", redact=lambda s: s.replace("secret", "***"))

        tc = ToolCall(name="auth", params={"creds": {"password": "secret123"}})
        result = ToolResult(name="auth", call_id=tc.id, payload="{}", success=True)
        await obs.on_tool_completed(tc, result, iteration=1)

        assert store.tool_calls[0]["params"]["creds"]["password"] == "***123"
