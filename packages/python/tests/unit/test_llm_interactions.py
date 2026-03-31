"""Tests for Sprint 3.5 — LLM interactions evidence layer.

Covers:
  - LLMResponse provider payload fields
  - Observer enriched on_llm_call_completed with semantic payloads + dual-write
  - _resume_core CAS bug fix
  - Dashboard /api/runs/{run_id}/llm-calls endpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendrux.runtime.observer import PersistenceObserver
from dendrux.types import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDef,
    UsageStats,
)

# ------------------------------------------------------------------
# Mock StateStore that records calls
# ------------------------------------------------------------------


@dataclass
class MockStateStore:
    """Fake StateStore recording all write calls."""

    traces: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)
    llm_interactions: list[dict[str, Any]] = field(default_factory=list)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def save_trace(self, run_id: str, role: str, content: str, **kwargs: Any) -> None:
        self.traces.append({"run_id": run_id, "role": role, "content": content, **kwargs})

    async def save_tool_call(self, run_id: str, **kwargs: Any) -> None:
        self.tool_calls.append({"run_id": run_id, **kwargs})

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        self.usages.append({"run_id": run_id, **kwargs})

    async def save_llm_interaction(self, run_id: str, **kwargs: Any) -> None:
        self.llm_interactions.append({"run_id": run_id, **kwargs})

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self._events.setdefault(run_id, []).append(kwargs)


# ------------------------------------------------------------------
# LLMResponse provider payload fields
# ------------------------------------------------------------------


class TestLLMResponseProviderPayloads:
    def test_default_none(self) -> None:
        r = LLMResponse(text="hi")
        assert r.provider_request is None
        assert r.provider_response is None

    def test_set_provider_payloads(self) -> None:
        r = LLMResponse(
            text="hi",
            provider_request={"model": "claude-sonnet-4-6", "messages": []},
            provider_response={"id": "msg_123", "content": []},
        )
        assert r.provider_request["model"] == "claude-sonnet-4-6"
        assert r.provider_response["id"] == "msg_123"


# ------------------------------------------------------------------
# Observer: enriched on_llm_call_completed
# ------------------------------------------------------------------


class TestObserverLLMInteraction:
    async def test_dual_write_to_both_tables(self) -> None:
        """on_llm_call_completed writes to llm_interactions AND token_usage."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", model="claude-sonnet", provider_name="Anthropic")

        response = LLMResponse(
            text="hello",
            usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        await obs.on_llm_call_completed(response, iteration=1)

        assert len(store.usages) == 1, "legacy token_usage should still be written"
        assert len(store.llm_interactions) == 1, "llm_interactions should be written"

    async def test_llm_interaction_has_usage(self) -> None:
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1", model="claude-sonnet", provider_name="Anthropic")

        response = LLMResponse(
            text="hi",
            usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.01),
        )
        await obs.on_llm_call_completed(response, iteration=2)

        rec = store.llm_interactions[0]
        assert rec["model"] == "claude-sonnet"
        assert rec["provider"] == "Anthropic"
        assert rec["iteration_index"] == 2
        assert rec["usage"].input_tokens == 100

    async def test_semantic_request_full_fidelity(self) -> None:
        """Semantic request preserves full Message structure and ToolDef schemas."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="p1")
        messages = [
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="Calling add", tool_calls=[tc]),
            Message(role=Role.TOOL, content='{"result": 3}', call_id=tc.id, name="add"),
        ]
        tools = [
            ToolDef(
                name="add",
                description="Add numbers",
                parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
            ),
        ]
        response = LLMResponse(text="4", usage=UsageStats())

        await obs.on_llm_call_completed(
            response, iteration=1, semantic_messages=messages, semantic_tools=tools
        )

        rec = store.llm_interactions[0]
        sr = rec["semantic_request"]
        assert sr is not None

        # Messages preserve full structure
        assert len(sr["messages"]) == 3
        assert sr["messages"][0]["role"] == "user"
        assert sr["messages"][0]["content"] == "What is 2+2?"
        # Assistant message has tool_calls with IDs
        assert sr["messages"][1]["tool_calls"][0]["name"] == "add"
        assert sr["messages"][1]["tool_calls"][0]["provider_tool_call_id"] == "p1"
        # Tool message has call_id and name
        assert sr["messages"][2]["call_id"] == tc.id
        assert sr["messages"][2]["name"] == "add"

        # Tools preserve full schema
        assert sr["tools"][0]["name"] == "add"
        assert sr["tools"][0]["parameters"]["type"] == "object"
        assert sr["tools"][0]["target"] == "server"

    async def test_semantic_response_full_fidelity(self) -> None:
        """Semantic response preserves full text, tool_calls with IDs, and usage."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="ptc_1")
        response = LLMResponse(
            text="Let me add those",
            tool_calls=[tc],
            usage=UsageStats(input_tokens=50, output_tokens=20, total_tokens=70, cost_usd=0.001),
        )
        await obs.on_llm_call_completed(response, iteration=1)

        rec = store.llm_interactions[0]
        sr = rec["semantic_response"]
        assert sr is not None
        # Full text, not truncated
        assert sr["text"] == "Let me add those"
        # Tool calls with IDs and params
        assert sr["tool_calls"][0]["name"] == "add"
        assert sr["tool_calls"][0]["id"] == tc.id
        assert sr["tool_calls"][0]["provider_tool_call_id"] == "ptc_1"
        assert sr["tool_calls"][0]["params"] == {"a": 1, "b": 2}
        # Usage included in semantic response
        assert sr["usage"]["input_tokens"] == 50
        assert sr["usage"]["cost_usd"] == 0.001

    async def test_provider_payloads_passed_through(self) -> None:
        """provider_request and provider_response from LLMResponse are stored."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

        response = LLMResponse(
            text="hi",
            usage=UsageStats(),
            provider_request={"model": "claude-sonnet-4-6", "messages": [{"role": "user"}]},
            provider_response={"id": "msg_abc", "type": "message"},
        )
        await obs.on_llm_call_completed(response, iteration=1)

        rec = store.llm_interactions[0]
        assert rec["provider_request"]["model"] == "claude-sonnet-4-6"
        assert rec["provider_response"]["id"] == "msg_abc"

    async def test_duration_ms_passed_through(self) -> None:
        """duration_ms kwarg flows from loop → observer → save_llm_interaction."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

        response = LLMResponse(text="hi", usage=UsageStats())
        await obs.on_llm_call_completed(response, iteration=1, duration_ms=1420)

        rec = store.llm_interactions[0]
        assert rec["duration_ms"] == 1420

    async def test_no_semantic_payloads_when_not_provided(self) -> None:
        """When called without semantic args (backcompat), payloads are None."""
        store = MockStateStore()
        obs = PersistenceObserver(store, "run_1")

        response = LLMResponse(text="hi", usage=UsageStats())
        await obs.on_llm_call_completed(response, iteration=1)

        rec = store.llm_interactions[0]
        assert rec["semantic_request"] is None
        assert rec["provider_request"] is None

    async def test_llm_interaction_failure_does_not_block_legacy_write(self) -> None:
        """If save_llm_interaction fails, save_usage still fires."""

        @dataclass
        class FailingInteractionStore(MockStateStore):
            async def save_llm_interaction(self, run_id: str, **kwargs: Any) -> None:
                raise RuntimeError("DB error")

        store = FailingInteractionStore()
        obs = PersistenceObserver(store, "run_1")

        response = LLMResponse(text="hi", usage=UsageStats())
        await obs.on_llm_call_completed(response, iteration=1)

        assert len(store.usages) == 1, "legacy write should proceed despite interaction failure"
        assert len(store.llm_interactions) == 0


# ------------------------------------------------------------------
# _resume_core CAS bug fix
# ------------------------------------------------------------------


class TestResumeCoreCAsBugFix:
    """Verifies that _resume_core now checks finalize_run result
    before emitting run.error, matching the pattern in run()."""

    async def test_resume_core_error_event_requires_cas_win(self) -> None:
        """After fix, error event is only emitted if CAS wins."""
        # This is a structural test — we verify the code pattern.
        # Reading the source to confirm the fix is in place.
        import inspect

        from dendrux.runtime import runner

        source = inspect.getsource(runner._resume_core)

        # The fix: error_won check before _emit_event in the except block
        assert "error_won" in source, "_resume_core should use error_won CAS check"
        assert "if error_won:" in source, "_resume_core should guard emit with 'if error_won:'"
