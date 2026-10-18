"""Stop reason normalization on every provider, batch and stream (#42)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage

from dendrux.llm._stop_reason import (
    anthropic_stop_reason,
    chat_stop_reason,
    responses_stop_reason,
)
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.openai import OpenAIProvider
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.llm.openrouter import OpenRouterProvider
from dendrux.types import LLMResponse, Message, Role, StreamEventType

_USER = [Message(role=Role.USER, content="hi")]


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------


class TestAnthropicMapping:
    @pytest.mark.parametrize(
        "raw",
        ["end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal", "pause_turn"],
    )
    def test_sdk_values_are_identity(self, raw: str) -> None:
        assert anthropic_stop_reason(raw) == (raw, raw)

    def test_unknown_value_is_other_with_raw_kept(self) -> None:
        assert anthropic_stop_reason("model_context_window_exceeded") == (
            "other",
            "model_context_window_exceeded",
        )

    @pytest.mark.parametrize("raw", [None, ""])
    def test_missing_is_none(self, raw: Any) -> None:
        assert anthropic_stop_reason(raw) == (None, None)


class TestChatMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("stop", "end_turn"),
            ("tool_calls", "tool_use"),
            ("function_call", "tool_use"),
            ("length", "max_tokens"),
            ("content_filter", "content_filter"),
            ("error", "other"),
        ],
    )
    def test_values(self, raw: str, expected: str) -> None:
        assert chat_stop_reason(raw) == (expected, raw)

    @pytest.mark.parametrize("raw", [None, ""])
    def test_missing_is_none(self, raw: Any) -> None:
        assert chat_stop_reason(raw) == (None, None)


def _resp(status: Any, *, reason: str | None = None, output: list[Any] | None = None) -> Any:
    details = SimpleNamespace(reason=reason) if reason is not None else None
    return SimpleNamespace(status=status, incomplete_details=details, output=output or [])


class TestResponsesMapping:
    def test_completed_text_is_end_turn(self) -> None:
        assert responses_stop_reason(_resp("completed")) == ("end_turn", "completed")

    def test_completed_with_function_call_is_tool_use(self) -> None:
        out = [SimpleNamespace(type="reasoning"), SimpleNamespace(type="function_call")]
        assert responses_stop_reason(_resp("completed", output=out)) == ("tool_use", "completed")

    def test_incomplete_max_output_tokens(self) -> None:
        assert responses_stop_reason(_resp("incomplete", reason="max_output_tokens")) == (
            "max_tokens",
            "max_output_tokens",
        )

    def test_incomplete_content_filter(self) -> None:
        assert responses_stop_reason(_resp("incomplete", reason="content_filter")) == (
            "content_filter",
            "content_filter",
        )

    def test_incomplete_other_reason(self) -> None:
        assert responses_stop_reason(_resp("incomplete", reason="something_new")) == (
            "other",
            "something_new",
        )

    def test_incomplete_without_reason(self) -> None:
        assert responses_stop_reason(_resp("incomplete")) == ("other", "incomplete")

    def test_other_status(self) -> None:
        assert responses_stop_reason(_resp("failed")) == ("other", "failed")

    def test_missing_status_is_none(self) -> None:
        assert responses_stop_reason(SimpleNamespace()) == (None, None)
        assert responses_stop_reason(_resp(None)) == (None, None)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _anthropic_message(content: list[Any], stop_reason: str | None) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_1",
        content=content,
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-test", model="claude-haiku-4-5")


class TestAnthropicProvider:
    def test_normalize_max_tokens(self, anthropic_provider: AnthropicProvider) -> None:
        msg = _anthropic_message([TextBlock(type="text", text="cut")], "max_tokens")
        resp = anthropic_provider._normalize_response(msg)
        assert resp.stop_reason == "max_tokens"
        assert resp.raw_stop_reason == "max_tokens"

    def test_normalize_tool_use(self, anthropic_provider: AnthropicProvider) -> None:
        block = ToolUseBlock(type="tool_use", id="t1", name="add", input={"a": 1})
        resp = anthropic_provider._normalize_response(_anthropic_message([block], "tool_use"))
        assert resp.stop_reason == "tool_use"

    def test_normalize_missing(self, anthropic_provider: AnthropicProvider) -> None:
        msg = _anthropic_message([TextBlock(type="text", text="x")], None)
        resp = anthropic_provider._normalize_response(msg)
        assert resp.stop_reason is None
        assert resp.raw_stop_reason is None

    def test_structured_forced_tool_reads_as_end_turn(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """The forced ``structured_output`` tool call IS the answer: a natural
        finish, not a request to run tools. The vendor string is kept."""
        block = ToolUseBlock(type="tool_use", id="t1", name="structured_output", input={"a": 1})
        resp = anthropic_provider._normalize_structured_response(
            _anthropic_message([block], "tool_use")
        )
        assert resp.stop_reason == "end_turn"
        assert resp.raw_stop_reason == "tool_use"

    def test_structured_truncated(self, anthropic_provider: AnthropicProvider) -> None:
        block = ToolUseBlock(type="tool_use", id="t1", name="structured_output", input={})
        resp = anthropic_provider._normalize_structured_response(
            _anthropic_message([block], "max_tokens")
        )
        assert resp.stop_reason == "max_tokens"

    async def test_complete_carries_stop_reason(
        self, anthropic_provider: AnthropicProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        msg = _anthropic_message([TextBlock(type="text", text="cut")], "max_tokens")
        monkeypatch.setattr(
            anthropic_provider._client.messages, "create", AsyncMock(return_value=msg)
        )
        resp = await anthropic_provider.complete(_USER)
        assert resp.stop_reason == "max_tokens"

    async def test_stream_reads_final_message(self, anthropic_provider: AnthropicProvider) -> None:
        final = _anthropic_message([TextBlock(type="text", text="cut")], "max_tokens")
        anthropic_provider._client.messages.stream = MagicMock(
            return_value=_FakeAnthropicStream([], final)
        )
        done = await _done(anthropic_provider)
        assert done.stop_reason == "max_tokens"
        assert done.raw_stop_reason == "max_tokens"

    async def test_stream_falls_back_to_message_delta(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """A final message without a stop_reason still gets the one the
        ``message_delta`` event announced."""
        final = _anthropic_message([TextBlock(type="text", text="ok")], None)
        events = [
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=None,
            )
        ]
        anthropic_provider._client.messages.stream = MagicMock(
            return_value=_FakeAnthropicStream(events, final)
        )
        done = await _done(anthropic_provider)
        assert done.stop_reason == "end_turn"


class _FakeAnthropicStream:
    def __init__(self, events: list[Any], final: AnthropicMessage) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        return self._iter()

    async def _iter(self) -> Any:
        for event in self._events:
            yield event

    async def get_final_message(self) -> AnthropicMessage:
        return self._final


async def _done(provider: Any, **kwargs: Any) -> LLMResponse:
    done: LLMResponse | None = None
    async for event in provider.complete_stream(_USER, **kwargs):
        if event.type == StreamEventType.DONE:
            done = event.raw
    assert done is not None
    return done


# ---------------------------------------------------------------------------
# OpenAI Chat + OpenRouter
# ---------------------------------------------------------------------------


@dataclass
class _ChatMessage:
    content: str | None = "ok"
    tool_calls: Any = None


@dataclass
class _ChatChoice:
    message: _ChatMessage = field(default_factory=_ChatMessage)
    finish_reason: str | None = "stop"


@dataclass
class _ChatCompletion:
    choices: list[Any] = field(default_factory=lambda: [_ChatChoice()])
    usage: Any = None

    def model_dump(self) -> dict[str, Any]:
        return {}


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: Any = None


@dataclass
class _StreamChoice:
    delta: _Delta = field(default_factory=_Delta)
    finish_reason: str | None = None


@dataclass
class _Chunk:
    choices: list[Any] = field(default_factory=list)
    usage: Any = None


async def _astream(chunks: list[Any]) -> Any:
    for chunk in chunks:
        yield chunk


def _chat_chunks(finish_reason: str | None) -> list[_Chunk]:
    return [
        _Chunk(choices=[_StreamChoice(delta=_Delta(content="cu"))]),
        _Chunk(choices=[_StreamChoice(delta=_Delta(content="t"), finish_reason=finish_reason)]),
    ]


def _chat_providers() -> list[Any]:
    return [
        OpenAIProvider(model="gpt-4.1-mini", api_key="sk-test"),
        OpenRouterProvider(model="deepseek/deepseek-chat", api_key="sk-or-test"),
    ]


class TestChatProviders:
    @pytest.mark.parametrize("provider", _chat_providers(), ids=["openai", "openrouter"])
    @pytest.mark.parametrize(
        ("raw", "expected"), [("length", "max_tokens"), ("stop", "end_turn"), (None, None)]
    )
    async def test_complete(
        self, provider: Any, raw: str | None, expected: str | None, monkeypatch: Any
    ) -> None:
        completion = _ChatCompletion(choices=[_ChatChoice(finish_reason=raw)])
        monkeypatch.setattr(
            provider._client.chat.completions, "create", AsyncMock(return_value=completion)
        )
        resp = await provider.complete(_USER)
        assert resp.stop_reason == expected
        assert resp.raw_stop_reason == raw

    @pytest.mark.parametrize("provider", _chat_providers(), ids=["openai", "openrouter"])
    @pytest.mark.parametrize(
        ("raw", "expected"), [("length", "max_tokens"), ("tool_calls", "tool_use"), (None, None)]
    )
    async def test_stream(
        self, provider: Any, raw: str | None, expected: str | None, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            provider._client.chat.completions,
            "create",
            AsyncMock(return_value=_astream(_chat_chunks(raw))),
        )
        done = await _done(provider)
        assert done.stop_reason == expected
        assert done.raw_stop_reason == raw


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


@dataclass
class _RUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    total_tokens: int = 15
    input_tokens_details: Any = None
    output_tokens_details: Any = None


@dataclass
class _RResponse:
    status: str | None = "completed"
    incomplete_details: Any = None
    output: list[Any] = field(default_factory=list)
    output_text: str = "cut"
    usage: Any = field(default_factory=_RUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"status": self.status}


@dataclass
class _REvent:
    type: str
    delta: str | None = None
    response: Any = None


@pytest.fixture
def responses_provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-4.1-mini", api_key="sk-test")


class TestResponsesProvider:
    async def test_complete_incomplete(
        self, responses_provider: OpenAIResponsesProvider, monkeypatch: Any
    ) -> None:
        response = _RResponse(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        monkeypatch.setattr(
            responses_provider._client.responses, "create", AsyncMock(return_value=response)
        )
        resp = await responses_provider.complete(_USER)
        assert resp.stop_reason == "max_tokens"
        assert resp.raw_stop_reason == "max_output_tokens"

    async def test_complete_completed(
        self, responses_provider: OpenAIResponsesProvider, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            responses_provider._client.responses, "create", AsyncMock(return_value=_RResponse())
        )
        resp = await responses_provider.complete(_USER)
        assert resp.stop_reason == "end_turn"
        assert resp.raw_stop_reason == "completed"

    async def test_stream_incomplete_event_keeps_usage_and_reason(
        self, responses_provider: OpenAIResponsesProvider, monkeypatch: Any
    ) -> None:
        """A truncated stream ends with ``response.incomplete`` (not
        ``response.completed``); its usage and stop reason must survive."""
        final = _RResponse(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        events = [
            _REvent(type="response.output_text.delta", delta="cut"),
            _REvent(type="response.incomplete", response=final),
        ]
        monkeypatch.setattr(
            responses_provider._client.responses,
            "create",
            AsyncMock(return_value=_astream(events)),
        )
        done = await _done(responses_provider)
        assert done.text == "cut"
        assert done.stop_reason == "max_tokens"
        assert done.raw_stop_reason == "max_output_tokens"
        assert done.usage.output_tokens == 5
        assert done.usage.usage_reported is True
        assert done.provider_response == {"status": "incomplete"}

    async def test_stream_completed(
        self, responses_provider: OpenAIResponsesProvider, monkeypatch: Any
    ) -> None:
        events = [
            _REvent(type="response.output_text.delta", delta="ok"),
            _REvent(type="response.completed", response=_RResponse()),
        ]
        monkeypatch.setattr(
            responses_provider._client.responses,
            "create",
            AsyncMock(return_value=_astream(events)),
        )
        done = await _done(responses_provider)
        assert done.stop_reason == "end_turn"

    async def test_stream_without_terminal_event_is_none(
        self, responses_provider: OpenAIResponsesProvider, monkeypatch: Any
    ) -> None:
        events = [_REvent(type="response.output_text.delta", delta="ok")]
        monkeypatch.setattr(
            responses_provider._client.responses,
            "create",
            AsyncMock(return_value=_astream(events)),
        )
        done = await _done(responses_provider)
        assert done.stop_reason is None
        assert done.raw_stop_reason is None


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class TestTypes:
    def test_stop_reason_exported(self) -> None:
        import dendrux
        from dendrux.types import StopReason

        assert dendrux.StopReason is StopReason

    def test_llm_response_defaults_to_none(self) -> None:
        resp = LLMResponse(text="x")
        assert resp.stop_reason is None
        assert resp.raw_stop_reason is None

    def test_run_result_truncated_property(self) -> None:
        from dendrux.types import RunResult, RunStatus

        assert RunResult(run_id="r", status=RunStatus.SUCCESS).stop_reason is None
        assert RunResult(run_id="r", status=RunStatus.SUCCESS).truncated is False
        cut = RunResult(run_id="r", status=RunStatus.SUCCESS, stop_reason="max_tokens")
        assert cut.truncated is True
        done = RunResult(run_id="r", status=RunStatus.SUCCESS, stop_reason="end_turn")
        assert done.truncated is False

    def test_run_event_mirrors_run_result_on_completed(self) -> None:
        from dendrux.types import RunEvent, RunEventType, RunResult, RunStatus

        cut = RunResult(run_id="r", status=RunStatus.SUCCESS, stop_reason="max_tokens")
        event = RunEvent(type=RunEventType.RUN_COMPLETED, run_result=cut)
        assert event.stop_reason == "max_tokens"
        assert event.truncated is True
        # Only the completed event carries it.
        paused = RunEvent(type=RunEventType.RUN_PAUSED, run_result=cut)
        assert paused.stop_reason is None
        assert paused.truncated is False
        assert RunEvent(type=RunEventType.TEXT_DELTA, text="x").stop_reason is None

    def test_structured_error_keeps_positional_signature(self) -> None:
        from dendrux.types import StructuredOutputValidationError, UsageStats

        err = StructuredOutputValidationError("{}", "Out", "bad")
        assert err.usage is None
        assert err.model is None
        assert err.raw_response == "{}"

        usage = UsageStats(input_tokens=3, output_tokens=2, total_tokens=5)
        err = StructuredOutputValidationError("{}", "Out", "bad", usage=usage, model="m")
        assert err.usage is usage
        assert err.model == "m"

    def test_structured_error_usage_is_keyword_only(self) -> None:
        from dendrux.types import StructuredOutputValidationError, UsageStats

        with pytest.raises(TypeError):
            StructuredOutputValidationError("{}", "Out", "bad", UsageStats())  # type: ignore[misc]
