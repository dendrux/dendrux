"""0.5.0 T2: providers, no silent failures.

Every test drives the real Anthropic / OpenAI SDKs through a fake httpx
transport (recorded-style bodies, real SDK exception classes). No network.

Covers: the typed error taxonomy (B5/B10, D-039), stream-incomplete detection
and the one pre-first-yield retry (D-017), Responses failures (D-018),
refusals and content filters (D-069), per-call deadlines (D-041), SDK
connection limits (D-080), upstream close on aclose/cancel (D-081), per-loop
clients (D-086), thinking + structured output (D-042), cache-write usage
(D-127) and ``reports_cost``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

from dendrux.errors import (  # noqa: E402
    ContentFilteredError,
    ContextLengthExceededError,
    ModelRefusalError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderStreamIncompleteError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from dendrux.llm import _resilience  # noqa: E402
from dendrux.llm._resilience import (  # noqa: E402
    Deadlines,
    StreamClock,
    WireRequest,
    resolve_deadlines,
    retry_unstarted_stream,
)
from dendrux.llm._usage import openai_wire_usage  # noqa: E402
from dendrux.llm.anthropic import AnthropicProvider  # noqa: E402
from dendrux.llm.base import LLMProvider  # noqa: E402
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402
from dendrux.llm.openrouter import OpenRouterProvider  # noqa: E402
from dendrux.types import Message, Role, StreamEvent, StreamEventType  # noqa: E402

Handler = Callable[[httpx.Request], Any]
SECRET = "VENDOR-SECRET-TEXT-123"
USER = [Message(role=Role.USER, content="hello")]


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


class Wire:
    """Counts requests and serves them from ``handler`` (sync or async)."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self.handler(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def _http(wire: Wire) -> httpx.AsyncClient:
    # The Dendrux call-state hooks are kept so backoff detection works.
    return httpx.AsyncClient(
        transport=httpx.MockTransport(wire),
        event_hooks={
            "request": [_resilience._on_request],
            "response": [_resilience._on_response],
        },
    )


def wire(provider: Any, handler: Handler) -> Wire:
    """Point the provider's SDK client at a fake transport (same SDK object)."""
    w = Wire(handler)
    provider._client._client = _http(w)
    if isinstance(provider, AnthropicProvider):
        provider._client.base_url = "http://anthropic.test"
    else:
        provider._client.base_url = "http://gateway.test/v1"
    return w


def json_handler(status: int, body: Any, headers: dict[str, str] | None = None) -> Handler:
    return lambda request: httpx.Response(status, json=body, headers=headers or {})


def sse(*events: tuple[str | None, Any]) -> bytes:
    out = []
    for name, data in events:
        line = f"event: {name}\n" if name else ""
        out.append(line + "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n")
    return "".join(out).encode()


class TimedStream(httpx.AsyncByteStream):
    """SSE body served part by part, with optional sleeps; records aclose."""

    def __init__(self, parts: list[bytes | float], then: Exception | None = None) -> None:
        self.parts = parts
        self.then = then
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            if isinstance(part, float):
                await asyncio.sleep(part)
            else:
                yield part
        if self.then is not None:
            raise self.then

    async def aclose(self) -> None:
        self.closed = True


def stream_handler(
    parts: list[bytes | float],
    then: Exception | None = None,
    record: list[TimedStream] | None = None,
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = TimedStream(parts, then)
        if record is not None:
            record.append(body)
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    return handler


async def drain(gen: Any) -> list[StreamEvent]:
    return [event async for event in gen]


def anthropic_provider(**kwargs: Any) -> AnthropicProvider:
    kwargs.setdefault("max_retries", 0)
    return AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key", **kwargs)


def chat_provider(**kwargs: Any) -> OpenAIProvider:
    kwargs.setdefault("max_retries", 0)
    return OpenAIProvider(model="gpt-4o", api_key="test-key", **kwargs)


def responses_provider(**kwargs: Any) -> OpenAIResponsesProvider:
    kwargs.setdefault("max_retries", 0)
    return OpenAIResponsesProvider(model="gpt-5", api_key="test-key", **kwargs)


def openrouter_provider(**kwargs: Any) -> OpenRouterProvider:
    kwargs.setdefault("max_retries", 0)
    return OpenRouterProvider(model="deepseek/deepseek-chat", api_key="sk-or-test", **kwargs)


# --- Anthropic wire ---------------------------------------------------------


def a_message(
    text: str = "hi", stop_reason: str | None = "end_turn", **usage: Any
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2, **usage},
    }


def a_start() -> bytes:
    return sse(("message_start", {"type": "message_start", "message": a_message("", None)}))


def a_text(text: str) -> bytes:
    return sse(
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )


def a_end(stop_reason: str = "end_turn") -> bytes:
    return sse(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def a_ping() -> bytes:
    return sse(("ping", {"type": "ping"}))


# --- Chat wire ----------------------------------------------------------------


def c_chunk(content: str | None = None, finish: str | None = None, **delta: Any) -> dict[str, Any]:
    if content is not None:
        delta["content"] = content
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def c_usage(**details: Any) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": details or None,
        },
    }


def c_completion(message: dict[str, Any], finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


# --- Responses wire -----------------------------------------------------------


def r_response(status: str = "completed", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5",
        "status": status,
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    body.update(extra)
    return body


def r_delta(text: str) -> dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "item_id": "m1",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
        "sequence_number": 1,
        "logprobs": [],
    }


def r_completed(**extra: Any) -> dict[str, Any]:
    return {"type": "response.completed", "response": r_response(**extra), "sequence_number": 9}


def r_in_progress() -> dict[str, Any]:
    return {
        "type": "response.in_progress",
        "response": r_response("in_progress"),
        "sequence_number": 0,
    }


# ---------------------------------------------------------------------------
# Error taxonomy (B5 / B10)
# ---------------------------------------------------------------------------

ANTHROPIC_CASES = [
    (401, "authentication_error", ProviderAuthError, "auth", False),
    (403, "permission_error", ProviderAuthError, "auth", False),
    (402, "billing_error", ProviderError, "credits", False),
    (429, "rate_limit_error", ProviderRateLimitError, "rate_limit", True),
    (500, "api_error", ProviderUnavailableError, "unavailable", True),
    (529, "overloaded_error", ProviderUnavailableError, "unavailable", True),
    (404, "not_found_error", ProviderError, "rejected", False),
    (400, "invalid_request_error", ProviderError, "rejected", False),
]


def a_error(error_type: str, message: str = SECRET) -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


class TestAnthropicTaxonomy:
    @pytest.mark.parametrize(("status", "etype", "cls", "category", "retryable"), ANTHROPIC_CASES)
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_status_errors_are_typed(
        self,
        status: int,
        etype: str,
        cls: type[ProviderError],
        category: str,
        retryable: bool,
        streaming: bool,
    ) -> None:
        provider = anthropic_provider()
        wire(provider, json_handler(status, a_error(etype), {"retry-after": "7"}))
        with pytest.raises(cls) as info:
            if streaming:
                await drain(provider.complete_stream(USER))
            else:
                await provider.complete(USER)
        err = info.value
        assert err.category == category
        assert err.retryable is retryable
        assert err.status_code == status
        assert err.provider == "anthropic"
        assert err.error_type == etype
        assert isinstance(err.__cause__, anthropic.APIStatusError)
        assert SECRET not in str(err) and SECRET not in err.safe_message
        assert err.safe_message == f"anthropic {category} (HTTP {status}, code n/a)"
        if cls is ProviderRateLimitError:
            assert err.retry_after == 7.0

    async def test_prompt_too_long_is_context_length(self) -> None:
        provider = anthropic_provider()
        body = a_error(
            "invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"
        )
        wire(provider, json_handler(400, body))
        with pytest.raises(ContextLengthExceededError) as info:
            await provider.complete(USER)
        assert info.value.provider == "anthropic"
        assert info.value.model == "claude-sonnet-4-6"
        assert info.value.safe_message.startswith("anthropic context_length (HTTP 400")

    async def test_in_stream_overloaded_after_delta(self) -> None:
        provider = anthropic_provider()
        body = a_start() + a_text("partial") + sse(("error", a_error("overloaded_error")))
        w = wire(provider, stream_handler([body]))
        events: list[StreamEvent] = []
        with pytest.raises(ProviderUnavailableError) as info:
            async for event in provider.complete_stream(USER):
                events.append(event)
        assert [e.text for e in events if e.type == StreamEventType.TEXT_DELTA] == ["partial"]
        assert info.value.status_code is None
        assert SECRET not in str(info.value)
        assert len(w.requests) == 1

    async def test_connect_timeout_is_retryable_connect_timeout(self) -> None:
        provider = anthropic_provider()

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("no route", request=request)

        wire(provider, boom)
        with pytest.raises(ProviderTimeoutError) as info:
            await provider.complete(USER)
        assert info.value.timeout_reason == "connect"
        assert info.value.retryable is True
        assert isinstance(info.value, TimeoutError)

    async def test_read_timeout_is_not_retryable(self) -> None:
        provider = anthropic_provider()

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        wire(provider, boom)
        with pytest.raises(ProviderTimeoutError) as info:
            await provider.complete(USER)
        assert info.value.timeout_reason == "call_deadline"
        assert info.value.retryable is False

    async def test_connection_error_is_unavailable(self) -> None:
        provider = anthropic_provider()

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        wire(provider, boom)
        with pytest.raises(ProviderUnavailableError) as info:
            await provider.complete(USER)
        assert isinstance(info.value, ConnectionError)
        assert isinstance(info.value.__cause__, anthropic.APIConnectionError)


CHAT_CASES = [
    (401, {"code": "invalid_api_key"}, ProviderAuthError, "auth"),
    (402, {"code": None}, ProviderError, "credits"),
    (429, {"code": "insufficient_quota"}, ProviderError, "credits"),
    (429, {"code": "rate_limit_exceeded"}, ProviderRateLimitError, "rate_limit"),
    (500, {"code": "server_error"}, ProviderUnavailableError, "unavailable"),
    (503, {"code": None}, ProviderUnavailableError, "unavailable"),
    (400, {"code": "invalid_value"}, ProviderError, "rejected"),
    (400, {"code": "content_filter"}, ContentFilteredError, "content_filter"),
    (400, {"code": "context_length_exceeded"}, ContextLengthExceededError, "context_length"),
]


class TestOpenAITaxonomy:
    @pytest.mark.parametrize(("status", "extra", "cls", "category"), CHAT_CASES)
    @pytest.mark.parametrize(
        "make", [chat_provider, responses_provider, openrouter_provider], ids=["chat", "resp", "or"]
    )
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_status_errors_are_typed(
        self,
        status: int,
        extra: dict[str, Any],
        cls: type[ProviderError],
        category: str,
        make: Callable[..., Any],
        streaming: bool,
    ) -> None:
        provider = make()
        body = {"error": {"message": SECRET, "type": "invalid_request_error", **extra}}
        wire(provider, json_handler(status, body, {"retry-after-ms": "1500"}))
        with pytest.raises(cls) as info:
            if streaming:
                await drain(provider.complete_stream(USER))
            else:
                await provider.complete(USER)
        err = info.value
        assert err.category == category
        assert err.status_code == status
        assert isinstance(err.__cause__, openai.APIStatusError)
        assert SECRET not in str(err)
        if cls is ProviderRateLimitError:
            assert err.retry_after == 1.5 and err.retryable is True

    async def test_openrouter_numeric_code_in_stream_error(self) -> None:
        provider = openrouter_provider()
        body = sse((None, c_chunk("x")), (None, {"error": {"message": SECRET, "code": 502}}))
        wire(provider, stream_handler([body]))
        with pytest.raises(ProviderUnavailableError) as info:
            await drain(provider.complete_stream(USER))
        assert info.value.provider == "openrouter"
        assert SECRET not in str(info.value)

    async def test_streaming_unsupported_stays_not_implemented(self) -> None:
        provider = chat_provider()
        body = {"error": {"message": "Streaming is not supported for this model"}}
        wire(provider, json_handler(400, body))
        with pytest.raises(NotImplementedError) as info:
            await drain(provider.complete_stream(USER))
        assert isinstance(info.value.__cause__, openai.BadRequestError)


class TestObservability:
    @pytest.mark.parametrize("make", [chat_provider, openrouter_provider, anthropic_provider])
    async def test_typed_errors_logged_once_with_run_id(
        self, make: Callable[..., Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = make()
        body = (
            a_error("rate_limit_error")
            if isinstance(provider, AnthropicProvider)
            else {"error": {"message": SECRET, "code": "rate_limit_exceeded"}}
        )
        wire(provider, json_handler(429, body))
        with caplog.at_level(logging.WARNING, logger="dendrux.llm._resilience"):
            with pytest.raises(ProviderRateLimitError):
                await provider.complete(USER, run_id="run_42")
            with pytest.raises(ProviderRateLimitError):
                await drain(provider.complete_stream(USER, run_id="run_42"))
        lines = [r.getMessage() for r in caplog.records if "provider.error" in r.getMessage()]
        assert len(lines) == 2
        assert all("run_id=run_42" in line and "category=rate_limit" in line for line in lines)
        assert SECRET not in caplog.text


# ---------------------------------------------------------------------------
# Stream incomplete (D-017) and the pre-first-yield retry
# ---------------------------------------------------------------------------


class TestStreamIncomplete:
    async def test_anthropic_cut_after_delta_raises_without_retry(self) -> None:
        provider = anthropic_provider()
        w = wire(provider, stream_handler([a_start() + a_text("partial")]))
        events: list[StreamEvent] = []
        with pytest.raises(ProviderStreamIncompleteError):
            async for event in provider.complete_stream(USER):
                events.append(event)
        assert [e.text for e in events if e.type == StreamEventType.TEXT_DELTA] == ["partial"]
        assert len(w.requests) == 1

    async def test_anthropic_cut_before_any_yield_is_retried_once(self) -> None:
        provider = anthropic_provider()
        bodies = iter([a_start(), a_start() + a_text("ok") + a_end()])
        w = wire(provider, lambda r: stream_handler([next(bodies)])(r))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].type == StreamEventType.DONE
        assert events[-1].raw.text == "ok"
        assert len(w.requests) == 2

    async def test_anthropic_twice_cut_before_yield_raises(self) -> None:
        provider = anthropic_provider()
        w = wire(provider, stream_handler([a_start()]))
        with pytest.raises(ProviderStreamIncompleteError):
            await drain(provider.complete_stream(USER))
        assert len(w.requests) == 2

    async def test_anthropic_empty_body_is_incomplete(self) -> None:
        provider = anthropic_provider()
        wire(provider, stream_handler([b""]))
        with pytest.raises(ProviderStreamIncompleteError):
            await drain(provider.complete_stream(USER))

    async def test_anthropic_transport_break_mid_stream(self) -> None:
        provider = anthropic_provider()
        broken = httpx.RemoteProtocolError("peer closed connection")
        w = wire(provider, stream_handler([a_start() + a_text("x")], then=broken))
        with pytest.raises(ProviderStreamIncompleteError) as info:
            await drain(provider.complete_stream(USER))
        assert isinstance(info.value.__cause__, httpx.RemoteProtocolError)
        assert len(w.requests) == 1

    async def test_anthropic_complete_stream_succeeds(self) -> None:
        provider = anthropic_provider()
        wire(provider, stream_handler([a_start() + a_text("ok") + a_end()]))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].raw.stop_reason == "end_turn"

    async def test_chat_cut_without_done_raises(self) -> None:
        provider = chat_provider()
        w = wire(provider, stream_handler([sse((None, c_chunk("partial")))]))
        with pytest.raises(ProviderStreamIncompleteError):
            await drain(provider.complete_stream(USER))
        assert len(w.requests) == 1

    async def test_chat_done_sentinel_alone_is_terminal(self) -> None:
        provider = chat_provider()
        wire(provider, stream_handler([sse((None, c_chunk("hi")), (None, "[DONE]"))]))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].raw.text == "hi"
        assert events[-1].raw.stop_reason is None

    async def test_chat_finish_reason_is_terminal(self) -> None:
        provider = chat_provider()
        wire(provider, stream_handler([sse((None, c_chunk("hi", finish="stop")))]))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].raw.stop_reason == "end_turn"

    async def test_chat_empty_stream_is_retried_then_succeeds(self) -> None:
        provider = openrouter_provider()
        bodies = iter([b"", sse((None, c_chunk("ok", finish="stop")), (None, "[DONE]"))])
        w = wire(provider, lambda r: stream_handler([next(bodies)])(r))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].raw.text == "ok"
        assert len(w.requests) == 2

    async def test_responses_without_completed_raises(self) -> None:
        provider = responses_provider()
        w = wire(provider, stream_handler([sse((None, r_delta("partial")))]))
        events: list[StreamEvent] = []
        with pytest.raises(ProviderStreamIncompleteError):
            async for event in provider.complete_stream(USER):
                events.append(event)
        assert [e.text for e in events if e.type == StreamEventType.TEXT_DELTA] == ["partial"]
        assert len(w.requests) == 1

    async def test_responses_completed_is_terminal(self) -> None:
        provider = responses_provider()
        wire(provider, stream_handler([sse((None, r_delta("ok")), (None, r_completed()))]))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].raw.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Responses failures (D-018)
# ---------------------------------------------------------------------------


class TestResponsesFailures:
    async def test_http_200_status_failed_raises_with_code(self) -> None:
        provider = responses_provider()
        body = r_response("failed", error={"code": "server_error", "message": SECRET})
        wire(provider, json_handler(200, body))
        with pytest.raises(ProviderUnavailableError) as info:
            await provider.complete(USER)
        assert info.value.provider_code == "server_error"
        assert SECRET not in str(info.value)

    async def test_status_failed_context_length(self) -> None:
        provider = responses_provider()
        body = r_response("failed", error={"code": "context_length_exceeded", "message": "x"})
        wire(provider, json_handler(200, body))
        with pytest.raises(ContextLengthExceededError) as info:
            await provider.complete(USER)
        assert info.value.provider == "openai_responses"

    async def test_status_cancelled_raises(self) -> None:
        provider = responses_provider()
        wire(provider, json_handler(200, r_response("cancelled")))
        with pytest.raises(ProviderError, match="cancelled"):
            await provider.complete(USER)

    async def test_error_event_raises_typed_after_delta(self) -> None:
        provider = responses_provider()
        error = {
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": SECRET,
            "param": None,
            "sequence_number": 2,
        }
        wire(provider, stream_handler([sse((None, r_delta("par")), (None, error))]))
        events: list[StreamEvent] = []
        with pytest.raises(ProviderRateLimitError) as info:
            async for event in provider.complete_stream(USER):
                events.append(event)
        assert events and events[0].text == "par"
        assert SECRET not in str(info.value)

    async def test_response_failed_event_keeps_its_real_code(self) -> None:
        provider = responses_provider()
        failed = {
            "type": "response.failed",
            "response": r_response(
                "failed", error={"code": "context_length_exceeded", "message": SECRET}
            ),
            "sequence_number": 3,
        }
        wire(provider, stream_handler([sse((None, failed))]))
        with pytest.raises(ContextLengthExceededError):
            await drain(provider.complete_stream(USER))

    async def test_agent_run_fails_instead_of_empty_success(self) -> None:
        """D-018: the run no longer reports success with an empty answer.

        (Persisting ``ErrorInfo`` on the run is T7's wiring.)
        """
        from dendrux.agent import Agent

        provider = responses_provider()
        body = r_response("failed", error={"code": "server_error", "message": "boom"})
        wire(provider, json_handler(200, body))
        agent = Agent(provider=provider, prompt="x")
        with pytest.raises(ProviderUnavailableError):
            await agent.run("hi")


# ---------------------------------------------------------------------------
# Refusals and content filters (D-069)
# ---------------------------------------------------------------------------


class TestRefusalAndFilter:
    async def test_anthropic_refusal_batch(self) -> None:
        provider = anthropic_provider()
        wire(provider, json_handler(200, a_message("I can't", stop_reason="refusal")))
        with pytest.raises(ModelRefusalError) as info:
            await provider.complete(USER)
        assert info.value.refusal_text == "I can't"
        assert "I can't" not in str(info.value)
        assert info.value.retryable is False

    async def test_anthropic_refusal_stream(self) -> None:
        provider = anthropic_provider()
        wire(provider, stream_handler([a_start() + a_text("No.") + a_end("refusal")]))
        with pytest.raises(ModelRefusalError) as info:
            await drain(provider.complete_stream(USER))
        assert info.value.refusal_text == "No."

    async def test_chat_refusal_batch(self) -> None:
        provider = chat_provider()
        body = c_completion({"content": None, "refusal": "I can't help with that."})
        wire(provider, json_handler(200, body))
        with pytest.raises(ModelRefusalError) as info:
            await provider.complete(USER)
        assert info.value.refusal_text == "I can't help with that."
        assert pickle.loads(pickle.dumps(info.value)).refusal_text is None

    async def test_chat_refusal_stream(self) -> None:
        provider = openrouter_provider()
        body = sse(
            (None, c_chunk(refusal="Nope")), (None, c_chunk(finish="stop")), (None, "[DONE]")
        )
        wire(provider, stream_handler([body]))
        with pytest.raises(ModelRefusalError):
            await drain(provider.complete_stream(USER))

    async def test_chat_content_filter_batch_and_stream(self) -> None:
        provider = chat_provider()
        wire(provider, json_handler(200, c_completion({"content": "x"}, finish="content_filter")))
        with pytest.raises(ContentFilteredError):
            await provider.complete(USER)
        wire(provider, stream_handler([sse((None, c_chunk("x", finish="content_filter")))]))
        with pytest.raises(ContentFilteredError):
            await drain(provider.complete_stream(USER))

    async def test_responses_refusal_and_filter(self) -> None:
        provider = responses_provider()
        refusal = {"type": "refusal", "refusal": "No can do"}
        message = {"type": "message", "id": "m1", "status": "completed", "role": "assistant"}
        wire(provider, json_handler(200, r_response(output=[{**message, "content": [refusal]}])))
        with pytest.raises(ModelRefusalError) as info:
            await provider.complete(USER)
        assert info.value.refusal_text == "No can do"
        filtered = r_response("incomplete", incomplete_details={"reason": "content_filter"})
        wire(provider, json_handler(200, filtered))
        with pytest.raises(ContentFilteredError):
            await provider.complete(USER)

    async def test_responses_refusal_stream(self) -> None:
        provider = responses_provider()
        delta = {
            "type": "response.refusal.delta",
            "item_id": "m1",
            "output_index": 0,
            "content_index": 0,
            "delta": "No",
            "sequence_number": 1,
        }
        wire(provider, stream_handler([sse((None, delta), (None, r_completed()))]))
        with pytest.raises(ModelRefusalError):
            await drain(provider.complete_stream(USER))


# ---------------------------------------------------------------------------
# Deadlines (D-041)
# ---------------------------------------------------------------------------


class TestDeadlineConfig:
    def test_defaults_and_read_timeout(self) -> None:
        provider = chat_provider()
        assert provider._deadlines == Deadlines(600.0, 600.0, None)
        assert provider._client.timeout.read == 630.0
        assert provider._client.timeout.connect == 10.0

    def test_read_timeout_never_unbounded(self) -> None:
        assert Deadlines(None, None, None).read_timeout == 600.0
        assert Deadlines(None, 20.0, None).read_timeout == 50.0
        assert Deadlines(100.0, 20.0, 5.0).read_timeout == 130.0

    @pytest.mark.parametrize(
        "make", [anthropic_provider, chat_provider, responses_provider, openrouter_provider]
    )
    def test_legacy_timeout_warns_and_sets_both(self, make: Callable[..., Any]) -> None:
        with pytest.warns(DeprecationWarning, match="call_timeout") as record:
            provider = make(timeout=45)
        assert provider._deadlines.call_timeout == 45.0
        assert provider._deadlines.stream_idle_timeout == 45.0
        # The warning points at the caller, not at Dendrux internals.
        assert record[0].filename == __file__

    def test_explicit_new_kwarg_wins_over_legacy(self) -> None:
        with pytest.warns(DeprecationWarning):
            provider = chat_provider(timeout=45, call_timeout=900)
        assert provider._deadlines.call_timeout == 900.0
        assert provider._deadlines.stream_idle_timeout == 45.0

    def test_no_warning_without_timeout(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            anthropic_provider(call_timeout=None, stream_total_timeout=3600)

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "10"])
    def test_invalid_values_raise(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="positive number"):
            resolve_deadlines(
                "X",
                timeout=_resilience.TIMEOUT_UNSET,
                call_timeout=bad,
                stream_idle_timeout=None,
                stream_total_timeout=None,
            )


class TestDeadlines:
    @pytest.mark.parametrize("make", [anthropic_provider, chat_provider, responses_provider])
    async def test_call_timeout_abandons_without_retry(self, make: Callable[..., Any]) -> None:
        provider = make(call_timeout=0.2, max_retries=3)

        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return httpx.Response(200, json={})

        w = wire(provider, slow)
        started = time.monotonic()
        with pytest.raises(ProviderTimeoutError, match="call_timeout=0.2s") as info:
            await provider.complete(USER)
        assert time.monotonic() - started < 2
        assert info.value.timeout_reason == "call_deadline"
        assert info.value.retryable is False
        assert info.value.__cause__ is None
        assert len(w.requests) == 1

    async def test_deadline_during_429_backoff_keeps_rate_limit(self) -> None:
        provider = anthropic_provider(call_timeout=0.3, max_retries=2)
        w = wire(provider, json_handler(429, a_error("rate_limit_error"), {"retry-after": "5"}))
        with pytest.raises(ProviderRateLimitError) as info:
            await provider.complete(USER)
        err = info.value
        assert err.timeout_reason == "deadline_during_backoff"
        assert err.status_code == 429
        assert err.retry_after == 5.0
        assert err.retryable is False
        assert len(w.requests) == 1

    async def test_deadline_during_529_backoff_keeps_unavailable(self) -> None:
        provider = chat_provider(call_timeout=0.3, max_retries=2)
        w = wire(provider, json_handler(529, {"error": {"message": "x"}}, {"retry-after": "5"}))
        with pytest.raises(ProviderUnavailableError) as info:
            await provider.complete(USER)
        assert info.value.timeout_reason == "deadline_during_backoff"
        assert len(w.requests) == 1

    async def test_call_timeout_none_opts_out(self) -> None:
        provider = chat_provider(call_timeout=None)

        async def slowish(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.3)
            return httpx.Response(200, json=c_completion({"content": "ok"}))

        wire(provider, slowish)
        assert (await provider.complete(USER)).text == "ok"

    async def test_stream_idle_fires_after_first_delta(self) -> None:
        provider = anthropic_provider(stream_idle_timeout=0.2)
        w = wire(provider, stream_handler([a_start() + a_text("a"), 2.0, a_end()]))
        events: list[StreamEvent] = []
        with pytest.raises(ProviderTimeoutError) as info:
            async for event in provider.complete_stream(USER):
                events.append(event)
        assert info.value.timeout_reason == "stream_idle"
        assert [e.text for e in events if e.type == StreamEventType.TEXT_DELTA] == ["a"]
        assert len(w.requests) == 1

    async def test_anthropic_ping_does_not_reset_idle(self) -> None:
        provider = anthropic_provider(stream_idle_timeout=0.3)
        pings: list[bytes | float] = []
        for _ in range(10):
            pings += [0.1, a_ping()]
        wire(provider, stream_handler([a_start(), *pings, a_text("late"), a_end()]))
        with pytest.raises(ProviderTimeoutError) as info:
            await drain(provider.complete_stream(USER))
        assert info.value.timeout_reason == "stream_idle"

    async def test_responses_in_progress_does_not_reset_idle(self) -> None:
        provider = responses_provider(stream_idle_timeout=0.3)
        parts: list[bytes | float] = []
        for _ in range(10):
            parts += [0.1, sse((None, r_in_progress()))]
        wire(provider, stream_handler([*parts, sse((None, r_completed()))]))
        with pytest.raises(ProviderTimeoutError):
            await drain(provider.complete_stream(USER))

    async def test_steady_events_keep_the_stream_alive(self) -> None:
        provider = chat_provider(stream_idle_timeout=0.3)
        parts: list[bytes | float] = []
        for i in range(6):
            parts += [0.1, sse((None, c_chunk(str(i))))]
        wire(provider, stream_handler([*parts, sse((None, c_chunk(finish="stop")))]))
        events = await drain(provider.complete_stream(USER))
        assert events[-1].raw.text == "012345"

    async def test_consumer_time_is_not_counted(self) -> None:
        provider = chat_provider(stream_idle_timeout=0.2)
        body = sse((None, c_chunk("a")), (None, c_chunk("b")), (None, c_chunk(finish="stop")))
        wire(provider, stream_handler([body]))
        texts = []
        async for event in provider.complete_stream(USER):
            texts.append(event.text)
            await asyncio.sleep(0.3)  # a slow consumer
        assert texts[:2] == ["a", "b"]

    async def test_stream_total_timeout(self) -> None:
        provider = chat_provider(stream_idle_timeout=5, stream_total_timeout=0.3)
        parts: list[bytes | float] = []
        for i in range(20):
            parts += [0.05, sse((None, c_chunk(str(i))))]
        wire(provider, stream_handler(parts))
        with pytest.raises(ProviderTimeoutError) as info:
            await drain(provider.complete_stream(USER))
        assert info.value.timeout_reason == "stream_total"

    async def test_idle_timeout_is_not_retried_before_first_yield(self) -> None:
        provider = responses_provider(stream_idle_timeout=0.2)
        w = wire(provider, stream_handler([2.0, sse((None, r_completed()))]))
        with pytest.raises(ProviderTimeoutError):
            await drain(provider.complete_stream(USER))
        assert len(w.requests) == 1


# ---------------------------------------------------------------------------
# Connection limits (D-080), upstream close (D-081), per-loop clients (D-086)
# ---------------------------------------------------------------------------


class TestTransport:
    @pytest.mark.parametrize(
        "make", [anthropic_provider, chat_provider, responses_provider, openrouter_provider]
    )
    def test_sdk_connection_limits_kept(self, make: Callable[..., Any]) -> None:
        pool = make()._client._client._transport._pool
        assert pool._max_connections == 1000
        assert pool._max_keepalive_connections == 100

    def test_http_client_hooks_include_telemetry(self) -> None:
        from dendrux.llm import _retry_telemetry

        hooks = chat_provider()._client._client.event_hooks
        assert _retry_telemetry._on_response in hooks["response"]
        assert _resilience._on_request in hooks["request"]

    def test_sdk_stream_exposes_sse_iterator(self) -> None:
        # The Chat [DONE] detection relies on this SDK seam; fail loudly if it moves.
        assert callable(getattr(openai.AsyncStream, "_iter_events", None))

    @pytest.mark.parametrize("make", [chat_provider, responses_provider, anthropic_provider])
    async def test_aclose_closes_upstream(self, make: Callable[..., Any]) -> None:
        provider = make()
        if isinstance(provider, AnthropicProvider):
            body = a_start() + a_text("a") + a_text("b") + a_end()
        elif isinstance(provider, OpenAIResponsesProvider):
            body = sse((None, r_delta("a")), (None, r_delta("b")), (None, r_completed()))
        else:
            body = sse((None, c_chunk("a")), (None, c_chunk("b")), (None, "[DONE]"))
        bodies: list[TimedStream] = []
        wire(provider, stream_handler([body, 5.0], record=bodies))
        gen = provider.complete_stream(USER)
        first = await gen.__anext__()
        assert first.text == "a"
        await gen.aclose()
        assert bodies[0].closed

    async def test_cancel_closes_upstream(self) -> None:
        provider = chat_provider()
        bodies: list[TimedStream] = []
        wire(provider, stream_handler([sse((None, c_chunk("a"))), 5.0], record=bodies))
        started = asyncio.Event()

        async def consume() -> None:
            async for _ in provider.complete_stream(USER):
                started.set()

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert bodies[0].closed

    def test_reuse_across_asyncio_run_rebuilds_transport(self) -> None:
        provider = chat_provider()
        sdk = provider._client
        first_http = sdk._client

        async def touch() -> Any:
            return provider._sdk()._client

        assert asyncio.run(touch()) is first_http
        second_http = asyncio.run(touch())
        assert provider._client is sdk  # same SDK object (patches survive)
        assert second_http is not first_http
        assert second_http._transport._pool._max_connections == 1000
        assert asyncio.run(touch()) is not second_http

    def test_concurrent_live_loops_get_their_own_client(self) -> None:
        provider = chat_provider()
        ready = threading.Event()
        release = threading.Event()
        seen: dict[str, Any] = {}

        def owner() -> None:
            async def main() -> None:
                seen["home"] = provider._sdk()
                ready.set()
                while not release.is_set():
                    await asyncio.sleep(0.01)

            asyncio.run(main())

        thread = threading.Thread(target=owner)
        thread.start()
        ready.wait(5)

        async def other() -> Any:
            return provider._sdk()

        try:
            seen["other"] = asyncio.run(other())
        finally:
            release.set()
            thread.join(5)
        assert seen["home"] is provider._client
        assert seen["other"] is not provider._client
        assert seen["other"]._client is not provider._client._client

    def test_close_after_loop_change_never_awaits_the_dead_transport(self) -> None:
        provider = chat_provider()

        async def touch() -> None:
            provider._sdk()

        asyncio.run(touch())
        old_http = provider._client._client
        asyncio.run(provider.close())
        assert provider._client._client is not old_http
        assert provider._client._client.is_closed
        assert not old_http.is_closed  # dropped, not awaited across loops


# ---------------------------------------------------------------------------
# D-042: Anthropic thinking + structured output
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
}


class TestAnthropicStructuredThinking:
    async def test_thinking_uses_output_config_format(self) -> None:
        provider = anthropic_provider(thinking=True, effort="high")
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=a_message('{"a": 1}'))

        wire(provider, handler)
        response = await provider.complete(USER, output_schema=SCHEMA)
        body = seen["body"]
        assert "tool_choice" not in body and "tools" not in body
        assert body["output_config"]["effort"] == "high"
        fmt = body["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["properties"] == SCHEMA["properties"]
        assert response.text == '{"a": 1}'
        assert response.tool_calls is None
        assert response.provider_request["output_config"] == body["output_config"]

    async def test_without_thinking_keeps_forced_tool(self) -> None:
        provider = anthropic_provider()
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            message = a_message("", stop_reason="tool_use")
            message["content"] = [
                {"type": "tool_use", "id": "t", "name": "structured_output", "input": {"a": 2}}
            ]
            return httpx.Response(200, json=message)

        wire(provider, handler)
        response = await provider.complete(USER, output_schema=SCHEMA)
        assert seen["body"]["tool_choice"] == {"type": "tool", "name": "structured_output"}
        assert "output_config" not in seen["body"]
        assert json.loads(response.text or "") == {"a": 2}


# ---------------------------------------------------------------------------
# D-127: cache writes and reports_cost
# ---------------------------------------------------------------------------


class _Extra:
    """A pydantic-like object whose extra fields live in ``model_extra``."""

    def __init__(self, typed: dict[str, Any], extra: dict[str, Any]) -> None:
        self.__dict__.update(typed)
        self.model_extra = extra


class TestUsage:
    def test_responses_cache_writes_partitioned(self) -> None:
        details = _Extra({"cached_tokens": 30}, {"cache_write_tokens": 50})
        usage = _Extra(
            {"input_tokens": 100, "output_tokens": 7, "input_tokens_details": details},
            {},
        )
        stats = openai_wire_usage(
            usage,
            input_field="input_tokens",
            output_field="output_tokens",
            details_field="input_tokens_details",
            output_details_field="output_tokens_details",
        )
        assert stats.input_tokens == 20
        assert stats.cache_read_input_tokens == 30
        assert stats.cache_creation_input_tokens == 50
        assert stats.total_tokens == 27
        assert stats.cost_unknown is False

    async def test_responses_wire_reports_cache_writes(self) -> None:
        provider = responses_provider()
        body = r_response()
        body["usage"]["input_tokens"] = 1000
        body["usage"]["input_tokens_details"] = {"cached_tokens": 200, "cache_write_tokens": 300}
        wire(provider, json_handler(200, body))
        usage = (await provider.complete(USER)).usage
        assert (usage.input_tokens, usage.cache_read_input_tokens) == (500, 200)
        assert usage.cache_creation_input_tokens == 300

    async def test_chat_wire_reports_cache_writes(self) -> None:
        provider = chat_provider()
        body = sse(
            (None, c_chunk("x", finish="stop")),
            (None, c_usage(cached_tokens=10, cache_write_tokens=40)),
            (None, "[DONE]"),
        )
        wire(provider, stream_handler([body]))
        usage = (await drain(provider.complete_stream(USER)))[-1].raw.usage
        assert usage.input_tokens == 50
        assert usage.cache_read_input_tokens == 10
        assert usage.cache_creation_input_tokens == 40

    def test_none_stays_distinct_from_zero(self) -> None:
        def stats(details: Any) -> Any:
            return openai_wire_usage(
                {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": details},
                input_field="prompt_tokens",
                output_field="completion_tokens",
                details_field="prompt_tokens_details",
                output_details_field="completion_tokens_details",
            )

        assert stats(None).cache_creation_input_tokens is None
        assert stats(None).cache_read_input_tokens is None
        zero = stats({"cached_tokens": 0, "cache_write_tokens": 0})
        assert zero.cache_creation_input_tokens == 0 and zero.cache_read_input_tokens == 0
        assert stats({"cached_tokens": 50, "cache_write_tokens": 50}).input_tokens == 0

    @pytest.mark.parametrize(
        "details",
        [
            {"cache_write_tokens": 5, "ttl_12h_tokens": 5},
            {"cache_write_tokens": 5, "cache_write_tokens_details": {"ttl_12h_tokens": 5}},
        ],
    )
    def test_long_ttl_writes_mark_cost_unknown(self, details: dict[str, Any]) -> None:
        stats = openai_wire_usage(
            {"input_tokens": 10, "output_tokens": 1, "input_tokens_details": details},
            input_field="input_tokens",
            output_field="output_tokens",
            details_field="input_tokens_details",
            output_details_field="output_tokens_details",
        )
        assert stats.cost_unknown is True

    def test_reports_cost(self) -> None:
        assert openrouter_provider().reports_cost is True
        for provider in (anthropic_provider(), chat_provider(), responses_provider()):
            assert provider.reports_cost is False
        assert LLMProvider.reports_cost.fget is not None


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


class TestMachinery:
    async def test_retry_wrapper_closes_each_attempt(self) -> None:
        closed: list[int] = []

        def attempt_factory() -> Callable[[], Any]:
            count = {"n": 0}

            def make() -> Any:
                count["n"] += 1
                n = count["n"]

                async def gen() -> Any:
                    try:
                        if n == 1:
                            raise ProviderStreamIncompleteError(provider="p")
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")
                    finally:
                        closed.append(n)

                return gen()

            return make

        events = await drain(retry_unstarted_stream(attempt_factory(), provider="p", model="m"))
        assert [e.text for e in events] == ["ok"]
        assert closed == [1, 2]

    async def test_stream_clock_step_without_deadline(self) -> None:
        clock = StreamClock(Deadlines(None, None, None), provider="p", model="m")

        async def value() -> int:
            return 7

        assert await clock.step(value()) == 7

    async def test_inner_timeout_error_is_not_mistaken_for_deadline(self) -> None:
        clock = StreamClock(Deadlines(None, 5.0, None), provider="p", model="m")

        async def boom() -> None:
            raise TimeoutError("inner")

        with pytest.raises(TimeoutError, match="inner") as info:
            await clock.step(boom())
        assert not isinstance(info.value, ProviderError)

    def test_wire_request_repr_hides_payload(self) -> None:
        request = WireRequest({"model": "m", "messages": [SECRET]}, {}, registry=None)
        assert SECRET not in repr(request)

    async def test_call_state_is_reset_after_call(self) -> None:
        async def call() -> int:
            assert _resilience._call_state.get() is not None
            return 1

        await _resilience.run_with_deadline(
            call, when=None, reason="call_deadline", limit=None, provider="p", model=None
        )
        assert _resilience._call_state.get() is None

    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"retry-after-ms": "250"}, 0.25),
            ({"retry-after": "3"}, 3.0),
            ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
            ({"retry-after": "-1"}, None),
            ({}, None),
        ],
    )
    def test_parse_retry_after(self, headers: dict[str, str], expected: float | None) -> None:
        assert _resilience.parse_retry_after(httpx.Headers(headers)) == expected

    def test_vendor_error_type_is_kept_only_when_code_like(self) -> None:
        err = _resilience.error_from_fields(
            provider="p", model=None, status_code=400, code=None, error_type="free text here!"
        )
        assert isinstance(err, ProviderError)
        assert err.error_type is None
