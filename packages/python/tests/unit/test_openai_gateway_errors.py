"""Gateway bug fixes on the OpenAI-compatible path (Dendrux 0.3.0 Track C).

B1  a stream 400 is only "streaming rejected" when it says so
B2  streamed tool calls survive a stream that never sends finish_reason
B4  context-length overflow -> ContextLengthExceededError (batch and stream)

Responses are recorded-style bodies served through a fake httpx transport and
the real OpenAI SDK, so status/body parsing matches what gateways send. No
network. OpenRouterProvider inherits all of it and is covered at the bottom.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.errors import ContextLengthExceededError  # noqa: E402
from dendrux.llm._helpers import (  # noqa: E402
    CONTEXT_LENGTH_CODES,
    CONTEXT_LENGTH_STATUSES,
    GatewayErrorKind,
    classify_gateway_error,
)
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openrouter import OpenRouterProvider  # noqa: E402
from dendrux.types import Message, Role, StreamEventType  # noqa: E402

PROMPT_SECRET = "TOP-SECRET-PROMPT-CONTENT"
MESSAGES = [Message(role=Role.USER, content=PROMPT_SECRET)]

Handler = Callable[[httpx.Request], httpx.Response]


def _wire(provider: OpenAIProvider, handler: Handler) -> list[httpx.Request]:
    """Point the provider's SDK client at a fake transport; return the request log."""
    seen: list[httpx.Request] = []

    def logging_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://gateway.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(logging_handler)),
        max_retries=0,
    )
    return seen


def _json(status: int, body: Any) -> Handler:
    return lambda request: httpx.Response(status, json=body)


def _sse(*payloads: Any) -> bytes:
    lines = ["data: " + (p if isinstance(p, str) else json.dumps(p)) + "\n\n" for p in payloads]
    return "".join(lines).encode()


def _sse_response(*payloads: Any) -> Handler:
    return lambda request: httpx.Response(
        200, content=_sse(*payloads), headers={"content-type": "text/event-stream"}
    )


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _tool_delta(
    index: int, *, id: str | None = None, name: str | None = None, args: str = ""
) -> dict:
    fn: dict[str, Any] = {"arguments": args}
    if name is not None:
        fn["name"] = name
    tc: dict[str, Any] = {"index": index, "function": fn}
    if id is not None:
        tc["id"] = id
        tc["type"] = "function"
    return {"tool_calls": [tc]}


_FIXTURES = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "gateway_errors.json").read_text()
)
CONTEXT_OVERFLOWS: dict[str, tuple[int, Any]] = {
    name: (f["status"], f.get("body")) for name, f in _FIXTURES["context_overflow"].items()
}
NOT_CONTEXT: dict[str, tuple[int, Any]] = {
    name: (f["status"], f.get("body")) for name, f in _FIXTURES["not_context"].items()
}


def _overflow_handler(status: int, body: Any) -> Handler:
    if body is None:
        return lambda request: httpx.Response(
            status, text="maximum context length exceeded", headers={"content-type": "text/plain"}
        )
    return _json(status, body)


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o", api_key="test-key", max_retries=0)


async def _drain(provider: OpenAIProvider) -> list[Any]:
    return [event async for event in provider.complete_stream(MESSAGES)]


# ---------------------------------------------------------------------------
# B4 — context-length detection
# ---------------------------------------------------------------------------
class TestContextLengthBatch:
    @pytest.mark.parametrize("name", sorted(CONTEXT_OVERFLOWS))
    async def test_overflow_raises_typed_error(self, provider: OpenAIProvider, name: str) -> None:
        status, body = CONTEXT_OVERFLOWS[name]
        _wire(provider, _overflow_handler(status, body))

        with pytest.raises(ContextLengthExceededError) as info:
            await provider.complete(MESSAGES)

        err = info.value
        assert isinstance(err, RuntimeError)
        assert err.status_code == status
        assert err.model == "gpt-4o"
        assert isinstance(err.__cause__, openai.APIStatusError)
        assert PROMPT_SECRET not in str(err)

    async def test_provider_code_is_carried(self, provider: OpenAIProvider) -> None:
        status, body = CONTEXT_OVERFLOWS["openai"]
        _wire(provider, _json(status, body))
        with pytest.raises(ContextLengthExceededError) as info:
            await provider.complete(MESSAGES)
        assert info.value.provider_code == "context_length_exceeded"
        assert info.value.status_code == 400

    @pytest.mark.parametrize("name", sorted(NOT_CONTEXT))
    async def test_other_errors_unchanged(self, provider: OpenAIProvider, name: str) -> None:
        status, body = NOT_CONTEXT[name]
        _wire(provider, _json(status, body))
        with pytest.raises(openai.APIStatusError) as info:
            await provider.complete(MESSAGES)
        assert not isinstance(info.value, ContextLengthExceededError)
        assert info.value.status_code == status


class TestContextLengthStream:
    @pytest.mark.parametrize("name", sorted(CONTEXT_OVERFLOWS))
    async def test_overflow_raises_typed_error(self, provider: OpenAIProvider, name: str) -> None:
        status, body = CONTEXT_OVERFLOWS[name]
        _wire(provider, _overflow_handler(status, body))

        with pytest.raises(ContextLengthExceededError) as info:
            await _drain(provider)

        assert info.value.status_code == status
        assert PROMPT_SECRET not in str(info.value)

    async def test_overflow_is_not_reported_as_streaming_rejected(
        self, provider: OpenAIProvider
    ) -> None:
        status, body = CONTEXT_OVERFLOWS["vllm_classic"]
        _wire(provider, _json(status, body))
        with pytest.raises(ContextLengthExceededError):
            await _drain(provider)

    @pytest.mark.parametrize("name", ["rate_limit", "auth", "server", "groq_tpm_413"])
    async def test_other_status_errors_unchanged(self, provider: OpenAIProvider, name: str) -> None:
        status, body = NOT_CONTEXT[name]
        _wire(provider, _json(status, body))
        with pytest.raises(openai.APIStatusError) as info:
            await _drain(provider)
        assert not isinstance(info.value, ContextLengthExceededError)
        assert info.value.status_code == status

    async def test_in_stream_error_event_with_context_code(self, provider: OpenAIProvider) -> None:
        _wire(
            provider,
            _sse_response(
                _chunk({"content": "partial"}),
                {
                    "error": {
                        "message": "Context window exceeded",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                },
            ),
        )
        events = []
        with pytest.raises(ContextLengthExceededError) as info:
            async for event in provider.complete_stream(MESSAGES):
                events.append(event)
        assert info.value.provider_code == "context_length_exceeded"
        assert info.value.status_code is None
        assert [e.text for e in events if e.type == StreamEventType.TEXT_DELTA] == ["partial"]

    async def test_in_stream_unrelated_error_event_unchanged(
        self, provider: OpenAIProvider
    ) -> None:
        _wire(
            provider,
            _sse_response(
                _chunk({"content": "x"}),
                {"error": {"message": "upstream exploded", "type": "server_error", "code": "boom"}},
            ),
        )
        with pytest.raises(openai.APIError) as info:
            await _drain(provider)
        assert not isinstance(info.value, ContextLengthExceededError)


class TestClassifier:
    """The shared pure classifier: ordered precedence and explicit accepted sets."""

    @staticmethod
    def _classify(**kw: Any) -> GatewayErrorKind:
        base: dict[str, Any] = {
            "status_code": 400,
            "code": None,
            "param": None,
            "message": "",
            "body": None,
        }
        return classify_gateway_error(**{**base, **kw}).kind

    def test_unrelated_message_is_other(self) -> None:
        assert self._classify(message="bad tool schema") is GatewayErrorKind.OTHER

    @pytest.mark.parametrize("status", [401, 404, 429, 500, 503])
    def test_unaccepted_status_is_never_context_length(self, status: int) -> None:
        kind = self._classify(
            status_code=status, code="context_length_exceeded", message="maximum context length"
        )
        assert kind is GatewayErrorKind.OTHER

    @pytest.mark.parametrize("status", sorted(CONTEXT_LENGTH_STATUSES) + [None])
    def test_accepted_statuses(self, status: int | None) -> None:
        assert (
            self._classify(status_code=status, code="context_length_exceeded")
            is GatewayErrorKind.CONTEXT_LENGTH
        )

    @pytest.mark.parametrize("code", sorted(CONTEXT_LENGTH_CODES))
    def test_accepted_codes_case_insensitive(self, code: str) -> None:
        assert self._classify(code=code.upper()) is GatewayErrorKind.CONTEXT_LENGTH

    @pytest.mark.parametrize(
        "text",
        [
            "THIS MODEL'S MAXIMUM CONTEXT LENGTH IS 10",
            "Please Reduce The Length Of The Messages",
            "the prompt is too long: 9",
            "exceeds the CONTEXT WINDOW",
        ],
    )
    def test_patterns_are_case_insensitive(self, text: str) -> None:
        assert self._classify(message=text) is GatewayErrorKind.CONTEXT_LENGTH

    def test_context_length_wins_over_streaming_unsupported(self) -> None:
        kind = self._classify(
            param="stream", message="maximum context length exceeded while streaming not supported"
        )
        assert kind is GatewayErrorKind.CONTEXT_LENGTH

    def test_streaming_unsupported_only_on_400(self) -> None:
        assert self._classify(param="stream") is GatewayErrorKind.STREAMING_UNSUPPORTED
        assert self._classify(status_code=500, param="stream") is GatewayErrorKind.OTHER
        assert self._classify(status_code=None, param="stream") is GatewayErrorKind.OTHER

    def test_normalizes_status_and_code_from_body(self) -> None:
        result = classify_gateway_error(
            status_code=400,
            code=None,
            param=None,
            message="x",
            body={"message": "maximum context length is 4096", "code": 400},
        )
        assert result.kind is GatewayErrorKind.CONTEXT_LENGTH
        assert (result.status_code, result.provider_code) == (400, "400")

    def test_error_class_exported(self) -> None:
        import dendrux.errors as errors

        assert "ContextLengthExceededError" in errors.__all__
        err = ContextLengthExceededError("x", status_code=400, provider_code="c", model="m")
        assert (err.status_code, err.provider_code, err.model) == (400, "c", "m")


# ---------------------------------------------------------------------------
# B1 — only real "streaming unsupported" 400s become NotImplementedError
# ---------------------------------------------------------------------------
class TestStreamRejected:
    @pytest.mark.parametrize(
        "body",
        [
            {"error": {"message": "Streaming is not supported for this model", "type": "x"}},
            {"error": {"message": "stream mode is unsupported by this deployment"}},
            {"error": {"message": "This endpoint does not support streaming responses."}},
            {
                "error": {
                    "message": "Unsupported value: 'stream' does not support true with this model.",
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": "unsupported_value",
                }
            },
            {"error": {"message": "Not allowed.", "param": "stream"}},
        ],
    )
    async def test_streaming_unsupported_maps_to_not_implemented(
        self, provider: OpenAIProvider, body: dict
    ) -> None:
        _wire(provider, _json(400, body))
        with pytest.raises(NotImplementedError, match="Streaming request rejected") as info:
            await _drain(provider)
        assert isinstance(info.value.__cause__, openai.BadRequestError)

    @pytest.mark.parametrize(
        "body",
        [
            {"error": {"message": "Invalid schema for function 'echo'", "param": "tools[0]"}},
            {"error": {"message": "Unrecognized request argument supplied: foo"}},
            {"error": {"message": "Invalid value for 'messages': must be non-empty"}},
            {"error": {"message": "stream_options only allowed when stream is true"}},
        ],
    )
    async def test_unrelated_400_propagates_unchanged(
        self, provider: OpenAIProvider, body: dict
    ) -> None:
        _wire(provider, _json(400, body))
        with pytest.raises(openai.BadRequestError) as info:
            await _drain(provider)
        assert info.value.status_code == 400


# ---------------------------------------------------------------------------
# B2 — end-of-stream flush of buffered tool calls
# ---------------------------------------------------------------------------
class TestEndOfStreamFlush:
    async def test_tool_call_flushed_when_no_finish_reason(self, provider: OpenAIProvider) -> None:
        _wire(
            provider,
            _sse_response(
                _chunk(_tool_delta(0, id="call_1", name="search", args="")),
                _chunk(_tool_delta(0, args='{"q": ')),
                _chunk(_tool_delta(0, args='"cats"}')),
                "[DONE]",
            ),
        )
        events = await _drain(provider)

        types = [e.type for e in events]
        assert types == [
            StreamEventType.TOOL_USE_START,
            StreamEventType.TOOL_USE_END,
            StreamEventType.DONE,
        ]
        end = events[1]
        assert end.tool_call is not None
        assert end.tool_call.name == "search"
        assert end.tool_call.params == {"q": "cats"}
        assert end.tool_call.provider_tool_call_id == "call_1"
        done = events[-1].raw
        assert done.tool_calls is not None and len(done.tool_calls) == 1

    async def test_multiple_tool_calls_flushed_in_index_order(
        self, provider: OpenAIProvider
    ) -> None:
        _wire(
            provider,
            _sse_response(
                _chunk(_tool_delta(1, id="b", name="second", args="{}")),
                _chunk(_tool_delta(0, id="a", name="first", args="{}")),
                "[DONE]",
            ),
        )
        events = await _drain(provider)
        ends = [e.tool_call.name for e in events if e.type == StreamEventType.TOOL_USE_END]
        assert ends == ["first", "second"]

    async def test_no_double_emission_when_finish_reason_arrives(
        self, provider: OpenAIProvider
    ) -> None:
        _wire(
            provider,
            _sse_response(
                _chunk(_tool_delta(0, id="call_1", name="search", args='{"q":"x"}')),
                _chunk({}, finish_reason="tool_calls"),
                "[DONE]",
            ),
        )
        events = await _drain(provider)
        assert [e.type for e in events].count(StreamEventType.TOOL_USE_END) == 1
        assert [e.type for e in events].count(StreamEventType.TOOL_USE_START) == 1
        done = events[-1].raw
        assert done.tool_calls is not None and len(done.tool_calls) == 1

    async def test_text_only_stream_without_finish_reason_has_no_tool_events(
        self, provider: OpenAIProvider
    ) -> None:
        _wire(provider, _sse_response(_chunk({"content": "hi"}), "[DONE]"))
        events = await _drain(provider)
        assert [e.type for e in events] == [StreamEventType.TEXT_DELTA, StreamEventType.DONE]


# ---------------------------------------------------------------------------
# OpenRouterProvider inherits all three fixes
# ---------------------------------------------------------------------------
@pytest.fixture
def openrouter() -> OpenRouterProvider:
    return OpenRouterProvider(model="deepseek/deepseek-chat", api_key="sk-or-test", max_retries=0)


class TestOpenRouterInherits:
    async def test_batch_context_overflow(self, openrouter: OpenRouterProvider) -> None:
        status, body = CONTEXT_OVERFLOWS["openai"]
        _wire(openrouter, _json(status, body))
        with pytest.raises(ContextLengthExceededError) as info:
            await openrouter.complete(MESSAGES)
        assert info.value.model == "deepseek/deepseek-chat"

    async def test_stream_context_overflow(self, openrouter: OpenRouterProvider) -> None:
        status, body = CONTEXT_OVERFLOWS["vllm_classic"]
        _wire(openrouter, _json(status, body))
        with pytest.raises(ContextLengthExceededError):
            await _drain(openrouter)

    async def test_stream_unrelated_400_propagates(self, openrouter: OpenRouterProvider) -> None:
        _wire(openrouter, _json(400, {"error": {"message": "Invalid schema", "param": "tools"}}))
        with pytest.raises(openai.BadRequestError):
            await _drain(openrouter)

    async def test_stream_unsupported_still_not_implemented(
        self, openrouter: OpenRouterProvider
    ) -> None:
        _wire(
            openrouter,
            _json(400, {"error": {"message": "Streaming is not supported for this model"}}),
        )
        with pytest.raises(NotImplementedError):
            await _drain(openrouter)

    async def test_end_of_stream_flush(self, openrouter: OpenRouterProvider) -> None:
        _wire(
            openrouter,
            _sse_response(_chunk(_tool_delta(0, id="c", name="echo", args="{}")), "[DONE]"),
        )
        events = await _drain(openrouter)
        assert [e.type for e in events].count(StreamEventType.TOOL_USE_END) == 1
