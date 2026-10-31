"""Gateway bug fixes on the OpenAI-compatible path.

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

from dendrux.errors import ContextLengthExceededError, ProviderError  # noqa: E402
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


async def _drain_messages(provider: OpenAIProvider) -> list[Any]:
    return await _drain(provider)


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
        with pytest.raises(ProviderError) as info:
            await provider.complete(MESSAGES)
        assert not isinstance(info.value, ContextLengthExceededError)
        assert info.value.status_code == status
        assert isinstance(info.value.__cause__, openai.APIStatusError)
        assert PROMPT_SECRET not in str(info.value)


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
        with pytest.raises(ProviderError) as info:
            await _drain(provider)
        assert not isinstance(info.value, ContextLengthExceededError)
        assert info.value.status_code == status
        assert isinstance(info.value.__cause__, openai.APIStatusError)

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
        with pytest.raises(ProviderError) as info:
            await _drain(provider)
        assert not isinstance(info.value, ContextLengthExceededError)
        assert info.value.category == "unavailable"
        assert "upstream exploded" not in str(info.value)
        assert isinstance(info.value.__cause__, openai.APIError)


class TestFalsePositiveVetoes:
    """Codex explicit non-context signals beat the message patterns."""

    @pytest.mark.parametrize(
        "name", ["review_rate_limit_413", "review_permission_403", "review_output_cap_400"]
    )
    def test_probe_fixtures_are_other(self, name: str) -> None:
        fx = _FIXTURES["not_context"][name]
        err = fx["body"]["error"]
        result = classify_gateway_error(
            status_code=fx["status"],
            code=err.get("code"),
            param=err.get("param"),
            message=err["message"],
            body=err,
        )
        assert result.kind is GatewayErrorKind.OTHER

    @pytest.mark.parametrize(
        "name", ["review_rate_limit_413", "review_permission_403", "review_output_cap_400"]
    )
    async def test_probe_fixtures_propagate_unchanged(
        self, provider: OpenAIProvider, name: str
    ) -> None:
        fx = _FIXTURES["not_context"][name]
        for call in (provider.complete, lambda m: _drain_messages(provider)):
            _wire(provider, _json(fx["status"], fx["body"]))
            with pytest.raises(ProviderError) as info:
                await call(MESSAGES)
            assert not isinstance(info.value, ContextLengthExceededError)
            assert isinstance(info.value.__cause__, openai.APIStatusError)

    def test_explicit_context_code_still_beats_veto_param(self) -> None:
        result = classify_gateway_error(
            status_code=400,
            code="context_length_exceeded",
            param="max_tokens",
            message="x",
        )
        assert result.kind is GatewayErrorKind.CONTEXT_LENGTH

    def test_output_cap_with_input_token_evidence_is_context_length(self) -> None:
        fx = _FIXTURES["context_overflow"]["vllm_output_cap_param"]["body"]["error"]
        result = classify_gateway_error(
            status_code=400, code=None, param=fx["param"], message=fx["message"], body=fx
        )
        assert result.kind is GatewayErrorKind.CONTEXT_LENGTH

    def test_403_accepts_only_together_formula(self) -> None:
        together = _FIXTURES["context_overflow"]["together_403"]["body"]["error"]
        assert (
            classify_gateway_error(
                status_code=403, code=None, param=None, message=together["message"], body=together
            ).kind
            is GatewayErrorKind.CONTEXT_LENGTH
        )
        generic = classify_gateway_error(
            status_code=403,
            code=None,
            param=None,
            message="This model's maximum context length is 4096 tokens",
        )
        assert generic.kind is GatewayErrorKind.OTHER


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

    def test_numeric_body_code_equal_to_status_is_dropped(self) -> None:
        result = classify_gateway_error(
            status_code=400,
            code=None,
            param=None,
            message="x",
            body={"message": "maximum context length is 4096", "code": 400},
        )
        assert result.kind is GatewayErrorKind.CONTEXT_LENGTH
        assert (result.status_code, result.provider_code) == (400, None)

    def test_error_class_exported(self) -> None:
        import dendrux.errors as errors

        assert "ContextLengthExceededError" in errors.__all__
        err = ContextLengthExceededError("x", status_code=400, provider_code="c", model="m")
        assert (err.status_code, err.provider_code, err.model) == (400, "provider_unknown", "m")
        assert err.provider_code_raw == "c"


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
        with pytest.raises(ProviderError) as info:
            await _drain(provider)
        assert info.value.status_code == 400
        assert info.value.category == "rejected"
        assert isinstance(info.value.__cause__, openai.BadRequestError)


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
        with pytest.raises(ProviderError) as info:
            await _drain(openrouter)
        assert info.value.provider == "openrouter"
        assert isinstance(info.value.__cause__, openai.BadRequestError)

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


# ---------------------------------------------------------------------------
# Codex review: P1-2 sanitized streaming message, P2-1 per-call model
# ---------------------------------------------------------------------------
class TestReviewFixRound:
    async def test_stream_unsupported_message_has_no_provider_text(
        self, provider: OpenAIProvider
    ) -> None:
        _wire(
            provider,
            _json(
                400,
                {
                    "error": {
                        "message": "Streaming is not supported for this deployment; "
                        f"request excerpt: {PROMPT_SECRET}",
                        "param": "stream",
                    }
                },
            ),
        )
        with pytest.raises(NotImplementedError) as info:
            await _drain(provider)
        assert PROMPT_SECRET not in str(info.value)
        assert isinstance(info.value.__cause__, openai.BadRequestError)
        assert PROMPT_SECRET in str(info.value.__cause__)

    async def test_batch_error_uses_per_call_model(self) -> None:
        provider = OpenAIProvider(model="configured-model", api_key="k", max_retries=0)
        status, body = CONTEXT_OVERFLOWS["openai"]
        _wire(provider, _json(status, body))
        with pytest.raises(ContextLengthExceededError) as info:
            await provider.complete(MESSAGES, model="per-call-model")
        assert info.value.model == "per-call-model"
        assert "per-call-model" in str(info.value)
        assert "configured-model" not in str(info.value)

    async def test_stream_creation_error_uses_per_call_model(self) -> None:
        provider = OpenAIProvider(model="configured-model", api_key="k", max_retries=0)
        status, body = CONTEXT_OVERFLOWS["openai"]
        _wire(provider, _json(status, body))
        with pytest.raises(ContextLengthExceededError) as info:
            async for _ in provider.complete_stream(MESSAGES, model="per-call-model"):
                pass
        assert info.value.model == "per-call-model"

    async def test_stream_iteration_error_uses_per_call_model(self) -> None:
        provider = OpenAIProvider(model="configured-model", api_key="k", max_retries=0)
        _wire(
            provider,
            _sse_response(
                _chunk({"content": "x"}),
                {"error": {"message": "too big", "code": "context_length_exceeded"}},
            ),
        )
        with pytest.raises(ContextLengthExceededError) as info:
            async for _ in provider.complete_stream(MESSAGES, model="per-call-model"):
                pass
        assert info.value.model == "per-call-model"

    async def test_openrouter_inherits_per_call_model(self, openrouter: OpenRouterProvider) -> None:
        status, body = CONTEXT_OVERFLOWS["openai"]
        _wire(openrouter, _json(status, body))
        with pytest.raises(ContextLengthExceededError) as info:
            await openrouter.complete(MESSAGES, model="other/model")
        assert info.value.model == "other/model"

    def test_top_level_export(self) -> None:
        import dendrux

        assert dendrux.ContextLengthExceededError is ContextLengthExceededError
        assert "ContextLengthExceededError" in dendrux.__all__


class TestProviderCodeIsNeverTheHttpStatus:
    """OpenRouter sends the HTTP status as an integer ``code`` in the error body."""

    OPENROUTER_BODY = {
        "error": {
            "message": "This endpoint's maximum context length is 8192 tokens.",
            "code": 400,
            "metadata": {"provider_name": "x"},
        }
    }

    def test_integer_status_code_is_not_a_provider_code(self) -> None:
        err = self.OPENROUTER_BODY["error"]
        result = classify_gateway_error(
            status_code=400, code=None, param=None, message=str(err["message"]), body=err
        )
        assert result.kind is GatewayErrorKind.CONTEXT_LENGTH
        assert result.provider_code is None

    @pytest.mark.parametrize("raw", ["400", 400.5, True])
    def test_string_or_different_numeric_body_code_is_kept(self, raw: Any) -> None:
        result = classify_gateway_error(
            status_code=400, code=None, param=None, message="x", body={"code": raw}
        )
        assert result.provider_code == str(raw)

    def test_explicit_string_code_equal_to_status_is_kept(self) -> None:
        result = classify_gateway_error(status_code=400, code="400", param=None, message="x")
        assert result.provider_code == "400"

    def test_float_status_duplicate_is_dropped(self) -> None:
        result = classify_gateway_error(
            status_code=400, code=None, param=None, message="x", body={"code": 400.0}
        )
        assert result.provider_code is None

    def test_real_string_code_is_kept(self) -> None:
        result = classify_gateway_error(
            status_code=400,
            code=None,
            param=None,
            message="x",
            body={"code": "context_length_exceeded"},
        )
        assert result.provider_code == "context_length_exceeded"

    async def test_openrouter_error_exposes_no_numeric_provider_code(self) -> None:
        provider = OpenRouterProvider(model="m/x", api_key="test-key", max_retries=0)
        _wire(provider, _json(400, self.OPENROUTER_BODY))
        with pytest.raises(ContextLengthExceededError) as info:
            await provider.complete(MESSAGES)
        assert info.value.status_code == 400
        assert info.value.provider_code is None
