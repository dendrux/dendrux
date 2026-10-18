"""Image-request error classification, locating and exception hygiene (0.4.0 Track C).

Spec §5.5 (hygiene), §5.6 (IMAGE_UNSUPPORTED), §5.7 (IMAGE_REJECTED + locating).
Recorded-style vendor bodies are served through fake httpx transports and the
real SDKs, batch and stream (creation-time and mid-stream after a delta).
Text-only requests keep v0.3.0 errors. No network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

from dendrux.errors import (  # noqa: E402
    ContextLengthExceededError,
    ImageRejectedError,
    ProviderError,
    UnsupportedModalityError,
)
from dendrux.llm._helpers import (  # noqa: E402
    GatewayErrorKind,
    classify_gateway_error,
    extract_error_path,
)
from dendrux.llm._images import sanitize_vendor_message  # noqa: E402
from dendrux.llm.anthropic import AnthropicProvider  # noqa: E402
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402
from dendrux.llm.openrouter import OpenRouterProvider  # noqa: E402
from dendrux.media import inspect_image  # noqa: E402
from dendrux.types import (  # noqa: E402
    ImagePart,
    ImageRef,
    Message,
    Role,
    StreamEventType,
    TextPart,
    ToolCall,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "images"
ZEBRA = (FIXTURES / "zebra42.png").read_bytes()
KIWI = (FIXTURES / "kiwi77.png").read_bytes()
ZEBRA_B64 = base64.b64encode(ZEBRA).decode("ascii")
LEAK_MARKERS = (ZEBRA_B64[:40], ZEBRA_B64[-60:], "iVBORw0K", "data:image")

Handler = Callable[[httpx.Request], httpx.Response]


def image_part(data: bytes) -> ImagePart:
    header = inspect_image(data)
    ref = ImageRef(
        sha256=hashlib.sha256(data).hexdigest(),
        media_type=header.media_type,
        byte_length=len(data),
        width=header.width,
        height=header.height,
    )
    return ImagePart(ref=ref, data=data)


Z = image_part(ZEBRA)
K = image_part(KIWI)

USER_IMAGE = [Message.with_parts(Role.USER, [TextPart("What does this say?"), Z])]
TEXT_ONLY = [Message(role=Role.USER, content="hello")]


def tool_image_history(provider_call_id: str) -> list[Message]:
    """user text, assistant tool call, TOOL message with one image (Z), user image (K)."""
    tc = ToolCall(name="take_screenshot", params={}, provider_tool_call_id=provider_call_id)
    return [
        Message(role=Role.USER, content="screenshot please"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
        Message.with_parts(Role.TOOL, [TextPart('"Captured."'), Z], name=tc.name, call_id=tc.id),
        Message.with_parts(Role.USER, [TextPart("and this?"), K]),
    ]


# A vendor message that echoes request bytes (some gateways do).
ECHO = f"bad request near {'data:image/png;base64,' + ZEBRA_B64[:300]} and {ZEBRA_B64[:400]}"


def leak_text(exc: BaseException) -> str:
    """Everything reachable from an exception graph that could carry request bytes."""
    seen: set[int] = set()
    out: list[str] = []

    def visit(e: BaseException | None) -> None:
        if e is None or id(e) in seen:
            return
        seen.add(id(e))
        out.append(str(e))
        out.append(repr(e.args))
        for key, value in vars(e).items():
            out.append(f"{key}={value!r}")
        for attr in ("request", "response"):
            obj = getattr(e, attr, None)
            if obj is not None:
                content = getattr(obj, "content", b"")
                out.append(content.decode("utf-8", "replace") if isinstance(content, bytes) else "")
        visit(e.__cause__)
        visit(e.__context__)

    visit(exc)
    return "\n".join(out)


def assert_clean(exc: BaseException) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None
    text = leak_text(exc)
    for marker in LEAK_MARKERS:
        assert marker not in text


def sse(*events: tuple[str | None, Any]) -> bytes:
    chunks = []
    for name, data in events:
        line = f"event: {name}\n" if name else ""
        chunks.append(
            line + "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"
        )
    return "".join(chunks).encode()


def sse_response(*events: tuple[str | None, Any]) -> Handler:
    return lambda request: httpx.Response(
        200, content=sse(*events), headers={"content-type": "text/event-stream"}
    )


def json_response(status: int, body: Any) -> Handler:
    return lambda request: httpx.Response(status, json=body)


def wire_anthropic(provider: AnthropicProvider, handler: Handler) -> None:
    provider._client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="http://anthropic.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def wire_openai(provider: Any, handler: Handler, base_url: str = "http://gateway.test/v1") -> None:
    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


async def drain(gen: Any) -> list[Any]:
    return [event async for event in gen]


# ---------------------------------------------------------------------------
# Recorded-style bodies (live probe report + vendor docs)
# ---------------------------------------------------------------------------
ANTHROPIC_REJECTED_LOCATED = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "messages.2.content.0.content.1.image.source.base64.data: "
        "At least one of the image dimensions exceed max allowed size: 8000 pixels",
    },
}
ANTHROPIC_REJECTED_UNLOCATED = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "Could not process image"},
}
ANTHROPIC_OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": ECHO}}
CHAT_UNSUPPORTED = {
    "error": {
        "message": "Invalid content type. image_url is only supported by certain models.",
        "type": "invalid_request_error",
        "param": "messages.[0].content.[1].type",
        "code": None,
    }
}
CHAT_REJECTED_LOCATED = {
    "error": {
        "message": "Invalid image.",
        "type": "invalid_request_error",
        "param": "messages[3].content[1].image_url",
        "code": "invalid_image",
    }
}
CHAT_REJECTED_UNLOCATED = {
    "error": {
        "message": "You uploaded an unsupported image. Please make sure your image is valid.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_image_format",
    }
}
RESPONSES_UNSUPPORTED = {
    "error": {
        "message": "Model 'o3-mini-2025-01-31' does not support image inputs. Try again with a "
        "vision model.",
        "type": "invalid_request_error",
        "param": "input",
        "code": None,
    }
}
RESPONSES_REJECTED_LOCATED = {
    "error": {
        "message": "Invalid image",
        "type": "invalid_request_error",
        "param": "input[2].output[1]",
        "code": "invalid_image",
    }
}
OPENROUTER_UNSUPPORTED = {
    "error": {
        "message": "No endpoints found that support image input",
        "code": 404,
        "metadata": {"failed_routing_step": "Filter by Image Support"},
    }
}
OPENROUTER_REJECTED = {
    "error": {
        "message": "Provider returned error",
        "code": 400,
        "metadata": {
            "raw": '{"type":"error","error":{"type":"invalid_request_error","message":'
            '"messages.1.content.1.image.source.base64: image exceeds 5 MB maximum"}}',
            "provider_name": "Anthropic",
        },
    }
}
CONTEXT_OVERFLOW = {
    "error": {
        "message": "This model's maximum context length is 128000 tokens.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }
}


# ---------------------------------------------------------------------------
# Classifier + path extraction (pure)
# ---------------------------------------------------------------------------
def _classify(status: int | None, inner: dict[str, Any], *, image: bool = True) -> GatewayErrorKind:
    meta = inner.get("metadata") or {}
    code = inner.get("code")
    return classify_gateway_error(
        status_code=status,
        code=code if isinstance(code, str) else None,
        param=inner.get("param"),
        message=inner.get("message", ""),
        body=inner,
        image_request=image,
        error_type=inner.get("type"),
        extra_text=meta.get("raw"),
    ).kind


class TestClassifier:
    @pytest.mark.parametrize(
        ("status", "body"),
        [(400, CHAT_UNSUPPORTED), (400, RESPONSES_UNSUPPORTED), (404, OPENROUTER_UNSUPPORTED)],
    )
    def test_unsupported(self, status: int, body: dict[str, Any]) -> None:
        assert _classify(status, body["error"]) is GatewayErrorKind.IMAGE_UNSUPPORTED
        assert _classify(None, body["error"]) is GatewayErrorKind.IMAGE_UNSUPPORTED

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (400, ANTHROPIC_REJECTED_LOCATED),
            (400, ANTHROPIC_REJECTED_UNLOCATED),
            (400, CHAT_REJECTED_LOCATED),
            (400, CHAT_REJECTED_UNLOCATED),
            (400, RESPONSES_REJECTED_LOCATED),
            (400, OPENROUTER_REJECTED),
        ],
    )
    def test_rejected(self, status: int, body: dict[str, Any]) -> None:
        assert _classify(status, body["error"]) is GatewayErrorKind.IMAGE_REJECTED
        assert _classify(None, body["error"]) is GatewayErrorKind.IMAGE_REJECTED

    @pytest.mark.parametrize(
        "message",
        [
            "image exceeds 5 MB maximum: 7340032 bytes > 5242880 bytes",
            "Image does not match the provided media type image/png",
            "messages.0.content.21.image.source.base64.data: At least one of the image "
            "dimensions exceed max allowed size for many-image requests: 2000 pixels",
            "Invalid base64 data in image",
        ],
    )
    def test_anthropic_keyword_rule(self, message: str) -> None:
        inner = {"type": "invalid_request_error", "message": message}
        assert _classify(400, inner) is GatewayErrorKind.IMAGE_REJECTED
        other_type = {"type": "overloaded_error", "message": message}
        assert _classify(400, other_type) is GatewayErrorKind.OTHER

    def test_text_only_requests_never_get_image_kinds(self) -> None:
        for body in (CHAT_UNSUPPORTED, CHAT_REJECTED_LOCATED, OPENROUTER_UNSUPPORTED):
            assert _classify(400, body["error"], image=False) is GatewayErrorKind.OTHER

    def test_precedence_context_then_unsupported_then_rejected(self) -> None:
        ctx = dict(CONTEXT_OVERFLOW["error"], message="maximum context length; invalid image")
        assert _classify(400, ctx) is GatewayErrorKind.CONTEXT_LENGTH
        both = {"message": "does not support image inputs; Invalid image", "code": "invalid_image"}
        assert _classify(400, both) is GatewayErrorKind.IMAGE_UNSUPPORTED

    def test_wrong_status_is_other(self) -> None:
        assert _classify(500, CHAT_REJECTED_LOCATED["error"]) is GatewayErrorKind.OTHER
        assert _classify(429, OPENROUTER_UNSUPPORTED["error"]) is GatewayErrorKind.OTHER

    @pytest.mark.parametrize(
        ("param", "text", "expected"),
        [
            ("messages[3].content[1].image_url", None, ("messages", 3, "content", 1, "image_url")),
            ("messages.[0].content.[1].type", None, ("messages", 0, "content", 1, "type")),
            (
                None,
                "messages.2.content.0.content.1.image.source.base64.data: too big",
                ("messages", 2, "content", 0, "content", 1, "image", "source", "base64", "data"),
            ),
            ("input[2].output[1]", None, ("input", 2, "output", 1)),
            ("input", "Invalid 'input[4].content[0]'", ("input", 4, "content", 0)),
            (None, "your messages resulted in 130000 tokens", None),
            (None, None, None),
        ],
    )
    def test_extract_error_path(self, param: Any, text: Any, expected: Any) -> None:
        assert extract_error_path(param, text) == expected

    def test_sanitize_vendor_message(self) -> None:
        clean = sanitize_vendor_message(ECHO)
        for marker in LEAK_MARKERS:
            assert marker not in clean
        assert len(sanitize_vendor_message("word " * 400)) == 500


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
@pytest.fixture
def claude() -> AnthropicProvider:
    return AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key")


def _anthropic_head() -> list[tuple[str, Any]]:
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
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
                "delta": {"type": "text_delta", "text": "The code"},
            },
        ),
    ]


class TestAnthropicErrors:
    async def test_rejected_located_batch(self, claude: AnthropicProvider) -> None:
        wire_anthropic(claude, json_response(400, ANTHROPIC_REJECTED_LOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await claude.complete(tool_image_history("toolu_P1"))
        err = info.value
        assert err.located is True
        assert err.image_ids == (Z.image_id,)
        assert err.retried is False
        assert err.status_code == 400
        assert err.provider == "anthropic"
        assert_clean(err)

    async def test_rejected_unlocated_batch(self, claude: AnthropicProvider) -> None:
        wire_anthropic(claude, json_response(400, ANTHROPIC_REJECTED_UNLOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await claude.complete(tool_image_history("toolu_P1"))
        assert info.value.located is False
        assert info.value.image_ids == ()
        assert_clean(info.value)

    async def test_other_error_is_sanitised_provider_error(self, claude: AnthropicProvider) -> None:
        wire_anthropic(claude, json_response(529, ANTHROPIC_OVERLOADED))
        with pytest.raises(ProviderError) as info:
            await claude.complete(USER_IMAGE)
        err = info.value
        assert err.status_code == 529
        assert err.error_type == "overloaded_error"
        assert "[redacted]" in err.message
        assert_clean(err)

    async def test_text_only_keeps_sdk_error(self, claude: AnthropicProvider) -> None:
        wire_anthropic(claude, json_response(400, ANTHROPIC_REJECTED_UNLOCATED))
        with pytest.raises(anthropic.BadRequestError):
            await claude.complete(TEXT_ONLY)

    async def test_timeout_has_no_request_attached(self, claude: AnthropicProvider) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        wire_anthropic(claude, boom)
        with pytest.raises(ProviderError) as info:
            await claude.complete(USER_IMAGE)
        assert info.value.error_type == "timeout"
        assert_clean(info.value)

    async def test_stream_creation_rejected(self, claude: AnthropicProvider) -> None:
        wire_anthropic(claude, json_response(400, ANTHROPIC_REJECTED_LOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await drain(claude.complete_stream(tool_image_history("toolu_P1")))
        assert info.value.image_ids == (Z.image_id,)
        assert_clean(info.value)

    async def test_mid_stream_rejection_after_delta(self, claude: AnthropicProvider) -> None:
        events = [*_anthropic_head(), ("error", ANTHROPIC_REJECTED_UNLOCATED)]
        wire_anthropic(claude, sse_response(*events))
        seen: list[Any] = []
        with pytest.raises(ImageRejectedError) as info:
            async for event in claude.complete_stream(USER_IMAGE):
                seen.append(event)
        assert [e.type for e in seen] == [StreamEventType.TEXT_DELTA]
        assert info.value.status_code is None  # in-stream
        assert info.value.located is False
        assert_clean(info.value)


# ---------------------------------------------------------------------------
# OpenAI Chat + OpenRouter
# ---------------------------------------------------------------------------
@pytest.fixture
def chat() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o-mini", api_key="test-key", max_retries=0)


def _chat_chunk(content: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


class TestChatErrors:
    async def test_unsupported_learns_no(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(400, CHAT_UNSUPPORTED))
        assert chat.learned_image_support("gpt-4o-mini") is None
        with pytest.raises(UnsupportedModalityError) as info:
            await chat.complete(USER_IMAGE)
        err = info.value
        assert err.source == "provider_response"
        assert err.image_count == 1
        assert err.status_code == 400
        assert chat.learned_image_support("gpt-4o-mini") == "no"
        assert chat.learned_image_support("gpt-4o") is None
        assert_clean(err)

    async def test_rejected_located_on_framed_message(self, chat: OpenAIProvider) -> None:
        # messages: 0 user, 1 assistant, 2 tool, 3 framed user (Z at content[2]), 4 user (K at [1])
        body = json.loads(json.dumps(CHAT_REJECTED_LOCATED))
        body["error"]["param"] = "messages[3].content[2].image_url"
        wire_openai(chat, json_response(400, body))
        with pytest.raises(ImageRejectedError) as info:
            await chat.complete(tool_image_history("call_P1"))
        assert info.value.image_ids == (Z.image_id,)
        body["error"]["param"] = "messages[4].content[1]"
        wire_openai(chat, json_response(400, body))
        with pytest.raises(ImageRejectedError) as info:
            await chat.complete(tool_image_history("call_P1"))
        assert info.value.image_ids == (K.image_id,)
        assert_clean(info.value)

    async def test_rejected_unlocated(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(400, CHAT_REJECTED_UNLOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await chat.complete(tool_image_history("call_P1"))
        assert info.value.located is False
        assert info.value.provider_code == "invalid_image_format"

    async def test_context_length_on_image_request_unchained(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(400, CONTEXT_OVERFLOW))
        with pytest.raises(ContextLengthExceededError) as info:
            await chat.complete(USER_IMAGE)
        assert_clean(info.value)

    async def test_text_only_context_length_still_chained(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(400, CONTEXT_OVERFLOW))
        with pytest.raises(ContextLengthExceededError) as info:
            await chat.complete(TEXT_ONLY)
        assert isinstance(info.value.__cause__, openai.BadRequestError)

    async def test_stream_creation_unsupported(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(400, CHAT_UNSUPPORTED))
        with pytest.raises(UnsupportedModalityError) as info:
            await drain(chat.complete_stream(USER_IMAGE))
        assert_clean(info.value)

    async def test_mid_stream_rejection_after_delta(self, chat: OpenAIProvider) -> None:
        wire_openai(
            chat,
            sse_response(
                (None, _chat_chunk("The code")),
                (None, {"error": CHAT_REJECTED_UNLOCATED["error"]}),
            ),
        )
        seen: list[Any] = []
        with pytest.raises(ImageRejectedError) as info:
            async for event in chat.complete_stream(USER_IMAGE):
                seen.append(event)
        assert [e.type for e in seen] == [StreamEventType.TEXT_DELTA]
        assert info.value.status_code is None
        assert_clean(info.value)

    async def test_other_status_echoing_bytes_is_redacted(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(500, {"error": {"message": ECHO, "type": "server_error"}}))
        with pytest.raises(ProviderError) as info:
            await chat.complete(USER_IMAGE)
        assert info.value.status_code == 500
        assert_clean(info.value)

    async def test_caplog_has_no_image_bytes(
        self, chat: OpenAIProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="dendrux")
        wire_openai(chat, json_response(400, CHAT_REJECTED_UNLOCATED))
        with pytest.raises(ImageRejectedError):
            await chat.complete(USER_IMAGE)
        for marker in LEAK_MARKERS:
            assert marker not in caplog.text


class TestOpenRouterErrors:
    @pytest.fixture
    def router(self) -> OpenRouterProvider:
        return OpenRouterProvider(model="deepseek/deepseek-chat", api_key="test-key", max_retries=0)

    async def test_404_unsupported(self, router: OpenRouterProvider) -> None:
        wire_openai(router, json_response(404, OPENROUTER_UNSUPPORTED))
        with pytest.raises(UnsupportedModalityError) as info:
            await router.complete(USER_IMAGE)
        assert info.value.provider == "openrouter"
        assert info.value.status_code == 404
        assert router.learned_image_support("deepseek/deepseek-chat") == "no"
        assert_clean(info.value)

    async def test_rejection_via_metadata_raw_never_located(
        self, router: OpenRouterProvider
    ) -> None:
        wire_openai(router, json_response(400, OPENROUTER_REJECTED))
        with pytest.raises(ImageRejectedError) as info:
            await router.complete(tool_image_history("call_P1"))
        assert info.value.located is False
        assert info.value.image_ids == ()
        assert_clean(info.value)


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------
@pytest.fixture
def responses() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-5-mini", api_key="test-key", max_retries=0)


class TestResponsesErrors:
    async def test_unsupported_batch(self, responses: OpenAIResponsesProvider) -> None:
        wire_openai(responses, json_response(400, RESPONSES_UNSUPPORTED))
        with pytest.raises(UnsupportedModalityError) as info:
            await responses.complete(USER_IMAGE)
        assert info.value.provider == "openai_responses"
        assert_clean(info.value)

    async def test_rejected_located_native_output(self, responses: OpenAIResponsesProvider) -> None:
        # input: 0 user, 1 function_call, 2 function_call_output (Z at output[1]), 3 user
        wire_openai(responses, json_response(400, RESPONSES_REJECTED_LOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await responses.complete(tool_image_history("call_P1"))
        assert info.value.image_ids == (Z.image_id,)
        assert_clean(info.value)

    async def test_stream_creation_bad_request_is_typed_not_notimplemented(
        self, responses: OpenAIResponsesProvider
    ) -> None:
        wire_openai(responses, json_response(400, RESPONSES_UNSUPPORTED))
        with pytest.raises(UnsupportedModalityError) as info:
            await drain(responses.complete_stream(USER_IMAGE))
        assert_clean(info.value)

    async def test_text_only_stream_bad_request_unchanged(
        self, responses: OpenAIResponsesProvider
    ) -> None:
        wire_openai(responses, json_response(400, RESPONSES_UNSUPPORTED))
        with pytest.raises(NotImplementedError):
            await drain(responses.complete_stream(TEXT_ONLY))

    @pytest.mark.parametrize("terminal", ["response.failed", "error"])
    async def test_mid_stream_rejection_after_delta(
        self, responses: OpenAIResponsesProvider, terminal: str
    ) -> None:
        delta = {"type": "response.output_text.delta", "item_id": "m1", "delta": "The code"}
        error = {"code": "invalid_image", "message": "Invalid image"}
        if terminal == "error":
            last = {"type": "error", **error, "param": None}
        else:
            last = {"type": "response.failed", "response": {"id": "r1", "error": error}}
        wire_openai(responses, sse_response((None, delta), (None, last)))
        seen: list[Any] = []
        with pytest.raises(ImageRejectedError) as info:
            async for event in responses.complete_stream(USER_IMAGE):
                seen.append(event)
        assert [e.type for e in seen] == [StreamEventType.TEXT_DELTA]
        assert info.value.status_code is None
        assert info.value.provider_code == "invalid_image"
        assert_clean(info.value)

    async def test_mid_stream_sdk_error(self, responses: OpenAIResponsesProvider) -> None:
        delta = {"type": "response.output_text.delta", "item_id": "m1", "delta": "x"}
        wire_openai(
            responses,
            sse_response((None, delta), (None, {"error": {"message": ECHO, "type": "server"}})),
        )
        with pytest.raises(ProviderError) as info:
            await drain(responses.complete_stream(USER_IMAGE))
        assert_clean(info.value)


# ---------------------------------------------------------------------------
# Fix round (review-C)
# ---------------------------------------------------------------------------
SDK_IMAGE_CODES = (
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "image_content_policy_violation",
    "invalid_image_mode",
    "image_file_too_large",
    "unsupported_image_media_type",
    "empty_image_file",
)


def assert_frames_clean(exc: BaseException) -> None:
    """No traceback frame local (as an error tracker would capture it) holds image bytes."""
    tb = exc.__traceback__
    assert tb is not None
    while tb is not None:
        frame = tb.tb_frame
        for name, value in frame.f_locals.items():
            text = repr(value)
            for marker in LEAK_MARKERS:
                assert marker not in text, (frame.f_code.co_name, name)
        tb = tb.tb_next


class RecordingStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False

    async def __aiter__(self) -> Any:
        yield self.data

    async def aclose(self) -> None:
        self.closed = True


class TestFixRoundErrors:
    @pytest.mark.parametrize("code", SDK_IMAGE_CODES)
    @pytest.mark.parametrize("status", [400, None])
    def test_sdk_image_codes_are_rejections(self, code: str, status: int | None) -> None:
        inner = {"message": "Something about the upload.", "code": code}
        assert _classify(status, inner) is GatewayErrorKind.IMAGE_REJECTED

    def test_non_image_sdk_codes_are_not(self) -> None:
        for code in ("server_error", "rate_limit_exceeded", "failed_to_download_image"):
            assert _classify(400, {"message": "x", "code": code}) is GatewayErrorKind.OTHER

    @pytest.mark.parametrize(
        "error",
        [
            {"code": "image_too_small", "message": "Image too small."},
            {"code": "empty_image_file", "message": "The file is empty."},
            {"code": None, "message": "Image too large for this model; exceeds limit"},
        ],
    )
    async def test_responses_error_event_rejections(
        self, responses: OpenAIResponsesProvider, error: dict[str, Any]
    ) -> None:
        last = {"type": "error", "param": None, **error}
        wire_openai(responses, sse_response((None, last)))
        with pytest.raises(ImageRejectedError) as info:
            await drain(responses.complete_stream(USER_IMAGE))
        assert_clean(info.value)
        failed = {"type": "response.failed", "response": {"id": "r", "error": error}}
        wire_openai(responses, sse_response((None, failed)))
        with pytest.raises(ImageRejectedError):
            await drain(responses.complete_stream(USER_IMAGE))

    async def test_frame_locals_clean_anthropic(self, claude: AnthropicProvider) -> None:
        wire_anthropic(claude, json_response(400, ANTHROPIC_REJECTED_LOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await claude.complete(tool_image_history("toolu_P1"))
        assert_frames_clean(info.value)
        with pytest.raises(ImageRejectedError) as info:
            await drain(claude.complete_stream(tool_image_history("toolu_P1")))
        assert_frames_clean(info.value)
        wire_anthropic(claude, sse_response(*_anthropic_head(), ("error", ANTHROPIC_OVERLOADED)))
        with pytest.raises(ProviderError) as perr:
            await drain(claude.complete_stream(USER_IMAGE))
        assert_frames_clean(perr.value)

    async def test_frame_locals_clean_chat(self, chat: OpenAIProvider) -> None:
        wire_openai(chat, json_response(400, CHAT_REJECTED_UNLOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await chat.complete(USER_IMAGE)
        assert_frames_clean(info.value)
        with pytest.raises(ImageRejectedError) as info:
            await drain(chat.complete_stream(USER_IMAGE))
        assert_frames_clean(info.value)
        wire_openai(
            chat,
            sse_response(
                (None, _chat_chunk("x")), (None, {"error": CHAT_REJECTED_UNLOCATED["error"]})
            ),
        )
        with pytest.raises(ImageRejectedError) as info:
            await drain(chat.complete_stream(USER_IMAGE))
        assert_frames_clean(info.value)

    async def test_frame_locals_clean_responses(self, responses: OpenAIResponsesProvider) -> None:
        wire_openai(responses, json_response(400, RESPONSES_REJECTED_LOCATED))
        with pytest.raises(ImageRejectedError) as info:
            await responses.complete(tool_image_history("call_P1"))
        assert_frames_clean(info.value)
        with pytest.raises(ImageRejectedError) as info:
            await drain(responses.complete_stream(tool_image_history("call_P1")))
        assert_frames_clean(info.value)
        delta = {"type": "response.output_text.delta", "item_id": "m1", "delta": "x"}
        failed = {
            "type": "response.failed",
            "response": {"id": "r", "error": {"code": "invalid_image", "message": "bad"}},
        }
        wire_openai(responses, sse_response((None, delta), (None, failed)))
        with pytest.raises(ImageRejectedError) as info:
            await drain(responses.complete_stream(USER_IMAGE))
        assert_frames_clean(info.value)

    async def test_timeout_keeps_hint(self, chat: OpenAIProvider) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        wire_openai(chat, boom)
        with pytest.raises(ProviderError) as info:
            await chat.complete(USER_IMAGE)
        assert "Increase timeout" in info.value.message
        assert_clean(info.value)

    async def test_in_stream_error_closes_stream(self, chat: OpenAIProvider) -> None:
        body = sse((None, _chat_chunk("x")), (None, {"error": CHAT_REJECTED_UNLOCATED["error"]}))
        recorder = RecordingStream(body)
        wire_openai(
            chat,
            lambda request: httpx.Response(
                200, stream=recorder, headers={"content-type": "text/event-stream"}
            ),
        )
        with pytest.raises(ImageRejectedError):
            await drain(chat.complete_stream(USER_IMAGE))
        assert recorder.closed is True

    def test_data_url_redaction_is_word_bounded(self) -> None:
        assert sanitize_vendor_message("metadata: x") == "metadata: x"
        assert sanitize_vendor_message("see data:image/png;base64,AAAA end") == "see [redacted] end"
