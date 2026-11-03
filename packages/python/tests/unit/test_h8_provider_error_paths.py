"""Error-path pins for the OpenAI-family adapters (H8).

Captured from the code before the H8 helper move, at the HTTP layer (httpx
mock transport, real SDK), for Chat, Responses and OpenRouter:

* a connection failure in ``complete`` and a broken stream after a delta;
* image-bearing rejections are typed and unchained, with no image bytes in
  errors or logs and no wire request retained in the streaming traceback;
* adapter labels, stream cleanup, lifecycle and extra-field readers.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Mapping
from types import AsyncGeneratorType, GeneratorType
from typing import TYPE_CHECKING, Any

import httpx
import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.errors import (  # noqa: E402
    ContentFilteredError,
    ImageRejectedError,
    ModelRefusalError,
    ProviderError,
    ProviderResponseError,
    ProviderStreamIncompleteError,
    ProviderTimeoutError,
)
from dendrux.llm._images import WireImageRegistry  # noqa: E402
from dendrux.llm._resilience import WireRequest  # noqa: E402
from dendrux.llm.anthropic import AnthropicProvider  # noqa: E402
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402
from dendrux.llm.openrouter import OpenRouterProvider  # noqa: E402
from dendrux.types import Message, Role, StreamEventType  # noqa: E402
from tests.unit.test_provider_images_errors import (  # noqa: E402
    ANTHROPIC_REJECTED_UNLOCATED,
    CHAT_REJECTED_UNLOCATED,
    LEAK_MARKERS,
    OPENROUTER_REJECTED,
    RESPONSES_REJECTED_LOCATED,
    USER_IMAGE,
    _anthropic_head,
    _chat_chunk,
    assert_clean,
    json_response,
    sse_response,
    tool_image_history,
    wire_anthropic,
    wire_openai,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SECRET = "SECRET-H8-7f3a"
TEXT = [Message(role=Role.USER, content="hi")]


def _chat() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o", api_key="test-key", max_retries=0)


def _responses() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-5-mini", api_key="test-key", max_retries=0)


def _router() -> OpenRouterProvider:
    return OpenRouterProvider(model="deepseek/deepseek-chat", api_key="test-key", max_retries=0)


FACTORIES: dict[str, tuple[Callable[[], Any], str]] = {
    "chat": (_chat, "openai"),
    "responses": (_responses, "openai_responses"),
    "openrouter": (_router, "openrouter"),
}


class _BrokenBody(httpx.AsyncByteStream):
    """An SSE body that delivers one event, then the connection drops."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self) -> Any:
        yield self._first
        raise httpx.ReadError(f"peer reset {SECRET}")


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError(f"refused {SECRET}", request=request)


def _first_delta(adapter: str) -> bytes:
    if adapter == "responses":
        event = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "logprobs": [],
        }
        return f"event: response.output_text.delta\ndata: {json.dumps(event)}\n\n".encode()
    return f"data: {json.dumps(_chat_chunk('Hello'))}\n\n".encode()


def _holders(err: BaseException) -> dict[str, list[str]]:
    """Frame name -> locals holding a wire request or image registry, along the traceback."""
    found: dict[str, list[str]] = {}
    tb = err.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        names = sorted(
            k for k, v in frame.f_locals.items() if isinstance(v, WireRequest | WireImageRegistry)
        )
        if names:
            found[frame.f_code.co_name] = names
        tb = tb.tb_next
    return found


def _reachable_request_holders(err: BaseException, sensitive_ids: set[int]) -> set[str]:
    found: set[str] = set()

    def visit(value: Any, path: str, depth: int, seen: set[int]) -> None:
        identity = id(value)
        if identity in sensitive_ids or isinstance(value, WireRequest | WireImageRegistry):
            found.add(path)
            return
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, functools.partial):
            visit(value.args, path + ".args", depth, seen)
            visit(value.keywords, path + ".keywords", depth, seen)
        elif isinstance(value, AsyncGeneratorType | GeneratorType):
            frame = value.ag_frame if isinstance(value, AsyncGeneratorType) else value.gi_frame
            if frame is not None:
                visit(frame.f_locals, path + ".frame", depth, seen)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}[{key}]", depth, seen)
        elif isinstance(value, tuple | list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth, seen)
        elif depth:
            for name, item in getattr(value, "__dict__", {}).items():
                visit(item, path + "." + name, depth - 1, seen)

    tb = err.__traceback__
    while tb is not None:
        for name, value in tb.tb_frame.f_locals.items():
            visit(value, tb.tb_frame.f_code.co_name + "." + name, 1, set())
        tb = tb.tb_next
    return found


def _frame_locals(err: BaseException, co_name: str) -> set[str]:
    tb = err.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == co_name:
            return set(tb.tb_frame.f_locals)
        tb = tb.tb_next
    raise AssertionError(f"no {co_name} frame in the traceback")


# ---------------------------------------------------------------------------
# Connection failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", sorted(FACTORIES))
async def test_complete_connection_error(adapter: str) -> None:
    factory, label = FACTORIES[adapter]
    provider = factory()
    wire_openai(provider, _refuse)
    with pytest.raises(ProviderError) as info:
        await provider.complete(TEXT)
    err = info.value
    assert type(err).__name__ == "ProviderUnavailableError"
    assert isinstance(err, ConnectionError)
    assert err.category == "unavailable"
    assert str(err) == (
        f"Provider '{label}' error: Could not reach the provider (connection failed after retries)."
    )
    assert isinstance(err.__cause__, openai.APIConnectionError)
    assert SECRET not in str(err)


@pytest.mark.parametrize("adapter", sorted(FACTORIES))
async def test_stream_connection_drop_after_delta(adapter: str) -> None:
    factory, label = FACTORIES[adapter]
    provider = factory()
    body = _first_delta(adapter)
    wire_openai(
        provider,
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_BrokenBody(body)
        ),
    )
    seen: list[Any] = []
    with pytest.raises(ProviderStreamIncompleteError) as info:
        async for event in provider.complete_stream(TEXT):
            seen.append(event)
    assert [e.type for e in seen] == [StreamEventType.TEXT_DELTA]
    err = info.value
    assert err.category == "stream_incomplete"
    assert str(err) == (
        f"Provider '{label}' error: The response stream broke before its terminal event."
    )
    assert isinstance(err.__cause__, httpx.ReadError)
    assert SECRET not in str(err)


# ---------------------------------------------------------------------------
# Image-bearing rejection
# ---------------------------------------------------------------------------

BATCH_REJECTIONS = {
    "chat": (USER_IMAGE, json_response(400, CHAT_REJECTED_UNLOCATED)),
    "responses": (tool_image_history("call_P1"), json_response(400, RESPONSES_REJECTED_LOCATED)),
    "openrouter": (tool_image_history("call_P1"), json_response(400, OPENROUTER_REJECTED)),
}


@pytest.mark.parametrize("adapter", sorted(FACTORIES))
async def test_image_rejection_batch_is_unchained_and_unreachable(
    adapter: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="dendrux")
    factory, label = FACTORIES[adapter]
    provider = factory()
    messages, handler = BATCH_REJECTIONS[adapter]
    wire_openai(provider, handler)
    with pytest.raises(ImageRejectedError) as info:
        await provider.complete(messages)
    err = info.value
    assert err.provider == label
    assert_clean(err)
    assert _holders(err) == {}, "the request and registry must not be frame locals"
    for marker in LEAK_MARKERS:
        assert marker not in caplog.text


def _chat_mid_stream_rejection() -> Callable[[httpx.Request], httpx.Response]:
    return sse_response(
        (None, _chat_chunk("The code")),
        (None, {"error": CHAT_REJECTED_UNLOCATED["error"]}),
    )


def _responses_mid_stream_rejection() -> Callable[[httpx.Request], httpx.Response]:
    delta = {
        "type": "response.output_text.delta",
        "delta": "The code",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "logprobs": [],
    }
    error = {
        "type": "error",
        "code": "invalid_image",
        "message": "Invalid image in input[0].content[1].",
        "param": "input[0].content[1]",
        "sequence_number": 2,
    }
    return sse_response(("response.output_text.delta", delta), ("error", error))


STREAM_REJECTIONS = {
    "chat": _chat_mid_stream_rejection,
    "responses": _responses_mid_stream_rejection,
    "openrouter": _chat_mid_stream_rejection,
}


@pytest.mark.parametrize("adapter", sorted(FACTORIES))
async def test_image_rejection_mid_stream(
    adapter: str,
    caplog: pytest.LogCaptureFixture,
    sensitive_ids: set[int],
) -> None:
    caplog.set_level(logging.DEBUG, logger="dendrux")
    factory, label = FACTORIES[adapter]
    provider = factory()
    wire_openai(provider, STREAM_REJECTIONS[adapter]())
    seen: list[Any] = []
    with pytest.raises(ImageRejectedError) as info:
        async for event in provider.complete_stream(USER_IMAGE):
            seen.append(event)
    assert [e.type for e in seen] == [StreamEventType.TEXT_DELTA]
    err = info.value
    assert err.provider == label
    assert err.status_code is None
    assert_clean(err)
    for marker in LEAK_MARKERS:
        assert marker not in caplog.text
    assert len(sensitive_ids) == 3
    assert _reachable_request_holders(err, sensitive_ids) == set()
    assert _holders(err) == {}
    assert "stream" not in _frame_locals(err, "_stream_attempt")
    assert "chunk_iter" not in _frame_locals(err, "_stream_attempt")
    assert "iterator" not in _frame_locals(err, "_stream_attempt")


@pytest.mark.parametrize("during_open", [False, True])
async def test_anthropic_stream_tail_drops_its_stream_local(
    during_open: bool, sensitive_ids: set[int]
) -> None:
    """Image-bearing failures drop the SDK stream before raising."""
    claude = AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key", max_retries=0)
    events = [*_anthropic_head(), ("error", ANTHROPIC_REJECTED_UNLOCATED)]
    handler = (
        json_response(400, ANTHROPIC_REJECTED_UNLOCATED) if during_open else sse_response(*events)
    )
    wire_anthropic(claude, handler)
    with pytest.raises(ImageRejectedError) as info:
        async for _ in claude.complete_stream(USER_IMAGE):
            pass
    assert_clean(info.value)
    assert _reachable_request_holders(info.value, sensitive_ids) == set()
    assert "stream" not in _frame_locals(info.value, "_stream_attempt")


# ---------------------------------------------------------------------------
# Lifecycle (B7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", sorted(FACTORIES))
async def test_async_context_manager_returns_self_and_closes(adapter: str) -> None:
    factory, _ = FACTORIES[adapter]
    provider = factory()
    closed: list[bool] = []
    original = provider._client.close

    async def close() -> None:
        closed.append(True)
        await original()

    provider._client.close = close
    async with provider as entered:
        assert entered is provider
    assert closed == [True]


@pytest.mark.parametrize(
    ("adapter", "family"),
    [("chat", "chat"), ("responses", "responses"), ("openrouter", "openrouter")],
)
def test_image_capability_key_family(adapter: str, family: str) -> None:
    factory, _ = FACTORIES[adapter]
    provider = factory()
    key = provider.image_capability_key("m")
    assert key[0] == family and key[2] == "m"


async def test_image_support_is_unknown_off_the_official_endpoint() -> None:
    chat = OpenAIProvider(model="gpt-4o", api_key="test-key", base_url="http://localhost:8000/v1")
    assert await chat.image_input_support("gpt-4o") == "unknown"
    assert await _chat().image_input_support("gpt-4o") == "yes"
    assert await _responses().image_input_support("gpt-4o") == "yes"


# ---------------------------------------------------------------------------
# B11: the extra-field readers (pinned as they are; they differ on dicts)
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, attrs: dict[str, Any], extra: dict[str, Any] | None) -> None:
        self.__dict__.update(attrs)
        self.model_extra = extra


EXTRA_CASES = [
    ("attribute", _Model({"cost": 0.5}, {"cost": 9}), 0.5, 0.5),
    ("model_extra", _Model({}, {"cost": 0.25}), 0.25, 0.25),
    ("attribute_none_falls_back", _Model({"cost": None}, {"cost": 1}), 1, 1),
    ("missing", _Model({}, {}), None, None),
    ("extra_not_a_dict", _Model({}, None), None, None),
    ("none", None, None, None),
    ("dict", {"cost": 0.75}, None, 0.75),
    ("dict_missing", {"other": 1}, None, None),
    ("zero_is_kept", _Model({"cost": 0}, {"cost": 5}), 0, 0),
]


@pytest.mark.parametrize(
    ("case", "obj", "chat_value", "router_value"), EXTRA_CASES, ids=[c[0] for c in EXTRA_CASES]
)
def test_extra_field_readers(case: str, obj: Any, chat_value: Any, router_value: Any) -> None:
    # The chat reasoning readers and the OpenRouter cost reader (shared in H8).
    from dendrux.llm._openai_family import extra_field, usage_extra_field

    assert extra_field(obj, "cost") == chat_value
    assert usage_extra_field(obj, "cost") == router_value


# ---------------------------------------------------------------------------
# B12: provider labels on tool-call warnings
# ---------------------------------------------------------------------------


def _malformed_tool_stream() -> Callable[[httpx.Request], httpx.Response]:
    chunk = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_X",
                            "type": "function",
                            "function": {"name": "add", "arguments": "{not json"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    return sse_response((None, chunk), (None, "[DONE]"))


@pytest.mark.parametrize(
    ("adapter", "model", "logged"),
    [
        ("chat", "gpt-4o", "provider=openai model=gpt-4o"),
        (
            "openrouter",
            "deepseek/deepseek-chat",
            "provider=openrouter model=deepseek/deepseek-chat",
        ),
    ],
)
async def test_malformed_stream_tool_call_label(
    adapter: str, model: str, logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    factory, _ = FACTORIES[adapter]
    provider = factory()
    wire_openai(provider, _malformed_tool_stream())
    events = [e async for e in provider.complete_stream(TEXT)]
    done = events[-1].raw
    assert done.tool_calls[0].parse_error is not None
    assert logged in caplog.text


async def test_openrouter_per_call_model_is_the_logged_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed tool-call logs identify the adapter and the model of this call."""
    caplog.set_level(logging.WARNING, logger="dendrux")
    provider = OpenRouterProvider(model="x/a", api_key="test-key", max_retries=0)
    wire_openai(provider, _malformed_tool_stream())
    _ = [e async for e in provider.complete_stream(TEXT, model="q/b")]
    assert "provider=openrouter model=q/b" in caplog.text


async def test_openrouter_missing_tool_name_uses_per_call_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    provider = OpenRouterProvider(model="x/a", api_key="test-key", max_retries=0)
    chunk = _chat_chunk("")
    chunk["choices"][0]["delta"] = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_X",
                "type": "function",
                "function": {"arguments": "{}"},
            }
        ]
    }
    chunk["choices"][0]["finish_reason"] = "tool_calls"
    wire_openai(provider, sse_response((None, chunk), (None, "[DONE]")))
    _ = [event async for event in provider.complete_stream(TEXT, model="q/b")]
    assert "Tool call completed with no name — provider=openrouter model=q/b" in caplog.text


async def test_openrouter_batch_tool_warning_uses_per_call_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    provider = OpenRouterProvider(model="x/a", api_key="test-key", max_retries=0)
    wire_openai(
        provider,
        json_response(
            200,
            {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "q/b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_X",
                                    "type": "function",
                                    "function": {"name": "add", "arguments": "{not json"},
                                }
                            ],
                        },
                    }
                ],
            },
        ),
    )
    response = await provider.complete(TEXT, model="q/b")
    assert response.tool_calls and response.tool_calls[0].parse_error is not None
    assert "provider=openrouter model=q/b" in caplog.text


def test_request_holder_scan_follows_partial_iterator_and_attributes() -> None:
    from types import SimpleNamespace

    registry = WireImageRegistry()
    api_kwargs = {"model": "m"}
    request = WireRequest(api_kwargs, {}, registry)
    sensitive_ids = {id(request), id(registry), id(api_kwargs)}

    def iterator_alias(request: WireRequest) -> Any:
        yield None

    iterator = iterator_alias(request)
    next(iterator)
    aliases = [
        functools.partial(id, request),
        iterator,
        SimpleNamespace(registry=registry, api_kwargs=api_kwargs),
    ]

    def fail(aliases: list[Any]) -> None:
        raise RuntimeError("holder probe")

    try:
        with pytest.raises(RuntimeError) as info:
            fail(aliases)
        holders = _reachable_request_holders(info.value, sensitive_ids)
        assert "fail.aliases[0].args[0]" in holders
        assert "fail.aliases[1].frame[request]" in holders
        assert "fail.aliases[2].registry" in holders
        assert "fail.aliases[2].api_kwargs" in holders
    finally:
        iterator.close()


async def test_retry_attempts_open_pull_and_close_in_consumer_task() -> None:
    import asyncio

    from dendrux.llm._resilience import retry_unstarted_stream
    from dendrux.types import StreamEvent

    owner = asyncio.current_task()
    calls: list[tuple[str, int, bool]] = []
    attempts = 0

    def make_attempt() -> Any:
        nonlocal attempts
        attempts += 1
        number = attempts
        calls.append(("open", number, asyncio.current_task() is owner))

        async def stream() -> Any:
            try:
                calls.append(("pull", number, asyncio.current_task() is owner))
                if number == 1:
                    raise ProviderStreamIncompleteError(provider="probe")
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")
            finally:
                calls.append(("close", number, asyncio.current_task() is owner))

        return stream()

    events = [
        event
        async for event in retry_unstarted_stream(
            make_attempt,
            provider="probe",
            model="m",
        )
    ]
    assert [event.text for event in events] == ["ok"]
    assert calls == [
        ("open", 1, True),
        ("pull", 1, True),
        ("close", 1, True),
        ("open", 2, True),
        ("pull", 2, True),
        ("close", 2, True),
    ]


@pytest.fixture
def sensitive_ids(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Keep objects alive in the fixture frame (outside the error traceback), so
    # CPython cannot reuse their ids before the strict scan, including on 3.13.
    held: list[Any] = []
    identities: set[int] = set()
    original = WireRequest.__init__

    def capture(request: WireRequest, *args: Any, **kwargs: Any) -> None:
        original(request, *args, **kwargs)
        held.extend((request, request.registry, request.api_kwargs))
        identities.update(map(id, held))

    monkeypatch.setattr(WireRequest, "__init__", capture)
    yield identities
    held.clear()


class _WaitingBody(httpx.AsyncByteStream):
    def __init__(self, first: bytes) -> None:
        self.first = first

    async def __aiter__(self) -> Any:
        yield self.first
        await asyncio.Event().wait()


def _failure_body(adapter: str, failure: str) -> bytes:
    if adapter == "anthropic":
        events: list[tuple[str | None, Any]] = list(_anthropic_head())
        if failure in {"filter", "refusal"}:
            events.extend(
                [
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "refusal",
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": 2},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
    elif adapter == "responses":
        events = [
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": "Hello",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 1,
                    "logprobs": [],
                },
            )
        ]
        if failure == "refusal":
            events.append(
                (
                    "response.refusal.delta",
                    {
                        "type": "response.refusal.delta",
                        "delta": "No",
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": 2,
                    },
                )
            )
        if failure in {"filter", "refusal"}:
            events.append(
                (
                    "response.incomplete",
                    {
                        "type": "response.incomplete",
                        "sequence_number": 3,
                        "response": {
                            "id": "r1",
                            "object": "response",
                            "created_at": 0,
                            "model": "m",
                            "output": [],
                            "status": "incomplete",
                            "incomplete_details": {"reason": "content_filter"},
                        },
                    },
                )
            )
    else:
        events = [(None, _chat_chunk("Hello"))]
        if failure in {"filter", "refusal"}:
            chunk = _chat_chunk("")
            chunk["choices"][0]["finish_reason"] = (
                "content_filter" if failure == "filter" else "stop"
            )
            if failure == "refusal":
                chunk["choices"][0]["delta"] = {"refusal": "No"}
            events.extend([(None, chunk), (None, "[DONE]")])
    return b"".join(
        (
            (f"event: {kind}\n" if kind else "")
            + "data: "
            + (value if isinstance(value, str) else json.dumps(value))
            + "\n\n"
        ).encode()
        for kind, value in events
    )


@pytest.mark.parametrize("adapter", [*FACTORIES, "anthropic"])
@pytest.mark.parametrize(
    "failure",
    [
        "incomplete",
        "broken",
        "filter",
        "refusal",
        "idle",
        "total",
        "size",
        "open_idle",
        "open_total",
    ],
)
async def test_image_stream_errors_drop_all_request_holders(
    adapter: str,
    failure: str,
    sensitive_ids: set[int],
) -> None:
    if adapter == "anthropic" and failure == "filter":
        pytest.skip("Anthropic reports refusal, not content_filter")
    provider = (
        AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key", max_retries=0)
        if adapter == "anthropic"
        else FACTORIES[adapter][0]()
    )
    from dataclasses import replace

    changes: dict[str, Any] = {}
    if "idle" in failure:
        changes["stream_idle_timeout"] = 1.0
    if "total" in failure:
        changes["stream_total_timeout"] = 1.0
    if failure == "size":
        changes["max_stream_bytes"] = 1
    provider._deadlines = replace(provider._deadlines, **changes)
    body = _failure_body(adapter, failure)
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if failure.startswith("open_"):
            await asyncio.Event().wait()
        stream = (
            _WaitingBody(body)
            if failure in {"idle", "total"}
            else _BrokenBody(body)
            if failure == "broken"
            else httpx.ByteStream(body)
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    (wire_anthropic if adapter == "anthropic" else wire_openai)(provider, handler)
    expected = {
        "filter": ContentFilteredError,
        "refusal": ModelRefusalError,
        "size": ProviderResponseError,
        "idle": ProviderTimeoutError,
        "total": ProviderTimeoutError,
        "open_idle": ProviderTimeoutError,
        "open_total": ProviderTimeoutError,
    }.get(failure, ProviderStreamIncompleteError)
    with pytest.raises(expected) as info:
        async for _ in provider.complete_stream(USER_IMAGE):
            pass
    if failure.startswith("open_"):
        # The open deadline covers SDK setup and can expire before the mock
        # transport handler is entered on a busy runner.
        assert len(calls) <= 1
    else:
        assert len(calls) == 1
    assert len(sensitive_ids) == 3
    assert _reachable_request_holders(info.value, sensitive_ids) == set()
    assert_clean(info.value)
