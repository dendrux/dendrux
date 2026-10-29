"""C1: real SDK parsers and retries over a deterministic, network-free wire."""

from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import logging
import tracemalloc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from dendrux.errors import ProviderResponseError
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.openai import OpenAIProvider
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.llm.openrouter import OpenRouterProvider

SENTINEL = "VENDOR-PRIVATE-SENTINEL"
MIB = 1024 * 1024


class Body(httpx.AsyncByteStream):
    def __init__(self, parts: list[bytes], *, stall: bool = False) -> None:
        self.parts = parts
        self.stall = stall
        self.reading = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            yield part
        if self.stall:
            self.reading.set()
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(params=["anthropic", "openai"])
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def provider(family: str, limit: int = 16 * MIB, event_limit: int = 16 * MIB) -> Any:
    cls = AnthropicProvider if family == "anthropic" else OpenAIProvider
    options: dict[str, Any] = {"max_stream_bytes": event_limit}
    # Allows the pre-implementation red run to exercise the missing wire guard.
    if "max_response_bytes" in inspect.signature(cls).parameters:
        options["max_response_bytes"] = limit
    return cls(model="test-model", api_key="test-key", max_retries=2, **options)


def install(p: Any, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    p._client._client._transport = httpx.MockTransport(handler)


async def call(p: Any, family: str, *, stream: bool = False) -> Any:
    options = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    if family == "anthropic":
        return await p._client.messages.create(**options, max_tokens=32, stream=stream)
    return await p._client.chat.completions.create(**options, stream=stream)


def document(family: str, size: int) -> bytes:
    if family == "anthropic":
        obj = {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "text", "text": "x" * size}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    else:
        obj = {
            "id": "m",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "x" * size},
                    "finish_reason": "stop",
                }
            ],
        }
    return json.dumps(obj).encode()


def event(family: str, size: int = 0, newline: bytes = b"\n") -> bytes:
    if family == "anthropic":
        obj = {"type": "ping", "padding": "x" * size}
        return b"event: ping" + newline + b"data: " + json.dumps(obj).encode() + newline * 2
    obj = {"id": "m", "object": "chat.completion.chunk", "choices": [], "padding": "x" * size}
    return b"data: " + json.dumps(obj).encode() + newline * 2


@pytest.mark.parametrize("compressed", [False, True])
async def test_total_decoded_limit_closes_without_sdk_retry(
    family: str,
    compressed: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = provider(family, limit=4096)
    obj = json.loads(document(family, 8192))
    obj["private"] = SENTINEL
    payload = json.dumps(obj).encode()
    if compressed:
        payload = gzip.compress(payload)
    body = Body([payload[:17], payload[17:]])
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            stream=body,
            headers={
                "content-type": "application/json",
                **({"content-encoding": "gzip"} if compressed else {}),
            },
        )

    install(p, respond)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ProviderResponseError) as caught:
            await call(p, family)
        assert caught.value.provider_code == "response_size_limit"
        assert caught.value.retryable is False
        assert requests == 1
        assert body.closed
        assert SENTINEL not in str(caught.value) + repr(caught.value) + caplog.text
        assert any(getattr(r, "code", None) == "response_size_limit" for r in caplog.records)
    finally:
        await p.close()


async def test_realistic_two_mib_nonstream_passes(family: str) -> None:
    p = provider(family)
    body = Body([document(family, 2 * MIB)])
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "application/json"})
    )
    try:
        result = await call(p, family)
        assert result.id == "m"
        assert body.closed
    finally:
        await p.close()


async def test_nonstream_cannot_evade_total_limit_with_sse_content_type(family: str) -> None:
    p = provider(family, limit=4096)
    body = Body([document(family, 8192)])
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
    )
    try:
        with pytest.raises(ProviderResponseError) as caught:
            await call(p, family)
        assert caught.value.provider_code == "response_size_limit"
        assert body.closed
    finally:
        await p.close()


async def test_stream_cannot_evade_event_limit_with_json_content_type(family: str) -> None:
    p = provider(family, event_limit=1024)
    body = Body([event(family, 2048)])
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "application/json"})
    )
    try:
        with pytest.raises(ProviderResponseError) as caught:
            result = await call(p, family, stream=True)
            async for _ in result:
                pass
        assert caught.value.provider_code == "stream_size_limit"
        assert body.closed
    finally:
        await p.close()


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
async def test_oversized_event_after_normal_events(family: str, newline: bytes) -> None:
    p = provider(family, event_limit=1024)
    normal = event(family, newline=newline)
    large = event(family, 2048, newline)
    body = Body([normal, normal, large[:1000], large[1000:]])
    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    install(p, respond)
    try:
        with pytest.raises(ProviderResponseError) as caught:
            upstream = await call(p, family, stream=True)
            async for _ in upstream:
                pass
        assert caught.value.provider_code == "stream_size_limit"
        assert caught.value.retryable is False
        assert requests == 1
        assert body.closed
    finally:
        await p.close()


async def test_long_stream_over_twenty_mib_is_not_a_total_wire_limit(family: str) -> None:
    p = provider(family, event_limit=4096)
    frame = event(family, 2048)
    body = Body([frame] * (21 * MIB // len(frame) + 1))
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
    )
    try:
        upstream = await call(p, family, stream=True)
        async for _ in upstream:
            pass
        assert body.closed
    finally:
        await p.close()


async def test_concurrent_attempts_do_not_share_a_counter(family: str) -> None:
    p = provider(family, limit=4096)
    arrived = 0
    both = asyncio.Event()

    class ConcurrentBody(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both.set()
            await both.wait()
            async for part in super().__aiter__():
                yield part

    bodies = [ConcurrentBody([document(family, 3000)]) for _ in range(2)]
    install(
        p,
        lambda _: httpx.Response(
            200, stream=bodies.pop(), headers={"content-type": "application/json"}
        ),
    )
    try:
        results = await asyncio.gather(call(p, family), call(p, family))
        assert len(results) == 2
    finally:
        await p.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_cancellation_closes_response(family: str, stream: bool) -> None:
    p = provider(family)
    body = Body([event(family)] if stream else [b"{"], stall=True)
    kind = "text/event-stream" if stream else "application/json"
    install(p, lambda _: httpx.Response(200, stream=body, headers={"content-type": kind}))

    async def consume() -> None:
        result = await call(p, family, stream=stream)
        if stream:
            async for _ in result:
                pass

    task = asyncio.create_task(consume())
    try:
        await body.reading.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
    finally:
        await p.close()


async def test_gzip_bomb_peak_memory_is_near_limit(family: str, record_property: Any) -> None:
    p = provider(family)
    install(p, lambda _: httpx.Response(200, json=json.loads(document(family, 16))))
    await call(p, family)  # Exclude lazy SDK/Pydantic imports and caches from body measurement.
    compressed = gzip.compress(b'"' + b" " * (64 * MIB) + b'"', compresslevel=9)
    body = Body([compressed])
    install(
        p,
        lambda _: httpx.Response(
            200,
            stream=body,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
        ),
    )
    tracemalloc.start()
    try:
        with pytest.raises(ProviderResponseError):
            await call(p, family)
        _, peak = tracemalloc.get_traced_memory()
        record_property("peak_bytes", peak)
        print(f"{family}: rejected 64 MiB gzip at 16 MiB; peak_bytes={peak}")
        assert peak < 24 * MIB
        assert body.closed
    finally:
        tracemalloc.stop()
        await p.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_exact_limit_and_split_delimiters(family: str, stream: bool) -> None:
    payload = event(family, 2048, b"\r\n") if stream else document(family, 2048)
    p = provider(family, limit=len(payload), event_limit=len(payload))
    # CR/LF and UTF-8 are byte boundaries, independent of transport chunks.
    parts = [payload[i : i + 1] for i in range(len(payload))]
    body = Body(parts * (3 if stream else 1))
    kind = "text/event-stream" if stream else "application/json"
    install(p, lambda _: httpx.Response(200, stream=body, headers={"content-type": kind}))
    try:
        result = await call(p, family, stream=stream)
        if stream:
            async for _ in result:
                pass
        assert body.closed
    finally:
        await p.close()


@pytest.mark.parametrize("tail", [b"", b"\r", b"\n", b"\r\n"])
async def test_unterminated_or_final_newline_cannot_exceed_event_limit(
    family: str,
    tail: bytes,
) -> None:
    payload = b"data: " + b"x" * 128 + tail
    p = provider(family, event_limit=len(payload) - 1)
    body = Body([payload])
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
    )
    try:
        result = await call(p, family, stream=True)
        # Inspect the SDK response iterator: rejection must precede delivery to
        # its SSE parser, including a final CR at the limit.
        with pytest.raises(ProviderResponseError):
            await result.response.aiter_bytes().__anext__()
        assert body.closed
    finally:
        await p.close()


async def test_compressed_stream_multiple_events_in_one_chunk(family: str) -> None:
    frame = event(family, 2048)
    p = provider(family, event_limit=len(frame))
    body = Body([gzip.compress(frame * 100)])
    install(
        p,
        lambda _: httpx.Response(
            200,
            stream=body,
            headers={
                "content-type": "text/event-stream",
                "content-encoding": "gzip",
            },
        ),
    )
    try:
        result = await call(p, family, stream=True)
        async for _ in result:
            pass
        assert body.closed
    finally:
        await p.close()


async def test_close_failure_preserves_limit_error_without_vendor_text(family: str) -> None:
    class FailingClose(Body):
        async def aclose(self) -> None:
            self.closed = True
            raise RuntimeError(SENTINEL)

    p = provider(family, limit=1024)
    body = FailingClose([document(family, 2048)])
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "application/json"})
    )
    try:
        with pytest.raises(ProviderResponseError) as caught:
            await call(p, family)
        assert SENTINEL not in str(caught.value)
        assert body.closed
    finally:
        await p.close()


@pytest.mark.parametrize(
    "cls", [AnthropicProvider, OpenAIProvider, OpenAIResponsesProvider, OpenRouterProvider]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_response_limit_validation(cls: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="max_response_bytes"):
        cls(model="test-model", api_key="test-key", max_response_bytes=value)
