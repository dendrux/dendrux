"""Terminal response policies across SDK status, cache and iterator surfaces."""

from __future__ import annotations

import asyncio
import gzip
import importlib
import inspect
import json
import logging
import traceback
import zlib
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from dendrux.errors import ProviderResponseError
from dendrux.llm._retry_telemetry import telemetry_context
from dendrux.loops._helpers import governance_sinks
from dendrux.types import Message, Role
from tests.unit.test_transport_bound import Body, call, document, event, install, provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SENTINEL = "TRANSPORT-BODY-MUST-STAY-PRIVATE"
MESSAGES = [Message(role=Role.USER, content="hi")]
RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


class ObservedBody(Body):
    def __init__(self, parts: list[bytes], *, fail_close: bool = False) -> None:
        super().__init__(parts)
        self.pulls = 0
        self.closes = 0
        self.fail_close = fail_close

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            self.pulls += 1
            yield part

    async def aclose(self) -> None:
        self.closed = True
        self.closes += 1
        if self.fail_close:
            raise RuntimeError(SENTINEL)


class Events:
    def __init__(self) -> None:
        self.values: list[Any] = []

    async def on_governance_event(self, *args: Any, **kwargs: Any) -> None:
        self.values.append((args, kwargs))


def assert_private(error: BaseException) -> None:
    assert SENTINEL not in "".join(traceback.format_exception(error))
    assert SENTINEL not in str(error) + repr(error)
    assert SENTINEL not in str(getattr(error, "transport_detail", None))
    # A terminal policy error must not carry the SDK's body-bearing error object.
    for linked in (error.__cause__, error.__context__):
        assert SENTINEL not in str(linked)
        assert not hasattr(linked, "body")


@pytest.mark.parametrize("family", ["anthropic", "openai"])
@pytest.mark.parametrize("status", [200, 400, 429, 500])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("encoding", [None, "unknown"])
async def test_policy_survives_status_and_body_modes(
    family: str,
    status: int,
    buffered: bool,
    stream: bool,
    encoding: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = provider(family, limit=1024, event_limit=1024)
    monkeypatch.setattr(p._client, "_calculate_retry_timeout", lambda *a, **k: 0)
    payload = json.dumps({"error": {"message": SENTINEL + "x" * 2048}}).encode()
    if status == 200 and stream:
        payload = b"event: error\ndata: " + payload + b"\n\n"
    bodies: list[ObservedBody] = []
    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        headers = {"content-type": "text/event-stream" if stream else "application/json"}
        if encoding:
            headers["content-encoding"] = encoding
        if buffered:
            return httpx.Response(status, content=payload, headers=headers)
        body = ObservedBody([payload, b"never pull this"])
        bodies.append(body)
        return httpx.Response(status, stream=body, headers=headers)

    install(p, respond)
    events = Events()
    emitted = []
    try:
        with (
            telemetry_context(
                run_id=RUN_ID,
                iteration=0,
                sinks=governance_sinks(events, events, RUN_ID),  # type: ignore[arg-type]
            ),
            caplog.at_level(logging.DEBUG),
            pytest.raises(ProviderResponseError) as caught,
        ):
            if stream:
                async for item in p.complete_stream(MESSAGES, run_id=RUN_ID):
                    emitted.append(item)
            else:
                await p.complete(MESSAGES, run_id=RUN_ID)
        code = (
            "response_encoding_unsupported"
            if encoding
            else ("stream_size_limit" if stream and status == 200 else "response_size_limit")
        )
        assert caught.value.provider_code == code
        assert caught.value.retryable is False
        assert_private(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert SENTINEL not in caplog.text + str(events.values) + str(emitted)
        records = [r for r in caplog.records if getattr(r, "code", None) == code]
        assert len(records) == 1
        assert records[0].run_id == RUN_ID
        unread_retry = stream and not buffered and not encoding and status in (429, 500)
        assert requests == (3 if unread_retry else 1)
        assert all(b.closed and b.closes == 1 for b in bodies)
        assert sum(b.pulls for b in bodies) == (0 if encoding or buffered else 1)
    finally:
        await p.close()


SURFACES = [
    "content",
    "text",
    "json",
    "read",
    "aread",
    "iter_bytes",
    "iter_text",
    "iter_lines",
    "iter_raw",
    "aiter_bytes",
    "aiter_text",
    "aiter_lines",
    "aiter_raw",
    "raise_for_status",
    "status_code",
    "is_success",
]


async def access(response: httpx.Response, name: str) -> Any:
    value = getattr(response, name)
    if callable(value):
        value = value()
    if inspect.isawaitable(value):
        return await value
    if hasattr(value, "__aiter__"):
        return [part async for part in value]
    if name.startswith("iter_"):
        return list(value)
    return value


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("buffered", [False, True])
async def test_refusal_persists_on_every_response_surface(surface: str, buffered: bool) -> None:
    p = provider("openai", limit=1024)
    body = ObservedBody([SENTINEL.encode() + b"x" * 2048, b"unread"])
    install(
        p,
        lambda _: (
            httpx.Response(400, content=body.parts[0])
            if buffered
            else httpx.Response(400, stream=body)
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        for name in ("aread", surface):
            with pytest.raises(ProviderResponseError) as caught:
                await access(response, name)
            assert caught.value.provider_code == "response_size_limit"
            assert_private(caught.value)
        assert body.pulls == (0 if buffered else 1)
    finally:
        await p.close()


@pytest.mark.parametrize("family", ["anthropic", "openai"])
@pytest.mark.parametrize("buffered", [False, True])
async def test_raw_sdk_success_cannot_render_refused_body(family: str, buffered: bool) -> None:
    p = provider(family, limit=1024)
    payload = document(family, 2048)
    install(
        p,
        lambda _: (
            httpx.Response(200, content=payload)
            if buffered
            else httpx.Response(200, stream=ObservedBody([payload]))
        ),
    )
    resource = p._client.messages if family == "anthropic" else p._client.chat.completions
    options: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if family == "anthropic":
        options["max_tokens"] = 32
    try:
        with pytest.raises(ProviderResponseError) as caught:
            raw = await resource.with_raw_response.create(**options)
            raw.parse()
        assert caught.value.provider_code == "response_size_limit"
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["identity", "gzip", "deflate"])
@pytest.mark.parametrize("raw", [False, True])
async def test_raw_and_decoded_ingress_stop_before_next_pull(encoding: str, raw: bool) -> None:
    payload = b"x" * 100000
    encoded = (
        gzip.compress(payload)
        if encoding == "gzip"
        else (zlib.compress(payload) if encoding == "deflate" else payload)
    )
    body = ObservedBody([encoded, b"unread"], fail_close=True)
    p = provider("openai", limit=1024)
    install(p, lambda _: httpx.Response(400, stream=body, headers={"content-encoding": encoding}))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        with pytest.raises(ProviderResponseError) as caught:
            await access(response, "aiter_raw" if raw else "aiter_bytes")
        assert caught.value.provider_code == "response_size_limit"
        assert_private(caught.value)
        assert body.pulls == body.closes == 1
        with pytest.raises(ProviderResponseError):
            await response.aread()
        assert body.pulls == body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["identity", "gzip", "deflate", "raw-deflate"])
@pytest.mark.parametrize("raw", [False, True])
async def test_accepted_encoded_bodies_keep_raw_and_decoded_meanings(
    encoding: str, raw: bool
) -> None:
    payload = b"x" * 4096
    encoded = payload
    if encoding == "gzip":
        encoded = gzip.compress(payload)
    elif encoding != "identity":
        encoded = zlib.compress(payload, wbits=-15 if encoding == "raw-deflate" else 15)
    body = ObservedBody([encoded[:1], encoded[1:7], encoded[7:]])
    p = provider("openai", limit=len(payload))
    install(
        p,
        lambda _: httpx.Response(
            400, stream=body, headers={"content-encoding": encoding.replace("raw-", "")}
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        iterator = response.aiter_raw if raw else response.aiter_bytes
        result = b"".join([part async for part in iterator(chunk_size=7)])
        assert result == (encoded if raw else payload)
        assert body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("family", ["anthropic", "openai"])
async def test_two_instances_have_independent_terminal_state(family: str) -> None:
    small = provider(family, limit=1024)
    large = provider(family, limit=8192)
    arrived = 0
    both = asyncio.Event()

    class Rendezvous(ObservedBody):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both.set()
            await both.wait()
            async for part in super().__aiter__():
                yield part

    bodies = [Rendezvous([document(family, 2048)]) for _ in range(2)]
    install(small, lambda _: httpx.Response(200, stream=bodies[0]))
    install(large, lambda _: httpx.Response(200, stream=bodies[1]))
    try:
        results = await asyncio.gather(
            small.complete(MESSAGES), large.complete(MESSAGES), return_exceptions=True
        )
        assert isinstance(results[0], ProviderResponseError)
        assert results[1].text == "x" * 2048
        assert all(body.closed for body in bodies)
    finally:
        await small.close()
        await large.close()


@pytest.mark.parametrize("family", ["anthropic", "openai"])
async def test_cancel_error_body_read_closes_once(family: str) -> None:
    p = provider(family)
    body = Body([b"{"], stall=True)
    install(p, lambda _: httpx.Response(400, stream=body))
    task = asyncio.create_task(p.complete_stream(MESSAGES).__anext__())
    try:
        await body.reading.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
    finally:
        await p.close()


async def test_cancel_during_close_preserves_cancellation() -> None:
    p = provider("openai", limit=1024)
    closing = asyncio.Event()

    class SlowClose(ObservedBody):
        async def aclose(self) -> None:
            closing.set()
            await asyncio.Event().wait()

    body = SlowClose([b"x" * 2048])
    response = httpx.Response(200, stream=body)
    install(p, lambda _: response)
    task = asyncio.create_task(p.complete(MESSAGES))
    try:
        await closing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProviderResponseError) as caught:
            await response.aread()
        assert_private(caught.value)
    finally:
        await p.close()


@pytest.mark.parametrize("surface", ["read", "aread", "json", "iter_raw", "aiter_raw"])
async def test_saved_accessors_cannot_bypass_a_later_refusal(surface: str) -> None:
    p = provider("openai", limit=1024)
    body = ObservedBody([b"x" * 2048])
    install(p, lambda _: httpx.Response(400, stream=body))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        saved = getattr(response, surface)
        with pytest.raises(ProviderResponseError):
            await response.aread()
        with pytest.raises(ProviderResponseError):
            value = saved()
            if inspect.isawaitable(value):
                await value
            elif hasattr(value, "__aiter__"):
                await anext(value)
            elif surface == "iter_raw":
                next(value)
    finally:
        await p.close()


@pytest.mark.parametrize("raw", [False, True])
async def test_decompression_uses_a_bounded_output_window(
    raw: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = zlib.decompressobj
    windows: list[int] = []

    class Decoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            windows.append(max_length)
            return self.inner.decompress(data, max_length)

    monkeypatch.setattr(zlib, "decompressobj", Decoder)
    p = provider("openai", limit=1024)
    body = ObservedBody([gzip.compress(b"x" * 100000)])
    install(p, lambda _: httpx.Response(400, stream=body, headers={"content-encoding": "gzip"}))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        with pytest.raises(ProviderResponseError):
            await access(response, "aiter_raw" if raw else "aiter_bytes")
        assert windows and all(0 < window <= 1025 for window in windows)
    finally:
        await p.close()


@pytest.mark.parametrize("family", ["anthropic", "openai"])
async def test_copied_sdk_client_keeps_terminal_refusals(family: str) -> None:
    p = provider(family, limit=1024)
    p._client = p._client.copy()
    body = ObservedBody([document(family, 2048)])
    install(p, lambda _: httpx.Response(500, stream=body))
    try:
        with pytest.raises(ProviderResponseError) as caught:
            await p.complete(MESSAGES)
        assert caught.value.provider_code == "response_size_limit"
        assert body.pulls == body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("raw", [False, True])
async def test_early_iterator_close_is_idempotent(raw: bool) -> None:
    p = provider("openai")
    body = ObservedBody([b"x" * 2048, b"unread"])
    install(p, lambda _: httpx.Response(200, stream=body))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        iterator = response.aiter_raw() if raw else response.aiter_bytes()
        await anext(iterator)
        await iterator.aclose()
        await response.aclose()
        assert body.pulls == body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("family", ["anthropic", "openai"])
async def test_cancel_unread_status_retry_preserves_cancellation(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = provider(family)
    retrying = asyncio.Event()

    def backoff(*args: Any, **kwargs: Any) -> float:
        retrying.set()
        return 600

    monkeypatch.setattr(p._client, "_calculate_retry_timeout", backoff)
    body = ObservedBody([b"unread"])
    install(p, lambda _: httpx.Response(500, stream=body))
    task = asyncio.create_task(p.complete_stream(MESSAGES).__anext__())
    try:
        await retrying.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closes == 1
        assert body.pulls == 0
    finally:
        await p.close()


async def test_cached_response_subclass_and_accepted_accessors_are_preserved() -> None:
    class Response(httpx.Response):
        def marker(self) -> str:
            return "original"

    p = provider("openai")
    response = Response(200, json={"ok": True})
    install(p, lambda _: response)
    try:
        actual = await p._client._client.get("https://provider.test")
        assert actual is response and isinstance(actual, Response)
        assert actual.marker() == "original"
        assert actual.json() == {"ok": True}
        assert actual.read() == await actual.aread()
        assert b"".join(actual.iter_bytes()) == actual.content
    finally:
        await p.close()


async def test_copied_context_cannot_retain_sdk_control_flow() -> None:
    p = provider("openai", limit=1024)
    contexts = []
    response = httpx.Response(200, content=b"x" * 2048)

    def respond(_: httpx.Request) -> httpx.Response:
        contexts.append(copy_context())
        return response

    install(p, respond)
    try:
        with pytest.raises(ProviderResponseError):
            await p.complete(MESSAGES)
        with pytest.raises(ProviderResponseError) as caught:
            contexts[0].run(lambda: response.content)
        assert_private(caught.value)
    finally:
        await p.close()


async def test_raw_deflate_can_start_with_zlib_header_bytes() -> None:
    payload = b"x" * 156
    # A legal raw stored block whose padding and length also look like a zlib header.
    encoded = b"\x78\x9c\x00\x63\xff" + payload + b"\x01\x00\x00\xff\xff"
    assert zlib.decompress(encoded, -15) == payload
    p = provider("openai", limit=len(payload))
    install(
        p,
        lambda _: httpx.Response(
            400, stream=ObservedBody([encoded]), headers={"content-encoding": "deflate"}
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        assert await response.aread() == payload
    finally:
        await p.close()


async def test_cached_sse_keeps_default_parser_chunks_bounded() -> None:
    payload = (b"data: " + b"x" * 512 + b"\n\n") * 1000
    p = provider("openai", event_limit=1024)
    install(p, lambda _: httpx.Response(200, content=payload))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        chunks = [chunk async for chunk in response.aiter_bytes()]
        assert b"".join(chunks) == payload
        assert max(map(len, chunks)) <= 65536
    finally:
        await p.close()


@pytest.mark.parametrize("family", ["anthropic", "openai"])
@pytest.mark.parametrize("encoding", ["identity", "gzip", "deflate"])
async def test_large_accepted_event_keeps_sdk_chunk_count_bounded(
    family: str, encoding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dendrux.llm._transport import _Budget

    window = 65536
    payload = event(family, 4 * 1024 * 1024)
    encoded = payload
    if encoding == "gzip":
        encoded = gzip.compress(payload)
    elif encoding == "deflate":
        encoded = zlib.compress(payload)
    body = ObservedBody(
        [encoded[start : start + window] for start in range(0, len(encoded), window)]
    )
    accounted: list[bytes] = []
    delivered: list[bytes] = []
    frames: list[int] = []
    account = _Budget.account
    decoder = importlib.import_module(f"{family}._streaming").SSEDecoder
    parse = decoder._aiter_chunks

    def observe_account(self: _Budget, chunk: bytes) -> None:
        account(self, chunk)
        accounted.append(chunk)

    async def observe_source(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for chunk in source:
            delivered.append(chunk)
            yield chunk

    async def observe_parser(self: Any, source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for frame in parse(self, observe_source(source)):
            frames.append(len(frame))
            yield frame

    monkeypatch.setattr(_Budget, "account", observe_account)
    monkeypatch.setattr(decoder, "_aiter_chunks", observe_parser)
    p = provider(family, event_limit=len(payload))
    install(
        p,
        lambda _: httpx.Response(
            200,
            stream=body,
            headers={"content-type": "text/event-stream", "content-encoding": encoding},
        ),
    )
    try:
        stream = await call(p, family, stream=True)
        async for _ in stream:
            pass
        assert frames == [len(payload)]
        assert accounted == delivered
        assert b"".join(delivered) == payload
        assert max(map(len, delivered)) <= window
        assert len(delivered) <= (len(payload) + window - 1) // window
        assert body.closes == 1
    finally:
        await p.close()
