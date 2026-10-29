"""Phase 2c review regressions, through installed SDKs without network access."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import random
import zlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest

from dendrux.errors import ProviderResponseError, ProviderTimeoutError
from dendrux.llm._transport import _Budget, deferred_client_type
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.types import Message, Role
from tests.unit.test_transport_bound import Body, call, document, install, provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

MESSAGES = [Message(role=Role.USER, content="hi")]
SECRET = "PRIVATE-TRANSPORT-SENTINEL"


@pytest.fixture(params=["anthropic", "openai"])
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.mark.parametrize(
    "error_type", [httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout]
)
async def test_p1_1_partial_body_transport_error_retries(
    family: str,
    error_type: type[httpx.TransportError],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(family)
    monkeypatch.setattr(p._client, "_sleep_for_retry", AsyncMock())
    requests = 0
    bodies: list[Body] = []

    class Broken(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield document(family, 32)[:10]
            raise error_type(SECRET)

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = Broken([]) if requests == 1 else Body([document(family, 32)])
        bodies.append(body)
        return httpx.Response(200, stream=body, headers={"content-type": "application/json"})

    install(p, respond)
    try:
        result = await p.complete(MESSAGES)
        assert result.text == "x" * 32
        assert requests == 2
        assert all(body.closed for body in bodies)
    finally:
        await p.close()


async def test_p1_1_exhausted_socket_timeout_is_not_a_call_deadline(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(family)
    monkeypatch.setattr(p._client, "_sleep_for_retry", AsyncMock())
    requests = 0

    class Broken(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"{"
            raise httpx.ReadTimeout(SECRET)

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, stream=Broken([]), headers={"content-type": "application/json"})

    install(p, respond)
    try:
        with pytest.raises(ProviderTimeoutError) as caught:
            await p.complete(MESSAGES)
        assert requests == 3
        assert caught.value.timeout_reason is None
        assert caught.value.retryable is False
        assert SECRET not in str(caught.value)
    finally:
        await p.close()


@pytest.mark.parametrize("separator", [b"\n\r", b"\n\r\n", b"\n\n"])
async def test_p1_2_sdk_buffer_cannot_grow_across_mixed_separators(
    family: str,
    separator: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = importlib.import_module(f"{family}._streaming").SSEDecoder
    original = decoder._aiter_chunks
    largest = 0

    async def observed(self: Any, source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        nonlocal largest
        async for chunk in original(self, source):
            largest = max(largest, len(chunk))
            yield chunk

    monkeypatch.setattr(decoder, "_aiter_chunks", observed)
    p = provider(family, event_limit=1024)
    body = Body([(b": " + b"x" * 1000 + separator)] * 2000)
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
    )
    try:
        result = await call(p, family, stream=True)
        if separator == b"\n\n":
            async for _ in result:
                pass
            assert largest == 1004
        else:
            with pytest.raises(ProviderResponseError) as caught:
                async for _ in result:
                    pass
            assert caught.value.provider_code == "stream_size_limit"
            assert caught.value.retryable is False
        assert largest <= 1024
        assert body.closed
    finally:
        await p.close()


@pytest.mark.parametrize("seed", [0, 1, 31, 7331])
async def test_p1_2_randomized_boundary_differential(family: str, seed: int) -> None:
    decoder = importlib.import_module(f"{family}._streaming").SSEDecoder
    rng = random.Random(seed)
    cases = [[b":xxxxxxxxxxxxxxxxxxxx\n\r"] * 3, [b"x\r", b"\r", b"\n"], [b"x\r\r\n"]]
    for _ in range(400):
        data = b"".join(rng.choice([b"x", b"abc", b"\r", b"\n", b"\r\n"]) for _ in range(80))
        cuts = sorted({0, len(data), *(rng.randrange(len(data) + 1) for _ in range(25))})
        cases.append([data[a:b] for a, b in zip(cuts, cuts[1:], strict=False)])

    for parts in cases:

        async def source(values: list[bytes] = parts) -> AsyncIterator[bytes]:
            for part in values:
                yield part

        expected = [chunk async for chunk in decoder()._aiter_chunks(source())]
        budget = _Budget(32, streaming=True, provider=family)

        async def guarded(counter: _Budget = budget) -> AsyncIterator[bytes]:
            async for part in source():
                counter.account(part)
                yield part

        if any(len(chunk) > 32 for chunk in expected):
            with pytest.raises(ProviderResponseError):
                async for _ in decoder()._aiter_chunks(guarded()):
                    pass
        else:
            actual = [chunk async for chunk in decoder()._aiter_chunks(guarded())]
            assert actual == expected
            assert all(len(chunk) <= 32 for chunk in actual)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("encoding", ["br", "zstd", "gzip, deflate", "identity, gzip", "unknown"])
async def test_p1_3_unsupported_encoding_is_refused_before_any_read(
    family: str,
    stream: bool,
    encoding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(family)
    monkeypatch.setattr(p._client, "_sleep_for_retry", AsyncMock())
    requests = 0
    reads = 0

    class Unread(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal reads
            reads += 1
            yield document(family, 32)

    body = Unread([])

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, stream=body, headers={"content-encoding": encoding})

    install(p, respond)
    try:
        with pytest.raises(ProviderResponseError) as caught:
            result = await call(p, family, stream=stream)
            if stream:
                async for _ in result:
                    pass
        assert caught.value.provider_code == "response_encoding_unsupported"
        assert caught.value.retryable is False
        assert requests == 1
        assert reads == 0
        assert body.closed
    finally:
        await p.close()


async def test_p1_3_encoding_offer_is_request_local(family: str) -> None:
    p = provider(family)
    client = p._client._client
    client.headers["accept-encoding"] = "br, zstd, identity"
    original = dict(client.headers)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "gzip, deflate"
        return httpx.Response(200, stream=Body([document(family, 32)]))

    install(p, respond)
    try:
        await call(p, family)
        assert dict(client.headers) == original
    finally:
        await p.close()


async def test_p1_1_actual_deadline_still_reports_call_deadline(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(family)
    body = Body([b"{"], stall=True)
    install(p, lambda _: httpx.Response(200, stream=body))
    loop = asyncio.get_running_loop()
    now = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: now[0])
    task = asyncio.create_task(p.complete(MESSAGES))
    try:
        await body.reading.wait()
        now[0] += 601
        with pytest.raises(ProviderTimeoutError) as caught:
            await task
        assert caught.value.timeout_reason == "call_deadline"
        assert body.closed
    finally:
        await p.close()


async def test_p2_super_send_override_is_preserved() -> None:
    class BaseClient(httpx.AsyncClient):
        async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            response = await super().send(request, **kwargs)
            response.extensions["base_send_called"] = True
            return response

    client_type = deferred_client_type(BaseClient)
    async with client_type(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    ) as client:
        result = await client.get("https://provider.test")
        assert result.extensions.get("base_send_called") is True


def test_p2_client_subclass_is_cached_per_base() -> None:
    assert deferred_client_type(httpx.AsyncClient) is deferred_client_type(httpx.AsyncClient)


async def test_p2_limit_has_one_structured_run_scoped_log(
    family: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = provider(family, limit=1024)
    install(p, lambda _: httpx.Response(200, stream=Body([document(family, 2048)])))
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ProviderResponseError):
            await p.complete(MESSAGES, run_id="run-limit")
        records = [r for r in caplog.records if getattr(r, "code", None) == "response_size_limit"]
        assert len(records) == 1
        assert records[0].run_id == "run-limit"
    finally:
        await p.close()


@pytest.mark.parametrize("chunk_size", [7, 65539])
async def test_p2_explicit_chunk_size_is_exact(family: str, chunk_size: int) -> None:
    payload = document(family, 150000)
    p = provider(family)
    body = Body([payload[:11], payload[11:123], payload[123:]])
    install(p, lambda _: httpx.Response(200, stream=body))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"),
            stream=True,
        )
        chunks = [part async for part in response.aiter_bytes(chunk_size=chunk_size)]
        assert b"".join(chunks) == payload
        assert all(len(part) == chunk_size for part in chunks[:-1])
        assert len(chunks[-1]) <= chunk_size
    finally:
        await p.close()


@pytest.mark.parametrize("status", [200, 400, 500])
@pytest.mark.parametrize("buffered", [False, True])
async def test_p1_1_limit_refusal_survives_cached_and_error_bodies(
    family: str,
    status: int,
    buffered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(family, limit=1024)
    monkeypatch.setattr(p._client, "_sleep_for_retry", AsyncMock())
    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        payload = document(family, 2048)
        if buffered:
            return httpx.Response(status, content=payload)
        return httpx.Response(status, stream=Body([payload]))

    install(p, respond)
    try:
        with pytest.raises(ProviderResponseError) as caught:
            await p.complete(MESSAGES)
        assert caught.value.provider_code == "response_size_limit"
        assert caught.value.retryable is False
        assert requests == 1
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["identity", "deflate"])
async def test_p1_3_allowed_encodings_parse_normally(family: str, encoding: str) -> None:
    p = provider(family)
    payload = document(family, 32)
    if encoding == "deflate":
        payload = zlib.compress(payload)
    body = Body([payload])
    install(p, lambda _: httpx.Response(200, stream=body, headers={"content-encoding": encoding}))
    try:
        result = await call(p, family)
        assert result.id == "m"
        assert body.closed
    finally:
        await p.close()


@pytest.mark.parametrize("limit", [1024, 8192])
async def test_p2_responses_completed_envelope_counts_in_wire_limit(limit: int) -> None:
    p = OpenAIResponsesProvider(model="test-model", api_key="test-key", max_stream_bytes=limit)
    response = {
        "id": "r",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "test-model",
        "instructions": "x" * 2048,
        "tools": [{"type": "function", "name": "echo", "parameters": {"type": "object"}}],
        "output": [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
    }
    payload = (
        b"event: response.completed\ndata: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": response,
            }
        ).encode()
        + b"\n\n"
    )
    body = Body([payload])
    install(
        p, lambda _: httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
    )
    try:
        upstream = await p._client.responses.create(model="test-model", input="hi", stream=True)
        if limit == 1024:
            with pytest.raises(ProviderResponseError) as caught:
                async for _ in upstream:
                    pass
            assert caught.value.provider_code == "stream_size_limit"
        else:
            events = [event async for event in upstream]
            assert events[-1].response.output[0].content[0].text == "ok"
        assert body.closed
    finally:
        await p.close()
