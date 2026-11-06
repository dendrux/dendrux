"""Compressed response work stays bounded before and after the end marker."""

from __future__ import annotations

import gzip
import random
import tracemalloc
import zlib
from typing import Any

import httpx
import pytest

from dendrux.errors import ProviderResponseError
from tests.unit.test_transport_bound import install, provider
from tests.unit.test_transport_refusals import ObservedBody, access


def compressed(payload: bytes, encoding: str) -> bytes:
    if encoding == "gzip":
        return gzip.compress(payload)
    return zlib.compress(payload, wbits=-15 if encoding == "raw-deflate" else 15)


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "raw-deflate"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("joined", [False, True])
async def test_compressed_trailer_stops_without_more_pulls(
    encoding: str, stream: bool, raw: bool, joined: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = zlib.decompressobj
    calls = 0
    after_eof = 0
    retained = 0

    class Decoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            nonlocal calls, after_eof, retained
            calls += 1
            after_eof += self.inner.eof
            result = self.inner.decompress(data, max_length)
            retained = max(retained, len(self.inner.unused_data))
            return result

    monkeypatch.setattr(zlib, "decompressobj", Decoder)
    head = compressed(b"data: ok\n\n" if stream else b"ok", encoding)
    # Separators in trailing wire bytes cannot reset a decoded event's budget.
    trailer = b"\n\n" * 4096
    parts = [head + trailer] if joined else [head, trailer]
    body = ObservedBody([*parts, *([trailer] * 128)])
    p = provider("openai", limit=1024, event_limit=1024)
    install(
        p,
        lambda _: httpx.Response(
            200 if stream else 400,
            stream=body,
            headers={"content-encoding": encoding.replace("raw-", "")},
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        tracemalloc.start()
        try:
            with pytest.raises(ProviderResponseError) as caught:
                await access(response, "aiter_raw" if raw else "aiter_bytes")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert caught.value.provider_code == (
            "stream_size_limit" if stream else "response_size_limit"
        )
        assert body.pulls <= len(parts)
        assert body.closes == 1
        assert calls <= 3 and after_eof == 0
        assert retained <= 1025
        assert peak < 2 * 1024 * 1024
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
@pytest.mark.parametrize("stream", [False, True])
async def test_empty_compressed_blocks_stop_before_more_wire(encoding: str, stream: bool) -> None:
    payload = b"data: ok\n\n" if stream else b"ok"
    full = compressed(payload, encoding)
    cut = 10 if encoding == "gzip" else 2
    empty_blocks = b"\x00\x00\x00\xff\xff" * 1024
    body = ObservedBody([full[:cut], empty_blocks, empty_blocks, full[cut:]])
    p = provider("openai", limit=1024, event_limit=1024)
    install(
        p,
        lambda _: httpx.Response(
            200 if stream else 400,
            stream=body,
            headers={"content-encoding": encoding},
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        with pytest.raises(ProviderResponseError) as caught:
            await response.aread()
        assert caught.value.provider_code == (
            "stream_size_limit" if stream else "response_size_limit"
        )
        assert body.pulls == 2 and body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "raw-deflate"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("size", [64, 10 * 64 * 1024])
async def test_real_compressed_bodies_keep_padding_and_large_payloads(
    encoding: str, stream: bool, size: int
) -> None:
    payload = random.Random(7331).randbytes(size).hex().encode()
    if stream:
        payload = b"data: " + payload + b"\n\n"
    encoded = compressed(payload, encoding) + b"\0\0\0"
    body = ObservedBody([encoded[:1], encoded[1:7], encoded[7:]])
    p = provider("openai", limit=len(payload) + 3, event_limit=len(payload) + 3)
    install(
        p,
        lambda _: httpx.Response(
            200 if stream else 400,
            stream=body,
            headers={"content-encoding": encoding.replace("raw-", "")},
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        assert await response.aread() == payload
        assert body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "raw-deflate"])
async def test_incompressible_body_at_decoded_limit_is_accepted(encoding: str) -> None:
    payload = random.Random(1).randbytes(10 * 64 * 1024)
    encoded = compressed(payload, encoding)
    assert len(encoded) > len(payload)
    body = ObservedBody([encoded])
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
        assert await response.aread() == payload
    finally:
        await p.close()


async def test_small_decoded_fragments_do_not_reset_wire_accounting() -> None:
    import struct

    payload = b"xxxx"
    head = gzip.compress(b"")[:10]
    blocks = b"\x00\x00\x00\xff\xff" * 200 + b"\x00\x01\x00\xfe\xffx"
    end = b"\x01\x00\x00\xff\xff" + struct.pack("<II", zlib.crc32(payload), len(payload))
    assert gzip.decompress(head + blocks * 4 + end) == payload
    body = ObservedBody([head, blocks, blocks, blocks, blocks, end])
    p = provider("openai", event_limit=1024)
    install(p, lambda _: httpx.Response(200, stream=body, headers={"content-encoding": "gzip"}))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        with pytest.raises(ProviderResponseError) as caught:
            await response.aread()
        assert caught.value.provider_code == "stream_size_limit"
        assert body.pulls == 4 and body.closes == 1
    finally:
        await p.close()


async def test_completed_stream_events_reset_wire_accounting() -> None:
    rng = random.Random(31)
    payload = b"".join(b"data: " + rng.randbytes(64).hex().encode() + b"\n\n" for _ in range(100))
    encoded = gzip.compress(payload)
    assert len(encoded) > 2048
    body = ObservedBody([encoded])
    p = provider("openai", event_limit=1024)
    install(p, lambda _: httpx.Response(200, stream=body, headers={"content-encoding": "gzip"}))
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"), stream=True
        )
        assert await response.aread() == payload
        assert body.closes == 1
    finally:
        await p.close()


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "raw-deflate"])
async def test_exact_decoder_window_at_eof_does_not_drain_again(
    encoding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = zlib.decompressobj
    calls = 0

    class Decoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            nonlocal calls
            assert not self.inner.eof
            calls += 1
            return self.inner.decompress(data, max_length)

    monkeypatch.setattr(zlib, "decompressobj", Decoder)
    payload = b"x" * 65536
    body = ObservedBody([compressed(payload, encoding) + b"\0\0\0"])
    p = provider("openai", limit=len(payload) + 3)
    install(
        p,
        lambda _: httpx.Response(
            400,
            stream=body,
            headers={"content-encoding": encoding.replace("raw-", "")},
        ),
    )
    try:
        response = await p._client._client.send(
            httpx.Request("GET", "https://provider.test"),
            stream=True,
        )
        assert await response.aread() == payload
        assert calls == 1
    finally:
        await p.close()
