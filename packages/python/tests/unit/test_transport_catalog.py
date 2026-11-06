"""Catalog discovery uses the same bounded response decoder as model calls."""

from __future__ import annotations

import gzip
import json
import logging
import zlib
from typing import Any

import httpx
import pytest

from dendrux.errors import ProviderResponseError
from dendrux.llm import openrouter
from tests.unit.test_transport_refusals import ObservedBody


def catalog_transport(monkeypatch: pytest.MonkeyPatch, encoded: bytes) -> list[ObservedBody]:
    bodies: list[ObservedBody] = []

    async def respond(self: Any, request: httpx.Request) -> httpx.Response:
        body = ObservedBody([encoded, b"never pull this"])
        bodies.append(body)
        return httpx.Response(200, stream=body, headers={"content-encoding": "gzip"})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    monkeypatch.setattr(openrouter, "_catalog_cache", {})
    return bodies


@pytest.mark.parametrize("native_check", [False, True])
async def test_oversized_catalog_stops_in_decoder(
    native_check: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    encoded = gzip.compress(b'{"data":[],"padding":"' + b"x" * (40 * 1024 * 1024) + b'"}')
    bodies = catalog_transport(monkeypatch, encoded)
    original = zlib.decompressobj
    windows: list[int] = []
    outputs: list[int] = []

    class Decoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            windows.append(max_length)
            value = self.inner.decompress(data, max_length)
            outputs.append(len(value))
            return value

    monkeypatch.setattr(zlib, "decompressobj", Decoder)
    p = openrouter.OpenRouterProvider(model="test/model", api_key="test-key")
    try:
        if native_check:
            with caplog.at_level(logging.WARNING):
                await p._ensure_native_tool_support("test/model")
            assert "catalog unavailable" in caplog.text
            assert openrouter._catalog_cache[p._models_url] is None
        else:
            with pytest.raises(ProviderResponseError) as caught:
                await p.list_models()
            assert caught.value.provider_code == "response_size_limit"
            assert caught.value.provider == "openrouter"
            assert caught.value.retryable is False
        assert windows and all(0 < size <= 65536 for size in windows)
        assert max(outputs) <= 65536
        assert sum(outputs) <= 32 * 1024 * 1024 + 65536
        assert len(bodies) == 1
        assert bodies[0].pulls == bodies[0].closes == 1
    finally:
        await p.close()


async def test_catalog_accepts_ten_times_a_generous_model_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Synthetic capacity case: 1,000 models with roughly 2 KiB of metadata each,
    # expanded tenfold. This is an allowance test, not a live catalog snapshot.
    entries = [
        {
            "id": f"vendor/model-{i}",
            "name": f"Model {i}",
            "description": "model description " * 100,
            "context_length": 131072,
            "supported_parameters": ["tools", "temperature", "top_p"],
            "pricing": {"prompt": "0.000002", "completion": "0.000006"},
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        }
        for i in range(10000)
    ]
    payload = json.dumps({"data": entries}).encode()
    assert 20 * 1024 * 1024 < len(payload) < 32 * 1024 * 1024
    bodies = catalog_transport(monkeypatch, gzip.compress(payload))

    # The accepted catalog has no unrelated trailing bytes.
    async def respond(self: Any, request: httpx.Request) -> httpx.Response:
        body = ObservedBody([gzip.compress(payload)])
        bodies.append(body)
        return httpx.Response(200, stream=body, headers={"content-encoding": "gzip"})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    p = openrouter.OpenRouterProvider(api_key="test-key")
    try:
        models = await p.list_models()
        assert len(models) == 10000
        assert models[-1].id == "vendor/model-9999"
        assert all(model.supports_tools and model.supports_image_input for model in models)
        assert bodies[0].closes == 1
    finally:
        await p.close()
