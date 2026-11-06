"""Response refusals provide a fixed explanation on every limit path."""

from __future__ import annotations

import pickle

import httpx
import pytest

from dendrux.errors import ProviderResponseError
from dendrux.llm._resilience import Deadlines, StreamClock
from tests.unit.test_transport_bound import Body, document, install, provider
from tests.unit.test_transport_refusals import MESSAGES, SENTINEL, assert_private

MESSAGE = "Response exceeds the local response contract."


@pytest.mark.parametrize("family", ["anthropic", "openai"])
async def test_transport_refusal_has_fixed_message(family: str) -> None:
    p = provider(family, limit=1024)
    install(p, lambda _: httpx.Response(500, stream=Body([document(family, 2048)])))
    try:
        with pytest.raises(ProviderResponseError) as caught:
            await p.complete(MESSAGES)
        assert caught.value.message == MESSAGE
        assert str(caught.value).endswith(": " + MESSAGE)
        assert caught.value.provider_code == "response_size_limit"
        assert caught.value.retryable is False
        assert_private(caught.value)
        restored = pickle.loads(pickle.dumps(caught.value))
        assert str(restored) == str(caught.value)
    finally:
        await p.close()


@pytest.mark.parametrize("tool_count", [False, True])
async def test_semantic_stream_refusal_has_fixed_message(tool_count: bool) -> None:
    clock = StreamClock(
        Deadlines(None, None, None, max_stream_bytes=1, max_stream_tool_calls=1),
        provider="openai",
        model="test-model",
    )
    with pytest.raises(ProviderResponseError) as caught:
        if tool_count:
            clock.check_tool_count(2)
        else:
            clock.account(SENTINEL)
    assert caught.value.message == MESSAGE
    assert str(caught.value).endswith(": " + MESSAGE)
    assert caught.value.provider_code == (
        "stream_tool_call_limit" if tool_count else "stream_size_limit"
    )
    assert caught.value.retryable is False
    assert_private(caught.value)
