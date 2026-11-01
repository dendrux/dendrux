"""Opt-in provider byte-bound matrix; keys are supplied by the coordinator.

Run with DENDRUX_LIVE=1 and DENDRUX_LIVE_RESULTS=<results.json>.
Model overrides: DENDRUX_LIVE_TRANSPORT_ANTHROPIC_MODEL and
DENDRUX_LIVE_TRANSPORT_OPENAI_MODEL. No key values are read by this module.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.errors import ProviderResponseError
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.openai import OpenAIProvider
from dendrux.types import Message, Role, StreamEventType

if TYPE_CHECKING:
    import httpx

PROMPT = [Message(role=Role.USER, content="Reply with exactly: transport ok")]


@pytest.fixture(params=["anthropic", "openai"])
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def provider(family: str, *, tiny: bool = False) -> Any:
    key_name = "ANTHROPIC_API_KEY" if family == "anthropic" else "OPENAI_API_KEY"
    if key_name not in os.environ:
        pytest.skip(f"{key_name} is not configured")
    options: dict[str, Any] = {"max_tokens": 128, "max_retries": 2}
    if tiny:
        options["max_response_bytes"] = 1
    if family == "anthropic":
        model = os.getenv("DENDRUX_LIVE_TRANSPORT_ANTHROPIC_MODEL", "claude-haiku-4-5")
        return AnthropicProvider(model=model, **options)
    model = os.getenv("DENDRUX_LIVE_TRANSPORT_OPENAI_MODEL", "gpt-4o-mini")
    return OpenAIProvider(model=model, **options)


async def test_default_nonstreaming_limit_passes(family: str, rec: dict[str, Any]) -> None:
    p = provider(family)
    rec.update(provider=family, model=p.model, scenario="transport_default_nonstreaming")
    try:
        result = await p.complete(PROMPT)
        assert result.text
    finally:
        await p.close()


async def test_default_streaming_limit_passes(family: str, rec: dict[str, Any]) -> None:
    p = provider(family)
    rec.update(provider=family, model=p.model, scenario="transport_default_streaming")
    try:
        done = None
        async for event in p.complete_stream(PROMPT):
            if event.type == StreamEventType.DONE:
                done = event.raw
        assert done is not None
        assert done.text
    finally:
        await p.close()


async def test_tiny_response_limit_is_not_retried(family: str, rec: dict[str, Any]) -> None:
    p = provider(family, tiny=True)
    rec.update(provider=family, model=p.model, scenario="transport_tiny_nonstreaming")
    attempts = 0

    async def count_request(request: httpx.Request) -> None:
        nonlocal attempts
        attempts += 1

    p._client._client.event_hooks["request"].append(count_request)
    try:
        with pytest.raises(ProviderResponseError) as caught:
            await p.complete(PROMPT)
        assert caught.value.provider_code == "response_size_limit"
        assert caught.value.retryable is False
        assert attempts == 1
        rec.update(provider_code="response_size_limit", attempts=attempts)
    finally:
        await p.close()
