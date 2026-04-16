"""Tests for cache token telemetry (PR 2).

UsageStats gains cache_read_input_tokens and cache_creation_input_tokens.
Each provider adapter populates whatever the vendor returns; OpenAI variants
also normalize input_tokens to mean fresh-only (excluding cached) so
cross-provider comparisons are honest.

Distinction kept across the codebase:
  - None  → vendor did not report this field
  - 0     → vendor reported zero
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    TextBlock,
    Usage,
)

from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import Message, Role, UsageStats

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402

# ---------------------------------------------------------------------------
# UsageStats — new cache fields
# ---------------------------------------------------------------------------


class TestUsageStatsCacheFields:
    """UsageStats has cache_read_input_tokens and cache_creation_input_tokens.

    None vs 0 distinction matters: None means the provider didn't tell us;
    0 means it told us zero.
    """

    def test_default_cache_fields_are_none(self) -> None:
        usage = UsageStats(input_tokens=10, output_tokens=5, total_tokens=15)
        assert usage.cache_read_input_tokens is None
        assert usage.cache_creation_input_tokens is None

    def test_can_construct_with_cache_fields(self) -> None:
        usage = UsageStats(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=200,
        )
        assert usage.cache_read_input_tokens == 100
        assert usage.cache_creation_input_tokens == 200

    def test_zero_distinguishable_from_none(self) -> None:
        usage_zero = UsageStats(cache_read_input_tokens=0)
        usage_none = UsageStats()
        assert usage_zero.cache_read_input_tokens == 0
        assert usage_none.cache_read_input_tokens is None
        assert usage_zero.cache_read_input_tokens != usage_none.cache_read_input_tokens


# ---------------------------------------------------------------------------
# Anthropic adapter — both fields populated
# ---------------------------------------------------------------------------


def _make_anthropic_response(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> AnthropicMessage:
    """Build a fake Anthropic Message with optional cache fields."""
    usage_kwargs: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read_input_tokens is not None:
        usage_kwargs["cache_read_input_tokens"] = cache_read_input_tokens
    if cache_creation_input_tokens is not None:
        usage_kwargs["cache_creation_input_tokens"] = cache_creation_input_tokens

    return AnthropicMessage(
        id="msg_test",
        content=[TextBlock(type="text", text="ok", citations=None)],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(**usage_kwargs),
    )


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-6")


class TestAnthropicCachePopulation:
    """Anthropic returns both cache_read_input_tokens and
    cache_creation_input_tokens on every Usage object."""

    def test_normalize_response_populates_both_cache_fields(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        response = _make_anthropic_response(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=300,
            cache_creation_input_tokens=400,
        )
        result = anthropic_provider._normalize_response(response)
        assert result.usage.cache_read_input_tokens == 300
        assert result.usage.cache_creation_input_tokens == 400

    def test_anthropic_zero_values_distinct_from_none(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """Anthropic always returns the cache fields, even as 0 — preserve
        that distinction (vs None which would mean 'not reported')."""
        response = _make_anthropic_response(
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        result = anthropic_provider._normalize_response(response)
        assert result.usage.cache_read_input_tokens == 0
        assert result.usage.cache_creation_input_tokens == 0

    def test_anthropic_input_tokens_unchanged(self, anthropic_provider: AnthropicProvider) -> None:
        """Anthropic input_tokens already excludes cached. No normalization
        needed — pass through verbatim."""
        response = _make_anthropic_response(
            input_tokens=500,  # Anthropic: this is fresh-only
            cache_read_input_tokens=1000,
        )
        result = anthropic_provider._normalize_response(response)
        assert result.usage.input_tokens == 500


# ---------------------------------------------------------------------------
# OpenAI Chat Completions adapter — read field, normalize input
# ---------------------------------------------------------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    type: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    role: str = "assistant"
    content: str | None = "ok"
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage = field(default_factory=_FakeMessage)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _FakeCachedDetails:
    cached_tokens: int


@dataclass
class _FakeOpenAIUsage:
    """Mirror of openai.types.CompletionUsage with optional cache details."""

    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150
    prompt_tokens_details: _FakeCachedDetails | None = None


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChoice] = field(default_factory=lambda: [_FakeChoice()])
    usage: _FakeOpenAIUsage = field(default_factory=_FakeOpenAIUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o", api_key="test-key")


class TestOpenAIChatCachePopulation:
    """OpenAI Chat Completions: cache_read_input_tokens populated from
    usage.prompt_tokens_details.cached_tokens. cache_creation stays None
    (OpenAI auto-caches, no manual creation API)."""

    async def test_complete_populates_cache_read_only(
        self, openai_provider: OpenAIProvider
    ) -> None:
        completion = _FakeChatCompletion(
            usage=_FakeOpenAIUsage(
                prompt_tokens=1500,
                completion_tokens=50,
                total_tokens=1550,
                prompt_tokens_details=_FakeCachedDetails(cached_tokens=1000),
            )
        )
        openai_provider._client.chat.completions.create = AsyncMock(return_value=completion)

        result = await openai_provider.complete([Message(role=Role.USER, content="hi")])

        assert result.usage.cache_read_input_tokens == 1000
        assert result.usage.cache_creation_input_tokens is None

    async def test_complete_normalizes_input_tokens_to_fresh(
        self, openai_provider: OpenAIProvider
    ) -> None:
        """input_tokens = prompt_tokens - cached_tokens.

        OpenAI reports prompt_tokens including cached (1500). After
        normalization input_tokens means fresh-only (500), matching
        Anthropic's semantics across providers.
        """
        completion = _FakeChatCompletion(
            usage=_FakeOpenAIUsage(
                prompt_tokens=1500,
                completion_tokens=50,
                total_tokens=1550,
                prompt_tokens_details=_FakeCachedDetails(cached_tokens=1000),
            )
        )
        openai_provider._client.chat.completions.create = AsyncMock(return_value=completion)
        result = await openai_provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.input_tokens == 500

    async def test_complete_no_cache_details_leaves_fields_none(
        self, openai_provider: OpenAIProvider
    ) -> None:
        """OpenAI-compatible backends (Groq, Together, vLLM) often omit
        prompt_tokens_details. Cache fields stay None and input_tokens
        passes through unchanged."""
        completion = _FakeChatCompletion(
            usage=_FakeOpenAIUsage(
                prompt_tokens=200,
                completion_tokens=30,
                total_tokens=230,
                prompt_tokens_details=None,
            )
        )
        openai_provider._client.chat.completions.create = AsyncMock(return_value=completion)
        result = await openai_provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.cache_read_input_tokens is None
        assert result.usage.cache_creation_input_tokens is None
        assert result.usage.input_tokens == 200

    async def test_complete_zero_cached_distinguishable_from_none(
        self, openai_provider: OpenAIProvider
    ) -> None:
        completion = _FakeChatCompletion(
            usage=_FakeOpenAIUsage(
                prompt_tokens=200,
                completion_tokens=30,
                total_tokens=230,
                prompt_tokens_details=_FakeCachedDetails(cached_tokens=0),
            )
        )
        openai_provider._client.chat.completions.create = AsyncMock(return_value=completion)
        result = await openai_provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.cache_read_input_tokens == 0
        # No cached → input_tokens equals prompt_tokens
        assert result.usage.input_tokens == 200


# ---------------------------------------------------------------------------
# OpenAI Responses adapter — same shape under input_tokens_details
# ---------------------------------------------------------------------------


@dataclass
class _FakeInputCachedDetails:
    cached_tokens: int


@dataclass
class _FakeResponsesUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150
    input_tokens_details: _FakeInputCachedDetails | None = None


@dataclass
class _FakeResponsesResponse:
    output: list = field(default_factory=list)
    usage: _FakeResponsesUsage = field(default_factory=_FakeResponsesUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


@pytest.fixture
def responses_provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-5", api_key="test-key")


class TestOpenAIResponsesCachePopulation:
    """OpenAI Responses API: cache_read_input_tokens from
    usage.input_tokens_details.cached_tokens."""

    async def test_complete_populates_cache_read_only(
        self, responses_provider: OpenAIResponsesProvider
    ) -> None:
        response = _FakeResponsesResponse(
            usage=_FakeResponsesUsage(
                input_tokens=2000,
                output_tokens=100,
                total_tokens=2100,
                input_tokens_details=_FakeInputCachedDetails(cached_tokens=1500),
            )
        )
        responses_provider._client.responses.create = AsyncMock(return_value=response)

        result = await responses_provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.cache_read_input_tokens == 1500
        assert result.usage.cache_creation_input_tokens is None

    async def test_complete_normalizes_input_tokens_to_fresh(
        self, responses_provider: OpenAIResponsesProvider
    ) -> None:
        response = _FakeResponsesResponse(
            usage=_FakeResponsesUsage(
                input_tokens=2000,
                output_tokens=100,
                total_tokens=2100,
                input_tokens_details=_FakeInputCachedDetails(cached_tokens=1500),
            )
        )
        responses_provider._client.responses.create = AsyncMock(return_value=response)
        result = await responses_provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.input_tokens == 500

    async def test_complete_no_cache_details_leaves_fields_none(
        self, responses_provider: OpenAIResponsesProvider
    ) -> None:
        response = _FakeResponsesResponse(
            usage=_FakeResponsesUsage(
                input_tokens=200,
                output_tokens=30,
                total_tokens=230,
                input_tokens_details=None,
            )
        )
        responses_provider._client.responses.create = AsyncMock(return_value=response)
        result = await responses_provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.cache_read_input_tokens is None
        assert result.usage.input_tokens == 200
