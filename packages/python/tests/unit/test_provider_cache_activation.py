"""Tests for provider cache activation (PR 3).

Anthropic: cache_control marker on system block + rolling marker on
the latest message. The marker on iteration N's last message warms
the cache for iteration N+1 — it does not discount the call that
writes it.

OpenAI Chat + Responses: prompt_cache_key derived from cache_key_prefix
or run_id, plus optional prompt_cache_retention pass-through.

Caller's message list must never be mutated (deep copy).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import Message, Role

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Anthropic cache_control placement
# ---------------------------------------------------------------------------


class TestAnthropicCacheControl:
    """AnthropicProvider applies cache_control to system block and to the
    last message's last content block."""

    def test_system_block_has_cache_control_default_no_ttl(self) -> None:
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6")
        messages = [
            Message(role=Role.SYSTEM, content="you are helpful"),
            Message(role=Role.USER, content="hello"),
        ]
        api_kwargs, _ = provider._build_api_kwargs(messages, tools=None, kwargs={})

        system = api_kwargs["system"]
        assert isinstance(system, list), "system must be block-form for cache_control"
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "you are helpful"
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_ttl_propagates_when_set(self) -> None:
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6", cache_ttl="1h")
        messages = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hi"),
        ]
        api_kwargs, _ = provider._build_api_kwargs(messages, tools=None, kwargs={})
        assert api_kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_last_message_gets_cache_control(self) -> None:
        """Rolling marker on the latest message warms next iteration's cache."""
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6")
        messages = [
            Message(role=Role.USER, content="first"),
            Message(role=Role.ASSISTANT, content="second"),
            Message(role=Role.USER, content="third"),
        ]
        api_kwargs, _ = provider._build_api_kwargs(messages, tools=None, kwargs={})

        last = api_kwargs["messages"][-1]
        # Content was a string; should be converted to block list with cache_control
        assert isinstance(last["content"], list)
        assert last["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_works_with_no_system_prompt(self) -> None:
        """No system message → no system block at all (NOT_GIVEN sentinel),
        but last message still gets cache_control."""
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6")
        messages = [Message(role=Role.USER, content="hello")]
        api_kwargs, _ = provider._build_api_kwargs(messages, tools=None, kwargs={})

        # No system → not present or NOT_GIVEN sentinel; either is acceptable
        # But last message still has cache_control
        last = api_kwargs["messages"][-1]
        assert isinstance(last["content"], list)
        assert last["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_does_not_mutate_caller_messages(self) -> None:
        """The Dendrux Message list passed in must remain unchanged
        after _build_api_kwargs runs."""
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6")
        messages = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hello"),
        ]
        before = copy.deepcopy(messages)
        provider._build_api_kwargs(messages, tools=None, kwargs={})
        assert messages == before

    def test_rolling_marker_across_simulated_iterations(self) -> None:
        """Each iteration's last message has cache_control. Two consecutive
        builds with growing message lists each produce a marker on the new
        latest message."""
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6")
        iter1_msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hi"),
        ]
        api_kwargs1, _ = provider._build_api_kwargs(iter1_msgs, tools=None, kwargs={})
        last1 = api_kwargs1["messages"][-1]
        assert last1["content"][-1].get("cache_control") == {"type": "ephemeral"}

        iter2_msgs = iter1_msgs + [
            Message(role=Role.ASSISTANT, content="hello"),
            Message(role=Role.USER, content="more"),
        ]
        api_kwargs2, _ = provider._build_api_kwargs(iter2_msgs, tools=None, kwargs={})
        last2 = api_kwargs2["messages"][-1]
        assert last2["content"][-1].get("cache_control") == {"type": "ephemeral"}
        # And iter1's source list remains unchanged
        assert all("cache_control" not in str(m.content) for m in iter1_msgs)


# ---------------------------------------------------------------------------
# OpenAI Chat Completions cache key + retention
# ---------------------------------------------------------------------------


@dataclass
class _FakeFunction:
    name: str = "noop"
    arguments: str = "{}"


@dataclass
class _FakeMessage:
    role: str = "assistant"
    content: str | None = "ok"
    tool_calls: list = field(default_factory=list)


@dataclass
class _FakeChoice:
    message: _FakeMessage = field(default_factory=_FakeMessage)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150
    prompt_tokens_details: Any = None


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChoice] = field(default_factory=lambda: [_FakeChoice()])
    usage: _FakeUsage = field(default_factory=_FakeUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


class TestOpenAIChatCacheActivation:
    """OpenAIProvider passes prompt_cache_key (from agent_name:model prefix
    or run_id fallback) and optional prompt_cache_retention."""

    async def test_passes_prompt_cache_key_from_prefix(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeChatCompletion:
            captured.update(kwargs)
            return _FakeChatCompletion()

        provider._client.chat.completions.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            cache_key_prefix="myagent:gpt-4o",
        )
        assert captured.get("prompt_cache_key") == "dendrux:myagent:gpt-4o"

    async def test_falls_back_to_run_id_when_no_prefix(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeChatCompletion:
            captured.update(kwargs)
            return _FakeChatCompletion()

        provider._client.chat.completions.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            run_id="run_abc",
        )
        assert captured.get("prompt_cache_key") == "dendrux:run_abc"

    async def test_no_cache_key_when_neither_provided(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeChatCompletion:
            captured.update(kwargs)
            return _FakeChatCompletion()

        provider._client.chat.completions.create = capture
        await provider.complete([Message(role=Role.USER, content="hi")])
        assert "prompt_cache_key" not in captured

    async def test_retention_passes_through_when_set(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test", prompt_cache_retention="24h")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeChatCompletion:
            captured.update(kwargs)
            return _FakeChatCompletion()

        provider._client.chat.completions.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            run_id="run_abc",
        )
        assert captured.get("prompt_cache_retention") == "24h"

    async def test_retention_omitted_when_default(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeChatCompletion:
            captured.update(kwargs)
            return _FakeChatCompletion()

        provider._client.chat.completions.create = capture
        await provider.complete([Message(role=Role.USER, content="hi")])
        assert "prompt_cache_retention" not in captured

    async def test_prefix_takes_priority_over_run_id(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeChatCompletion:
            captured.update(kwargs)
            return _FakeChatCompletion()

        provider._client.chat.completions.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            cache_key_prefix="agent:model",
            run_id="run_abc",
        )
        assert captured.get("prompt_cache_key") == "dendrux:agent:model"


# ---------------------------------------------------------------------------
# OpenAI Responses cache key + retention
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponsesUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150
    input_tokens_details: Any = None


@dataclass
class _FakeResponsesResponse:
    output: list = field(default_factory=list)
    usage: _FakeResponsesUsage = field(default_factory=_FakeResponsesUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


class TestOpenAIResponsesCacheActivation:
    async def test_passes_prompt_cache_key_from_prefix(self) -> None:
        provider = OpenAIResponsesProvider(model="gpt-5", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeResponsesResponse:
            captured.update(kwargs)
            return _FakeResponsesResponse()

        provider._client.responses.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            cache_key_prefix="myagent:gpt-5",
        )
        assert captured.get("prompt_cache_key") == "dendrux:myagent:gpt-5"

    async def test_falls_back_to_run_id(self) -> None:
        provider = OpenAIResponsesProvider(model="gpt-5", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeResponsesResponse:
            captured.update(kwargs)
            return _FakeResponsesResponse()

        provider._client.responses.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            run_id="r_xyz",
        )
        assert captured.get("prompt_cache_key") == "dendrux:r_xyz"

    async def test_retention_passes_through(self) -> None:
        provider = OpenAIResponsesProvider(
            model="gpt-5", api_key="test", prompt_cache_retention="24h"
        )
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeResponsesResponse:
            captured.update(kwargs)
            return _FakeResponsesResponse()

        provider._client.responses.create = capture
        await provider.complete(
            [Message(role=Role.USER, content="hi")],
            run_id="r_xyz",
        )
        assert captured.get("prompt_cache_retention") == "24h"

    async def test_no_cache_kwargs_when_nothing_provided(self) -> None:
        provider = OpenAIResponsesProvider(model="gpt-5", api_key="test")
        captured: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> _FakeResponsesResponse:
            captured.update(kwargs)
            return _FakeResponsesResponse()

        provider._client.responses.create = capture
        await provider.complete([Message(role=Role.USER, content="hi")])
        assert "prompt_cache_key" not in captured
        assert "prompt_cache_retention" not in captured


# ---------------------------------------------------------------------------
# Anthropic ignores OpenAI-shaped cache args without crashing
# ---------------------------------------------------------------------------


class TestAnthropicAcceptsButIgnoresCacheArgs:
    """Anthropic gets cache_control via _build_api_kwargs, not via run_id /
    cache_key_prefix, but the contract still requires it accepts those args."""

    async def test_anthropic_accepts_run_id_and_prefix(self) -> None:
        from anthropic.types import Message as AnthropicMessage
        from anthropic.types import TextBlock, Usage

        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-6")

        async def fake_create(**kwargs: Any) -> AnthropicMessage:
            return AnthropicMessage(
                id="msg_1",
                content=[TextBlock(type="text", text="ok", citations=None)],
                model="claude-opus-4-6",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=10, output_tokens=5),
            )

        provider._client.messages.create = AsyncMock(side_effect=fake_create)

        # Must not raise even though Anthropic doesn't use these
        result = await provider.complete(
            [Message(role=Role.USER, content="hi")],
            run_id="r1",
            cache_key_prefix="agent:model",
        )
        assert result.text == "ok"
