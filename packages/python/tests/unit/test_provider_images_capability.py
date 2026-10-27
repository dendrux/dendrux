"""Image capability surface of the providers.

Static tables, OpenRouter catalogue, custom-provider default, learned "no"
cache (keyed by surface + base URL + model; per instance; thread-safe),
many-image rule, thinking flag, tool-result image mode and MockLLM options.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from dendrux.llm import openrouter as openrouter_mod
from dendrux.llm._images import normalize_base_url, openai_image_support
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.llm.openai import OpenAIProvider
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.llm.openrouter import OpenRouterModel, OpenRouterProvider
from dendrux.types import LLMResponse, Message, Role


class CustomProvider(LLMProvider):
    @property
    def model(self) -> str:
        return "custom"

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        return LLMResponse(text="ok")


def or_model(model_id: str, modalities: tuple[str, ...]) -> OpenRouterModel:
    return OpenRouterModel(
        id=model_id,
        name=model_id,
        context_length=None,
        supported_parameters=("tools",),
        prompt_price=None,
        completion_price=None,
        input_modalities=modalities,
        output_modalities=("text",),
    )


class TestStaticTables:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "yes"),
            ("gpt-4o-mini-2024-07-18", "yes"),
            ("chatgpt-4o-latest", "yes"),
            ("gpt-4.1-nano", "yes"),
            ("gpt-4.5-preview", "yes"),
            ("gpt-4-turbo-2024-04-09", "yes"),
            ("gpt-5-mini", "yes"),
            ("gpt-6", "yes"),
            ("o1", "yes"),
            ("o1-pro", "yes"),
            ("o3", "yes"),
            ("o4-mini", "yes"),
            ("gpt-3.5-turbo", "no"),
            ("gpt-4", "no"),
            ("gpt-4-0314", "no"),
            ("gpt-4-0613", "no"),
            ("gpt-4-32k-0613", "no"),
            ("o1-mini", "no"),
            ("o1-preview-2024-09-12", "no"),
            ("o3-mini", "no"),
            ("o3-mini-2025-01-31", "no"),
            ("gpt-4-1106-preview", "unknown"),
            ("llama-3.3-70b", "unknown"),
            ("davinci-002", "unknown"),
        ],
    )
    def test_openai_table(self, model: str, expected: str) -> None:
        assert openai_image_support(model) == expected

    async def test_chat_and_responses_official_endpoint(self) -> None:
        chat = OpenAIProvider(model="gpt-4o", api_key="k")
        responses = OpenAIResponsesProvider(model="gpt-4o", api_key="k")
        for provider in (chat, responses):
            assert await provider.image_input_support("gpt-5-mini") == "yes"
            assert await provider.image_input_support("o3-mini") == "no"
            assert await provider.image_input_support("mystery") == "unknown"

    async def test_chat_compatible_endpoint_is_unknown(self) -> None:
        chat = OpenAIProvider(model="gpt-4o", api_key="k", base_url="http://localhost:8000/v1")
        assert await chat.image_input_support("gpt-4o") == "unknown"
        assert await chat.image_input_support("gpt-3.5-turbo") == "unknown"

    async def test_anthropic(self) -> None:
        provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="k")
        assert await provider.image_input_support("claude-haiku-4-5") == "yes"
        assert await provider.image_input_support("my-proxy-model") == "unknown"

    async def test_custom_provider_defaults(self) -> None:
        custom = CustomProvider()
        assert custom.capabilities.supports_multimodal is False
        assert await custom.image_input_support("custom") == "no"
        assert custom.image_capability_source() == "adapter"
        assert custom.tool_result_images == "portable"
        assert custom.thinking_enabled is False
        assert custom.many_image_rule("custom") is None
        assert custom.image_capability_key("m") == ("CustomProvider", "", "m")

    def test_builtins_declare_multimodal(self) -> None:
        for cls in (AnthropicProvider, OpenAIProvider, OpenAIResponsesProvider, OpenRouterProvider):
            assert cls.capabilities.supports_multimodal is True


class TestOpenRouterCatalogue:
    @pytest.fixture
    def router(self, monkeypatch: pytest.MonkeyPatch) -> OpenRouterProvider:
        provider = OpenRouterProvider(model="anthropic/claude-haiku-4.5", api_key="k")
        catalog = {
            "anthropic/claude-haiku-4.5": or_model(
                "anthropic/claude-haiku-4.5", ("text", "image", "file")
            ),
            "deepseek/deepseek-chat": or_model("deepseek/deepseek-chat", ("text",)),
            "openai/o3-mini": or_model("openai/o3-mini", ("text", "file")),
        }
        monkeypatch.setitem(openrouter_mod._catalog_cache, provider._models_url, catalog)
        return provider

    async def test_catalogue_verdicts(self, router: OpenRouterProvider) -> None:
        assert await router.image_input_support("anthropic/claude-haiku-4.5") == "yes"
        assert await router.image_input_support("deepseek/deepseek-chat") == "no"
        assert await router.image_input_support("openai/o3-mini") == "no"
        assert await router.image_input_support("not/listed") == "unknown"
        assert router.image_capability_source() == "catalog"

    async def test_catalogue_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = OpenRouterProvider(model="x/y", api_key="k")
        monkeypatch.setitem(openrouter_mod._catalog_cache, provider._models_url, None)
        assert await provider.image_input_support("x/y") == "unknown"

    def test_supports_image_input_vs_is_multimodal(self) -> None:
        audio_only = or_model("a", ("text", "audio"))
        assert audio_only.is_multimodal is True
        assert audio_only.supports_image_input is False
        assert or_model("b", ("image", "text")).supports_image_input is True
        assert or_model("c", ()).supports_image_input is False


class TestManyImageRuleAndFlags:
    def test_many_image_rule(self) -> None:
        router = OpenRouterProvider(model="openai/gpt-4o-mini", api_key="k")
        assert AnthropicProvider(model="claude-sonnet-4-6", api_key="k").many_image_rule(
            "claude-sonnet-4-6"
        ) == (20, 2000)
        assert router.many_image_rule("anthropic/claude-haiku-4.5") == (20, 2000)
        assert router.many_image_rule("~anthropic/claude-sonnet-latest") == (20, 2000)
        assert router.many_image_rule("openai/gpt-4o-mini") is None
        assert OpenAIProvider(model="gpt-4o", api_key="k").many_image_rule("gpt-4o") is None
        assert OpenAIResponsesProvider(model="gpt-4o", api_key="k").many_image_rule("x") is None

    def test_thinking_enabled(self) -> None:
        assert AnthropicProvider(model="c", api_key="k", thinking=True).thinking_enabled is True
        assert AnthropicProvider(model="c", api_key="k").thinking_enabled is False
        assert OpenAIResponsesProvider(model="m", api_key="k", thinking=True).thinking_enabled
        assert OpenAIProvider(model="m", api_key="k").thinking_enabled is False

    def test_tool_result_image_modes(self) -> None:
        assert AnthropicProvider(model="c", api_key="k").tool_result_images == "native"
        assert OpenAIProvider(model="m", api_key="k").tool_result_images == "portable"
        assert OpenRouterProvider(model="a/b", api_key="k").tool_result_images == "portable"
        assert OpenAIResponsesProvider(model="m", api_key="k").tool_result_images == "native"
        portable = OpenAIResponsesProvider(model="m", api_key="k", tool_result_images="portable")
        assert portable.tool_result_images == "portable"
        assert OpenAIResponsesProvider.tool_result_images == "portable"  # class default untouched


class TestLearnedCache:
    def test_capability_keys(self) -> None:
        assert AnthropicProvider(model="c", api_key="k").image_capability_key("c") == (
            "anthropic",
            "https://api.anthropic.com",
            "c",
        )
        assert OpenAIProvider(model="m", api_key="k").image_capability_key("m") == (
            "chat",
            "https://api.openai.com/v1",
            "m",
        )
        assert OpenAIResponsesProvider(model="m", api_key="k").image_capability_key("m")[0] == (
            "responses"
        )
        assert OpenRouterProvider(model="a/b", api_key="k").image_capability_key("a/b") == (
            "openrouter",
            "https://openrouter.ai/api/v1",
            "a/b",
        )

    def test_normalize_base_url(self) -> None:
        assert (
            normalize_base_url("HTTPS://user:pw@Gateway.Example:8443/v1/?key=abc#x")
            == "https://gateway.example:8443/v1"
        )
        assert normalize_base_url("http://[::1]:8000/v1/") == "http://[::1]:8000/v1"
        assert normalize_base_url(None) == ""

    def test_isolated_by_base_url_and_instance(self) -> None:
        a = OpenAIProvider(model="m", api_key="k", base_url="http://one.test/v1")
        b = OpenAIProvider(model="m", api_key="k", base_url="http://two.test/v1")
        a2 = OpenAIProvider(model="m", api_key="k", base_url="http://one.test/v1")
        a.learn_image_unsupported("vision-less")
        assert a.learned_image_support("vision-less") == "no"
        assert a.learned_image_support("other") is None
        assert b.learned_image_support("vision-less") is None
        assert a2.learned_image_support("vision-less") is None  # never shared across instances

    def test_concurrent_learns_are_safe(self) -> None:
        provider = OpenAIProvider(model="m", api_key="k")
        models = [f"model-{i}" for i in range(50)]

        def learn(chunk: list[str]) -> None:
            for name in chunk:
                provider.learn_image_unsupported(name)
                assert provider.learned_image_support(name) == "no"

        threads = [threading.Thread(target=learn, args=(models[i::5],)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(provider.learned_image_support(m) == "no" for m in models)


class TestMockLLM:
    async def test_default_yes_and_wire_messages(self) -> None:
        llm = MockLLM([LLMResponse(text="a"), LLMResponse(text="b")])
        assert llm.capabilities.supports_multimodal is True
        assert await llm.image_input_support("mock") == "yes"
        msgs = [Message(role=Role.USER, content="hi")]
        await llm.complete(msgs)
        await llm.complete([*msgs, Message(role=Role.ASSISTANT, content="a")])
        assert len(llm.wire_messages) == 2
        assert llm.wire_messages[0][0] is msgs[0]
        assert len(llm.wire_messages[1]) == 2

    @pytest.mark.parametrize(("support", "multimodal"), [("no", False), ("unknown", True)])
    async def test_configured_support(self, support: Any, multimodal: bool) -> None:
        llm = MockLLM([], image_support=support)
        assert llm.capabilities.supports_multimodal is multimodal
        assert await llm.image_input_support("mock") == support

    def test_invalid_support(self) -> None:
        with pytest.raises(ValueError, match="image_support"):
            MockLLM([], image_support="maybe")  # type: ignore[arg-type]


class TestFixRoundCapability:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o-audio-preview",
            "gpt-4o-realtime-preview-2024-12-17",
            "gpt-4o-mini-audio-preview",
            "gpt-4o-mini-realtime-preview",
        ],
    )
    def test_gpt4o_audio_realtime_not_yes(self, model: str) -> None:
        assert openai_image_support(model) == "unknown"

    async def test_mock_subclass_capabilities_kept(self) -> None:
        from dendrux.types import ProviderCapabilities

        class ToolMock(MockLLM):
            capabilities = ProviderCapabilities(
                supports_native_tools=True, supports_streaming=True, max_context_tokens=1234
            )

        llm = ToolMock([])
        assert llm.capabilities.supports_native_tools is True
        assert llm.capabilities.supports_streaming is True
        assert llm.capabilities.max_context_tokens == 1234
        assert llm.capabilities.supports_multimodal is True
        assert ToolMock.capabilities.supports_multimodal is False  # class attr untouched
        off = ToolMock([], image_support="no")
        assert off.capabilities.supports_native_tools is True
        assert off.capabilities.supports_multimodal is False
