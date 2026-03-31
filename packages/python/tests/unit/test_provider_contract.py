"""Tests for the LLMProvider contract — G0: Provider Contract.

Verifies that the base class exposes model, close(), and async context manager,
and that concrete providers (AnthropicProvider, MockLLM) satisfy the contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic

from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse

# ------------------------------------------------------------------
# Base class contract
# ------------------------------------------------------------------


class TestLLMProviderContract:
    """LLMProvider base class exposes the full provider contract."""

    def test_model_is_abstract(self) -> None:
        """Cannot instantiate a provider without implementing model."""

        class NoModelProvider(LLMProvider):
            async def complete(self, messages, tools=None, **kwargs):
                return LLMResponse(text="ok")

        try:
            NoModelProvider()
            raise AssertionError("Should not be able to instantiate without model")  # noqa: TRY301
        except TypeError as e:
            assert "model" in str(e)

    async def test_base_close_is_noop(self) -> None:
        """Base close() is a no-op — doesn't raise."""
        llm = MockLLM([LLMResponse(text="hi")])
        await llm.close()  # should not raise

    async def test_base_context_manager_calls_close(self) -> None:
        """async with on base class calls close() on exit."""

        class TrackingProvider(LLMProvider):
            closed = False

            @property
            def model(self) -> str:
                return "test"

            async def complete(self, messages, tools=None, **kwargs):
                return LLMResponse(text="ok")

            async def close(self) -> None:
                self.closed = True

        provider = TrackingProvider()
        async with provider as p:
            assert p is provider
            assert not provider.closed
        assert provider.closed

    def test_concurrency_contract_in_docstring(self) -> None:
        """The concurrency-safety contract is documented."""
        assert "concurrent" in LLMProvider.__doc__.lower()


# ------------------------------------------------------------------
# MockLLM contract
# ------------------------------------------------------------------


class TestMockLLMContract:
    def test_model_defaults_to_mock(self) -> None:
        llm = MockLLM([LLMResponse(text="hi")])
        assert llm.model == "mock"

    def test_model_is_configurable(self) -> None:
        llm = MockLLM([LLMResponse(text="hi")], model="custom-test-model")
        assert llm.model == "custom-test-model"


# ------------------------------------------------------------------
# AnthropicProvider contract
# ------------------------------------------------------------------


class TestAnthropicProviderContract:
    def test_model_property_returns_configured_model(self) -> None:
        from dendrux.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-6")
        assert provider.model == "claude-sonnet-4-6"

    def test_constructs_without_explicit_api_key(self) -> None:
        """AnthropicProvider passes api_key=None to SDK, letting it read env."""
        from dendrux.llm.anthropic import AnthropicProvider

        with patch.object(anthropic, "AsyncAnthropic", return_value=MagicMock()) as mock_cls:
            AnthropicProvider(model="claude-sonnet-4-6")
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["api_key"] is None

    def test_explicit_api_key_forwarded_to_sdk(self) -> None:
        """When api_key is provided, it's passed through to the SDK."""
        from dendrux.llm.anthropic import AnthropicProvider

        with patch.object(anthropic, "AsyncAnthropic", return_value=MagicMock()) as mock_cls:
            AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-explicit")
            mock_cls.assert_called_once()
            assert mock_cls.call_args.kwargs["api_key"] == "sk-explicit"

    async def test_close_closes_client(self) -> None:
        from dendrux.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test", model="test")
        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_awaited_once()

    async def test_context_manager_closes_on_exit(self) -> None:
        from dendrux.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test", model="test")
        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            async with provider:
                pass
            mock_close.assert_awaited_once()
