"""LLM provider base class.

Defines the contract that all LLM providers must implement.
Provides a default complete_stream() fallback that wraps complete().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from dendrite.types import ProviderCapabilities, StreamEvent, StreamEventType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dendrite.types import LLMResponse, Message, ToolDef


class LLMProvider(ABC):
    """Base class for LLM providers.

    Subclasses must implement complete() and declare capabilities.
    The complete_stream() method has a default fallback that calls complete()
    and yields the result as events. Override for real token-by-token streaming.

    Usage:
        class MyProvider(LLMProvider):
            capabilities = ProviderCapabilities(supports_native_tools=True, ...)

            async def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
                # call your LLM API here
                ...
    """

    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the LLM and return a complete response.

        Args:
            messages: Conversation history in Dendrite's universal format.
            tools: Tool definitions the LLM can call. Provider converts these
                   to its native format internally.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            Normalized LLMResponse with text, tool_calls, and usage stats.
        """

    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream LLM response as events.

        Default implementation calls complete() and yields the result as a
        TEXT_DELTA + DONE sequence. Override for real token-by-token streaming.

        Args:
            messages: Conversation history in Dendrite's universal format.
            tools: Tool definitions the LLM can call.
            **kwargs: Provider-specific options.

        Yields:
            StreamEvent objects as they arrive from the LLM.
        """
        response = await self.complete(messages, tools, **kwargs)
        if response.text:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=response.text)
        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name=tc.name,
                )
        yield StreamEvent(type=StreamEventType.DONE)
