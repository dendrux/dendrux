"""LLM provider base class.

Defines the contract that all LLM providers must implement.
Provides a default complete_stream() fallback that wraps complete().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from dendrux.types import ProviderCapabilities, StreamEvent, StreamEventType

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrux.types import LLMResponse, Message, ToolDef


class LLMProvider(ABC):
    """Base class for LLM providers.

    Subclasses must implement complete() and declare capabilities.
    The complete_stream() method has a default fallback that calls complete()
    and yields the result as events. Override for real token-by-token streaming.

    **Concurrency contract:** Providers must be safe for concurrent use across
    multiple ``agent.run()`` calls. This means the underlying HTTP client (or
    equivalent) must support concurrent requests. ``httpx.AsyncClient`` and the
    Anthropic/OpenAI SDKs satisfy this by default via connection pooling.

    Usage:
        class MyProvider(LLMProvider):
            capabilities = ProviderCapabilities(supports_native_tools=True, ...)

            async def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
                # call your LLM API here
                ...
    """

    capabilities: ProviderCapabilities = ProviderCapabilities()

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier this provider is configured to use.

        Subclasses must override. Returned value is persisted in run records
        and logged — a blank string is a silent bug, so we enforce this.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release any resources held by this provider.

        No-op by default. Providers that hold HTTP clients or connection pools
        should override to close them cleanly.
        """

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the LLM and return a complete response.

        Args:
            messages: Conversation history in Dendrux's universal format.
            tools: Tool definitions the LLM can call. Provider converts these
                   to its native format internally.
            output_schema: JSON Schema for structured output. When provided,
                the provider uses its native mechanism to constrain the
                response to this schema (Anthropic: tool-use trick, OpenAI:
                response_format). The structured data is normalized into
                LLMResponse.text as a JSON string.
            run_id: The current run's id. OpenAI providers use it as a
                fallback ``prompt_cache_key`` when ``cache_key_prefix`` is
                not supplied. Anthropic ignores it (caching is byte-based).
            cache_key_prefix: Stable identifier for the cache pool, typically
                ``f"{agent_name}:{model}"``. OpenAI providers prefer this over
                ``run_id`` so all runs of the same agent share a pool.
                Anthropic ignores it.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            Normalized LLMResponse with text, tool_calls, and usage stats.
        """

    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM response as events.

        Default implementation calls complete() and yields the result as a
        TEXT_DELTA + DONE sequence. Override for real token-by-token streaming.

        Args:
            messages: Conversation history in Dendrux's universal format.
            tools: Tool definitions the LLM can call.
            run_id, cache_key_prefix: See ``complete()``.
            **kwargs: Provider-specific options.

        Yields:
            StreamEvent objects as they arrive from the LLM.
        """
        response = await self.complete(
            messages,
            tools,
            output_schema=output_schema,
            run_id=run_id,
            cache_key_prefix=cache_key_prefix,
            **kwargs,
        )
        if response.text:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=response.text)
        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name=tc.name,
                )
        # DONE always carries the full LLMResponse so the loop can
        # consume usage stats and provider payloads uniformly.
        yield StreamEvent(type=StreamEventType.DONE, raw=response)
