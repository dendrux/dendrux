"""LLM provider base class.

Defines the contract that all LLM providers must implement.
Provides a default complete_stream() fallback that wraps complete().
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

from dendrux.types import ProviderCapabilities, StreamEvent, StreamEventType

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrux.types import LLMResponse, Message, ToolDef

ImageSupport = Literal["yes", "no", "unknown"]
ImageCapabilitySource = Literal["capability", "catalog", "adapter"]

# Guards every provider instance's learned-capability set. Providers are shared
# across concurrent runs, threads and event loops; learns are rare, so one lock
# is enough.
_LEARNED_LOCK = threading.Lock()
_LEARNED_ATTR = "_dendrux_learned_image_no"


def _typed_complete(method: Any) -> Any:
    @wraps(method)
    async def complete(*args: Any, **kwargs: Any) -> Any:
        from dendrux.errors import provider_error_in

        try:
            return await method(*args, **kwargs)
        except ExceptionGroup as group:
            failure = provider_error_in(group)
            if failure is not None:
                raise failure from None
            raise

    return complete


def _typed_stream(method: Any) -> Any:
    @wraps(method)
    async def stream(*args: Any, **kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        from dendrux.errors import provider_error_in

        source = method(*args, **kwargs)
        try:
            async for event in source:
                yield event
        except ExceptionGroup as group:
            failure = provider_error_in(group)
            if failure is not None:
                raise failure from None
            raise
        finally:
            await _close_typed_stream(source)

    return stream


async def _close_typed_stream(source: Any) -> None:
    from dendrux.errors import provider_error_in

    try:
        await source.aclose()
    except ExceptionGroup as group:
        failure = provider_error_in(group)
        if failure is not None:
            raise failure from None
        raise


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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, wrapper in (("complete", _typed_complete), ("complete_stream", _typed_stream)):
            if name in cls.__dict__:
                setattr(cls, name, wrapper(cls.__dict__[name]))

    capabilities: ProviderCapabilities = ProviderCapabilities()

    tool_result_images: Literal["native", "portable"] = "portable"
    """How images in tool results reach the model: inside the vendor's
    tool-result block (``"native"``) or as a text tool message plus one framed
    user message after the tool block (``"portable"``)."""

    # ------------------------------------------------------------------
    # Image input (0.4.0)
    # ------------------------------------------------------------------

    async def image_input_support(self, model: str) -> ImageSupport:
        """Whether ``model`` accepts image input on this adapter: yes / no / unknown.

        Default: ``"unknown"`` when the adapter converts image parts
        (``capabilities.supports_multimodal``), else ``"no"`` — a custom
        provider that never opted in can never silently drop image parts.
        """
        return "unknown" if self.capabilities.supports_multimodal else "no"

    def image_capability_source(self) -> ImageCapabilitySource:
        """What :meth:`image_input_support` is based on, for typed errors.

        ``"capability"`` (static table), ``"catalog"`` (a provider model
        catalogue) or ``"adapter"`` (the adapter cannot convert images).
        """
        return "capability" if self.capabilities.supports_multimodal else "adapter"

    @property
    def thinking_enabled(self) -> bool:
        """True when this provider sends thinking/reasoning blocks by default."""
        return False

    @property
    def reports_cost(self) -> bool:
        """True when responses carry the vendor's own cost (``UsageStats.cost_usd``).

        A dollar budget needs either this or a ``PriceTable`` entry.
        """
        return False

    def many_image_rule(self, model: str) -> tuple[int, int] | None:
        """``(max_images, max_long_side)`` rule for many-image requests, or None.

        When a request carries more than ``max_images`` visible images, every
        image must have a long side of at most ``max_long_side`` pixels.
        """
        return None

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        """``(api_surface, normalised base URL, model)`` keying the learned cache."""
        return (type(self).__name__, "", model)

    def learned_image_support(self, model: str) -> Literal["no"] | None:
        """``"no"`` when this instance learned that ``model`` rejects images, else None.

        Learned only from a vendor "image input unsupported" response on this
        instance; process lifetime, never shared or persisted.
        """
        learned: set[tuple[str, str, str]] | None = self.__dict__.get(_LEARNED_ATTR)
        if not learned:
            return None
        key = self.image_capability_key(model)
        with _LEARNED_LOCK:
            return "no" if key in learned else None

    def learn_image_unsupported(self, model: str) -> None:
        """Record that ``model`` rejected image input (called by adapters on
        an ``IMAGE_UNSUPPORTED`` classification). Thread-safe and idempotent."""
        key = self.image_capability_key(model)
        with _LEARNED_LOCK:
            learned = self.__dict__.setdefault(_LEARNED_ATTR, set())
            learned.add(key)

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
            Set ``stop_reason`` (normalized :data:`~dendrux.types.StopReason`)
            and ``raw_stop_reason`` when the vendor reports why the call
            stopped; leave both ``None`` otherwise. ``"max_tokens"`` marks a
            run's final answer as truncated (``RunResult.truncated``).
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
