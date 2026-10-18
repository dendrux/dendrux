"""Mock LLM provider for deterministic testing.

Returns predetermined responses in order. No API calls, no network,
no API key needed. Every test in Dendrux uses this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux.llm.base import LLMProvider
from dendrux.types import ProviderCapabilities

if TYPE_CHECKING:
    from dendrux.llm.base import ImageSupport
    from dendrux.types import LLMResponse, Message, ToolDef


class MockLLM(LLMProvider):
    """Returns predetermined responses for testing.

    Usage:
        llm = MockLLM([
            LLMResponse(text=None, tool_calls=[ToolCall("add", {"a": 1, "b": 2})]),
            LLMResponse(text="The answer is 3"),
        ])
        response = await llm.complete(messages)  # returns first response
        response = await llm.complete(messages)  # returns second response

    ``image_support`` is what :meth:`image_input_support` reports (default
    ``"yes"``); ``"no"`` also turns ``supports_multimodal`` off, like a custom
    provider that never opted in. ``wire_messages`` records the exact message
    lists each call received (image parts carry their wire bytes there).
    """

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        model: str = "mock",
        image_support: ImageSupport = "yes",
    ) -> None:
        if image_support not in ("yes", "no", "unknown"):
            raise ValueError(f"image_support must be yes/no/unknown, got {image_support!r}")
        self._responses = list(responses)
        self._model = model
        self._call_count = 0
        self._image_support: ImageSupport = image_support
        self.capabilities = ProviderCapabilities(supports_multimodal=image_support != "no")
        self.call_history: list[dict[str, Any]] = []
        self.wire_messages: list[list[Message]] = []

    @property
    def model(self) -> str:
        """The model identifier (default ``"mock"``)."""
        return self._model

    async def image_input_support(self, model: str) -> ImageSupport:
        """The configured ``image_support`` (for any model)."""
        return self._image_support

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
        """Return the next predetermined response.

        Raises:
            IndexError: If all responses have been consumed.
        """
        if self._call_count >= len(self._responses):
            msg = (
                f"MockLLM exhausted: {self._call_count} calls made "
                f"but only {len(self._responses)} responses provided"
            )
            raise IndexError(msg)

        self.wire_messages.append(list(messages))
        self.call_history.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools else tools,
                "output_schema": output_schema,
                "run_id": run_id,
                "cache_key_prefix": cache_key_prefix,
                "kwargs": kwargs,
            }
        )

        response = self._responses[self._call_count]
        self._call_count += 1
        # Mirror real providers: honor a per-call ``model=`` override.
        response.model = kwargs.get("model") or self._model
        return response

    @property
    def calls_made(self) -> int:
        """Number of complete() calls made so far."""
        return self._call_count

    @property
    def exhausted(self) -> bool:
        """True if all responses have been consumed."""
        return self._call_count >= len(self._responses)
