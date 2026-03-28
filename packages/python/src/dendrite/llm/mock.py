"""Mock LLM provider for deterministic testing.

Returns predetermined responses in order. No API calls, no network,
no API key needed. Every test in Dendrite uses this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrite.llm.base import LLMProvider

if TYPE_CHECKING:
    from dendrite.types import LLMResponse, Message, ToolDef


class MockLLM(LLMProvider):
    """Returns predetermined responses for testing.

    Usage:
        llm = MockLLM([
            LLMResponse(text=None, tool_calls=[ToolCall("add", {"a": 1, "b": 2})]),
            LLMResponse(text="The answer is 3"),
        ])
        response = await llm.complete(messages)  # returns first response
        response = await llm.complete(messages)  # returns second response
    """

    def __init__(self, responses: list[LLMResponse], *, model: str = "mock") -> None:
        self._responses = list(responses)
        self._model = model
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        """The model identifier (default ``"mock"``)."""
        return self._model

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
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

        self.call_history.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools else tools,
                "kwargs": kwargs,
            }
        )

        response = self._responses[self._call_count]
        self._call_count += 1
        return response

    @property
    def calls_made(self) -> int:
        """Number of complete() calls made so far."""
        return self._call_count

    @property
    def exhausted(self) -> bool:
        """True if all responses have been consumed."""
        return self._call_count >= len(self._responses)
