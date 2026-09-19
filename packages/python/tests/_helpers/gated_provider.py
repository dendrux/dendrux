"""A streaming provider whose output can be paused mid-stream.

Used by stream-interruption tests to hold a run inside an in-flight
provider stream deterministically: the provider emits ``first``, sets
``first_sent``, then blocks on ``gate`` until the test releases it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from dendrux.llm.base import LLMProvider
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    StreamEvent,
    StreamEventType,
    UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class GatedStreamLLM(LLMProvider):
    """Streams ``before`` responses instantly, then one gated response.

    The gated response yields ``first`` as a text delta, sets
    ``first_sent``, waits on ``gate``, then yields ``second`` and DONE.

    ``closed`` flips when the gated stream's generator is finalized;
    ``cancelled`` flips if a ``CancelledError`` was thrown into it.
    """

    capabilities = ProviderCapabilities(supports_native_tools=True, supports_streaming=True)

    def __init__(
        self,
        *,
        first: str = "First sentence. ",
        second: str = "Second sentence.",
        before: list[LLMResponse] | None = None,
    ) -> None:
        self.first = first
        self.second = second
        self._before = list(before or [])
        self.gate = asyncio.Event()
        self.first_sent = asyncio.Event()
        self.closed = False
        self.cancelled = False
        self.call_count = 0

    @property
    def model(self) -> str:
        return "gated-stream"

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        if self._before:
            return self._before.pop(0)
        return LLMResponse(text=self.first + self.second)

    async def complete_stream(
        self, messages: Any, tools: Any = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        self.call_count += 1
        if self._before:
            resp = self._before.pop(0)
            if resp.text:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=resp.text)
            for tc in resp.tool_calls or []:
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_name=tc.name,
                    tool_call_id=tc.provider_tool_call_id or tc.id,
                )
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name=tc.name,
                    tool_call_id=tc.provider_tool_call_id or tc.id,
                )
            yield StreamEvent(type=StreamEventType.DONE, raw=resp)
            return

        try:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=self.first)
            self.first_sent.set()
            await self.gate.wait()
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=self.second)
            yield StreamEvent(
                type=StreamEventType.DONE,
                raw=LLMResponse(
                    text=self.first + self.second,
                    usage=UsageStats(input_tokens=10, output_tokens=8, total_tokens=18),
                ),
            )
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.closed = True
