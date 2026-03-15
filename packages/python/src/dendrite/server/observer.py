"""Server-side observers for transport and composition.

CompositeObserver fans out to multiple observers (persistence + transport).
TransportObserver pushes events to an asyncio.Queue for SSE consumption.
"""

from __future__ import annotations

import asyncio  # noqa: TC003 — used at runtime in Queue type hints
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dendrite.loops.base import LoopObserver

if TYPE_CHECKING:
    from dendrite.types import LLMResponse, Message, ToolCall, ToolResult


logger = logging.getLogger(__name__)


class CompositeObserver(LoopObserver):
    """Fans out loop events to multiple observers.

    The loop sees one observer; CompositeObserver dispatches to all
    registered observers. If one fails, the others still fire.
    """

    def __init__(self, observers: list[LoopObserver]) -> None:
        self._observers = list(observers)

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        for obs in self._observers:
            try:
                await obs.on_message_appended(message, iteration)
            except Exception:
                logger.warning("CompositeObserver: on_message_appended failed", exc_info=True)

    async def on_llm_call_completed(self, response: LLMResponse, iteration: int) -> None:
        for obs in self._observers:
            try:
                await obs.on_llm_call_completed(response, iteration)
            except Exception:
                logger.warning("CompositeObserver: on_llm_call_completed failed", exc_info=True)

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        for obs in self._observers:
            try:
                await obs.on_tool_completed(tool_call, tool_result, iteration)
            except Exception:
                logger.warning("CompositeObserver: on_tool_completed failed", exc_info=True)


@dataclass
class ServerEvent:
    """An event pushed to the SSE queue."""

    event: str  # e.g., "run.started", "run.step", "run.paused", "run.completed"
    data: dict[str, Any] = field(default_factory=dict)


class TransportObserver(LoopObserver):
    """Pushes loop events onto an asyncio.Queue for SSE streaming.

    The SSE endpoint reads from this queue and yields Server-Sent Events.
    Terminal events (run.completed, run.error, run.paused) are also buffered
    so late SSE subscribers can receive them.
    """

    def __init__(self, queue: asyncio.Queue[ServerEvent]) -> None:
        self._queue = queue

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        await self._queue.put(
            ServerEvent(
                event="run.step",
                data={
                    "role": message.role.value,
                    "content": message.content[:500],  # Truncate for SSE
                    "iteration": iteration,
                },
            )
        )

    async def on_llm_call_completed(self, response: LLMResponse, iteration: int) -> None:
        await self._queue.put(
            ServerEvent(
                event="run.llm_done",
                data={
                    "iteration": iteration,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        )

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        await self._queue.put(
            ServerEvent(
                event="run.tool_done",
                data={
                    "tool_name": tool_call.name,
                    "success": tool_result.success,
                    "iteration": iteration,
                },
            )
        )
