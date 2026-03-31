"""Bridge observers for transport and composition.

CompositeObserver fans out to multiple observers (persistence + transport).
TransportObserver pushes events to an asyncio.Queue for SSE consumption.
"""

from __future__ import annotations

import asyncio  # noqa: TC003 — used at runtime in Queue type hints
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import LoopObserver

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.types import LLMResponse, Message, ToolCall, ToolResult


logger = logging.getLogger(__name__)


# CompositeObserver lives in dendrux.observers.composite (no bridge dependency).
# Re-exported here for backward compatibility with bridge internals.
from dendrux.observers.composite import CompositeObserver


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

    Args:
        queue: The asyncio.Queue to push events to.
        redact: Optional redaction function. When provided, message content
            is scrubbed before being pushed to the SSE queue — ensuring
            the transport path has the same redaction posture as persistence.
    """

    def __init__(
        self,
        queue: asyncio.Queue[ServerEvent],
        *,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self._queue = queue
        self._redact = redact

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        content = message.content[:500]
        if self._redact:
            content = self._redact(content)
        await self._queue.put(
            ServerEvent(
                event="run.step",
                data={
                    "role": message.role.value,
                    "content": content,
                    "iteration": iteration,
                },
            )
        )

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: Any | None = None,
        duration_ms: int | None = None,
    ) -> None:
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
