"""Composite observer — fans out to multiple observers.

Lives in the core observers package (no bridge/transport dependency)
so it can be used in script-mode runs without requiring FastAPI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import LoopObserver

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, ToolCall, ToolResult

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

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: Any | None = None,
        duration_ms: int | None = None,
    ) -> None:
        for obs in self._observers:
            try:
                await obs.on_llm_call_completed(
                    response,
                    iteration,
                    semantic_messages=semantic_messages,
                    semantic_tools=semantic_tools,
                    duration_ms=duration_ms,
                )
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
