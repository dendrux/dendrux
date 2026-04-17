"""Composite notifier — fans out to multiple notifiers.

Lives in the core notifiers package (no FastAPI dependency) so it
can be used in script-mode runs without the ``http`` extra.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import LoopNotifier

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult

logger = logging.getLogger(__name__)


class CompositeNotifier(LoopNotifier):
    """Fans out loop events to multiple notifiers.

    The loop sees one notifier; CompositeNotifier dispatches to all
    registered notifiers. If one fails, the others still fire.
    """

    def __init__(self, notifiers: list[LoopNotifier]) -> None:
        self._notifiers = list(notifiers)

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_message_appended(message, iteration)
            except Exception:
                logger.warning("CompositeNotifier: on_message_appended failed", exc_info=True)

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_llm_call_completed(
                    response,
                    iteration,
                    semantic_messages=semantic_messages,
                    semantic_tools=semantic_tools,
                    duration_ms=duration_ms,
                )
            except Exception:
                logger.warning("CompositeNotifier: on_llm_call_completed failed", exc_info=True)

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_tool_completed(tool_call, tool_result, iteration)
            except Exception:
                logger.warning("CompositeNotifier: on_tool_completed failed", exc_info=True)

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_governance_event(
                    event_type, iteration, data, correlation_id=correlation_id
                )
            except Exception:
                logger.warning("CompositeNotifier: on_governance_event failed", exc_info=True)
