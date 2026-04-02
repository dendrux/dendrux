"""Shared observer notification helpers for loop implementations.

These fire observer callbacks at the exact points where history mutates
and provider.complete() returns. Observer failures are logged but never
fatal — observability must not kill agent runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.loops.base import LoopObserver
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult

logger = logging.getLogger(__name__)


async def notify_message(
    observer: LoopObserver | None,
    message: Message,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify observer of a message append, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_message_appended(message, iteration)
    except Exception:
        logger.warning("Observer.on_message_appended failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_message_appended failed at iteration {iteration}")


async def notify_llm(
    observer: LoopObserver | None,
    response: LLMResponse,
    iteration: int,
    warnings: list[str] | None = None,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
    duration_ms: int | None = None,
) -> None:
    """Notify observer of an LLM call completion, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_llm_call_completed(
            response,
            iteration,
            semantic_messages=semantic_messages,
            semantic_tools=semantic_tools,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.warning("Observer.on_llm_call_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_completed failed at iteration {iteration}")


async def notify_tool(
    observer: LoopObserver | None,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify observer of a tool completion, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_tool_completed(tool_call, tool_result, iteration)
    except Exception:
        logger.warning("Observer.on_tool_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_tool_completed failed at iteration {iteration}")
