"""Shared notification helpers for loop implementations.

Two seams with different failure contracts:

  record_* — call LoopRecorder (internal persistence). Exceptions PROPAGATE.
             If persistence fails, the run stops.

  notify_* — call LoopNotifier (best-effort notifications). Exceptions SWALLOWED.
             Console printing, Slack, SSE — if they fail, the run continues.

At each event point, the loop calls record first, then notify.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult

logger = logging.getLogger(__name__)


def build_cache_key_prefix(agent: Agent) -> str | None:
    """Stable cache-pool identifier per agent + model.

    OpenAI providers use this to scope ``prompt_cache_key`` so all runs of
    the same agent share a cache pool. Returns ``None`` when the agent has
    no name; the provider then falls back to ``run_id``.
    """
    if not agent.name:
        return None
    return f"{agent.name}:{agent.model}"


# ------------------------------------------------------------------
# Recorder helpers — fail-closed (exceptions propagate)
# ------------------------------------------------------------------


async def record_message(
    recorder: LoopRecorder | None,
    message: Message,
    iteration: int,
) -> None:
    """Record message to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_message_appended(message, iteration)


async def record_llm(
    recorder: LoopRecorder | None,
    response: LLMResponse,
    iteration: int,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
    duration_ms: int | None = None,
    guardrail_findings: dict[str, Any] | None = None,
) -> None:
    """Record LLM completion to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_llm_call_completed(
        response,
        iteration,
        semantic_messages=semantic_messages,
        semantic_tools=semantic_tools,
        duration_ms=duration_ms,
        guardrail_findings=guardrail_findings,
    )


async def record_tool(
    recorder: LoopRecorder | None,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
) -> None:
    """Record tool completion to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_tool_completed(tool_call, tool_result, iteration)


async def record_governance(
    recorder: LoopRecorder | None,
    event_type: str,
    iteration: int,
    data: dict[str, Any],
    correlation_id: str | None = None,
) -> None:
    """Record governance event to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_governance_event(event_type, iteration, data, correlation_id=correlation_id)


# ------------------------------------------------------------------
# Notifier helpers — best-effort (exceptions swallowed)
# ------------------------------------------------------------------


async def notify_message(
    notifier: LoopNotifier | None,
    message: Message,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify notifier of a message append, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_message_appended(message, iteration)
    except Exception:
        logger.warning("Notifier.on_message_appended failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_message_appended failed at iteration {iteration}")


async def notify_llm(
    notifier: LoopNotifier | None,
    response: LLMResponse,
    iteration: int,
    warnings: list[str] | None = None,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
    duration_ms: int | None = None,
) -> None:
    """Notify notifier of an LLM call completion, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_llm_call_completed(
            response,
            iteration,
            semantic_messages=semantic_messages,
            semantic_tools=semantic_tools,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.warning("Notifier.on_llm_call_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_completed failed at iteration {iteration}")


async def notify_tool(
    notifier: LoopNotifier | None,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify notifier of a tool completion, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_tool_completed(tool_call, tool_result, iteration)
    except Exception:
        logger.warning("Notifier.on_tool_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_tool_completed failed at iteration {iteration}")


async def notify_governance(
    notifier: LoopNotifier | None,
    event_type: str,
    iteration: int,
    data: dict[str, Any],
    correlation_id: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Notify notifier of a governance event, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_governance_event(
            event_type, iteration, data, correlation_id=correlation_id
        )
    except Exception:
        logger.warning("Notifier.on_governance_event failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_governance_event failed at iteration {iteration}")
