"""Shared notification helpers for loop implementations.

Two seams with different failure contracts:

  record_* — call LoopRecorder (internal persistence). Exceptions PROPAGATE.
             If persistence fails, the run stops.

  notify_* — call LoopNotifier (best-effort notifications). Exceptions SWALLOWED.
             Console printing, Slack, SSE — if they fail, the run continues.

At each event point, the loop calls record first, then notify. Every helper
takes ``run_id`` as the first argument after the recorder/notifier so a
shared instance can disambiguate concurrent runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.guardrails._engine import GuardrailEngine
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.types import (
        LLMResponse,
        Message,
        RunResult,
        ToolCall,
        ToolDef,
        ToolResult,
    )

logger = logging.getLogger(__name__)


def guardrail_meta(
    g_engine: GuardrailEngine | None,
    notifier_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the RunResult.meta payload that must reach the runner's
    terminal finalize call.

    Two pieces travel this path:

      - ``pii_mapping``: the audit key. Must appear on every terminal
        RunResult — not just the happy path — or the runner's finalize
        call writes ``NULL`` into ``agent_runs.pii_mapping`` and the
        raw traces persisted during the run lose their LLM-eye view.
      - ``notifier_warnings``: best-effort-notifier failures collected
        during the run, surfaced to the caller on RunResult.meta.
    """
    meta: dict[str, Any] = {}
    if g_engine is not None:
        meta["pii_mapping"] = g_engine.get_pii_mapping()
    if notifier_warnings:
        meta["notifier_warnings"] = notifier_warnings
    return meta


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


async def record_run_started(
    recorder: LoopRecorder | None,
    run_id: str,
    *,
    agent_name: str | None = None,
    agent_model: str | None = None,
) -> None:
    """Record run start to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_run_started(run_id, agent_name=agent_name, agent_model=agent_model)


async def record_run_finished(
    recorder: LoopRecorder | None,
    run_id: str,
    result: RunResult,
) -> None:
    """Record run finish (any non-error terminal). Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_run_finished(run_id, result)


async def record_run_failed(
    recorder: LoopRecorder | None,
    run_id: str,
    error: BaseException,
    *,
    iteration: int | None = None,
) -> None:
    """Record run failure (unhandled exception). Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_run_failed(run_id, error, iteration=iteration)


async def record_message(
    recorder: LoopRecorder | None,
    run_id: str,
    message: Message,
    iteration: int,
) -> None:
    """Record message to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_message_appended(run_id, message, iteration)


async def record_llm_started(
    recorder: LoopRecorder | None,
    run_id: str,
    iteration: int,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
) -> None:
    """Record LLM call start to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_llm_call_started(
        run_id,
        iteration,
        semantic_messages=semantic_messages,
        semantic_tools=semantic_tools,
    )


async def record_llm(
    recorder: LoopRecorder | None,
    run_id: str,
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
        run_id,
        response,
        iteration,
        semantic_messages=semantic_messages,
        semantic_tools=semantic_tools,
        duration_ms=duration_ms,
        guardrail_findings=guardrail_findings,
    )


async def record_llm_failed(
    recorder: LoopRecorder | None,
    run_id: str,
    iteration: int,
    error: BaseException,
    *,
    duration_ms: int | None = None,
) -> None:
    """Record LLM call failure (provider exception). Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_llm_call_failed(run_id, iteration, error, duration_ms=duration_ms)


async def record_tool_started(
    recorder: LoopRecorder | None,
    run_id: str,
    tool_call: ToolCall,
    iteration: int,
) -> None:
    """Record tool dispatch start to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_tool_started(run_id, tool_call, iteration)


async def record_tool(
    recorder: LoopRecorder | None,
    run_id: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
) -> None:
    """Record tool completion to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_tool_completed(run_id, tool_call, tool_result, iteration)


async def record_governance(
    recorder: LoopRecorder | None,
    run_id: str,
    event_type: str,
    iteration: int,
    data: dict[str, Any],
    *,
    correlation_id: str | None = None,
) -> None:
    """Record governance event to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_governance_event(
        run_id, event_type, iteration, data, correlation_id=correlation_id
    )


# ------------------------------------------------------------------
# Notifier helpers — best-effort (exceptions swallowed)
# ------------------------------------------------------------------


async def notify_run_started(
    notifier: LoopNotifier | None,
    run_id: str,
    *,
    agent_name: str | None = None,
    agent_model: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Notify of run start, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_run_started(run_id, agent_name=agent_name, agent_model=agent_model)
    except Exception:
        logger.warning("Notifier.on_run_started failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_run_started failed for run {run_id}")


async def notify_run_finished(
    notifier: LoopNotifier | None,
    run_id: str,
    result: RunResult,
    warnings: list[str] | None = None,
) -> None:
    """Notify of run finish, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_run_finished(run_id, result)
    except Exception:
        logger.warning("Notifier.on_run_finished failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_run_finished failed for run {run_id}")


async def notify_run_failed(
    notifier: LoopNotifier | None,
    run_id: str,
    error: BaseException,
    *,
    iteration: int | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Notify of run failure, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_run_failed(run_id, error, iteration=iteration)
    except Exception:
        logger.warning("Notifier.on_run_failed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_run_failed failed for run {run_id}")


async def notify_message(
    notifier: LoopNotifier | None,
    run_id: str,
    message: Message,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify notifier of a message append, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_message_appended(run_id, message, iteration)
    except Exception:
        logger.warning("Notifier.on_message_appended failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_message_appended failed at iteration {iteration}")


async def notify_llm_started(
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    warnings: list[str] | None = None,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
) -> None:
    """Notify of an LLM call start, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_llm_call_started(
            run_id,
            iteration,
            semantic_messages=semantic_messages,
            semantic_tools=semantic_tools,
        )
    except Exception:
        logger.warning("Notifier.on_llm_call_started failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_started failed at iteration {iteration}")


async def notify_llm(
    notifier: LoopNotifier | None,
    run_id: str,
    response: LLMResponse,
    iteration: int,
    warnings: list[str] | None = None,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
    duration_ms: int | None = None,
    guardrail_findings: dict[str, Any] | None = None,
) -> None:
    """Notify notifier of an LLM call completion, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_llm_call_completed(
            run_id,
            response,
            iteration,
            semantic_messages=semantic_messages,
            semantic_tools=semantic_tools,
            duration_ms=duration_ms,
            guardrail_findings=guardrail_findings,
        )
    except Exception:
        logger.warning("Notifier.on_llm_call_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_completed failed at iteration {iteration}")


async def notify_llm_failed(
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    error: BaseException,
    warnings: list[str] | None = None,
    *,
    duration_ms: int | None = None,
) -> None:
    """Notify of an LLM call failure, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_llm_call_failed(run_id, iteration, error, duration_ms=duration_ms)
    except Exception:
        logger.warning("Notifier.on_llm_call_failed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_failed failed at iteration {iteration}")


async def notify_tool_started(
    notifier: LoopNotifier | None,
    run_id: str,
    tool_call: ToolCall,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify of tool dispatch start, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_tool_started(run_id, tool_call, iteration)
    except Exception:
        logger.warning("Notifier.on_tool_started failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_tool_started failed at iteration {iteration}")


async def notify_tool(
    notifier: LoopNotifier | None,
    run_id: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify notifier of a tool completion, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_tool_completed(run_id, tool_call, tool_result, iteration)
    except Exception:
        logger.warning("Notifier.on_tool_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_tool_completed failed at iteration {iteration}")


async def notify_governance(
    notifier: LoopNotifier | None,
    run_id: str,
    event_type: str,
    iteration: int,
    data: dict[str, Any],
    *,
    correlation_id: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Notify notifier of a governance event, swallowing exceptions."""
    if notifier is None:
        return
    try:
        await notifier.on_governance_event(
            run_id, event_type, iteration, data, correlation_id=correlation_id
        )
    except Exception:
        logger.warning("Notifier.on_governance_event failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_governance_event failed at iteration {iteration}")
