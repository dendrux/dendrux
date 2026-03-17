"""ReAct loop — think, act, observe, repeat.

The core agent execution loop. Orchestrates the cycle of:
  1. Strategy builds messages
  2. Provider calls the LLM
  3. Strategy parses the response into an AgentStep
  4. If ToolCall → execute → format result → append → repeat
  5. If Finish → return RunResult
  6. If max_iterations → return RunResult with MAX_ITERATIONS

The loop never touches provider-specific APIs or prompt formatting.
It operates entirely on Dendrite's universal types.

Observer hooks fire at three kinds of points:
  - After each history.append() → on_message_appended
  - After provider.complete() → on_llm_call_completed
  - After _execute_tool() → on_tool_completed
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from dendrite.loops.base import Loop
from dendrite.tool import get_tool_def
from dendrite.types import (
    AgentStep,
    Clarification,
    Finish,
    Message,
    PauseState,
    Role,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolTarget,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.loops.base import LoopObserver
    from dendrite.strategies.base import Strategy
    from dendrite.types import LLMResponse, ToolDef

logger = logging.getLogger(__name__)


async def _notify_message(
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


async def _notify_llm(
    observer: LoopObserver | None,
    response: LLMResponse,
    iteration: int,
    warnings: list[str] | None = None,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
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
        )
    except Exception:
        logger.warning("Observer.on_llm_call_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_completed failed at iteration {iteration}")


async def _notify_tool(
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


class ReActLoop(Loop):
    """Think → act → observe → repeat.

    The default loop for most agent tasks. Runs until the strategy returns
    a Finish action or max_iterations is reached.

    Usage:
        loop = ReActLoop()
        result = await loop.run(
            agent=agent,
            provider=provider,
            strategy=strategy,
            user_input="What is 15 + 27?",
        )
    """

    async def run(
        self,
        *,
        agent: Agent,
        provider: LLMProvider,
        strategy: Strategy,
        user_input: str,
        run_id: str | None = None,
        observer: LoopObserver | None = None,
        initial_history: list[Message] | None = None,
        initial_steps: list[AgentStep] | None = None,
        iteration_offset: int = 0,
        initial_usage: UsageStats | None = None,
    ) -> RunResult:
        """Execute the ReAct loop, optionally resuming from a pause."""
        resolved_run_id = run_id or generate_ulid()
        tool_defs = agent.get_tool_defs()
        tool_lookup, target_lookup, timeout_lookup = _build_tool_lookup(agent.tools)

        if initial_history is not None:
            # Resume from pause — use provided history, skip user message creation
            history = list(initial_history)
        else:
            # Fresh run — create initial user message
            user_msg = Message(role=Role.USER, content=user_input)
            history = [user_msg]
            await _notify_message(observer, user_msg, 0)

        steps: list[AgentStep] = list(initial_steps) if initial_steps else []
        total_usage = UsageStats(
            input_tokens=initial_usage.input_tokens if initial_usage else 0,
            output_tokens=initial_usage.output_tokens if initial_usage else 0,
            total_tokens=initial_usage.total_tokens if initial_usage else 0,
            cost_usd=initial_usage.cost_usd if initial_usage else None,
        )

        observer_warnings: list[str] = []

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            # 1. Build messages via strategy
            messages, tools = strategy.build_messages(
                system_prompt=agent.prompt,
                history=history,
                tool_defs=tool_defs,
            )

            # 2. Call the LLM
            response = await provider.complete(messages, tools=tools)
            await _notify_llm(
                observer,
                response,
                iteration,
                observer_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
            )

            # Accumulate usage
            total_usage.input_tokens += response.usage.input_tokens
            total_usage.output_tokens += response.usage.output_tokens
            total_usage.total_tokens += response.usage.total_tokens
            if response.usage.cost_usd is not None:
                if total_usage.cost_usd is None:
                    total_usage.cost_usd = 0.0
                total_usage.cost_usd += response.usage.cost_usd

            # 3. Parse response into AgentStep
            step = strategy.parse_response(response)
            steps.append(step)

            # 4. Check action type
            if isinstance(step.action, Finish):
                # Persist the final assistant message before returning
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
                history.append(assistant_msg)
                await _notify_message(observer, assistant_msg, iteration, observer_warnings)

                meta = {}
                if observer_warnings:
                    meta["observer_warnings"] = observer_warnings
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.SUCCESS,
                    answer=step.action.answer,
                    steps=steps,
                    iteration_count=iteration,
                    usage=total_usage,
                    meta=meta,
                )

            if isinstance(step.action, Clarification):
                # Persist the assistant message before returning
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
                history.append(assistant_msg)
                await _notify_message(observer, assistant_msg, iteration, observer_warnings)

                # Build PauseState for resume via resume_with_input()
                clarification_pause = PauseState(
                    agent_name=agent.name,
                    pending_tool_calls=[],  # No tool calls — resume is a user message
                    history=list(history),
                    steps=list(steps),
                    iteration=iteration,
                    trace_order_offset=len(history),
                    usage=UsageStats(
                        input_tokens=total_usage.input_tokens,
                        output_tokens=total_usage.output_tokens,
                        total_tokens=total_usage.total_tokens,
                        cost_usd=total_usage.cost_usd,
                    ),
                )

                clarification_meta: dict[str, Any] = {"pause_state": clarification_pause}
                if observer_warnings:
                    clarification_meta["observer_warnings"] = observer_warnings
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.WAITING_HUMAN_INPUT,
                    answer=step.action.question,
                    steps=steps,
                    iteration_count=iteration,
                    usage=total_usage,
                    meta=clarification_meta,
                )

            if isinstance(step.action, ToolCall):
                # Append assistant message with all tool_calls to history
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
                history.append(assistant_msg)
                await _notify_message(observer, assistant_msg, iteration, observer_warnings)

                # Split tool calls into server (execute now) vs non-server (pause)
                all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
                server_calls: list[ToolCall] = []
                pending_calls: list[ToolCall] = []
                for tc in all_calls:
                    target = target_lookup.get(tc.name, ToolTarget.SERVER)
                    if target == ToolTarget.SERVER:
                        server_calls.append(tc)
                    else:
                        pending_calls.append(tc)

                # Execute server tools immediately
                for tc in server_calls:
                    tool_result = await _execute_tool(
                        tc, tool_lookup, target_lookup, timeout_lookup
                    )
                    await _notify_tool(observer, tc, tool_result, iteration, observer_warnings)
                    result_msg = strategy.format_tool_result(tool_result)
                    history.append(result_msg)
                    await _notify_message(observer, result_msg, iteration, observer_warnings)

                # If non-server tools remain, pause the loop
                if pending_calls:
                    # Approximate trace offset from history length. This can
                    # diverge from the actual DB trace count if observer
                    # notifications were swallowed (see _notify_message).
                    # Group 2's resume() must load the real max order_index
                    # from the DB, not trust this value blindly.
                    trace_offset = len(history)

                    # Build target map for pending calls
                    pending_targets = {
                        tc.id: target_lookup.get(tc.name, ToolTarget.SERVER).value
                        for tc in pending_calls
                    }

                    pause_state = PauseState(
                        agent_name=agent.name,
                        pending_tool_calls=pending_calls,
                        pending_targets=pending_targets,
                        history=list(history),
                        steps=list(steps),
                        iteration=iteration,
                        trace_order_offset=trace_offset,
                        usage=UsageStats(
                            input_tokens=total_usage.input_tokens,
                            output_tokens=total_usage.output_tokens,
                            total_tokens=total_usage.total_tokens,
                            cost_usd=total_usage.cost_usd,
                        ),
                    )

                    pause_meta: dict[str, Any] = {"pause_state": pause_state}
                    if observer_warnings:
                        pause_meta["observer_warnings"] = observer_warnings
                    return RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.WAITING_CLIENT_TOOL,
                        steps=steps,
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=pause_meta,
                    )

        # Max iterations reached
        meta = {}
        if observer_warnings:
            meta["observer_warnings"] = observer_warnings
        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.MAX_ITERATIONS,
            steps=steps,
            iteration_count=agent.max_iterations,
            usage=total_usage,
            meta=meta,
        )


def _build_tool_lookup(
    tools: list[Callable[..., Any]],
) -> tuple[dict[str, Callable[..., Any]], dict[str, ToolTarget], dict[str, float]]:
    """Build name → function, name → target, and name → timeout lookups."""
    fn_lookup: dict[str, Callable[..., Any]] = {}
    target_lookup: dict[str, ToolTarget] = {}
    timeout_lookup: dict[str, float] = {}
    for fn in tools:
        td = get_tool_def(fn)
        if td.name in fn_lookup:
            raise ValueError(
                f"Duplicate tool name '{td.name}'. "
                f"Each tool registered on an agent must have a unique name."
            )
        fn_lookup[td.name] = fn
        target_lookup[td.name] = td.target
        timeout_lookup[td.name] = td.timeout_seconds
    return fn_lookup, target_lookup, timeout_lookup


_DEFAULT_TOOL_TIMEOUT = 30.0


async def _execute_tool(
    tool_call: ToolCall,
    tool_lookup: dict[str, Callable[..., Any]],
    target_lookup: dict[str, ToolTarget] | None = None,
    timeout_lookup: dict[str, float] | None = None,
) -> ToolResult:
    """Execute a server tool function and return a ToolResult.

    Non-server tools are filtered out by the loop before reaching this
    function. They trigger a pause instead of execution.
    """
    fn = tool_lookup.get(tool_call.name)
    if fn is None:
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    timeout_s = _DEFAULT_TOOL_TIMEOUT
    if timeout_lookup is not None:
        timeout_s = timeout_lookup.get(tool_call.name, _DEFAULT_TOOL_TIMEOUT)

    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(fn):
            coro = fn(**tool_call.params)
        else:
            coro = asyncio.to_thread(fn, **tool_call.params)
        result = await asyncio.wait_for(coro, timeout=timeout_s)

        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = json.dumps(result, default=str)
        except (TypeError, ValueError):
            payload = json.dumps(str(result))

        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=payload,
            success=True,
            duration_ms=duration_ms,
        )
    except TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        error_msg = f"Tool '{tool_call.name}' timed out after {timeout_s}s"
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": error_msg}),
            success=False,
            error=error_msg,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": str(e)}),
            success=False,
            error=str(e),
            duration_ms=duration_ms,
        )
