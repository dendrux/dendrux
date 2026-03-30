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
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrite.loops.base import Loop
from dendrite.tool import DEFAULT_TOOL_TIMEOUT, get_tool_def
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
        lookups = _build_tool_lookups(agent.tools)

        if initial_history is not None:
            # Resume from pause — use provided history, skip user message creation
            history = list(initial_history)
        else:
            # Fresh run — create initial user message
            user_msg = Message(role=Role.USER, content=user_input)
            history = [user_msg]
            await _notify_message(observer, user_msg, 0)

        # Derive per-tool call counts from history (resume-safe).
        # Counts Role.TOOL messages only — not assistant tool_call requests.
        call_counts: dict[str, int] = {}
        for msg in history:
            if msg.role == Role.TOOL and msg.name is not None:
                call_counts[msg.name] = call_counts.get(msg.name, 0) + 1

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
            t0 = time.monotonic()
            response = await provider.complete(messages, tools=tools)
            llm_duration_ms = int((time.monotonic() - t0) * 1000)
            await _notify_llm(
                observer,
                response,
                iteration,
                observer_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
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

                # Enforce max_calls_per_run before server/client split.
                # Tools that exceed their limit get an immediate graceful result.
                # Uses a provisional counter so duplicate tool calls within a
                # single LLM batch are correctly counted (not just the baseline).
                all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
                allowed_calls: list[ToolCall] = []
                batch_counts: dict[str, int] = {}
                for tc in all_calls:
                    max_calls = lookups.max_calls.get(tc.name)
                    current = call_counts.get(tc.name, 0) + batch_counts.get(tc.name, 0)
                    if max_calls is not None and current >= max_calls:
                        limit_msg = (
                            f"Tool '{tc.name}' has reached its maximum of "
                            f"{max_calls} calls for this run."
                        )
                        limit_result = ToolResult(
                            name=tc.name,
                            call_id=tc.id,
                            payload=json.dumps({"limit": limit_msg}),
                            success=False,
                            error=limit_msg,
                        )
                        await _notify_tool(
                            observer, tc, limit_result, iteration, observer_warnings,
                        )
                        result_msg = strategy.format_tool_result(limit_result)
                        history.append(result_msg)
                        await _notify_message(
                            observer, result_msg, iteration, observer_warnings,
                        )
                    else:
                        batch_counts[tc.name] = batch_counts.get(tc.name, 0) + 1
                        allowed_calls.append(tc)

                # Split allowed calls into server (execute now) vs non-server (pause)
                server_calls: list[ToolCall] = []
                pending_calls: list[ToolCall] = []
                for tc in allowed_calls:
                    target = lookups.target.get(tc.name, ToolTarget.SERVER)
                    if target == ToolTarget.SERVER:
                        server_calls.append(tc)
                    else:
                        pending_calls.append(tc)

                # Execute server tools — parallel where safe, sequential barriers.
                # Group contiguous parallel=True tools for concurrent execution.
                # parallel=False tools act as barriers (run alone in order).
                exec_groups = _build_execution_groups(server_calls, lookups.parallel)
                # Collect (tc, result) pairs in original order for deterministic
                # history append after all groups complete.
                all_results: list[tuple[ToolCall, ToolResult]] = []
                for group, is_parallel in exec_groups:
                    if is_parallel and len(group) > 1:
                        # Run concurrently — each notifies observer on completion
                        pairs = await asyncio.gather(*[
                            _execute_and_notify(
                                tc, lookups, observer, iteration, observer_warnings,
                            )
                            for tc in group
                        ])
                        all_results.extend(pairs)
                    else:
                        # Sequential — single tool or parallel=False barrier
                        for tc in group:
                            tool_result = await _execute_tool(tc, lookups)
                            await _notify_tool(
                                observer, tc, tool_result, iteration, observer_warnings,
                            )
                            all_results.append((tc, tool_result))

                # Append history in original order (deterministic for LLM)
                for tc, tool_result in all_results:
                    call_counts[tc.name] = call_counts.get(tc.name, 0) + 1
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
                        tc.id: lookups.target.get(tc.name, ToolTarget.SERVER).value
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


class ToolLookups(NamedTuple):
    """Pre-computed lookups for tool execution — built once per run."""

    fn: dict[str, Callable[..., Any]]
    target: dict[str, ToolTarget]
    timeout: dict[str, float]
    explicit_timeout: dict[str, bool]
    max_calls: dict[str, int | None]
    parallel: dict[str, bool]


def _build_tool_lookups(tools: list[Callable[..., Any]]) -> ToolLookups:
    """Build all tool lookups from a list of @tool-decorated functions."""
    fn: dict[str, Callable[..., Any]] = {}
    target: dict[str, ToolTarget] = {}
    timeout: dict[str, float] = {}
    explicit_timeout: dict[str, bool] = {}
    max_calls: dict[str, int | None] = {}
    parallel: dict[str, bool] = {}
    for func in tools:
        td = get_tool_def(func)
        if td.name in fn:
            raise ValueError(
                f"Duplicate tool name '{td.name}'. "
                f"Each tool registered on an agent must have a unique name."
            )
        fn[td.name] = func
        target[td.name] = td.target
        timeout[td.name] = td.timeout_seconds
        explicit_timeout[td.name] = td.has_explicit_timeout
        max_calls[td.name] = td.max_calls_per_run
        parallel[td.name] = td.parallel
    return ToolLookups(fn, target, timeout, explicit_timeout, max_calls, parallel)


def _build_execution_groups(
    tool_calls: list[ToolCall],
    parallel_lookup: dict[str, bool],
) -> list[tuple[list[ToolCall], bool]]:
    """Group contiguous tool calls by their parallel flag.

    Scans left to right, building contiguous groups of parallel=True tools.
    A parallel=False tool is a barrier — always forms its own singleton group.

    Returns list of (group, is_parallel) tuples in original order.
    Example: [A(par), B(par), C(nonpar), D(par)] →
             [([A,B], True), ([C], False), ([D], True)]
    Example: [A(seq), B(seq)] →
             [([A], False), ([B], False)]
    """
    if not tool_calls:
        return []

    groups: list[tuple[list[ToolCall], bool]] = []
    current_parallel_group: list[ToolCall] = []

    for tc in tool_calls:
        is_parallel = parallel_lookup.get(tc.name, True)
        if not is_parallel:
            # Flush any accumulated parallel group first
            if current_parallel_group:
                groups.append((current_parallel_group, True))
                current_parallel_group = []
            # Barrier: always a singleton group
            groups.append(([tc], False))
        else:
            current_parallel_group.append(tc)

    # Flush remaining parallel group
    if current_parallel_group:
        groups.append((current_parallel_group, True))

    return groups


async def _execute_and_notify(
    tool_call: ToolCall,
    lookups: ToolLookups,
    observer: LoopObserver | None,
    iteration: int,
    observer_warnings: list[str],
) -> tuple[ToolCall, ToolResult]:
    """Execute a tool and notify observer immediately on completion.

    Used for parallel execution — each tool streams its SSE event
    as soon as it finishes, without waiting for siblings.
    """
    result = await _execute_tool(tool_call, lookups)
    await _notify_tool(observer, tool_call, result, iteration, observer_warnings)
    return tool_call, result


async def _execute_tool(
    tool_call: ToolCall,
    lookups: ToolLookups,
) -> ToolResult:
    """Execute a server tool function and return a ToolResult.

    Non-server tools are filtered out by the loop before reaching this
    function. They trigger a pause instead of execution.
    """
    fn = lookups.fn.get(tool_call.name)
    if fn is None:
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    timeout_s = lookups.timeout.get(tool_call.name, DEFAULT_TOOL_TIMEOUT)
    is_explicit = lookups.explicit_timeout.get(tool_call.name, False)

    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(fn):
            coro = fn(**tool_call.params)
        else:
            coro = asyncio.to_thread(fn, **tool_call.params)
        result = await asyncio.wait_for(coro, timeout=timeout_s)

        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = json.dumps(result)
        except (TypeError, ValueError) as ser_err:
            raise TypeError(
                f"Tool '{tool_call.name}' returned a non-JSON-serializable value: "
                f"{type(result).__name__}. Tool return values must be JSON-serializable "
                f"(str, int, float, bool, list, dict, or None)."
            ) from ser_err

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
        if is_explicit:
            logger.warning("%s", error_msg)
        else:
            logger.warning(
                "%s (default). Set an explicit timeout: @tool(timeout_seconds=N)",
                error_msg,
            )
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
