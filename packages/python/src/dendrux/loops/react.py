"""ReAct loop — think, act, observe, repeat.

The core agent execution loop. Orchestrates the cycle of:
  1. Strategy builds messages
  2. Provider calls the LLM
  3. Strategy parses the response into an AgentStep
  4. If ToolCall → execute → format result → append → repeat
  5. If Finish → return RunResult
  6. If max_iterations → return RunResult with MAX_ITERATIONS

The loop never touches provider-specific APIs or prompt formatting.
It operates entirely on Dendrux's universal types.

Notifier hooks fire at three kinds of points:
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

from dendrux.loops._helpers import (
    notify_llm,
    notify_message,
    notify_tool,
    record_llm,
    record_message,
    record_tool,
)
from dendrux.loops.base import Loop
from dendrux.tool import DEFAULT_TOOL_TIMEOUT, get_tool_def
from dendrux.types import (
    AgentStep,
    Clarification,
    Finish,
    Message,
    PauseState,
    Role,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
    StreamEventType,
    ToolCall,
    ToolResult,
    ToolTarget,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from pydantic import BaseModel

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.strategies.base import Strategy
    from dendrux.types import LLMResponse

logger = logging.getLogger(__name__)


_record_message = record_message
_record_llm = record_llm
_record_tool = record_tool
_notify_message = notify_message
_notify_llm = notify_llm
_notify_tool = notify_tool


class _ToolCallOutcome(NamedTuple):
    """Result of processing tool calls in a single iteration."""

    # All (tool_call, tool_result) pairs — limits + executed, in order
    all_results: list[tuple[ToolCall, ToolResult]]
    # Non-server tools that need pause (empty if all were server tools)
    pending_calls: list[ToolCall]


def _accumulate_usage(total: UsageStats, step_usage: UsageStats) -> None:
    """Add per-call usage to running total. Mutates total in place."""
    total.input_tokens += step_usage.input_tokens
    total.output_tokens += step_usage.output_tokens
    total.total_tokens += step_usage.total_tokens
    if step_usage.cost_usd is not None:
        if total.cost_usd is None:
            total.cost_usd = 0.0
        total.cost_usd += step_usage.cost_usd


async def _append_assistant(
    response: LLMResponse,
    history: list[Message],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    iteration: int,
    warnings: list[str],
) -> Message:
    """Create assistant message, append to history, record then notify."""
    msg = Message(
        role=Role.ASSISTANT,
        content=response.text or "",
        tool_calls=response.tool_calls,
    )
    history.append(msg)
    await _record_message(recorder, msg, iteration)
    await _notify_message(notifier, msg, iteration, warnings)
    return msg


def _snapshot_usage(usage: UsageStats) -> UsageStats:
    """Create an independent copy of usage stats for pause state."""
    return UsageStats(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
    )


def _build_pause(
    *,
    agent_name: str,
    pending_calls: list[ToolCall],
    target_lookup: dict[str, ToolTarget],
    history: list[Message],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
) -> PauseState:
    """Build PauseState for client tool or clarification pause."""
    pending_targets = (
        {tc.id: target_lookup.get(tc.name, ToolTarget.SERVER).value for tc in pending_calls}
        if pending_calls
        else {}
    )

    return PauseState(
        agent_name=agent_name,
        pending_tool_calls=pending_calls,
        pending_targets=pending_targets,
        history=list(history),
        steps=list(steps),
        iteration=iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(usage),
    )


class _LoopState(NamedTuple):
    """Mutable iteration state — initialized once, shared across iterations."""

    history: list[Message]
    call_counts: dict[str, int]
    steps: list[AgentStep]
    usage: UsageStats
    warnings: list[str]


async def _init_loop_state(
    *,
    user_input: str,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    initial_history: list[Message] | None,
    initial_steps: list[AgentStep] | None,
    initial_usage: UsageStats | None,
) -> _LoopState:
    """Initialize mutable loop state — shared between run() and run_stream()."""
    if initial_history is not None:
        history = list(initial_history)
    else:
        user_msg = Message(role=Role.USER, content=user_input)
        history = [user_msg]
        await _record_message(recorder, user_msg, 0)
        await _notify_message(notifier, user_msg, 0)

    call_counts: dict[str, int] = {}
    for msg in history:
        if msg.role == Role.TOOL and msg.name is not None:
            call_counts[msg.name] = call_counts.get(msg.name, 0) + 1

    steps: list[AgentStep] = list(initial_steps) if initial_steps else []
    usage = UsageStats(
        input_tokens=initial_usage.input_tokens if initial_usage else 0,
        output_tokens=initial_usage.output_tokens if initial_usage else 0,
        total_tokens=initial_usage.total_tokens if initial_usage else 0,
        cost_usd=initial_usage.cost_usd if initial_usage else None,
    )

    return _LoopState(
        history=history,
        call_counts=call_counts,
        steps=steps,
        usage=usage,
        warnings=[],
    )


async def _process_tool_calls(
    *,
    step: AgentStep,
    call_counts: dict[str, int],
    lookups: ToolLookups,
    strategy: Strategy,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    iteration: int,
    history: list[Message],
    warnings: list[str],
) -> _ToolCallOutcome:
    """Execute tool calls: enforce limits, run server tools, update history.

    Returns a _ToolCallOutcome with all results and any pending (non-server)
    calls. Both run() and run_stream() call this — the caller decides
    whether to yield events from the results.

    Side effects: mutates call_counts, appends to history, records then notifies.
    """
    all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
    all_results: list[tuple[ToolCall, ToolResult]] = []

    # --- Phase 1: Enforce max_calls_per_run ---
    allowed_calls: list[ToolCall] = []
    batch_counts: dict[str, int] = {}
    for tc in all_calls:
        max_calls = lookups.max_calls.get(tc.name)
        current = call_counts.get(tc.name, 0) + batch_counts.get(tc.name, 0)
        if max_calls is not None and current >= max_calls:
            limit_msg = (
                f"Tool '{tc.name}' has reached its maximum of {max_calls} calls for this run."
            )
            limit_result = ToolResult(
                name=tc.name,
                call_id=tc.id,
                payload=json.dumps({"limit": limit_msg}),
                success=False,
                error=limit_msg,
            )
            await _record_tool(recorder, tc, limit_result, iteration)
            await _notify_tool(notifier, tc, limit_result, iteration, warnings)
            result_msg = strategy.format_tool_result(limit_result)
            history.append(result_msg)
            await _record_message(recorder, result_msg, iteration)
            await _notify_message(notifier, result_msg, iteration, warnings)
            all_results.append((tc, limit_result))
        else:
            batch_counts[tc.name] = batch_counts.get(tc.name, 0) + 1
            allowed_calls.append(tc)

    # --- Phase 2: Split into server (execute now) vs non-server (pause) ---
    server_calls: list[ToolCall] = []
    pending_calls: list[ToolCall] = []
    for tc in allowed_calls:
        target = lookups.target.get(tc.name, ToolTarget.SERVER)
        if target == ToolTarget.SERVER:
            server_calls.append(tc)
        else:
            pending_calls.append(tc)

    # --- Phase 3: Execute server tools ---
    exec_groups = _build_execution_groups(server_calls, lookups.parallel)
    executed: list[tuple[ToolCall, ToolResult]] = []
    for group, is_parallel in exec_groups:
        if is_parallel and len(group) > 1:
            pairs = await asyncio.gather(
                *[
                    _execute_record_notify(
                        tc,
                        lookups,
                        recorder,
                        notifier,
                        iteration,
                        warnings,
                    )
                    for tc in group
                ]
            )
            executed.extend(pairs)
        else:
            for tc in group:
                tool_result = await _execute_tool(tc, lookups)
                await _record_tool(recorder, tc, tool_result, iteration)
                await _notify_tool(notifier, tc, tool_result, iteration, warnings)
                executed.append((tc, tool_result))

    # --- Phase 4: Append history in original order (deterministic for LLM) ---
    for tc, tool_result in executed:
        call_counts[tc.name] = call_counts.get(tc.name, 0) + 1
        result_msg = strategy.format_tool_result(tool_result)
        history.append(result_msg)
        await _record_message(recorder, result_msg, iteration)
        await _notify_message(notifier, result_msg, iteration, warnings)
        all_results.append((tc, tool_result))

    return _ToolCallOutcome(all_results=all_results, pending_calls=pending_calls)


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
        recorder: LoopRecorder | None = None,
        notifier: LoopNotifier | None = None,
        initial_history: list[Message] | None = None,
        initial_steps: list[AgentStep] | None = None,
        iteration_offset: int = 0,
        initial_usage: UsageStats | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
    ) -> RunResult:
        """Execute the ReAct loop, optionally resuming from a pause."""
        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        tool_defs = agent.get_tool_defs()
        lookups = _build_tool_lookups(agent.tools)
        state = await _init_loop_state(
            user_input=user_input,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            initial_usage=initial_usage,
        )
        history, call_counts, steps = state.history, state.call_counts, state.steps
        total_usage, notifier_warnings = state.usage, state.warnings

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            messages, tools = strategy.build_messages(
                system_prompt=agent.prompt,
                history=history,
                tool_defs=tool_defs,
            )

            # LLM call — batch
            t0 = time.monotonic()
            response = await provider.complete(messages, tools=tools, **_pkw)
            llm_duration_ms = int((time.monotonic() - t0) * 1000)
            await _record_llm(
                recorder,
                response,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            await _notify_llm(
                notifier,
                response,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            _accumulate_usage(total_usage, response.usage)

            step = strategy.parse_response(response)
            steps.append(step)

            if isinstance(step.action, Finish):
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                meta = {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
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
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                pause = _build_pause(
                    agent_name=agent.name,
                    pending_calls=[],
                    target_lookup=lookups.target,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
                )
                meta: dict[str, Any] = {"pause_state": pause}
                if notifier_warnings:
                    meta["notifier_warnings"] = notifier_warnings
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.WAITING_HUMAN_INPUT,
                    answer=step.action.question,
                    steps=steps,
                    iteration_count=iteration,
                    usage=total_usage,
                    meta=meta,
                )

            if isinstance(step.action, ToolCall):
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                outcome = await _process_tool_calls(
                    step=step,
                    call_counts=call_counts,
                    lookups=lookups,
                    strategy=strategy,
                    recorder=recorder,
                    notifier=notifier,
                    iteration=iteration,
                    history=history,
                    warnings=notifier_warnings,
                )
                if outcome.pending_calls:
                    pause = _build_pause(
                        agent_name=agent.name,
                        pending_calls=outcome.pending_calls,
                        target_lookup=lookups.target,
                        history=history,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                    )
                    meta = {"pause_state": pause}
                    if notifier_warnings:
                        meta["notifier_warnings"] = notifier_warnings
                    return RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.WAITING_CLIENT_TOOL,
                        steps=steps,
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    )

        meta = {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.MAX_ITERATIONS,
            steps=steps,
            iteration_count=agent.max_iterations,
            usage=total_usage,
            meta=meta,
        )

    async def run_stream(
        self,
        *,
        agent: Agent,
        provider: LLMProvider,
        strategy: Strategy,
        user_input: str,
        run_id: str | None = None,
        recorder: LoopRecorder | None = None,
        notifier: LoopNotifier | None = None,
        initial_history: list[Message] | None = None,
        initial_steps: list[AgentStep] | None = None,
        iteration_offset: int = 0,
        initial_usage: UsageStats | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream the ReAct loop as RunEvents.

        Same iteration logic as run() — shared helpers ensure consistency.
        Swaps provider.complete() for complete_stream(), forwarding text
        deltas live and yielding TOOL_RESULT events after execution.
        Terminal event is RUN_COMPLETED or RUN_PAUSED.
        """
        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        tool_defs = agent.get_tool_defs()
        lookups = _build_tool_lookups(agent.tools)
        state = await _init_loop_state(
            user_input=user_input,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            initial_usage=initial_usage,
        )
        history, call_counts, steps = state.history, state.call_counts, state.steps
        total_usage, notifier_warnings = state.usage, state.warnings

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            messages, tools = strategy.build_messages(
                system_prompt=agent.prompt,
                history=history,
                tool_defs=tool_defs,
            )

            # LLM call — streaming
            t0 = time.monotonic()
            llm_response: LLMResponse | None = None
            provider_stream = provider.complete_stream(messages, tools=tools, **_pkw)
            try:
                async for event in provider_stream:
                    if event.type == StreamEventType.TEXT_DELTA:
                        yield RunEvent(type=RunEventType.TEXT_DELTA, text=event.text)
                    elif event.type == StreamEventType.TOOL_USE_START:
                        yield RunEvent(
                            type=RunEventType.TOOL_USE_START,
                            tool_name=event.tool_name,
                            tool_call_id=event.tool_call_id,
                        )
                    elif event.type == StreamEventType.TOOL_USE_END:
                        yield RunEvent(
                            type=RunEventType.TOOL_USE_END,
                            tool_call=event.tool_call,
                            tool_name=event.tool_name,
                            tool_call_id=event.tool_call_id,
                        )
                    elif event.type == StreamEventType.DONE:
                        llm_response = event.raw
            finally:
                await provider_stream.aclose()

            llm_duration_ms = int((time.monotonic() - t0) * 1000)

            if llm_response is None:
                raise RuntimeError(
                    f"Provider stream ended without DONE event at iteration {iteration}. "
                    f"complete_stream() must yield StreamEvent(type=DONE, raw=LLMResponse)."
                )

            await _record_llm(
                recorder,
                llm_response,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            await _notify_llm(
                notifier,
                llm_response,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            _accumulate_usage(total_usage, llm_response.usage)

            step = strategy.parse_response(llm_response)
            steps.append(step)

            if isinstance(step.action, Finish):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                meta: dict[str, Any] = (
                    {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
                )
                yield RunEvent(
                    type=RunEventType.RUN_COMPLETED,
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.SUCCESS,
                        answer=step.action.answer,
                        steps=steps,
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    ),
                )
                return

            if isinstance(step.action, Clarification):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                pause = _build_pause(
                    agent_name=agent.name,
                    pending_calls=[],
                    target_lookup=lookups.target,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
                )
                meta = {"pause_state": pause}
                if notifier_warnings:
                    meta["notifier_warnings"] = notifier_warnings
                yield RunEvent(
                    type=RunEventType.RUN_PAUSED,
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.WAITING_HUMAN_INPUT,
                        answer=step.action.question,
                        steps=steps,
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    ),
                )
                return

            if isinstance(step.action, ToolCall):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                outcome = await _process_tool_calls(
                    step=step,
                    call_counts=call_counts,
                    lookups=lookups,
                    strategy=strategy,
                    recorder=recorder,
                    notifier=notifier,
                    iteration=iteration,
                    history=history,
                    warnings=notifier_warnings,
                )
                for tc, tool_result in outcome.all_results:
                    yield RunEvent(
                        type=RunEventType.TOOL_RESULT,
                        tool_call=tc,
                        tool_result=tool_result,
                    )
                if outcome.pending_calls:
                    pause = _build_pause(
                        agent_name=agent.name,
                        pending_calls=outcome.pending_calls,
                        target_lookup=lookups.target,
                        history=history,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                    )
                    meta = {"pause_state": pause}
                    if notifier_warnings:
                        meta["notifier_warnings"] = notifier_warnings
                    yield RunEvent(
                        type=RunEventType.RUN_PAUSED,
                        run_result=RunResult(
                            run_id=resolved_run_id,
                            status=RunStatus.WAITING_CLIENT_TOOL,
                            steps=steps,
                            iteration_count=iteration,
                            usage=total_usage,
                            meta=meta,
                        ),
                    )
                    return

        meta = {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
        yield RunEvent(
            type=RunEventType.RUN_COMPLETED,
            run_result=RunResult(
                run_id=resolved_run_id,
                status=RunStatus.MAX_ITERATIONS,
                steps=steps,
                iteration_count=agent.max_iterations,
                usage=total_usage,
                meta=meta,
            ),
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


async def _execute_record_notify(
    tool_call: ToolCall,
    lookups: ToolLookups,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    iteration: int,
    notifier_warnings: list[str],
) -> tuple[ToolCall, ToolResult]:
    """Execute a tool, record, then notify — for parallel execution.

    Each tool records and streams its event as soon as it finishes,
    without waiting for siblings.
    """
    result = await _execute_tool(tool_call, lookups)
    await _record_tool(recorder, tool_call, result, iteration)
    await _notify_tool(notifier, tool_call, result, iteration, notifier_warnings)
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
