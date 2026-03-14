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
    from dendrite.types import LLMResponse

logger = logging.getLogger(__name__)


async def _notify_message(observer: LoopObserver | None, message: Message, iteration: int) -> None:
    """Notify observer of a message append, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_message_appended(message, iteration)
    except Exception:
        logger.warning("Observer.on_message_appended failed", exc_info=True)


async def _notify_llm(observer: LoopObserver | None, response: LLMResponse, iteration: int) -> None:
    """Notify observer of an LLM call completion, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_llm_call_completed(response, iteration)
    except Exception:
        logger.warning("Observer.on_llm_call_completed failed", exc_info=True)


async def _notify_tool(
    observer: LoopObserver | None, tool_call: ToolCall, tool_result: ToolResult, iteration: int
) -> None:
    """Notify observer of a tool completion, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_tool_completed(tool_call, tool_result, iteration)
    except Exception:
        logger.warning("Observer.on_tool_completed failed", exc_info=True)


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
    ) -> RunResult:
        """Execute the ReAct loop."""
        resolved_run_id = run_id or generate_ulid()
        tool_defs = agent.get_tool_defs()
        tool_lookup, target_lookup = _build_tool_lookup(agent.tools)

        user_msg = Message(role=Role.USER, content=user_input)
        history: list[Message] = [user_msg]
        await _notify_message(observer, user_msg, 0)

        steps: list[AgentStep] = []
        total_usage = UsageStats()

        for iteration in range(1, agent.max_iterations + 1):
            # 1. Build messages via strategy
            messages, tools = strategy.build_messages(
                system_prompt=agent.prompt,
                history=history,
                tool_defs=tool_defs,
            )

            # 2. Call the LLM
            response = await provider.complete(messages, tools=tools)
            await _notify_llm(observer, response, iteration)

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
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.SUCCESS,
                    answer=step.action.answer,
                    steps=steps,
                    iteration_count=iteration,
                    usage=total_usage,
                )

            if isinstance(step.action, Clarification):
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.SUCCESS,
                    answer=step.action.question,
                    steps=steps,
                    iteration_count=iteration,
                    usage=total_usage,
                )

            if isinstance(step.action, ToolCall):
                # Append assistant message with all tool_calls to history
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
                history.append(assistant_msg)
                await _notify_message(observer, assistant_msg, iteration)

                # Execute all tool calls from this turn and append results
                # in the same order the assistant requested them.
                # step.meta["all_tool_calls"] contains the full list;
                # step.action is only the first (AgentStep models one action).
                all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
                for tc in all_calls:
                    tool_result = await _execute_tool(tc, tool_lookup, target_lookup)
                    await _notify_tool(observer, tc, tool_result, iteration)
                    result_msg = strategy.format_tool_result(tool_result)
                    history.append(result_msg)
                    await _notify_message(observer, result_msg, iteration)

        # Max iterations reached
        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.MAX_ITERATIONS,
            steps=steps,
            iteration_count=agent.max_iterations,
            usage=total_usage,
        )


def _build_tool_lookup(
    tools: list[Callable[..., Any]],
) -> tuple[dict[str, Callable[..., Any]], dict[str, ToolTarget]]:
    """Build name → function and name → target lookups from the agent's tool list."""
    fn_lookup: dict[str, Callable[..., Any]] = {}
    target_lookup: dict[str, ToolTarget] = {}
    for fn in tools:
        td = get_tool_def(fn)
        fn_lookup[td.name] = fn
        target_lookup[td.name] = td.target
    return fn_lookup, target_lookup


async def _execute_tool(
    tool_call: ToolCall,
    tool_lookup: dict[str, Callable[..., Any]],
    target_lookup: dict[str, ToolTarget] | None = None,
) -> ToolResult:
    """Execute a tool function and return a ToolResult."""
    # Guard: refuse to execute non-server tools locally
    if target_lookup is not None:
        target = target_lookup.get(tool_call.name)
        if target is not None and target != ToolTarget.SERVER:
            return ToolResult(
                name=tool_call.name,
                call_id=tool_call.id,
                payload=json.dumps(
                    {
                        "error": f"Tool '{tool_call.name}' has target={target.value!r} "
                        f"and cannot be executed server-side. "
                        f"Client/human/agent tool execution is not yet implemented."
                    }
                ),
                success=False,
                error=f"Non-server tool target: {target.value}",
            )

    fn = tool_lookup.get(tool_call.name)
    if fn is None:
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(fn):
            result = await fn(**tool_call.params)
        else:
            result = await asyncio.to_thread(fn, **tool_call.params)

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
