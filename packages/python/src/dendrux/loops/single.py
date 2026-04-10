"""SingleCall loop — one LLM call, no tools, no iteration.

For agents that don't need tools or iteration: classification,
summarization, extraction, one-turn Q&A.

Constraints:
  - Agent must have zero tools (validated at Agent creation time).
  - If the provider unexpectedly returns tool_calls or streams TOOL_USE_*
    events, SingleCall raises RuntimeError rather than silently ignoring.
  - Never pauses — no WAITING_CLIENT_TOOL or WAITING_HUMAN_INPUT states.
    resume() is not applicable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from dendrux.loops._helpers import notify_llm, notify_message, record_llm, record_message
from dendrux.loops.base import Loop
from dendrux.loops.react import _check_budget
from dendrux.types import (
    Message,
    Role,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
    StreamEventType,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic import BaseModel

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.strategies.base import Strategy
    from dendrux.types import AgentStep, LLMResponse


class SingleCall(Loop):
    """One LLM call, no tools, no iteration.

    Use for classification, summarization, extraction, one-turn Q&A —
    any task where the agent produces a response in a single LLM call
    without needing to use tools.

    SingleCall never pauses, so resume() / resume_stream() are not
    applicable. If called on a SingleCall run, the existing CAS guard
    on run status will reject it (status is SUCCESS or ERROR, never a
    waiting state).

    Usage:
        from dendrux import Agent
        from dendrux.loops import SingleCall

        classifier = Agent(
            provider=AnthropicProvider(model="claude-haiku-4-5"),
            loop=SingleCall(),
            prompt="Classify the input as: positive, negative, or neutral.",
        )
        result = await classifier.run("I love this product!")
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
        """Execute a single LLM call and return the result."""
        is_resume = (
            initial_history is not None
            or initial_steps
            or iteration_offset != 0
            or initial_usage is not None
        )
        if is_resume:
            raise RuntimeError(
                "SingleCall does not support resume. It never pauses "
                "(no tools, no waiting states), so receiving resume parameters "
                "(initial_history, initial_steps, iteration_offset, initial_usage) "
                "indicates a bug in the caller."
            )

        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}

        user_msg = Message(role=Role.USER, content=user_input)
        history = [user_msg]
        await record_message(recorder, user_msg, 0)
        await notify_message(notifier, user_msg, 0)

        messages, _tools = strategy.build_messages(
            system_prompt=agent.prompt,
            history=history,
            tool_defs=[],
        )

        t0 = time.monotonic()

        # Structured output path: use the structured helper
        validated_output: Any = None
        if output_type is not None:
            from dendrux.llm.structured import structured_complete

            response, validated_output = await structured_complete(
                provider, messages, output_type, **_pkw
            )
        else:
            response = await provider.complete(messages, tools=None, **_pkw)

            if response.tool_calls:
                raise RuntimeError(
                    f"SingleCall received unexpected tool_calls from provider "
                    f"({len(response.tool_calls)} calls). SingleCall agents must "
                    f"have zero tools — the provider should not produce tool calls."
                )

        llm_duration_ms = int((time.monotonic() - t0) * 1000)

        await record_llm(
            recorder,
            response,
            1,
            semantic_messages=messages,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
        )
        await notify_llm(
            notifier,
            response,
            1,
            semantic_messages=messages,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
        )

        assistant_msg = Message(role=Role.ASSISTANT, content=response.text or "")
        history.append(assistant_msg)
        await record_message(recorder, assistant_msg, 1)
        await notify_message(notifier, assistant_msg, 1)

        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            cost_usd=response.usage.cost_usd,
        )

        await _check_budget(
            agent.budget,
            usage,
            [],
            recorder,
            notifier,
            1,
            [],
        )

        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.SUCCESS,
            answer=response.text,
            output=validated_output,
            steps=[],
            iteration_count=1,
            usage=usage,
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
        """Stream a single LLM call as RunEvents.

        Yields TEXT_DELTA events live. The terminal RunResult.answer
        is built from the final DONE event's LLMResponse, not from
        concatenated deltas.

        Structured output (output_type) is not supported in streaming
        mode in this version. Use agent.run() for structured output.
        """
        if output_type is not None:
            raise NotImplementedError(
                "Structured output streaming is not supported in this version. "
                "Use agent.run() for structured output, or agent.stream() without output_type."
            )

        is_resume = (
            initial_history is not None
            or initial_steps
            or iteration_offset != 0
            or initial_usage is not None
        )
        if is_resume:
            raise RuntimeError(
                "SingleCall does not support resume. It never pauses "
                "(no tools, no waiting states), so receiving resume parameters "
                "(initial_history, initial_steps, iteration_offset, initial_usage) "
                "indicates a bug in the caller."
            )

        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}

        user_msg = Message(role=Role.USER, content=user_input)
        history = [user_msg]
        await record_message(recorder, user_msg, 0)
        await notify_message(notifier, user_msg, 0)

        messages, _tools = strategy.build_messages(
            system_prompt=agent.prompt,
            history=history,
            tool_defs=[],
        )

        t0 = time.monotonic()
        llm_response: LLMResponse | None = None
        provider_stream = provider.complete_stream(messages, tools=None, **_pkw)
        try:
            async for event in provider_stream:
                if event.type == StreamEventType.TEXT_DELTA:
                    yield RunEvent(type=RunEventType.TEXT_DELTA, text=event.text)
                elif event.type in (StreamEventType.TOOL_USE_START, StreamEventType.TOOL_USE_END):
                    raise RuntimeError(
                        f"SingleCall received unexpected {event.type.value} event "
                        f"from provider stream. SingleCall agents must have zero "
                        f"tools — the provider should not produce tool events."
                    )
                elif event.type == StreamEventType.DONE:
                    llm_response = event.raw
        finally:
            await provider_stream.aclose()

        llm_duration_ms = int((time.monotonic() - t0) * 1000)

        if llm_response is None:
            raise RuntimeError(
                "Provider stream ended without DONE event. "
                "complete_stream() must yield StreamEvent(type=DONE, raw=LLMResponse)."
            )

        if llm_response.tool_calls:
            raise RuntimeError(
                f"SingleCall received unexpected tool_calls from provider "
                f"({len(llm_response.tool_calls)} calls). SingleCall agents must "
                f"have zero tools — the provider should not produce tool calls."
            )

        await record_llm(
            recorder,
            llm_response,
            1,
            semantic_messages=messages,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
        )
        await notify_llm(
            notifier,
            llm_response,
            1,
            semantic_messages=messages,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
        )

        assistant_msg = Message(role=Role.ASSISTANT, content=llm_response.text or "")
        history.append(assistant_msg)
        await record_message(recorder, assistant_msg, 1)
        await notify_message(notifier, assistant_msg, 1)

        usage = UsageStats(
            input_tokens=llm_response.usage.input_tokens,
            output_tokens=llm_response.usage.output_tokens,
            total_tokens=llm_response.usage.total_tokens,
            cost_usd=llm_response.usage.cost_usd,
        )

        await _check_budget(
            agent.budget,
            usage,
            [],
            recorder,
            notifier,
            1,
            [],
        )

        yield RunEvent(
            type=RunEventType.RUN_COMPLETED,
            run_result=RunResult(
                run_id=resolved_run_id,
                status=RunStatus.SUCCESS,
                answer=llm_response.text,
                steps=[],
                iteration_count=1,
                usage=usage,
            ),
        )
