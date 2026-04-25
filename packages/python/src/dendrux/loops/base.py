"""Loop protocol — how the agent iterates.

A loop orchestrates the cycle of calling the LLM, interpreting the response,
executing tools, and feeding results back. It uses a Strategy for LLM
communication and a Provider for actual LLM calls.

The loop never touches provider-specific APIs or prompt formatting — that's
the strategy's job. The loop is pure orchestration.

Two event seams:
  - LoopRecorder: internal, authoritative persistence. Fail-closed.
  - LoopNotifier: external, best-effort notifications. Exceptions swallowed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic import BaseModel

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        AgentStep,
        LLMResponse,
        Message,
        RunEvent,
        RunResult,
        ToolCall,
        ToolDef,
        ToolResult,
        UsageStats,
    )


# ------------------------------------------------------------------
# Internal: authoritative persistence (fail-closed)
# ------------------------------------------------------------------


@runtime_checkable
class LoopRecorder(Protocol):
    """Internal persistence hooks — authoritative evidence recording.

    NOT a public extension point. Used only by the framework's
    PersistenceRecorder. Exceptions propagate — if persistence fails,
    the run stops.

    The recorder decides internally which writes are fail-closed
    (trace, tool_call, run_event) vs best-effort (usage, interaction,
    touch_progress). The loop just calls the hooks; the recorder owns
    the durability policy.
    """

    async def on_message_appended(self, message: Message, iteration: int) -> None: ...

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None: ...

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None: ...

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None: ...


# ------------------------------------------------------------------
# Public: best-effort notifications (exceptions swallowed)
# ------------------------------------------------------------------


@runtime_checkable
class LoopNotifier(Protocol):
    """Notifier for loop events — best-effort notification hook.

    The loop fires these callbacks at the exact points where history mutates
    and provider.complete() returns. Implementations decide what to do:
    log, emit metrics, stream to SSE, print to console, etc.

    Failure policy: notifiers should not raise. If they do, the loop logs
    a warning and continues execution. Notification failures must not
    kill agent runs.
    """

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        """Called when a message is appended to the conversation history.

        Fires for: initial user message (iteration=0), assistant responses,
        and tool result messages. SYSTEM prompt is not in history — it's
        rebuilt by the strategy each iteration.
        """
        ...

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Called after provider.complete() returns.

        Carries token usage in response.usage, plus optional semantic
        payloads (the Dendrux-normalized messages and tool defs sent
        to the LLM). Provider-level payloads are on response itself
        (response.provider_request, response.provider_response).

        Args:
            duration_ms: Wall-clock time for the provider.complete() call.
        """
        ...

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        """Called after _execute_tool() returns.

        Consumes tool_result.duration_ms — no re-timing. Records both
        tool_call.id (Dendrux ULID) and tool_call.provider_tool_call_id.
        """
        ...

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        """Called when a governance action fires (deny, approval, budget, guardrail).

        Implementations decide what to do: log, emit metrics, stream to SSE, etc.
        Best-effort — failures must not kill agent runs.

        Args:
            event_type: Stable string like ``policy.denied``, ``approval.requested``.
            iteration: Current loop iteration.
            data: Structured payload (tool_name, reason, etc.).
            correlation_id: Optional join key (e.g. tool call ID for dashboard joins).
        """
        ...


class Loop(ABC):
    """Base class for agent execution loops.

    Built-in implementations:
        ReActLoop  — think → act → observe → repeat (requires tools)
        SingleCall — one LLM call, no tools, no iteration
    """

    @abstractmethod
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
        initial_pii_mapping: dict[str, str] | None = None,
        state_store: StateStore | None = None,
    ) -> RunResult:
        """Execute the agent loop until completion.

        Args:
            agent: Agent definition (tools, prompt, limits).
            provider: LLM provider to call.
            strategy: Strategy for message building and response parsing.
            user_input: The user's input to process.
            run_id: Optional runner-provided ID. If None, loop generates one.
            recorder: Internal persistence hooks (fail-closed).
            notifier: Optional notifier for best-effort notifications.
            initial_history: Pre-existing conversation history. Used for
                two cases: (1) resuming a paused run with messages already
                persisted in the DB, and (2) seeding a fresh run with chat
                history passed by the caller (typically via
                ``agent.run(history=...)``). In BOTH cases the loop treats
                this list as already-known and does NOT call
                ``record_message`` / ``notify_message`` for any of its
                contents — the caller is responsible for recording any
                new messages it expects to land in ``react_traces``.
                For chatbot-style fresh runs, ``runner.run`` records the
                new ``user_input`` itself before invoking the loop.
            initial_steps: Pre-existing AgentSteps from before a pause.
                Merged with new steps so the final RunResult.steps spans
                the entire run.
            iteration_offset: Iteration number to resume from. The loop
                starts counting from iteration_offset + 1.
            initial_usage: Cumulative token usage from before a pause.
            provider_kwargs: Extra kwargs forwarded to provider.complete()
                (e.g. temperature, max_tokens). Passed through by the runner.

        Returns:
            RunResult with status, answer, steps, and usage.
        """

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
        state_store: StateStore | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream agent execution as RunEvents.

        Default implementation calls run() and yields the result as a
        single RUN_COMPLETED event. Override for real token-by-token streaming.

        The runner owns lifecycle events (RUN_STARTED, RUN_ERROR, cancellation).
        The loop yields execution outcomes (RUN_COMPLETED, RUN_PAUSED) and
        intermediate events (TEXT_DELTA, TOOL_USE_*, TOOL_RESULT).

        Args:
            Same as run().

        Yields:
            RunEvent objects. Terminal event is RUN_COMPLETED or RUN_PAUSED.
        """
        from dendrux.types import RunEvent as _RunEvent
        from dendrux.types import RunEventType as _RunEventType

        result = await self.run(
            agent=agent,
            provider=provider,
            strategy=strategy,
            user_input=user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            iteration_offset=iteration_offset,
            initial_usage=initial_usage,
            provider_kwargs=provider_kwargs,
            output_type=output_type,
            state_store=state_store,
        )

        # Determine terminal event type from run status
        from dendrux.types import RunStatus as _RunStatus

        if result.status in (
            _RunStatus.WAITING_CLIENT_TOOL,
            _RunStatus.WAITING_HUMAN_INPUT,
            _RunStatus.WAITING_APPROVAL,
        ):
            yield _RunEvent(type=_RunEventType.RUN_PAUSED, run_result=result)
        elif result.status == _RunStatus.CANCELLED:
            yield _RunEvent(type=_RunEventType.RUN_CANCELLED, run_result=result)
        else:
            yield _RunEvent(type=_RunEventType.RUN_COMPLETED, run_result=result)
