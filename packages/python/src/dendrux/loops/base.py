"""Loop protocol — how the agent iterates.

A loop orchestrates the cycle of calling the LLM, interpreting the response,
executing tools, and feeding results back. It uses a Strategy for LLM
communication and a Provider for actual LLM calls.

The loop never touches provider-specific APIs or prompt formatting — that's
the strategy's job. The loop is pure orchestration.

Two event seams (symmetric surface, asymmetric failure policy):
  - LoopRecorder: internal, authoritative persistence. Fail-closed.
  - LoopNotifier: external, best-effort notifications. Exceptions swallowed.

Both protocols share the same hook signatures so a single emission point
in the loop can fan out to both rails. The first positional argument of
every hook is the ``run_id`` so shared notifier instances can disambiguate
concurrent runs without leaning on contextvars or implicit state.

Concrete no-op base classes (:class:`BaseRecorder`, :class:`BaseNotifier`)
are provided for ergonomics — subclass and override only the hooks you
care about. New hooks added in future protocol revisions land as no-op
defaults so existing implementations keep working.
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

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None: ...

    async def on_run_finished(self, run_id: str, result: RunResult) -> None: ...

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None: ...

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None: ...

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None: ...

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None: ...

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None: ...

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None: ...

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None: ...

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
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

    The runner emits the run-level hooks (started / finished / failed); the
    loop emits everything else. Every hook carries ``run_id`` as its first
    positional argument so shared notifier instances can disambiguate
    concurrent runs without contextvars.
    """

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        """Called once at the top of a run, before any loop work begins.

        Fires for fresh runs and for resumes — both transit through the
        runner. Use this to open a root span, emit a "run started" log,
        prime a dashboard row, etc.
        """
        ...

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        """Called when a run reaches a non-error terminal state.

        Includes success, paused (waiting_*), cancelled, max_iterations.
        ``result.status`` carries the distinction. For exceptions use
        :meth:`on_run_failed` instead.
        """
        ...

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        """Called when a run errors out via an unhandled exception.

        Mutually exclusive with :meth:`on_run_finished` — exactly one of
        the two fires per run. Use this to mark spans as ERROR, page
        on-call, etc.
        """
        ...

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        """Called when a message is appended to the conversation history.

        Fires for: initial user message (iteration=0), assistant responses,
        and tool result messages. SYSTEM prompt is not in history — it's
        rebuilt by the strategy each iteration.
        """
        ...

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        """Called immediately before ``provider.complete*()``.

        Carries the Dendrux-normalized request the provider is about to
        receive. Use this to start a span, log the outgoing request, etc.
        Pairs with either :meth:`on_llm_call_completed` (success) or
        :meth:`on_llm_call_failed` (provider exception).
        """
        ...

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        """Called after ``provider.complete*()`` returns successfully.

        Carries token usage in ``response.usage``, plus optional semantic
        payloads (the Dendrux-normalized messages and tool defs sent to
        the LLM). Provider-level payloads are on response itself
        (``response.provider_request``, ``response.provider_response``).

        Args:
            duration_ms: Wall-clock time for the provider.complete() call.
            guardrail_findings: Detector hits from the incoming + outgoing
                guardrail scans for this LLM call (None when no guardrails
                or no hits). Mirrors :class:`LoopRecorder.on_llm_call_completed`
                so notifiers (OTel, dashboards, webhooks) can attach
                guardrail context to LLM events without joining through
                the separate ``guardrail.*`` governance event stream.
        """
        ...

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        """Called when ``provider.complete*()`` raises.

        Fires once between :meth:`on_llm_call_started` and the propagating
        exception. Use this to mark spans as ERROR, surface provider
        failures to ops, etc. The original exception still re-raises
        after the notifier rail finishes.
        """
        ...

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        """Called immediately before tool dispatch.

        Pairs with :meth:`on_tool_completed`. Tool failures arrive on
        completion as ``ToolResult(success=False)`` — there is no
        separate ``on_tool_failed`` hook because the loop already
        converts every dispatch error into a ToolResult.
        """
        ...

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        """Called after _execute_tool() returns.

        Consumes tool_result.duration_ms — no re-timing. Records both
        tool_call.id (Dendrux ULID) and tool_call.provider_tool_call_id.
        """
        ...

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
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


# ------------------------------------------------------------------
# Concrete no-op bases (ergonomic subclassing)
# ------------------------------------------------------------------


class BaseRecorder:
    """Concrete no-op base for :class:`LoopRecorder`.

    Subclass and override only the hooks you care about. New hooks added
    in future protocol revisions land as no-op defaults so existing
    implementations keep working without changes.
    """

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        return None

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        return None

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        return None

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        return None

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        return None

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        return None

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        return None

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        return None

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        return None


class BaseNotifier:
    """Concrete no-op base for :class:`LoopNotifier`.

    Subclass and override only the hooks you care about. New hooks added
    in future protocol revisions land as no-op defaults so existing
    implementations keep working without changes.
    """

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        return None

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        return None

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        return None

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        return None

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        return None

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        return None

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        return None

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        return None

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        return None


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
