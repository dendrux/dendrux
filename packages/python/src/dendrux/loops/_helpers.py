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

import asyncio
import inspect
import logging
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from dendrux.errors import safe_exception_info
from dendrux.pricing import price_usage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from dendrux.agent import Agent
    from dendrux.guardrails._engine import GuardrailEngine
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.pricing import PriceTable
    from dendrux.types import (
        LLMResponse,
        Message,
        RunResult,
        StopReason,
        StreamEvent,
        ToolCall,
        ToolDef,
        ToolResult,
    )

logger = logging.getLogger(__name__)


def order_observations(recorder: LoopRecorder | None, history: list[Message]) -> None:
    """Restore format-2 observations to their assistant's original call order."""
    order = getattr(recorder, "order_observations", None)
    if order is not None:
        order(history)


class StreamInterruptedError(Exception):
    """Raised inside a loop when an in-flight provider stream is interrupted.

    Carries no payload. The loop that catches it owns the partial output
    (the text deltas it already yielded) and turns the interruption into
    a ``RUN_CANCELLED`` outcome.
    """

    def __init__(self) -> None:
        super().__init__("provider stream interrupted by cancel_run")


class _InterruptibleStream:
    """Async iterator that lets an interrupt signal abort a pending provider pull.

    The provider generator is always advanced *in the caller's task* (never
    in a helper task), so ContextVars a provider sets inside its generator —
    retry attempt counters, telemetry bindings — live in the same context
    they are later reset in. To interrupt a pull that is blocked on the
    network, a watcher task cancels the caller's task, exactly the way
    ``asyncio.timeout()`` does; the cancellation is absorbed here with
    ``Task.uncancel()`` and surfaced as :class:`StreamInterruptedError`.

    Not an async generator on purpose: it holds no state that needs
    finalizing, so closing the loop generator around it leaves nothing
    behind for asyncio's async-generator finalizer hooks.
    """

    def __init__(
        self,
        stream: AsyncGenerator[StreamEvent, None],
        interrupt: asyncio.Event,
    ) -> None:
        self._stream = stream
        self._interrupt = interrupt

    def __aiter__(self) -> _InterruptibleStream:
        return self

    async def __anext__(self) -> StreamEvent:
        task = asyncio.current_task()
        if task is None:  # pragma: no cover — always inside a task under asyncio
            return await self._stream.__anext__()

        armed = True
        fired = False

        def _on_signal(waiter: asyncio.Future[Any]) -> None:
            nonlocal fired
            if armed and not waiter.cancelled():
                fired = True
                task.cancel()

        # Always race, even when the signal is already set: the pull below
        # runs first and anything the provider hands over without blocking
        # (an event or end-of-stream) is delivered. The watcher only bites
        # once the pull actually suspends.
        watcher: asyncio.Future[Any] = asyncio.ensure_future(self._interrupt.wait())
        watcher.add_done_callback(_on_signal)
        try:
            return await self._stream.__anext__()
        except asyncio.CancelledError:
            if fired:
                task.uncancel()
                if task.cancelling() == 0:
                    raise StreamInterruptedError() from None
            raise
        finally:
            armed = False
            watcher.cancel()

    async def aclose(self) -> None:
        await self._stream.aclose()


def interruptible(
    stream: AsyncGenerator[StreamEvent, None],
    interrupt: asyncio.Event | None,
) -> AsyncIterator[StreamEvent]:
    """Iterate ``stream``, raising :class:`StreamInterruptedError` when ``interrupt`` fires.

    With ``interrupt=None`` the stream is returned unchanged (zero overhead).
    Otherwise each pull races against the signal, so an interrupt arriving
    while the provider is silent is honoured immediately rather than at the
    next token. Anything the provider can hand over without blocking (an
    event or end-of-stream) is always delivered, even after the signal fired.
    The provider generator always runs in the caller's task and context.
    """
    if interrupt is None:
        return stream
    return _InterruptibleStream(stream, interrupt)


class _LeasedStream:
    def __init__(self, stream: AsyncIterator[StreamEvent], recorder: LoopRecorder | None) -> None:
        self.stream = stream
        self.recorder = recorder
        self.scope: AbstractAsyncContextManager[None] | None = None
        self.closed = False

    def __aiter__(self) -> _LeasedStream:
        return self

    async def __anext__(self) -> StreamEvent:
        if self.closed:
            raise StopAsyncIteration
        try:
            if self.scope is None:
                scope = lease_scope(self.recorder)
                await scope.__aenter__()
                self.scope = scope
            check = getattr(self.recorder, "check_lease", None)
            if check is not None:
                check()
            return await anext(self.stream)
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            close = getattr(self.stream, "aclose", None)
            if close is not None:
                await close()
        finally:
            if self.scope is not None:
                scope, self.scope = self.scope, None
                await scope.__aexit__(None, None, None)


def leased_stream(
    stream: AsyncIterator[StreamEvent], recorder: LoopRecorder | None
) -> _LeasedStream:
    """Hold one lease until the provider stream ends, fails, or is explicitly closed."""
    return _LeasedStream(stream, recorder)


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


async def project_message(recorder: LoopRecorder | None, message: Message) -> Message:
    """Prepare provider form in the loop, before any persistence transaction."""
    engine = getattr(recorder, "guardrail_engine", None)
    if engine is None:
        return message
    take = getattr(recorder, "take_observation", None)
    observation = take(message.call_id) if take is not None else None
    if observation is not None:
        return replace(cast("Message", observation), meta={**observation.meta, **message.meta})
    return cast("Message", (await engine.project_incoming_message(message))[0])


async def record_message(
    recorder: LoopRecorder | None,
    run_id: str,
    message: Message,
    iteration: int,
) -> Message:
    """Project then record a message; callers retain the returned provider form."""
    message = await project_message(recorder, message)
    if recorder is not None:
        await recorder.on_message_appended(run_id, message, iteration)
    return message


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


def final_stop_reason(
    response: LLMResponse, *, run_id: str, default_model: str | None
) -> StopReason | None:
    """Stop reason of the call that produced a run's final answer.

    Logs a warning when the answer was cut off at the output-token limit:
    the run still succeeds, so this is the only signal outside
    ``RunResult.truncated``.
    """
    if response.stop_reason == "max_tokens":
        logger.warning(
            "Run %s: final answer was cut off at the output-token limit "
            "(model=%s, stop_reason=max_tokens). The run succeeded with a "
            "truncated answer; raise max_tokens to get the full answer.",
            run_id,
            response.model or default_model,
        )
    return response.stop_reason


def price_response(
    response: LLMResponse,
    *,
    pricing: PriceTable | None,
    default_model: str | None,
    model: str | None = None,
) -> LLMResponse:
    """Resolve ``response.usage.cost_usd`` from the agent's price table.

    ``model`` is the per-call requested model (``run(model=...)``), or None
    when the call used the provider's configured ``default_model``. The table
    is consulted for ``response.model`` first, then for the requested model, so
    a dated snapshot id returned by the vendor still prices through the entry
    written for the alias that was requested.

    Applied once per LLM call, before the response is recorded, so the
    persisted usage, the ``llm.completed`` event, the budget check and the
    run-level rollup all see the same number. A provider-reported cost is
    kept as-is; see :func:`dendrux.pricing.price_usage`.
    """
    priced = price_usage(
        response.usage,
        model=response.model,
        fallback_models=(model or default_model,),
        pricing=pricing,
    )
    if priced is response.usage:
        return response
    return replace(response, usage=priced)


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
    projected = getattr(recorder, "on_projected_tool_completed", None)
    if projected is not None and getattr(recorder, "guardrail_engine", None) is not None:
        formatter = getattr(recorder, "format_observation", None)
        assert formatter is not None
        scope = getattr(recorder, "observation_scope", None)
        assert scope is not None
        async with scope():
            message = await project_message(recorder, formatter(tool_result))
            await projected(run_id, tool_call, tool_result, iteration, message)
    else:
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


def notifier_trace_link(notifier: LoopNotifier | None, run_id: str) -> str | None:
    """Read the optional synchronous run-link hook without requiring it in the protocol."""
    return _notifier_link(notifier, "trace_link", run_id)


def notifier_span_link(notifier: LoopNotifier | None, run_id: str, tool_call_id: str) -> str | None:
    """Read an optional tool-span link after on_tool_started has created its span."""
    return _notifier_link(notifier, "current_span_link", run_id, tool_call_id)


def _notifier_link(notifier: LoopNotifier | None, name: str, *args: str) -> str | None:
    try:
        hook = getattr(notifier, name, None)
        value = hook(*args) if callable(hook) else None
        return value if isinstance(value, str) else None
    except Exception:
        logger.warning("Notifier.%s failed", name, exc_info=safe_exception_info())
        return None


async def notify_run_started(
    notifier: LoopNotifier | None,
    run_id: str,
    *,
    agent_name: str | None = None,
    agent_model: str | None = None,
    warnings: list[str] | None = None,
    trace_link: str | None = None,
) -> None:
    """Notify of run start, swallowing exceptions."""
    if notifier is None:
        return
    try:
        kwargs: dict[str, Any] = {"agent_name": agent_name, "agent_model": agent_model}
        try:
            params = dict(inspect.signature(notifier.on_run_started).parameters)
        except (TypeError, ValueError):
            params = {}
        parameter = params.get("trace_link")
        if (parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            kwargs["trace_link"] = trace_link
        await notifier.on_run_started(run_id, **kwargs)
    except Exception:
        logger.warning("Notifier.on_run_started failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_run_finished failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_run_failed failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_message_appended failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_llm_call_started failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_llm_call_completed failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_llm_call_failed failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_tool_started failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_tool_completed failed", exc_info=safe_exception_info())
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
        logger.warning("Notifier.on_governance_event failed", exc_info=safe_exception_info())
        if warnings is not None:
            warnings.append(f"on_governance_event failed at iteration {iteration}")


def lease_scope(recorder: LoopRecorder | None) -> AbstractAsyncContextManager[None]:
    """Ask the recorder for a call-scoped renewal without depending on the runtime layer."""
    scope = getattr(recorder, "lease_scope", None)
    return scope() if scope is not None else nullcontext()
