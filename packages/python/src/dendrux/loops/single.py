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
from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux._lifecycle import (
    FailureReason,
    GovernanceEventType,
)
from dendrux.guardrails._engine import GuardrailEngine
from dendrux.llm._calls import complete_call
from dendrux.llm._retry_telemetry import telemetry_context
from dendrux.loops._budget import BudgetTracker
from dendrux.loops._helpers import (
    build_cache_key_prefix,
    final_stop_reason,
    governance_sinks,
    guardrail_meta,
    lease_scope,
    notify_governance,
    notify_llm,
    notify_llm_started,
    notify_message,
    price_response,
    record_governance,
    record_llm,
    record_llm_started,
    record_message,
)
from dendrux.loops._images import ImageCallSite, release_wire
from dendrux.loops._llm_call import EventRelay, ProviderCallSite, StreamCall, stream_call_events
from dendrux.loops.base import Loop
from dendrux.loops.react import (
    _bind_checkpoint,
    _blocked_data,
    _image_block_data,
    _image_policy,
    _incoming_location,
    _interrupted_result,
    _LoopState,
    _read_control,
)
from dendrux.types import (
    ImageViewState,
    LLMResponse,
    Message,
    Role,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
    StreamEventType,
    StructuredOutputValidationError,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from pydantic import BaseModel

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import AgentStep, Budget, StreamEvent


async def _single_call_budget(
    budget: Budget | None,
    usage: UsageStats,
    run_id: str,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    warnings: list[str],
) -> None:
    """Budget events for the one LLM call.

    There is no checkpoint after it, so ``on_exceed="pause"``/``"cancel"``
    degrade to warn (``budget.exceeded`` reports ``action="warn"``).
    """
    if budget is None:
        return
    # The tracker seeds its spend from the usage it is given: here, the call.
    tracker = BudgetTracker(
        budget,
        run_id=run_id,
        usage=usage,
        recorder=recorder,
        notifier=notifier,
        warnings=warnings,
        enforce=False,
    )
    await tracker.evaluate(1)


def _call_usage(usage: UsageStats) -> UsageStats:
    """Run usage for the one call; an unpriced call marks the total unknown."""
    return replace(
        usage,
        cost_unknown=usage.cost_unknown or usage.cost_usd is None,
        cost_usd=None if usage.cost_unknown else usage.cost_usd,
    )


async def _record_rejected_call(
    exc: StructuredOutputValidationError,
    *,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    warnings: list[str],
    semantic_messages: list[Message] | None,
    duration_ms: int,
    model: str | None = None,
) -> None:
    """Record a structured call whose output failed validation as a completed call.

    The call was made and billed: it is priced, recorded (``llm.completed``
    and its ``llm_interactions`` row) and checked against the budget like any
    other call. ``exc.usage`` / ``exc.model`` are updated to the priced usage.
    The caller re-raises, so the run still fails.

    With guardrails configured the output was never scanned, so it is
    recorded scrubbed (like a guardrail block): placeholder text and no
    provider response, keeping usage, model, cost and stop reason.
    """
    response = exc._response or LLMResponse(
        text=exc.raw_response,
        usage=exc.usage or UsageStats(usage_reported=False),
        model=exc.model,
        stop_reason=exc.stop_reason,
    )
    if agent.guardrails:
        response = LLMResponse(
            text="[structured output rejected]",
            tool_calls=None,
            usage=response.usage,
            provider_request=response.provider_request,
            provider_response=None,
            model=response.model,
            stop_reason=response.stop_reason,
            raw_stop_reason=response.raw_stop_reason,
        )
    response = price_response(
        response, pricing=agent.pricing, default_model=provider.model, model=model
    )
    exc.usage = response.usage
    exc.model = response.model or provider.model
    await record_llm(
        recorder,
        run_id,
        response,
        1,
        semantic_messages=semantic_messages,
        semantic_tools=None,
        duration_ms=duration_ms,
    )
    await notify_llm(
        notifier,
        run_id,
        response,
        1,
        warnings,
        semantic_messages=semantic_messages,
        semantic_tools=None,
        duration_ms=duration_ms,
    )
    await _single_call_budget(
        agent.budget, _call_usage(response.usage), run_id, recorder, notifier, warnings
    )


def _single_checkpoint(
    recorder: LoopRecorder | None,
    agent: Agent,
    strategy: Strategy,
    run_id: str,
    history: list[Message],
    images: ImageCallSite,
    provider_kwargs: dict[str, Any],
    engine: GuardrailEngine | None,
    notifier: LoopNotifier | None,
    warnings: list[str],
) -> tuple[_LoopState, BudgetTracker]:
    state = _LoopState(history, {}, [], UsageStats(), warnings, images.state)
    budget = BudgetTracker(
        agent.budget,
        run_id=run_id,
        usage=state.usage,
        recorder=recorder,
        notifier=notifier,
        warnings=warnings,
        enforce=False,
    )
    _bind_checkpoint(
        recorder,
        state=state,
        budget=budget,
        provider_kwargs=provider_kwargs,
        tool_defs=[],
        engine=engine,
        agent=agent,
        strategy=strategy,
        loop_type="single_call",
    )
    return state, budget


def _refuse_resume(
    initial_steps: list[AgentStep] | None,
    iteration_offset: int,
    initial_usage: UsageStats | None,
) -> None:
    if initial_steps or iteration_offset != 0 or initial_usage is not None:
        raise RuntimeError(
            "SingleCall does not support resume. It never pauses "
            "(no tools, no waiting states), so receiving resume parameters "
            "(initial_steps, iteration_offset, initial_usage) "
            "indicates a bug in the caller."
        )


class _SinglePrelude(NamedTuple):
    run_id: str
    pkw: dict[str, Any]
    cache_key_prefix: str | None
    warnings: list[str]
    history: list[Message]
    messages: list[Message]
    images: ImageCallSite
    calls: ProviderCallSite


async def _seed_history(
    initial_history: list[Message] | None,
    user_input: str,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    warnings: list[str],
) -> list[Message]:
    if initial_history is not None:
        # Seeded chat history (or resume) — caller already persisted what's
        # supposed to land in react_traces. Use as-is, do not re-record.
        return list(initial_history)
    from dendrux.context_blocks import wire_user_input

    user_msg = Message(role=Role.USER, content=wire_user_input(user_input, [], None))
    history = [user_msg]
    await record_message(recorder, run_id, user_msg, 0)
    await notify_message(notifier, run_id, user_msg, 0, warnings)
    return history


async def _single_prelude(
    loop: Loop,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy,
    user_input: str,
    *,
    run_id: str | None,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    initial_history: list[Message] | None,
    provider_kwargs: dict[str, Any] | None,
    image_runtime: ImageRuntime | None,
    image_view_state: ImageViewState | None,
) -> _SinglePrelude:
    """Shared start of ``run`` and ``run_stream``: ids, history, the request, call sites."""
    resolved_run_id = run_id or generate_ulid()
    pkw = provider_kwargs or {}
    cache_key_prefix = build_cache_key_prefix(agent)
    warnings: list[str] = []
    history = await _seed_history(
        initial_history, user_input, recorder, notifier, resolved_run_id, warnings
    )
    messages, _tools = strategy.build_messages(
        system_prompt=agent.get_system_prompt(loop=loop),
        history=history,
        tool_defs=[],
    )
    images = _image_site(
        agent,
        provider,
        image_runtime,
        image_view_state,
        pkw,
        recorder,
        notifier,
        resolved_run_id,
        warnings,
    )
    calls = ProviderCallSite(
        images,
        agent=agent,
        provider=provider,
        pkw=pkw,
        recorder=recorder,
        notifier=notifier,
        run_id=resolved_run_id,
        warnings=warnings,
    )
    return _SinglePrelude(
        resolved_run_id, pkw, cache_key_prefix, warnings, history, messages, images, calls
    )


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
        initial_pii_mapping: dict[str, str] | None = None,
        state_store: StateStore | None = None,
        image_runtime: ImageRuntime | None = None,
        image_view_state: ImageViewState | None = None,
    ) -> RunResult:
        """Execute a single LLM call and return the result."""
        if output_type is not None and agent.guardrails:
            raise ValueError(
                "guardrails with output_type is not supported in this version. "
                "Structured output is parsed before guardrail scanning, so PII "
                "in the typed model would bypass redaction. Use guardrails "
                "without output_type, or output_type without guardrails."
            )

        _refuse_resume(initial_steps, iteration_offset, initial_usage)
        pre = await _single_prelude(
            self,
            agent,
            provider,
            strategy,
            user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            provider_kwargs=provider_kwargs,
            image_runtime=image_runtime,
            image_view_state=image_view_state,
        )
        resolved_run_id, _pkw, cache_key_prefix = pre.run_id, pre.pkw, pre.cache_key_prefix
        notifier_warnings, history, messages = pre.warnings, pre.history, pre.messages
        images, calls = pre.images, pre.calls
        control_reader = _cancel_reader(state_store, resolved_run_id)

        # Incoming guardrail — scan all messages
        g_engine = getattr(recorder, "guardrail_engine", None) or (
            GuardrailEngine(agent.guardrails, pii_mapping=initial_pii_mapping)
            if agent.guardrails
            else None
        )
        checkpoint_state, checkpoint_budget = _single_checkpoint(
            recorder,
            agent,
            strategy,
            resolved_run_id,
            history,
            images,
            _pkw,
            g_engine,
            notifier,
            notifier_warnings,
        )
        if g_engine is not None:
            all_in_findings: list[Any] = []
            _in_was_redacted = False
            for msg_idx, msg in enumerate(messages):
                if not msg.content:
                    continue
                cleaned_msg, findings, block_err = await g_engine.scan_incoming_message(
                    msg, index=msg_idx
                )
                if findings:
                    all_in_findings.extend(findings)
                if block_err is not None:
                    await record_governance(
                        recorder,
                        resolved_run_id,
                        GovernanceEventType.GUARDRAIL_BLOCKED.value,
                        1,
                        _blocked_data(
                            "incoming", block_err, findings, _incoming_location(msg, msg_idx)
                        ),
                    )
                    await notify_governance(
                        notifier,
                        resolved_run_id,
                        GovernanceEventType.GUARDRAIL_BLOCKED.value,
                        1,
                        _blocked_data(
                            "incoming", block_err, findings, _incoming_location(msg, msg_idx)
                        ),
                        warnings=notifier_warnings,
                    )
                    return RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.ERROR,
                        error=block_err,
                        steps=[],
                        iteration_count=1,
                        usage=UsageStats(),
                        meta={
                            **guardrail_meta(g_engine, notifier_warnings),
                            "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value,
                        },
                    )
                if cleaned_msg is not msg:
                    _in_was_redacted = True
                    # The engine's replace() preserves the origin envelope
                    # (kind/placement/source) through redaction.
                    messages[msg_idx] = cleaned_msg
            if all_in_findings:
                _in_entities = list({f.entity_type for f in all_in_findings})
                await record_governance(
                    recorder,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_DETECTED.value,
                    1,
                    {
                        "direction": "incoming",
                        "findings_count": len(all_in_findings),
                        "entities": _in_entities,
                    },
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_DETECTED.value,
                    1,
                    {
                        "direction": "incoming",
                        "findings_count": len(all_in_findings),
                        "entities": _in_entities,
                    },
                    warnings=notifier_warnings,
                )
            if _in_was_redacted:
                _red_entities = list({f.entity_type for f in all_in_findings})
                await record_governance(
                    recorder,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_REDACTED.value,
                    1,
                    {"direction": "incoming", "entities": _red_entities},
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_REDACTED.value,
                    1,
                    {"direction": "incoming", "entities": _red_entities},
                    warnings=notifier_warnings,
                )

        prepared = await images.prepare(messages, iteration=1, guardrails=agent.guardrails or ())
        if prepared.block_error is not None:
            await images.emit(
                GovernanceEventType.GUARDRAIL_BLOCKED.value,
                _image_block_data(prepared.block_error, images.state),
                1,
            )
            return RunResult(
                run_id=resolved_run_id,
                status=RunStatus.ERROR,
                error=prepared.block_error,
                steps=[],
                iteration_count=1,
                usage=UsageStats(),
                meta={
                    **guardrail_meta(g_engine, notifier_warnings),
                    "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value,
                },
            )

        # Structured output path: use the structured helper
        validated_output: Any = None
        retried = False
        while True:
            # LLM call (lifecycle: started → completed/failed)
            await record_llm_started(
                recorder,
                resolved_run_id,
                1,
                semantic_messages=prepared.semantic,
                semantic_tools=None,
            )
            await notify_llm_started(
                notifier,
                resolved_run_id,
                1,
                notifier_warnings,
                semantic_messages=prepared.semantic,
                semantic_tools=None,
            )
            t0 = time.monotonic()
            failure: Exception | None = None
            try:
                with telemetry_context(
                    run_id=resolved_run_id,
                    iteration=1,
                    sinks=governance_sinks(recorder, notifier, resolved_run_id),
                ):
                    async with lease_scope(recorder):
                        if output_type is not None:
                            from dendrux.llm.structured import structured_complete

                            response, validated_output = await structured_complete(
                                provider,
                                prepared.wire,
                                output_type,
                                run_id=resolved_run_id,
                                cache_key_prefix=cache_key_prefix,
                                **_pkw,
                            )
                        else:
                            response = await complete_call(
                                provider,
                                prepared.wire,
                                tools=None,
                                run_id=resolved_run_id,
                                cache_key_prefix=cache_key_prefix,
                                **_pkw,
                            )

                    if response.tool_calls:
                        raise RuntimeError(
                            f"SingleCall received unexpected tool_calls from provider "
                            f"({len(response.tool_calls)} calls). SingleCall agents must "
                            f"have zero tools — the provider should not produce tool calls."
                        )
            except StructuredOutputValidationError as _invalid:
                await _record_rejected_call(
                    _invalid,
                    agent=agent,
                    provider=provider,
                    run_id=resolved_run_id,
                    recorder=recorder,
                    notifier=notifier,
                    warnings=notifier_warnings,
                    semantic_messages=prepared.semantic,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    model=_pkw.get("model"),
                )
                raise
            except Exception as _llm_exc:
                await calls.record_failure(_llm_exc, t0, 1)
                if not prepared.slots:
                    raise
                failure = _llm_exc
                prepared = release_wire(prepared)
            if failure is None:
                break
            stop, error, decision = await calls.decide(
                failure,
                prepared,
                1,
                retried=retried,
                visible_deltas=0,
                control_reader=control_reader,
            )
            del failure
            if stop is not None:
                # SingleCall honours cancel only: nothing ran, so iteration 0.
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.CANCELLED,
                    steps=[],
                    iteration_count=0,
                    usage=UsageStats(),
                    meta=guardrail_meta(g_engine, notifier_warnings),
                )
            if error is not None:
                raise error
            retried = True
            prepared = await images.prepare(
                messages,
                iteration=1,
                guardrails=agent.guardrails or (),
                force_tombstones=decision.tombstones,
            )

        prepared, llm_duration_ms, response = calls.settle(prepared, response, t0)

        # Output guardrail — detection-only. Persistence stores raw
        # (DB is ground truth); block policies abort the run, findings
        # emit governance events. Redaction for any follow-up LLM call
        # happens at the next scan_incoming.
        _sc_findings: dict[str, Any] = {}
        if g_engine is not None and all_in_findings:
            _sc_findings["incoming"] = [
                {"entity_type": f.entity_type, "score": f.score} for f in all_in_findings
            ]
        if g_engine is not None and (response.text or response.tool_calls):
            out_findings, out_block = await g_engine.scan_outgoing(response.text or "")
            if out_findings:
                _sc_findings["outgoing"] = [
                    {"entity_type": f.entity_type, "score": f.score} for f in out_findings
                ]
                _out_data = {
                    "direction": "outgoing",
                    "findings_count": len(out_findings),
                    "entities": list({f.entity_type for f in out_findings}),
                }
                await record_governance(
                    recorder,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_DETECTED.value,
                    1,
                    _out_data,
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_DETECTED.value,
                    1,
                    _out_data,
                    warnings=notifier_warnings,
                )
            if out_block is not None:
                # Scrub response before recording — block means the
                # original text/params contain the PII that triggered it.
                _blocked_resp = LLMResponse(
                    text="[blocked by guardrail]",
                    tool_calls=None,
                    raw=response.raw,
                    usage=response.usage,
                    provider_request=response.provider_request,
                    provider_response=None,
                    stop_reason=response.stop_reason,
                    raw_stop_reason=response.raw_stop_reason,
                )
                await record_llm(
                    recorder,
                    resolved_run_id,
                    _blocked_resp,
                    1,
                    semantic_messages=prepared.semantic,
                    semantic_tools=None,
                    duration_ms=llm_duration_ms,
                    guardrail_findings=images.findings(prepared, _sc_findings),
                )
                await notify_llm(
                    notifier,
                    resolved_run_id,
                    _blocked_resp,
                    1,
                    notifier_warnings,
                    semantic_messages=prepared.semantic,
                    semantic_tools=None,
                    duration_ms=llm_duration_ms,
                    guardrail_findings=images.findings(prepared, _sc_findings),
                )
                await _single_call_budget(
                    agent.budget,
                    response.usage,
                    resolved_run_id,
                    recorder,
                    notifier,
                    notifier_warnings,
                )
                await record_governance(
                    recorder,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_BLOCKED.value,
                    1,
                    _blocked_data("outgoing", out_block, out_findings, "output"),
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    GovernanceEventType.GUARDRAIL_BLOCKED.value,
                    1,
                    _blocked_data("outgoing", out_block, out_findings, "output"),
                    warnings=notifier_warnings,
                )
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.ERROR,
                    error=out_block,
                    steps=[],
                    iteration_count=1,
                    usage=response.usage,
                    meta={
                        **guardrail_meta(g_engine, notifier_warnings),
                        "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value,
                    },
                )

        # Record the LLM response (raw — DB is ground truth).
        await record_llm(
            recorder,
            resolved_run_id,
            response,
            1,
            semantic_messages=prepared.semantic,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
            guardrail_findings=images.findings(prepared, _sc_findings),
        )
        await notify_llm(
            notifier,
            resolved_run_id,
            response,
            1,
            notifier_warnings,
            semantic_messages=prepared.semantic,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
            guardrail_findings=images.findings(prepared, _sc_findings),
        )

        usage = _call_usage(response.usage)
        vars(checkpoint_state.usage).update(vars(usage))
        checkpoint_budget.add_response(response.usage)

        assistant_msg = Message(
            role=Role.ASSISTANT, content=response.text or "", meta={"phase": "final_answer"}
        )
        history.append(assistant_msg)
        await record_message(recorder, resolved_run_id, assistant_msg, 1)
        await notify_message(notifier, resolved_run_id, assistant_msg, 1, notifier_warnings)

        await checkpoint_budget.evaluate(1)

        sc_meta = guardrail_meta(g_engine, notifier_warnings)
        if response.reasoning is not None:
            sc_meta["reasoning"] = response.reasoning
        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.SUCCESS,
            answer=response.text,
            output=validated_output,
            steps=[],
            iteration_count=1,
            usage=usage,
            meta=sc_meta,
            stop_reason=final_stop_reason(
                response, run_id=resolved_run_id, default_model=provider.model
            ),
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
        state_store: StateStore | None = None,
        interrupt: asyncio.Event | None = None,
        image_runtime: ImageRuntime | None = None,
        image_view_state: ImageViewState | None = None,
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

        _refuse_resume(initial_steps, iteration_offset, initial_usage)
        pre = await _single_prelude(
            self,
            agent,
            provider,
            strategy,
            user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            provider_kwargs=provider_kwargs,
            image_runtime=image_runtime,
            image_view_state=image_view_state,
        )
        resolved_run_id, _pkw, cache_key_prefix = pre.run_id, pre.pkw, pre.cache_key_prefix
        notifier_warnings, history, messages = pre.warnings, pre.history, pre.messages
        images, calls = pre.images, pre.calls
        checkpoint_state, checkpoint_budget = _single_checkpoint(
            recorder,
            agent,
            strategy,
            resolved_run_id,
            history,
            images,
            _pkw,
            None,
            notifier,
            notifier_warnings,
        )
        control_reader = _cancel_reader(state_store, resolved_run_id, interrupt)

        if interrupt is not None and interrupt.is_set():
            # cancel_run landed before the provider call started — don't spend it.
            yield RunEvent(
                type=RunEventType.RUN_CANCELLED,
                run_result=RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.CANCELLED,
                    iteration_count=0,
                    usage=UsageStats(),
                    meta={"notifier_warnings": notifier_warnings} if notifier_warnings else {},
                ),
            )
            return

        # LLM call (lifecycle: started → completed/failed), driven here in the
        # consumer's task and closed with this generator.
        call = StreamCall(
            calls,
            await images.prepare(messages, iteration=1),
            messages,
            tools=None,
            iteration=1,
            interrupt=interrupt,
            control_reader=control_reader,
            cache_key_prefix=cache_key_prefix,
            translate=_single_delta,
            check=_require_single_done,
        )
        relay = EventRelay(stream_call_events(call))
        async with aclosing(relay.events):
            while (event := await relay.next()) is not None:
                try:
                    yield event
                except Exception as thrown:
                    relay.throw(thrown)

        if call.stop is not None:
            yield RunEvent(
                type=RunEventType.RUN_CANCELLED,
                run_result=RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.CANCELLED,
                    iteration_count=0,
                    usage=UsageStats(),
                    meta={"notifier_warnings": notifier_warnings} if notifier_warnings else {},
                ),
            )
            return

        if call.interrupted:
            yield RunEvent(
                type=RunEventType.RUN_CANCELLED,
                run_result=_interrupted_result(
                    run_id=resolved_run_id,
                    partial_text=call.partial_text,
                    steps=[],
                    iteration=1,
                    usage=UsageStats(),
                    notifier_warnings=notifier_warnings,
                ),
            )
            return

        # The call raised if no DONE arrived; only the interrupted path
        # (handled above) leaves this None.
        assert call.response is not None
        prepared, llm_duration_ms, llm_response = calls.settle(
            call.prepared, call.response, call.t0
        )

        await record_llm(
            recorder,
            resolved_run_id,
            llm_response,
            1,
            semantic_messages=prepared.semantic,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
        )
        await notify_llm(
            notifier,
            resolved_run_id,
            llm_response,
            1,
            notifier_warnings,
            semantic_messages=prepared.semantic,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
        )

        usage = _call_usage(llm_response.usage)
        vars(checkpoint_state.usage).update(vars(usage))
        checkpoint_budget.add_response(llm_response.usage)

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=llm_response.text or "",
            meta={"phase": "final_answer"},
        )
        history.append(assistant_msg)
        await record_message(recorder, resolved_run_id, assistant_msg, 1)
        await notify_message(notifier, resolved_run_id, assistant_msg, 1, notifier_warnings)

        await checkpoint_budget.evaluate(1)

        sc_stream_meta: dict[str, Any] = {}
        if llm_response.reasoning is not None:
            sc_stream_meta["reasoning"] = llm_response.reasoning
        yield RunEvent(
            type=RunEventType.RUN_COMPLETED,
            phase="final_answer",
            run_result=RunResult(
                run_id=resolved_run_id,
                status=RunStatus.SUCCESS,
                answer=llm_response.text,
                steps=[],
                iteration_count=1,
                usage=usage,
                meta=sc_stream_meta,
                stop_reason=final_stop_reason(
                    llm_response, run_id=resolved_run_id, default_model=provider.model
                ),
            ),
        )


def _single_delta(event: StreamEvent) -> RunEvent | None:
    """SingleCall's event for a streamed delta; a tool event breaks its contract."""
    if event.type == StreamEventType.TEXT_DELTA:
        return RunEvent(
            type=RunEventType.TEXT_DELTA, text=event.text, iteration=1, phase="final_answer"
        )
    if event.type == StreamEventType.REASONING_DELTA:
        return RunEvent(type=RunEventType.REASONING_DELTA, text=event.text, iteration=1)
    if event.type in (StreamEventType.TOOL_USE_START, StreamEventType.TOOL_USE_END):
        raise RuntimeError(
            f"SingleCall received unexpected {event.type.value} event "
            f"from provider stream. SingleCall agents must have zero "
            f"tools — the provider should not produce tool events."
        )
    return None


def _require_single_done(response: LLMResponse | None, iteration: int) -> None:
    # Contract violations still pair with on_llm_call_failed (started already fired).
    if response is None:
        raise RuntimeError(
            "Provider stream ended without DONE event. "
            "complete_stream() must yield StreamEvent(type=DONE, raw=LLMResponse)."
        )
    if response.tool_calls:
        raise RuntimeError(
            f"SingleCall received unexpected tool_calls from provider "
            f"({len(response.tool_calls)} calls). SingleCall agents must "
            f"have zero tools — the provider should not produce tool calls."
        )


def _image_site(
    agent: Agent,
    provider: LLMProvider,
    image_runtime: ImageRuntime | None,
    image_view_state: ImageViewState | None,
    provider_kwargs: dict[str, Any],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    warnings: list[str],
) -> ImageCallSite:
    return ImageCallSite(
        state=image_view_state if image_view_state is not None else ImageViewState(),
        policy=_image_policy(agent, image_runtime),
        runtime=image_runtime,
        provider=provider,
        model=str(provider_kwargs.get("model") or provider.model),
        recorder=recorder,
        notifier=notifier,
        run_id=run_id,
        warnings=warnings,
    )


def _cancel_reader(
    state_store: StateStore | None,
    run_id: str,
    interrupt: asyncio.Event | None = None,
) -> Callable[[], Awaitable[Any]]:
    """Control reader for an image retry: SingleCall honours cancel only (it never pauses).

    An in-process ``interrupt`` (stream) counts as cancel.
    """

    async def read() -> Any:
        if interrupt is not None and interrupt.is_set():
            return _InProcessCancel()
        if state_store is None:
            return None
        control = await _read_control(state_store, run_id)
        return control if control.cancel_requested else None

    return read


class _InProcessCancel:
    cancel_requested = True
    pause_requested = False
