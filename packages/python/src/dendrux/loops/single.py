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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from dendrux.guardrails._engine import GuardrailEngine
from dendrux.llm._calls import complete_call, stream_call
from dendrux.llm._evidence import persistable_response
from dendrux.llm._retry_telemetry import telemetry_context, telemetry_stream
from dendrux.loops._budget import BudgetTracker
from dendrux.loops._helpers import (
    StreamInterruptedError,
    build_cache_key_prefix,
    final_stop_reason,
    guardrail_meta,
    interruptible,
    lease_scope,
    leased_stream,
    notify_governance,
    notify_llm,
    notify_llm_failed,
    notify_llm_started,
    notify_message,
    price_response,
    record_governance,
    record_llm,
    record_llm_failed,
    record_llm_started,
    record_message,
)
from dendrux.loops._images import ImageCallSite, control_stop, release_wire
from dendrux.loops.base import Loop
from dendrux.loops.react import (
    _bind_checkpoint,
    _blocked_data,
    _drop_frame_locals,
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
    from dendrux.types import AgentStep, Budget


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

        is_resume = initial_steps or iteration_offset != 0 or initial_usage is not None
        if is_resume:
            raise RuntimeError(
                "SingleCall does not support resume. It never pauses "
                "(no tools, no waiting states), so receiving resume parameters "
                "(initial_steps, iteration_offset, initial_usage) "
                "indicates a bug in the caller."
            )

        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        cache_key_prefix = build_cache_key_prefix(agent)
        notifier_warnings: list[str] = []

        if initial_history is not None:
            # Seeded chat history (or resume) — caller already persisted what's
            # supposed to land in react_traces. Use as-is, do not re-record.
            history: list[Message] = list(initial_history)
        else:
            from dendrux.context_blocks import wire_user_input

            user_msg = Message(role=Role.USER, content=wire_user_input(user_input, [], None))
            history = [user_msg]
            await record_message(recorder, resolved_run_id, user_msg, 0)
            await notify_message(notifier, resolved_run_id, user_msg, 0, notifier_warnings)

        messages, _tools = strategy.build_messages(
            system_prompt=agent.get_system_prompt(loop=self),
            history=history,
            tool_defs=[],
        )
        images = _image_site(
            agent,
            provider,
            image_runtime,
            image_view_state,
            _pkw,
            recorder,
            notifier,
            resolved_run_id,
            notifier_warnings,
        )
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
                        "guardrail.blocked",
                        1,
                        _blocked_data(
                            "incoming", block_err, findings, _incoming_location(msg, msg_idx)
                        ),
                    )
                    await notify_governance(
                        notifier,
                        resolved_run_id,
                        "guardrail.blocked",
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
                            "failure_reason": "guardrail_blocked",
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
                    "guardrail.detected",
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
                    "guardrail.detected",
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
                    "guardrail.redacted",
                    1,
                    {"direction": "incoming", "entities": _red_entities},
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    "guardrail.redacted",
                    1,
                    {"direction": "incoming", "entities": _red_entities},
                    warnings=notifier_warnings,
                )

        prepared = await images.prepare(messages, iteration=1, guardrails=agent.guardrails or ())
        if prepared.block_error is not None:
            await images.emit(
                "guardrail.blocked",
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
                    "failure_reason": "guardrail_blocked",
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
                    recorder=recorder,
                    notifier=notifier,
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
                _llm_fail_ms = int((time.monotonic() - t0) * 1000)
                await record_llm_failed(
                    recorder,
                    resolved_run_id,
                    1,
                    _llm_exc,
                    duration_ms=_llm_fail_ms,
                )
                await notify_llm_failed(
                    notifier,
                    resolved_run_id,
                    1,
                    _llm_exc,
                    notifier_warnings,
                    duration_ms=_llm_fail_ms,
                )
                if not prepared.slots:
                    raise
                failure = _llm_exc
                prepared = release_wire(prepared)
            if failure is None:
                break
            decision = await images.on_failure(
                failure,
                prepared,
                iteration=1,
                retried=retried,
                visible_deltas=0,
                control_reader=control_reader,
            )
            if control_stop(decision) is not None:
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.CANCELLED,
                    steps=[],
                    iteration_count=0,
                    usage=UsageStats(),
                    meta=guardrail_meta(g_engine, notifier_warnings),
                )
            if decision.action == "propagate":
                _drop_frame_locals(failure)
                raise failure
            if decision.action == "raise":
                assert decision.error is not None
                del failure
                raise decision.error
            del failure
            retried = True
            prepared = await images.prepare(
                messages,
                iteration=1,
                guardrails=agent.guardrails or (),
                force_tombstones=decision.tombstones,
            )

        prepared = images.mark_sent(prepared)
        llm_duration_ms = int((time.monotonic() - t0) * 1000)
        response = persistable_response(response)
        response = price_response(
            response,
            pricing=agent.pricing,
            default_model=provider.model,
            model=_pkw.get("model"),
        )

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
                    "guardrail.detected",
                    1,
                    _out_data,
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    "guardrail.detected",
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
                    "guardrail.blocked",
                    1,
                    _blocked_data("outgoing", out_block, out_findings, "output"),
                )
                await notify_governance(
                    notifier,
                    resolved_run_id,
                    "guardrail.blocked",
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
                        "failure_reason": "guardrail_blocked",
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

        is_resume = initial_steps or iteration_offset != 0 or initial_usage is not None
        if is_resume:
            raise RuntimeError(
                "SingleCall does not support resume. It never pauses "
                "(no tools, no waiting states), so receiving resume parameters "
                "(initial_steps, iteration_offset, initial_usage) "
                "indicates a bug in the caller."
            )

        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        cache_key_prefix = build_cache_key_prefix(agent)
        notifier_warnings: list[str] = []

        if initial_history is not None:
            # Seeded chat history (or resume) — caller already persisted what's
            # supposed to land in react_traces. Use as-is, do not re-record.
            history: list[Message] = list(initial_history)
        else:
            from dendrux.context_blocks import wire_user_input

            user_msg = Message(role=Role.USER, content=wire_user_input(user_input, [], None))
            history = [user_msg]
            await record_message(recorder, resolved_run_id, user_msg, 0)
            await notify_message(notifier, resolved_run_id, user_msg, 0, notifier_warnings)

        messages, _tools = strategy.build_messages(
            system_prompt=agent.get_system_prompt(loop=self),
            history=history,
            tool_defs=[],
        )
        images = _image_site(
            agent,
            provider,
            image_runtime,
            image_view_state,
            _pkw,
            recorder,
            notifier,
            resolved_run_id,
            notifier_warnings,
        )
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

        prepared = await images.prepare(messages, iteration=1)

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
        llm_response: LLMResponse | None = None
        provider_stream = telemetry_stream(
            stream_call(
                provider,
                prepared.wire,
                tools=None,
                run_id=resolved_run_id,
                cache_key_prefix=cache_key_prefix,
                **_pkw,
            ),
            run_id=resolved_run_id,
            iteration=1,
            recorder=recorder,
            notifier=notifier,
        )
        partial_text: list[str] = []
        interrupted = False
        retried = False
        cancelled = False
        try:
            while True:
                failure: Exception | None = None
                visible_deltas = 0
                active_stream = leased_stream(interruptible(provider_stream, interrupt), recorder)
                try:
                    async for event in active_stream:
                        if event.type == StreamEventType.TEXT_DELTA:
                            partial_text.append(event.text or "")
                            visible_deltas += 1
                            yield RunEvent(
                                type=RunEventType.TEXT_DELTA,
                                text=event.text,
                                iteration=1,
                                phase="final_answer",
                            )
                        elif event.type == StreamEventType.REASONING_DELTA:
                            visible_deltas += 1
                            yield RunEvent(
                                type=RunEventType.REASONING_DELTA, text=event.text, iteration=1
                            )
                        elif event.type in (
                            StreamEventType.TOOL_USE_START,
                            StreamEventType.TOOL_USE_END,
                        ):
                            raise RuntimeError(
                                f"SingleCall received unexpected {event.type.value} event "
                                f"from provider stream. SingleCall agents must have zero "
                                f"tools — the provider should not produce tool events."
                            )
                        elif event.type == StreamEventType.DONE:
                            llm_response = event.raw
                    # Validate inside the same try so contract violations still
                    # pair with on_llm_call_failed (started already fired).
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
                except Exception as _stream_exc:
                    _stream_fail_ms = int((time.monotonic() - t0) * 1000)
                    await record_llm_failed(
                        recorder,
                        resolved_run_id,
                        1,
                        _stream_exc,
                        duration_ms=_stream_fail_ms,
                    )
                    await notify_llm_failed(
                        notifier,
                        resolved_run_id,
                        1,
                        _stream_exc,
                        notifier_warnings,
                        duration_ms=_stream_fail_ms,
                    )
                    if isinstance(_stream_exc, StreamInterruptedError):
                        interrupted = True
                    elif not prepared.slots:
                        raise
                    else:
                        failure = _stream_exc
                        prepared = release_wire(prepared)
                if failure is None:
                    break
                decision = await images.on_failure(
                    failure,
                    prepared,
                    iteration=1,
                    retried=retried,
                    visible_deltas=visible_deltas,
                    control_reader=control_reader,
                )
                if control_stop(decision) is not None:
                    cancelled = True
                    break
                if decision.action == "propagate":
                    _drop_frame_locals(failure)
                    raise failure
                if decision.action == "raise":
                    assert decision.error is not None
                    del failure
                    raise decision.error
                del failure
                retried = True
                await provider_stream.aclose()
                prepared = await images.prepare(
                    messages, iteration=1, force_tombstones=decision.tombstones
                )
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
                provider_stream = telemetry_stream(
                    stream_call(
                        provider,
                        prepared.wire,
                        tools=None,
                        run_id=resolved_run_id,
                        cache_key_prefix=cache_key_prefix,
                        **_pkw,
                    ),
                    run_id=resolved_run_id,
                    iteration=1,
                    recorder=recorder,
                    notifier=notifier,
                )
        finally:
            await active_stream.aclose()
            await provider_stream.aclose()

        if cancelled:
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

        if interrupted:
            yield RunEvent(
                type=RunEventType.RUN_CANCELLED,
                run_result=_interrupted_result(
                    run_id=resolved_run_id,
                    partial_text=partial_text,
                    steps=[],
                    iteration=1,
                    usage=UsageStats(),
                    notifier_warnings=notifier_warnings,
                ),
            )
            return

        # The try-block raised if no DONE arrived; only the interrupted path
        # (handled above) leaves this None.
        assert llm_response is not None
        prepared = images.mark_sent(prepared)
        llm_duration_ms = int((time.monotonic() - t0) * 1000)
        llm_response = persistable_response(llm_response)
        llm_response = price_response(
            llm_response,
            pricing=agent.pricing,
            default_model=provider.model,
            model=_pkw.get("model"),
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
