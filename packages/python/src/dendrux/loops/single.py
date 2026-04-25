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

from dendrux.guardrails._engine import GuardrailEngine
from dendrux.llm._retry_telemetry import telemetry_context
from dendrux.loops._helpers import (
    build_cache_key_prefix,
    guardrail_meta,
    notify_governance,
    notify_llm,
    notify_message,
    record_governance,
    record_llm,
    record_message,
)
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
    from dendrux.runtime.state import StateStore
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
        initial_pii_mapping: dict[str, str] | None = None,
        state_store: StateStore | None = None,  # noqa: ARG002 — SingleCall has no checkpoints
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

        if initial_history is not None:
            # Seeded chat history (or resume) — caller already persisted what's
            # supposed to land in react_traces. Use as-is, do not re-record.
            history: list[Message] = list(initial_history)
        else:
            user_msg = Message(role=Role.USER, content=user_input)
            history = [user_msg]
            await record_message(recorder, user_msg, 0)
            await notify_message(notifier, user_msg, 0)

        messages, _tools = strategy.build_messages(
            system_prompt=agent.get_system_prompt(),
            history=history,
            tool_defs=[],
        )

        # Incoming guardrail — scan all messages
        g_engine = GuardrailEngine(agent.guardrails) if agent.guardrails else None
        if g_engine is not None:
            all_in_findings: list[Any] = []
            _in_was_redacted = False
            for msg_idx, msg in enumerate(messages):
                if not msg.content:
                    continue
                cleaned, findings, block_err = await g_engine.scan_incoming(msg.content)
                if findings:
                    all_in_findings.extend(findings)
                if block_err is not None:
                    await record_governance(
                        recorder,
                        "guardrail.blocked",
                        1,
                        {"direction": "incoming", "error": block_err},
                    )
                    await notify_governance(
                        notifier,
                        "guardrail.blocked",
                        1,
                        {"direction": "incoming", "error": block_err},
                    )
                    return RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.ERROR,
                        error=block_err,
                        steps=[],
                        iteration_count=1,
                        usage=UsageStats(),
                        meta=guardrail_meta(g_engine),
                    )
                if cleaned != msg.content:
                    _in_was_redacted = True
                    messages[msg_idx] = Message(
                        role=msg.role,
                        content=cleaned,
                        name=msg.name,
                        tool_calls=msg.tool_calls,
                        call_id=msg.call_id,
                        meta=msg.meta,
                    )
            if all_in_findings:
                _in_entities = list({f.entity_type for f in all_in_findings})
                await record_governance(
                    recorder,
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
                    "guardrail.detected",
                    1,
                    {
                        "direction": "incoming",
                        "findings_count": len(all_in_findings),
                        "entities": _in_entities,
                    },
                )
            if _in_was_redacted:
                _red_entities = list({f.entity_type for f in all_in_findings})
                await record_governance(
                    recorder,
                    "guardrail.redacted",
                    1,
                    {"direction": "incoming", "entities": _red_entities},
                )
                await notify_governance(
                    notifier,
                    "guardrail.redacted",
                    1,
                    {"direction": "incoming", "entities": _red_entities},
                )

        t0 = time.monotonic()

        # Structured output path: use the structured helper
        validated_output: Any = None
        with telemetry_context(
            run_id=resolved_run_id,
            iteration=1,
            recorder=recorder,
            notifier=notifier,
        ):
            if output_type is not None:
                from dendrux.llm.structured import structured_complete

                response, validated_output = await structured_complete(
                    provider,
                    messages,
                    output_type,
                    run_id=resolved_run_id,
                    cache_key_prefix=cache_key_prefix,
                    **_pkw,
                )
            else:
                response = await provider.complete(
                    messages,
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

        llm_duration_ms = int((time.monotonic() - t0) * 1000)

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
                    "guardrail.detected",
                    1,
                    _out_data,
                )
                await notify_governance(
                    notifier,
                    "guardrail.detected",
                    1,
                    _out_data,
                )
            if out_block is not None:
                # Scrub response before recording — block means the
                # original text/params contain the PII that triggered it.
                from dendrux.types import LLMResponse as _LLMResp2

                _blocked_resp = _LLMResp2(
                    text="[blocked by guardrail]",
                    tool_calls=None,
                    raw=response.raw,
                    usage=response.usage,
                    provider_request=response.provider_request,
                    provider_response=None,
                )
                await record_llm(
                    recorder,
                    _blocked_resp,
                    1,
                    semantic_messages=messages,
                    semantic_tools=None,
                    duration_ms=llm_duration_ms,
                    guardrail_findings=_sc_findings or None,
                )
                await notify_llm(
                    notifier,
                    _blocked_resp,
                    1,
                    semantic_messages=messages,
                    semantic_tools=None,
                    duration_ms=llm_duration_ms,
                )
                await _check_budget(
                    agent.budget,
                    response.usage,
                    [],
                    recorder,
                    notifier,
                    1,
                    [],
                )
                await record_governance(
                    recorder,
                    "guardrail.blocked",
                    1,
                    {"direction": "outgoing", "error": out_block},
                )
                await notify_governance(
                    notifier,
                    "guardrail.blocked",
                    1,
                    {"direction": "outgoing", "error": out_block},
                )
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.ERROR,
                    error=out_block,
                    steps=[],
                    iteration_count=1,
                    usage=response.usage,
                    meta=guardrail_meta(g_engine),
                )

        # Record the LLM response (raw — DB is ground truth).
        await record_llm(
            recorder,
            response,
            1,
            semantic_messages=messages,
            semantic_tools=None,
            duration_ms=llm_duration_ms,
            guardrail_findings=_sc_findings or None,
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
            cache_read_input_tokens=response.usage.cache_read_input_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
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
            meta=guardrail_meta(g_engine),
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
        state_store: StateStore | None = None,  # noqa: ARG002 — SingleCall has no checkpoints
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

        if initial_history is not None:
            # Seeded chat history (or resume) — caller already persisted what's
            # supposed to land in react_traces. Use as-is, do not re-record.
            history: list[Message] = list(initial_history)
        else:
            user_msg = Message(role=Role.USER, content=user_input)
            history = [user_msg]
            await record_message(recorder, user_msg, 0)
            await notify_message(notifier, user_msg, 0)

        messages, _tools = strategy.build_messages(
            system_prompt=agent.get_system_prompt(),
            history=history,
            tool_defs=[],
        )

        t0 = time.monotonic()
        llm_response: LLMResponse | None = None
        _stream_telemetry = telemetry_context(
            run_id=resolved_run_id,
            iteration=1,
            recorder=recorder,
            notifier=notifier,
        )
        _stream_telemetry.__enter__()
        provider_stream = provider.complete_stream(
            messages,
            tools=None,
            run_id=resolved_run_id,
            cache_key_prefix=cache_key_prefix,
            **_pkw,
        )
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
            _stream_telemetry.__exit__(None, None, None)

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
            cache_read_input_tokens=llm_response.usage.cache_read_input_tokens,
            cache_creation_input_tokens=llm_response.usage.cache_creation_input_tokens,
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
