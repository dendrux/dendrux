"""Agent runner — the entry point for executing agents.

Takes an Agent definition and runs it through the loop with an explicit
provider and strategy. This is the top-level API developers interact with.

Sprint 1: caller provides the LLM provider instance, defaults to
NativeToolCalling strategy and ReActLoop. Future sprints add provider
registry (model string → provider resolution), strategy selection from
agent config, and more loop types.

Sprint 2 adds optional state_store for persistence. When provided:
  - Runner owns the run_id (generates it, passes to loop)
  - PersistenceObserver records traces, tool calls, and usage
  - finalize_run() is called in try/finally to guarantee persistence
  - Observer failures are logged, never kill the run
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling
from dendrite.types import PauseState, RunStatus, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.loops.base import Loop
    from dendrite.runtime.state import StateStore
    from dendrite.strategies.base import Strategy
    from dendrite.types import RunResult, ToolResult

logger = logging.getLogger(__name__)


class EventSequencer:
    """Monotonic sequence counter for run_events within a single run.

    Shared between the runner (run-level events) and the PersistenceObserver
    (loop-level events) to guarantee a globally unique, ordered sequence_index
    per run — including across pause/resume boundaries.

    On resume, initialize with the max existing sequence_index from the DB.
    """

    def __init__(self, initial: int = 0) -> None:
        self._seq = initial

    def next(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    @property
    def current(self) -> int:
        return self._seq


async def _emit_event(
    state_store: StateStore | None,
    run_id: str,
    event_type: str,
    sequencer: EventSequencer | None = None,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Record a durable run-level event. Failures are logged, never fatal."""
    if state_store is None:
        return
    seq = sequencer.next() if sequencer else 0
    try:
        await state_store.save_run_event(
            run_id,
            event_type=event_type,
            sequence_index=seq,
            correlation_id=correlation_id,
            data=data,
        )
    except Exception:
        logger.warning("Failed to record event %s for run %s", event_type, run_id, exc_info=True)


async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    redact: Callable[[str], str] | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run an agent to completion.

    This is the primary API for executing a Dendrite agent. It wires
    together the agent definition, LLM provider, strategy, and loop,
    then executes the loop until completion.

    Args:
        agent: Agent definition (model, tools, prompt, limits).
        provider: LLM provider to use for this run.
        user_input: The user's input to process.
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop. Defaults to ReActLoop.
        state_store: Optional persistence backend. If provided, the run
            is persisted to the database with full traces.
        tenant_id: Optional tenant ID for multi-tenant isolation.
        metadata: Optional developer linking data (thread_id, user_id, etc.).
            Stored in agent_runs.meta — Dendrite stores it, never reads it.
        redact: Optional string scrubber applied to all persisted content
            (trace text, tool params, result payloads, error messages).
            Receives a plain string, must return a plain string.
        **kwargs: Reserved for future use.

    Returns:
        RunResult with status, answer, steps, and usage stats.

    Usage:
        from dendrite import Agent, tool, run
        from dendrite.llm import AnthropicProvider

        @tool()
        async def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(
            model="claude-sonnet-4-6",
            tools=[add],
            prompt="You are a calculator.",
        )
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        result = await run(agent, provider=provider, user_input="What is 15 + 27?")
        print(result.answer)
    """
    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or ReActLoop()

    # Runner owns run_id — single source of truth
    run_id = generate_ulid()
    observer = None
    # Shared sequence counter for run_events — monotonic across runner + observer
    sequencer = EventSequencer()

    if state_store is not None:
        # Create the run record before the loop starts
        # Apply redaction to user input before persistence
        redacted_input = redact(user_input) if redact else user_input
        await state_store.create_run(
            run_id,
            agent.name,
            input_data={"input": redacted_input},
            model=agent.model,
            strategy=type(resolved_strategy).__name__,
            tenant_id=tenant_id,
            meta=metadata,
        )

        # Create persistence observer with shared sequencer
        from dendrite.runtime.observer import PersistenceObserver
        from dendrite.tool import get_tool_def

        target_lookup = {}
        for fn in agent.tools:
            td = get_tool_def(fn)
            target_lookup[td.name] = td.target
        observer = PersistenceObserver(
            state_store,
            run_id,
            model=agent.model,
            provider_name=type(provider).__name__,
            target_lookup=target_lookup,
            redact=redact,
            event_sequencer=sequencer,
        )

    await _emit_event(
        state_store,
        run_id,
        "run.started",
        sequencer,
        {"agent_name": agent.name, "system_prompt": agent.prompt},
    )

    try:
        result = await resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=resolved_strategy,
            user_input=user_input,
            run_id=run_id,
            observer=observer,
        )

        if state_store is not None:
            if result.status in (RunStatus.WAITING_CLIENT_TOOL, RunStatus.WAITING_HUMAN_INPUT):
                # Pause — persist state for resume
                pause_state: PauseState = result.meta["pause_state"]
                await state_store.pause_run(
                    run_id,
                    status=result.status.value,
                    pause_data=pause_state.to_dict(),
                    iteration_count=result.iteration_count,
                )
                await _emit_event(
                    state_store,
                    run_id,
                    "run.paused",
                    sequencer,
                    {
                        "status": result.status.value,
                        "pending_tool_calls": [
                            {
                                "id": tc.id,
                                "name": tc.name,
                                "target": pause_state.pending_targets.get(tc.id),
                            }
                            for tc in pause_state.pending_tool_calls
                        ],
                    },
                )
            else:
                # Finalize with success or max_iterations.
                # Conditional: only if still running (prevents cancel race).
                redacted_answer = (
                    redact(result.answer) if redact and result.answer else result.answer
                )
                finalize_won = await state_store.finalize_run(
                    run_id,
                    status=result.status.value,
                    answer=redacted_answer,
                    iteration_count=result.iteration_count,
                    total_usage=result.usage,
                    expected_current_status="running",
                )
                # Only the CAS winner emits the terminal event — prevents
                # duplicate run.completed when cancel races with finish.
                if finalize_won:
                    await _emit_event(
                        state_store,
                        run_id,
                        "run.completed",
                        sequencer,
                        {"status": result.status.value},
                    )

        return result

    except Exception as exc:
        # Persist ERROR status before re-raising.
        # Conditional: only if still running (prevents cancel race).
        if state_store is not None:
            error_won = False
            try:
                redacted_err = redact(str(exc)) if redact else str(exc)
                error_won = await state_store.finalize_run(
                    run_id,
                    status=RunStatus.ERROR.value,
                    error=redacted_err,
                    total_usage=None,
                    expected_current_status="running",
                )
            except Exception:
                logger.warning("Failed to persist ERROR status for run %s", run_id, exc_info=True)
            # Only the CAS winner emits the error event
            if error_won:
                await _emit_event(
                    state_store, run_id, "run.error", sequencer, {"error": str(exc)[:500]}
                )
        raise


async def resume(
    run_id: str,
    tool_results: list[ToolResult],
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    redact: Callable[[str], str] | None = None,
    extra_observer: Any | None = None,
) -> RunResult:
    """Resume a paused run by providing client tool results.

    Only works on runs with status WAITING_CLIENT_TOOL. Uses an atomic
    claim to prevent double-resume races.

    Args:
        run_id: The paused run's ID.
        tool_results: Results for the pending tool calls. Each must have
            a call_id matching one of the pending_tool_calls.
        state_store: Persistence backend (required for resume).
        agent: Agent definition (must match the paused run's agent).
        provider: LLM provider for continuing the run.
        strategy: Strategy override. Defaults to NativeToolCalling.
        loop: Loop override. Defaults to ReActLoop.
        redact: Redaction policy for persistence.
        extra_observer: Optional additional observer for SSE streaming.
    """
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        redact=redact,
        expected_status=RunStatus.WAITING_CLIENT_TOOL.value,
        tool_results=tool_results,
        extra_observer=extra_observer,
    )


async def resume_with_input(
    run_id: str,
    user_input: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    redact: Callable[[str], str] | None = None,
    extra_observer: Any | None = None,
) -> RunResult:
    """Resume a paused run by providing clarification input.

    Only works on runs with status WAITING_HUMAN_INPUT. Appends the
    user's response as a normal USER message and re-enters the loop.

    Args:
        run_id: The paused run's ID.
        user_input: Free-text response to the agent's clarification question.
        state_store: Persistence backend (required for resume).
        agent: Agent definition (must match the paused run's agent).
        provider: LLM provider for continuing the run.
        strategy: Strategy override. Defaults to NativeToolCalling.
        loop: Loop override. Defaults to ReActLoop.
        redact: Redaction policy for persistence.
        extra_observer: Optional additional observer for SSE streaming.
    """
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        redact=redact,
        expected_status=RunStatus.WAITING_HUMAN_INPUT.value,
        user_input=user_input,
        extra_observer=extra_observer,
    )


async def _resume_core(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    redact: Callable[[str], str] | None = None,
    expected_status: str,
    tool_results: list[ToolResult] | None = None,
    user_input: str | None = None,
    extra_observer: Any | None = None,
) -> RunResult:
    """Shared resume logic for tool results and clarification input.

    Args:
        extra_observer: Optional additional LoopObserver (e.g. TransportObserver)
            to compose with the PersistenceObserver for SSE streaming during resume.
    """
    from dendrite.runtime.observer import PersistenceObserver
    from dendrite.tool import get_tool_def
    from dendrite.types import Message, Role

    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or ReActLoop()

    # 1. Load pause state
    raw_pause = await state_store.get_pause_state(run_id)
    if raw_pause is None:
        raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
    pause_state = PauseState.from_dict(raw_pause)

    # 2. Validate tool results BEFORE claiming (prevents stuck RUNNING on bad input)
    if tool_results is not None:
        pending_ids = {tc.id for tc in pause_state.pending_tool_calls}
        provided_ids = {tr.call_id for tr in tool_results}
        if provided_ids != pending_ids:
            raise ValueError(
                f"Tool result call_ids {provided_ids} do not match "
                f"pending tool call_ids {pending_ids}."
            )

    # 3. Atomic claim — transition WAITING → RUNNING
    #    Done after validation so a bad request doesn't leave the run stuck.
    claimed = await state_store.claim_paused_run(run_id, expected_status=expected_status)
    if not claimed:
        raise ValueError(
            f"Run '{run_id}' is not in status '{expected_status}' — "
            f"cannot resume. It may have been claimed by another caller."
        )

    # 4. Initialize sequencer from DB max (continues across pause boundaries)
    existing_events = await state_store.get_run_events(run_id)
    max_seq = max((e.sequence_index for e in existing_events), default=-1)
    sequencer = EventSequencer(initial=max_seq + 1)

    # 5. Record resume event with enriched payload for dashboard
    resume_data: dict[str, Any] = {"resumed_from": expected_status}
    if tool_results is not None:
        resume_data["submitted_results"] = [
            {"call_id": tr.call_id, "name": tr.name, "success": tr.success} for tr in tool_results
        ]
    elif user_input is not None:
        resume_data["user_input"] = redact(user_input) if redact else user_input
    await _emit_event(state_store, run_id, "run.resumed", sequencer, resume_data)

    # 6. Build resume history
    history = list(pause_state.history)

    if tool_results is not None:
        # Inject tool results into history as TOOL messages
        for tr in tool_results:
            result_msg = Message(
                role=Role.TOOL,
                content=tr.payload,
                name=tr.name,
                call_id=tr.call_id,
                meta={"is_error": True} if not tr.success else {},
            )
            history.append(result_msg)
    elif user_input is not None:
        # Append clarification response as USER message
        history.append(Message(role=Role.USER, content=user_input))

    # 7. Load real trace order offset from DB (not the approximate one in pause_state)
    traces = await state_store.get_traces(run_id)
    trace_order_offset = max((t.order_index for t in traces), default=-1) + 1

    # 8. Create observer for resumed run with shared sequencer
    target_lookup = {}
    for fn in agent.tools:
        td = get_tool_def(fn)
        target_lookup[td.name] = td.target
    persistence_obs = PersistenceObserver(
        state_store,
        run_id,
        model=agent.model,
        provider_name=type(provider).__name__,
        target_lookup=target_lookup,
        redact=redact,
        initial_order_index=trace_order_offset,
        event_sequencer=sequencer,
    )

    # Compose with extra observer (e.g. TransportObserver for SSE)
    observer: Any
    if extra_observer is not None:
        from dendrite.server.observer import CompositeObserver

        observer = CompositeObserver([persistence_obs, extra_observer])
    else:
        observer = persistence_obs

    # 7. Notify observer of injected messages and tool completions
    if tool_results is not None:
        pending_by_id = {tc.id: tc for tc in pause_state.pending_tool_calls}
        injected_start = len(pause_state.history)
        for i, tr in enumerate(tool_results):
            # Persist the TOOL message trace
            await observer.on_message_appended(history[injected_start + i], pause_state.iteration)
            # Persist the tool_call record (params, result, success, target)
            await observer.on_tool_completed(pending_by_id[tr.call_id], tr, pause_state.iteration)
    elif user_input is not None:
        await observer.on_message_appended(history[-1], pause_state.iteration)

    # 8. Re-enter loop
    try:
        result = await resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=resolved_strategy,
            user_input="",  # Not used when initial_history is provided
            run_id=run_id,
            observer=observer,
            initial_history=history,
            initial_steps=pause_state.steps,
            iteration_offset=pause_state.iteration,
            initial_usage=pause_state.usage,
        )

        # 9. Finalize or pause again
        if result.status in (RunStatus.WAITING_CLIENT_TOOL, RunStatus.WAITING_HUMAN_INPUT):
            new_pause: PauseState = result.meta["pause_state"]
            await state_store.pause_run(
                run_id,
                status=result.status.value,
                pause_data=new_pause.to_dict(),
                iteration_count=result.iteration_count,
            )
            await _emit_event(
                state_store,
                run_id,
                "run.paused",
                sequencer,
                {
                    "status": result.status.value,
                    "pending_tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "target": new_pause.pending_targets.get(tc.id),
                        }
                        for tc in new_pause.pending_tool_calls
                    ],
                },
            )
        else:
            redacted_answer = redact(result.answer) if redact and result.answer else result.answer
            finalize_won = await state_store.finalize_run(
                run_id,
                status=result.status.value,
                answer=redacted_answer,
                iteration_count=result.iteration_count,
                total_usage=result.usage,
                expected_current_status="running",
            )
            # Surface CAS result so callers (server) know if they own the terminal event
            result.meta["_finalize_won"] = finalize_won
            if finalize_won:
                await _emit_event(
                    state_store, run_id, "run.completed", sequencer, {"status": result.status.value}
                )

        return result

    except Exception as exc:
        if state_store is not None:
            error_won = False
            try:
                redacted_err = redact(str(exc)) if redact else str(exc)
                error_won = await state_store.finalize_run(
                    run_id,
                    status=RunStatus.ERROR.value,
                    error=redacted_err,
                    total_usage=None,
                    expected_current_status="running",
                )
            except Exception:
                logger.warning("Failed to persist ERROR status for run %s", run_id, exc_info=True)
            # Only the CAS winner emits the error event
            if error_won:
                await _emit_event(
                    state_store, run_id, "run.error", sequencer, {"error": str(exc)[:500]}
                )
        raise
