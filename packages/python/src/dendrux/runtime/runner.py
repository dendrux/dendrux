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

from dendrux.loops.react import ReActLoop
from dendrux.runtime.context import (
    DelegationContext,
    get_delegation_context,
    get_store_identity,
    reset_delegation_context,
    resolve_parent_link,
    set_delegation_context,
)
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import PauseState, RunEventType, RunStatus, generate_ulid

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import Message, RunEvent, RunResult, RunStream, ToolResult

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
    extra_observer: Any | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run an agent to completion.

    This is the primary API for executing a Dendrux agent. It wires
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
            Stored in agent_runs.meta — Dendrux stores it, never reads it.
        redact: Optional string scrubber applied to all persisted content
            (trace text, tool params, result payloads, error messages).
            Receives a plain string, must return a plain string.
        **kwargs: Reserved for future use.

    Returns:
        RunResult with status, answer, steps, and usage stats.

    Usage:
        from dendrux import Agent, tool, run
        from dendrux.llm import AnthropicProvider

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
    resolved_loop = loop or agent.loop or ReActLoop()

    # Extract provider kwargs (temperature, max_tokens, etc.)
    # before they get consumed. kwargs dict is forwarded to the loop
    # which forwards to provider.complete().
    provider_kwargs = dict(kwargs) if kwargs else {}

    # Runner owns run_id — single source of truth
    run_id = generate_ulid()
    observer = None
    # Shared sequence counter for run_events — monotonic across runner + observer
    sequencer = EventSequencer()

    # --- Delegation context ---
    parent_ctx = get_delegation_context()
    parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)

    if state_store is not None:
        # Create the run record before the loop starts
        # Apply redaction to user input before persistence
        redacted_input = redact(user_input) if redact else user_input
        # Merge loop type into developer metadata for dashboard/debugging
        run_meta = dict(metadata) if metadata else {}
        run_meta["dendrux.loop"] = type(resolved_loop).__name__
        await state_store.create_run(
            run_id,
            agent.name,
            input_data={"input": redacted_input},
            model=provider.model,
            strategy=type(resolved_strategy).__name__,
            parent_run_id=parent_run_id,
            delegation_level=delegation_level,
            tenant_id=tenant_id,
            meta=run_meta,
        )

        # Create persistence observer with shared sequencer
        from dendrux.runtime.observer import PersistenceObserver
        from dendrux.tool import get_tool_def

        target_lookup = {}
        for fn in agent.tools:
            td = get_tool_def(fn)
            target_lookup[td.name] = td.target
        observer = PersistenceObserver(
            state_store,
            run_id,
            model=provider.model,
            provider_name=type(provider).__name__,
            target_lookup=target_lookup,
            redact=redact,
            event_sequencer=sequencer,
        )

    # Compose with extra observer (e.g. ConsoleObserver, TransportObserver)
    composed_observer = observer
    if extra_observer is not None:
        from dendrux.observers.composite import CompositeObserver

        if observer is not None:
            composed_observer = CompositeObserver([observer, extra_observer])
        else:
            composed_observer = extra_observer

    await _emit_event(
        state_store,
        run_id,
        "run.started",
        sequencer,
        {"agent_name": agent.name, "system_prompt": agent.prompt},
    )

    # Set delegation context for the duration of this run so nested
    # agent.run() calls inside tools inherit the parent link.
    this_ctx = DelegationContext(
        run_id=run_id,
        delegation_level=delegation_level,
        persisted=state_store is not None,
        store_identity=get_store_identity(state_store),
    )
    ctx_token = set_delegation_context(this_ctx)

    try:
        result = await resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=resolved_strategy,
            user_input=user_input,
            run_id=run_id,
            observer=composed_observer,
            provider_kwargs=provider_kwargs or None,
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

    finally:
        reset_delegation_context(ctx_token)


def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    redact: Callable[[str], str] | None = None,
    extra_observer: Any | None = None,
    **kwargs: Any,
) -> RunStream:
    """Stream an agent run as RunEvents, returning a RunStream.

    Synchronous — returns immediately with run_id available. All async
    setup (DB row, observers) runs lazily on first iteration.

    Accepts either an already-resolved ``state_store`` or an async
    ``state_store_resolver`` callable. The resolver is called once in
    the generator preamble.

    Error semantics: exceptions are caught, persisted, and yielded as
    RUN_ERROR events. No exception is raised to the consumer.

    Lifecycle ownership:
      - Runner emits RUN_STARTED, RUN_ERROR, and handles cancellation cleanup.
      - Loop yields execution outcomes (RUN_COMPLETED, RUN_PAUSED) and
        intermediate events (TEXT_DELTA, TOOL_USE_*, TOOL_RESULT).
      - Runner persists loop outcomes before forwarding them.
    """
    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or agent.loop or ReActLoop()
    provider_kwargs = dict(kwargs) if kwargs else {}

    run_id = generate_ulid()

    # Shared mutable state — generator populates on first iteration,
    # cleanup reads if the consumer abandons after setup.
    _shared: dict[str, Any] = {"state_store": state_store, "sequencer": EventSequencer()}

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        """Inner generator — lazy async setup, then loop stream with lifecycle.

        The entire body is wrapped in try/except so that setup failures
        (state_store_resolver, create_run, observer init) are also caught
        and yielded as RUN_ERROR — honoring the "no exception to consumer" contract.
        """
        # store is captured in the closure for the error handler.
        # Starts as the pre-provided value (often None for the lazy path).
        store = state_store
        sequencer = _shared["sequencer"]
        ctx_token = None

        try:
            # 1. Resolve state store lazily (may create DB engine)
            if store is None and state_store_resolver is not None:
                store = await state_store_resolver()
            _shared["state_store"] = store

            # --- Delegation context ---
            parent_ctx = get_delegation_context()
            parent_run_id, delegation_level = resolve_parent_link(parent_ctx, store)

            observer = None

            if store is not None:
                redacted_input = redact(user_input) if redact else user_input
                run_meta = dict(metadata) if metadata else {}
                run_meta["dendrux.loop"] = type(resolved_loop).__name__
                await store.create_run(
                    run_id,
                    agent.name,
                    input_data={"input": redacted_input},
                    model=provider.model,
                    strategy=type(resolved_strategy).__name__,
                    parent_run_id=parent_run_id,
                    delegation_level=delegation_level,
                    tenant_id=tenant_id,
                    meta=run_meta,
                )

                from dendrux.runtime.observer import PersistenceObserver
                from dendrux.tool import get_tool_def

                target_lookup = {}
                for fn in agent.tools:
                    td = get_tool_def(fn)
                    target_lookup[td.name] = td.target
                observer = PersistenceObserver(
                    store,
                    run_id,
                    model=provider.model,
                    provider_name=type(provider).__name__,
                    target_lookup=target_lookup,
                    redact=redact,
                    event_sequencer=sequencer,
                )

            composed_observer = observer
            if extra_observer is not None:
                from dendrux.observers.composite import CompositeObserver

                if observer is not None:
                    composed_observer = CompositeObserver([observer, extra_observer])
                else:
                    composed_observer = extra_observer

            # 2. Set delegation context for the generator lifetime
            this_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=delegation_level,
                persisted=store is not None,
                store_identity=get_store_identity(store),
            )
            ctx_token = set_delegation_context(this_ctx)

            # 3. Emit lifecycle start
            await _emit_event(
                store,
                run_id,
                "run.started",
                sequencer,
                {"agent_name": agent.name, "system_prompt": agent.prompt},
            )

            yield RunEvent(type=RunEventType.RUN_STARTED, run_id=run_id)

            # 3. Stream the loop
            async for event in resolved_loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=resolved_strategy,
                user_input=user_input,
                run_id=run_id,
                observer=composed_observer,
                provider_kwargs=provider_kwargs or None,
            ):
                # Persist loop outcomes before forwarding
                if event.type == RunEventType.RUN_COMPLETED and event.run_result:
                    if store is not None:
                        result = event.run_result
                        redacted_answer = (
                            redact(result.answer) if redact and result.answer else result.answer
                        )
                        finalize_won = await store.finalize_run(
                            run_id,
                            status=result.status.value,
                            answer=redacted_answer,
                            iteration_count=result.iteration_count,
                            total_usage=result.usage,
                            expected_current_status="running",
                        )
                        if finalize_won:
                            await _emit_event(
                                store,
                                run_id,
                                "run.completed",
                                sequencer,
                                {"status": result.status.value},
                            )
                    yield event

                elif event.type == RunEventType.RUN_PAUSED and event.run_result:
                    if store is not None:
                        result = event.run_result
                        pause_state_obj: PauseState = result.meta["pause_state"]
                        await store.pause_run(
                            run_id,
                            status=result.status.value,
                            pause_data=pause_state_obj.to_dict(),
                            iteration_count=result.iteration_count,
                        )
                        await _emit_event(
                            store,
                            run_id,
                            "run.paused",
                            sequencer,
                            {
                                "status": result.status.value,
                                "pending_tool_calls": [
                                    {
                                        "id": tc.id,
                                        "name": tc.name,
                                        "target": pause_state_obj.pending_targets.get(tc.id),
                                    }
                                    for tc in pause_state_obj.pending_tool_calls
                                ],
                            },
                        )
                    yield event

                else:
                    # TEXT_DELTA, TOOL_USE_START, TOOL_USE_END, TOOL_RESULT — pass through
                    yield event

        except Exception as exc:
            # Persist error, yield RUN_ERROR, return cleanly. No re-raise.
            # Covers both setup failures and loop execution errors.
            if store is not None:
                error_won = False
                try:
                    redacted_err = redact(str(exc)) if redact else str(exc)
                    error_won = await store.finalize_run(
                        run_id,
                        status=RunStatus.ERROR.value,
                        error=redacted_err,
                        total_usage=None,
                        expected_current_status="running",
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist ERROR status for run %s", run_id, exc_info=True
                    )
                if error_won:
                    await _emit_event(
                        store, run_id, "run.error", sequencer, {"error": str(exc)[:500]}
                    )

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=RunStatus.ERROR,
                    error=str(exc),
                ),
                error=str(exc),
            )

        finally:
            if ctx_token is not None:
                reset_delegation_context(ctx_token)

    async def _cleanup() -> None:
        """CAS-guarded cancellation — only if run is still RUNNING.

        Uses the shared state dict to access the state store resolved
        by the generator. If the generator never started, state_store
        is None (or the pre-provided value) and cleanup is a no-op for
        the lazy case.
        """
        store = _shared.get("state_store")
        sequencer = _shared.get("sequencer")
        if store is not None:
            try:
                cancel_won = await store.finalize_run(
                    run_id,
                    status=RunStatus.CANCELLED.value,
                    expected_current_status="running",
                )
                if cancel_won and sequencer:
                    await _emit_event(
                        store, run_id, "run.cancelled", sequencer, {}
                    )
            except Exception:
                logger.warning(
                    "Failed to cancel run %s during stream cleanup", run_id, exc_info=True
                )

    return _RunStream(run_id=run_id, generator=_generate(), cleanup=_cleanup)


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


class _ResumeContext:
    """Prepared state for re-entering the loop after a resume.

    Built by _prepare_resume(), consumed by both _resume_core() (batch)
    and resume_stream() (streaming). Does NOT own claiming — that happens
    before prepare is called.
    """

    __slots__ = (
        "history", "observer", "sequencer", "pause_state",
        "resolved_loop", "resolved_strategy",
    )

    def __init__(
        self,
        *,
        history: list[Message],
        observer: Any,
        sequencer: EventSequencer,
        pause_state: PauseState,
        resolved_loop: Loop,
        resolved_strategy: Strategy,
    ) -> None:
        self.history = history
        self.observer = observer
        self.sequencer = sequencer
        self.pause_state = pause_state
        self.resolved_loop = resolved_loop
        self.resolved_strategy = resolved_strategy


async def _prepare_resume(
    run_id: str,
    pause_state: PauseState,
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
) -> _ResumeContext:
    """Build everything needed to re-enter the loop after a resume.

    Receives an already-loaded PauseState — does NOT load pause data or
    claim the run. Those steps are owned by the caller (_resume_core for
    direct resume, resume_stream for streaming, resume_claimed for bridge).

    Steps performed:
      1. Initialize sequencer from DB (continues across pause boundaries)
      2. Record durable run.resumed event
      3. Build resume history (inject tool results or user input)
      4. Create observer with correct trace offset
      5. Notify observer of injected messages
    """
    from dendrux.runtime.observer import PersistenceObserver
    from dendrux.tool import get_tool_def
    from dendrux.types import Message, Role

    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or agent.loop or ReActLoop()

    # 1. Initialize sequencer from DB max (continues across pause boundaries)
    existing_events = await state_store.get_run_events(run_id)
    max_seq = max((e.sequence_index for e in existing_events), default=-1)
    sequencer = EventSequencer(initial=max_seq + 1)

    # 2. Record resume event with enriched payload for dashboard
    resume_data: dict[str, Any] = {"resumed_from": expected_status}
    if tool_results is not None:
        resume_data["submitted_results"] = [
            {"call_id": tr.call_id, "name": tr.name, "success": tr.success} for tr in tool_results
        ]
    elif user_input is not None:
        resume_data["user_input"] = redact(user_input) if redact else user_input
    await _emit_event(state_store, run_id, "run.resumed", sequencer, resume_data)

    # 3. Build resume history
    history = list(pause_state.history)

    if tool_results is not None:
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
        history.append(Message(role=Role.USER, content=user_input))

    # 4. Create observer with correct trace offset
    traces = await state_store.get_traces(run_id)
    trace_order_offset = max((t.order_index for t in traces), default=-1) + 1

    target_lookup = {}
    for fn in agent.tools:
        td = get_tool_def(fn)
        target_lookup[td.name] = td.target
    persistence_obs = PersistenceObserver(
        state_store,
        run_id,
        model=provider.model,
        provider_name=type(provider).__name__,
        target_lookup=target_lookup,
        redact=redact,
        initial_order_index=trace_order_offset,
        event_sequencer=sequencer,
    )

    observer: Any
    if extra_observer is not None:
        from dendrux.observers.composite import CompositeObserver

        observer = CompositeObserver([persistence_obs, extra_observer])
    else:
        observer = persistence_obs

    # 5. Notify observer of injected messages and tool completions
    if tool_results is not None:
        pending_by_id = {tc.id: tc for tc in pause_state.pending_tool_calls}
        injected_start = len(pause_state.history)
        for i, tr in enumerate(tool_results):
            await observer.on_message_appended(history[injected_start + i], pause_state.iteration)
            await observer.on_tool_completed(pending_by_id[tr.call_id], tr, pause_state.iteration)
    elif user_input is not None:
        await observer.on_message_appended(history[-1], pause_state.iteration)

    return _ResumeContext(
        history=history,
        observer=observer,
        sequencer=sequencer,
        pause_state=pause_state,
        resolved_loop=resolved_loop,
        resolved_strategy=resolved_strategy,
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
    _skip_claim: bool = False,
) -> RunResult:
    """Shared resume logic for tool results and clarification input.

    Args:
        extra_observer: Optional additional LoopObserver (e.g. TransportObserver)
            to compose with the PersistenceObserver for SSE streaming during resume.
        _skip_claim: Internal flag — skip the atomic claim step when the caller
            has already claimed via submit_and_claim(). Used by resume_claimed().
    """
    # 1. Load pause state
    raw_pause = await state_store.get_pause_state(run_id)
    if raw_pause is None:
        raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
    pause_state = PauseState.from_dict(raw_pause)

    # 1b. Verify agent identity — fail closed on mismatch (H-003)
    if pause_state.agent_name and pause_state.agent_name != agent.name:
        raise ValueError(
            f"Agent name mismatch: run '{run_id}' was paused by "
            f"'{pause_state.agent_name}', but resume called with "
            f"'{agent.name}'. This can cause state corruption."
        )

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
    if not _skip_claim:
        claimed = await state_store.claim_paused_run(run_id, expected_status=expected_status)
        if not claimed:
            raise ValueError(
                f"Run '{run_id}' is not in status '{expected_status}' — "
                f"cannot resume. It may have been claimed by another caller."
            )

    # 4-8. Prepare history, observer, sequencer (shared with resume_stream)
    ctx = await _prepare_resume(
        run_id,
        pause_state,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        redact=redact,
        expected_status=expected_status,
        tool_results=tool_results,
        user_input=user_input,
        extra_observer=extra_observer,
    )

    # 9. Set delegation context — resumed run is the active parent for any
    #    nested agent.run() calls spawned during this resume cycle.
    run_record = await state_store.get_run(run_id)
    resume_delegation_level = run_record.delegation_level if run_record else 0
    resume_ctx = DelegationContext(
        run_id=run_id,
        delegation_level=resume_delegation_level,
        persisted=True,
        store_identity=get_store_identity(state_store),
    )
    ctx_token = set_delegation_context(resume_ctx)

    # 10. Re-enter loop (batch)
    try:
        result = await ctx.resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=ctx.resolved_strategy,
            user_input="",
            run_id=run_id,
            observer=ctx.observer,
            initial_history=ctx.history,
            initial_steps=ctx.pause_state.steps,
            iteration_offset=ctx.pause_state.iteration,
            initial_usage=ctx.pause_state.usage,
        )

        # 10. Finalize or pause again
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
                ctx.sequencer,
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
            result.meta["_finalize_won"] = finalize_won
            if finalize_won:
                await _emit_event(
                    state_store, run_id, "run.completed", ctx.sequencer,
                    {"status": result.status.value},
                )

        return result

    except Exception as exc:
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
        if error_won:
            await _emit_event(
                state_store, run_id, "run.error", ctx.sequencer, {"error": str(exc)[:500]}
            )
        raise

    finally:
        reset_delegation_context(ctx_token)


def resume_stream(
    run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tool_results: list[ToolResult] | None = None,
    user_input: str | None = None,
    redact: Callable[[str], str] | None = None,
    extra_observer: Any | None = None,
) -> RunStream:
    """Stream a resumed run as RunEvents, returning a RunStream.

    Synchronous — returns immediately. All async setup (store resolution,
    pause state load, claim, history build) runs lazily on first iteration.

    Same error contract as run_stream(): exceptions are caught, persisted,
    and yielded as RUN_ERROR events. No exception is raised to the consumer.

    First event is RUN_RESUMED (not RUN_STARTED) carrying the existing run_id.
    """
    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    _shared: dict[str, Any] = {"state_store": state_store, "sequencer": None}

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        store = state_store
        ctx_token = None

        try:
            # 1. Resolve state store
            if store is None and state_store_resolver is not None:
                store = await state_store_resolver()
            _shared["state_store"] = store

            if store is None:
                raise ValueError(
                    "resume_stream() requires persistence. "
                    "Pass database_url or state_store to the agent."
                )

            # 2. Load pause state
            raw_pause = await store.get_pause_state(run_id)
            if raw_pause is None:
                raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
            pause_state = PauseState.from_dict(raw_pause)

            # 2b. Verify agent identity
            if pause_state.agent_name and pause_state.agent_name != agent.name:
                raise ValueError(
                    f"Agent name mismatch: run '{run_id}' was paused by "
                    f"'{pause_state.agent_name}', but resume called with "
                    f"'{agent.name}'."
                )

            # 3. Validate tool results
            if tool_results is not None:
                pending_ids = {tc.id for tc in pause_state.pending_tool_calls}
                provided_ids = {tr.call_id for tr in tool_results}
                if provided_ids != pending_ids:
                    raise ValueError(
                        f"Tool result call_ids {provided_ids} do not match "
                        f"pending tool call_ids {pending_ids}."
                    )

            # 4. Determine expected status and claim
            if tool_results is not None:
                expected = RunStatus.WAITING_CLIENT_TOOL.value
            else:
                expected = RunStatus.WAITING_HUMAN_INPUT.value

            claimed = await store.claim_paused_run(run_id, expected_status=expected)
            if not claimed:
                raise ValueError(
                    f"Run '{run_id}' is not in status '{expected}' — "
                    f"cannot resume. It may have been claimed by another caller."
                )

            # 5. Prepare (shared helper — history, observer, sequencer)
            ctx = await _prepare_resume(
                run_id,
                pause_state,
                state_store=store,
                agent=agent,
                provider=provider,
                redact=redact,
                expected_status=expected,
                tool_results=tool_results,
                user_input=user_input,
                extra_observer=extra_observer,
            )
            _shared["sequencer"] = ctx.sequencer

            # 6. Set delegation context for the generator lifetime
            run_record = await store.get_run(run_id)
            resume_level = run_record.delegation_level if run_record else 0
            resume_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=resume_level,
                persisted=True,
                store_identity=get_store_identity(store),
            )
            ctx_token = set_delegation_context(resume_ctx)

            # 7. Emit RUN_RESUMED as first event
            yield RunEvent(type=RunEventType.RUN_RESUMED, run_id=run_id)

            # 8. Stream the loop
            async for event in ctx.resolved_loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=ctx.resolved_strategy,
                user_input="",
                run_id=run_id,
                observer=ctx.observer,
                initial_history=ctx.history,
                initial_steps=ctx.pause_state.steps,
                iteration_offset=ctx.pause_state.iteration,
                initial_usage=ctx.pause_state.usage,
            ):
                # Persist loop outcomes before forwarding
                if event.type == RunEventType.RUN_COMPLETED and event.run_result:
                    result = event.run_result
                    redacted_answer = (
                        redact(result.answer) if redact and result.answer else result.answer
                    )
                    finalize_won = await store.finalize_run(
                        run_id,
                        status=result.status.value,
                        answer=redacted_answer,
                        iteration_count=result.iteration_count,
                        total_usage=result.usage,
                        expected_current_status="running",
                    )
                    if finalize_won:
                        await _emit_event(
                            store, run_id, "run.completed", ctx.sequencer,
                            {"status": result.status.value},
                        )
                    yield event

                elif event.type == RunEventType.RUN_PAUSED and event.run_result:
                    result = event.run_result
                    pause_state_obj: PauseState = result.meta["pause_state"]
                    await store.pause_run(
                        run_id,
                        status=result.status.value,
                        pause_data=pause_state_obj.to_dict(),
                        iteration_count=result.iteration_count,
                    )
                    await _emit_event(
                        store,
                        run_id,
                        "run.paused",
                        ctx.sequencer,
                        {
                            "status": result.status.value,
                            "pending_tool_calls": [
                                {
                                    "id": tc.id,
                                    "name": tc.name,
                                    "target": pause_state_obj.pending_targets.get(tc.id),
                                }
                                for tc in pause_state_obj.pending_tool_calls
                            ],
                        },
                    )
                    yield event

                else:
                    yield event

        except Exception as exc:
            store = _shared.get("state_store")
            sequencer = _shared.get("sequencer")
            if store is not None:
                error_won = False
                try:
                    redacted_err = redact(str(exc)) if redact else str(exc)
                    error_won = await store.finalize_run(
                        run_id,
                        status=RunStatus.ERROR.value,
                        error=redacted_err,
                        total_usage=None,
                        expected_current_status="running",
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist ERROR status for run %s", run_id, exc_info=True
                    )
                if error_won:
                    await _emit_event(
                        store, run_id, "run.error", sequencer,
                        {"error": str(exc)[:500]},
                    )

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=RunStatus.ERROR,
                    error=str(exc),
                ),
                error=str(exc),
            )

        finally:
            if ctx_token is not None:
                reset_delegation_context(ctx_token)

    async def _cleanup() -> None:
        """CAS-guarded cancellation for abandoned resume streams."""
        store = _shared.get("state_store")
        sequencer = _shared.get("sequencer")
        if store is not None:
            try:
                cancel_won = await store.finalize_run(
                    run_id,
                    status=RunStatus.CANCELLED.value,
                    expected_current_status="running",
                )
                if cancel_won and sequencer:
                    await _emit_event(
                        store, run_id, "run.cancelled", sequencer, {}
                    )
            except Exception:
                logger.warning(
                    "Failed to cancel run %s during resume stream cleanup",
                    run_id,
                    exc_info=True,
                )

    return _RunStream(run_id=run_id, generator=_generate(), cleanup=_cleanup)


async def resume_claimed(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    redact: Callable[[str], str] | None = None,
    extra_observer: Any | None = None,
) -> RunResult:
    """Resume a run that was already claimed via submit_and_claim().

    Internal helper for the bridge's persist-first handoff. NOT a public API.

    The bridge's background task calls submit_and_claim() to atomically save
    submitted data and transition WAITING → RUNNING. Once that succeeds, it
    calls this helper to actually re-enter the loop.

    Validates before proceeding:
      - pause_data exists (run was paused)
      - submitted_tool_results or submitted_user_input present in blob
      - for tool results: call_ids match pending_tool_calls

    Args:
        run_id: The run ID (already in RUNNING status after submit_and_claim).
        state_store: Persistence backend.
        agent: Agent definition.
        provider: LLM provider.
        redact: Optional redaction policy.
        extra_observer: Optional TransportObserver for SSE streaming.

    Returns:
        RunResult from the resumed loop.

    Raises:
        ValueError: If pause_data is missing, no submitted data found,
            or submitted call_ids don't match pending tool calls.
    """
    from dendrux.types import ToolResult as ToolResultType

    # 1. Load pause_data (submit_and_claim merged submitted data into it)
    raw_pause = await state_store.get_pause_state(run_id)
    if raw_pause is None:
        raise ValueError(
            f"Run '{run_id}' has no pause state after claim — "
            "this indicates a bug in submit_and_claim."
        )

    # 1b. Verify agent identity (H-003)
    paused_agent = raw_pause.get("agent_name", "")
    if paused_agent and paused_agent != agent.name:
        raise ValueError(
            f"Agent name mismatch: run '{run_id}' was paused by "
            f"'{paused_agent}', but resume called with '{agent.name}'."
        )

    # 2. Detect submission type and extract data
    submitted_results = raw_pause.get("submitted_tool_results")
    submitted_input = raw_pause.get("submitted_user_input")

    if submitted_results is not None:
        # 3a. Convert dicts to typed ToolResult objects
        tool_results = [
            ToolResultType(
                name=r["name"],
                call_id=r["call_id"],
                payload=r["payload"],
                success=r.get("success", True),
                error=r.get("error"),
                duration_ms=r.get("duration_ms", 0),
            )
            for r in submitted_results
        ]

        # 3b. Validate call_ids match pending tool calls
        pending_ids = {tc["id"] for tc in raw_pause.get("pending_tool_calls", [])}
        provided_ids = {tr.call_id for tr in tool_results}
        if provided_ids != pending_ids:
            raise ValueError(
                f"Submitted tool result call_ids {provided_ids} do not match "
                f"pending tool call_ids {pending_ids}."
            )

        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            redact=redact,
            expected_status="running",
            tool_results=tool_results,
            extra_observer=extra_observer,
            _skip_claim=True,
        )

    if submitted_input is not None:
        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            redact=redact,
            expected_status="running",
            user_input=submitted_input,
            extra_observer=extra_observer,
            _skip_claim=True,
        )

    raise ValueError(
        f"Run '{run_id}' has pause_data but no submitted_tool_results "
        "or submitted_user_input — nothing to resume with."
    )
