"""Agent runner — the entry point for executing agents.

Takes an Agent definition and runs it through the loop with an explicit
provider and strategy. This is the top-level API developers interact with.

Sprint 1: caller provides the LLM provider instance, defaults to
NativeToolCalling strategy and ReActLoop. Future sprints add provider
registry (model string → provider resolution), strategy selection from
agent config, and more loop types.

Sprint 2 adds optional state_store for persistence. When provided:
  - Runner owns the run_id (generates it, passes to loop)
  - PersistenceRecorder records traces, tool calls, and usage (fail-closed)
  - finalize_run() is called in try/finally to guarantee persistence
  - Developer's extra_notifier is passed separately (best-effort)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload

from dendrux._sentinel import _UnsetType
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
from dendrux.types import (
    PauseState,
    RunAlreadyActiveError,
    RunEventType,
    RunStatus,
    UsageStats,
    compute_idempotency_fingerprint,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import Message, RunEvent, RunResult, RunStream, ToolResult

logger = logging.getLogger(__name__)


class EventSequencer:
    """Monotonic sequence counter for run_events within a single run.

    Shared between the runner (run-level events) and the PersistenceRecorder
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
    """Record a durable run-level lifecycle event. Fail-closed with retry.

    Runner lifecycle events (run.started, run.paused, run.completed,
    run.error, run.resumed, run.cancelled) are authoritative truth —
    they get the same retry/propagate contract as PersistenceRecorder's
    fail-closed writes.
    """
    if state_store is None:
        return
    seq = sequencer.next() if sequencer else 0

    from dendrux.runtime.persistence import _retry_critical

    async def _write() -> None:
        await state_store.save_run_event(
            run_id,
            event_type=event_type,
            sequence_index=seq,
            correlation_id=correlation_id,
            data=data,
        )

    await _retry_critical(_write, label=f"emit_{event_type}", run_id=run_id)


async def _emit_event_safe(
    state_store: StateStore | None,
    run_id: str,
    event_type: str,
    sequencer: EventSequencer | None = None,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Best-effort lifecycle event — for use inside streaming error handlers.

    In streaming generators, the error handler itself must not raise
    (the stream contract promises no exception escapes). This variant
    logs and swallows persistence failures instead of propagating.
    Used ONLY in the ``except`` block of streaming generators.
    """
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
        logger.warning(
            "Failed to record %s for run %s (in error handler)", event_type, run_id, exc_info=True
        )


_UNSET_DEPTH = _UnsetType()


def _resolve_max_delegation_depth(
    requested: int | None | _UnsetType,
    parent_ctx: DelegationContext | None,
) -> int | None:
    """Resolve effective max_delegation_depth from explicit value + parent context.

    Pure policy helper — no I/O, no persistence, no parent linking.

    Resolution order:
      1. Explicit requested value (even None = unbounded)
      2. Inherit from parent context
      3. Default 10 (root run)

    Tighten-only rule: a child can never loosen the parent's limit.
    Warns (deduplicated per parent run) when an explicit child value is clamped.
    """
    # --- Validation ---
    if (
        not isinstance(requested, _UnsetType)
        and requested is not None
        and (not isinstance(requested, int) or requested < 0)
    ):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, got {requested!r}"
        )

    # --- Resolution ---
    if not isinstance(requested, _UnsetType):
        effective = requested
    elif parent_ctx is not None:
        effective = parent_ctx.max_delegation_depth
    else:
        effective = 10

    # --- Tighten-only clamp ---
    if parent_ctx is not None and parent_ctx.max_delegation_depth is not None:
        parent_limit = parent_ctx.max_delegation_depth
        if effective is None or (isinstance(effective, int) and effective > parent_limit):
            # Warn only on explicit loosen attempts, deduplicated per parent run.
            if (
                not isinstance(requested, _UnsetType)
                and "depth_clamped" not in parent_ctx.warned_mismatches
            ):
                logger.warning(
                    "max_delegation_depth=%r clamped to %d (inherited parent limit)",
                    effective,
                    parent_limit,
                )
                parent_ctx.warned_mismatches.add("depth_clamped")
            effective = parent_limit

    return effective


@overload
async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    tenant_id: str | None = ...,
    metadata: dict[str, Any] | None = ...,
    redact: Callable[[str], str] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    idempotency_key: str | None = ...,
    **kwargs: Any,
) -> RunResult: ...


@overload
async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    tenant_id: str | None = ...,
    metadata: dict[str, Any] | None = ...,
    redact: Callable[[str], str] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    idempotency_key: str | None = ...,
    **kwargs: Any,
) -> RunResult: ...


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
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    idempotency_key: str | None = None,
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
        extra_notifier: Optional additional LoopNotifier for best-effort
            notifications (e.g. ConsoleNotifier, TransportNotifier for SSE).
        max_delegation_depth: Maximum allowed delegation depth for the
            run tree. Default 10. None means unbounded. Propagated to
            all child runs via contextvar. Raises DelegationDepthExceededError
            if a child run would exceed this depth.
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
    # Validate + resolve depth early — before any side effects.
    # _resolve_max_delegation_depth is a pure policy helper; side effect
    # is limited to dedup warning state on parent_ctx.warned_mismatches.
    parent_ctx = get_delegation_context()
    effective_max_depth = _resolve_max_delegation_depth(max_delegation_depth, parent_ctx)

    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or agent.loop or ReActLoop()
    provider_kwargs = dict(kwargs) if kwargs else {}

    run_id = generate_ulid()
    recorder: LoopRecorder | None = None
    sequencer = EventSequencer()

    # --- Delegation context ---
    parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)

    if state_store is not None:
        # Create the run record before the loop starts
        # Apply redaction to user input before persistence
        redacted_input = redact(user_input) if redact else user_input
        # Merge loop type + depth limit into developer metadata
        run_meta = dict(metadata) if metadata else {}
        run_meta["dendrux.loop"] = type(resolved_loop).__name__
        if effective_max_depth is not None:
            run_meta["dendrux.max_delegation_depth"] = effective_max_depth

        # Compute idempotency fingerprint if key is provided
        idem_fingerprint: str | None = None
        if idempotency_key is not None:
            idem_fingerprint = compute_idempotency_fingerprint(agent.name, user_input)

        create_result = await state_store.create_run(
            run_id,
            agent.name,
            input_data={"input": redacted_input},
            model=provider.model,
            strategy=type(resolved_strategy).__name__,
            parent_run_id=parent_run_id,
            delegation_level=delegation_level,
            tenant_id=tenant_id,
            meta=run_meta,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idem_fingerprint,
        )

        # Handle idempotency outcomes
        if create_result.outcome == "existing_terminal":
            return await _build_cached_result(state_store, create_result.run_id)
        if create_result.outcome == "existing_active":
            raise RunAlreadyActiveError(create_result.run_id, create_result.status)

        # "created" — use the run_id from the result (same as generated)
        run_id = create_result.run_id

        # Create persistence recorder with shared sequencer
        from dendrux.runtime.persistence import PersistenceRecorder
        from dendrux.tool import get_tool_def

        target_lookup = {}
        for fn in agent.tools:
            td = get_tool_def(fn)
            target_lookup[td.name] = td.target
        recorder = PersistenceRecorder(
            state_store,
            run_id,
            model=provider.model,
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

    # Set delegation context for the duration of this run so nested
    # agent.run() calls inside tools inherit the parent link.
    this_ctx = DelegationContext(
        run_id=run_id,
        delegation_level=delegation_level,
        persisted=state_store is not None,
        store_identity=get_store_identity(state_store),
        max_delegation_depth=effective_max_depth,
    )
    ctx_token = set_delegation_context(this_ctx)

    try:
        result = await resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=resolved_strategy,
            user_input=user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=extra_notifier,
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
                logger.error("Failed to persist ERROR status for run %s", run_id, exc_info=True)
            # Only the CAS winner emits the error event
            if error_won:
                await _emit_event(
                    state_store, run_id, "run.error", sequencer, {"error": str(exc)[:500]}
                )
        raise

    finally:
        reset_delegation_context(ctx_token)


async def _raise_resume_claim_failure(
    state_store: StateStore, run_id: str, expected_status: str
) -> None:
    """Raise a descriptive error when resume claim fails.

    Checks the actual run status to give a specific message for swept
    (abandoned/stale) runs instead of the generic "claimed by another caller."
    """
    run = await state_store.get_run(run_id)
    if run is not None and run.status == "error" and run.failure_reason:
        reason = run.failure_reason
        if reason == "abandoned_waiting":
            raise ValueError(
                f"Run '{run_id}' was abandoned (swept as expired waiting run). "
                f"It cannot be resumed."
            )
        if reason in ("stale_running", "never_started"):
            raise ValueError(f"Run '{run_id}' was swept as stale ({reason}). It cannot be resumed.")
    raise ValueError(
        f"Run '{run_id}' is not in status '{expected_status}' — "
        f"cannot resume. It may have been claimed by another caller."
    )


async def _build_cached_result(state_store: StateStore, run_id: str) -> RunResult:
    """Rebuild a summary RunResult from persisted DB state for idempotent dedup.

    This is a *summary*, not a faithful replay of the original RunResult:
    - steps is always [] (AgentStep history is not persisted in recoverable form)
    - meta is empty (developer meta is on the run row, not in RunResult.meta)
    - usage is reconstructed from aggregate columns (total_input_tokens, etc.),
      which may be zero for errored runs where finalize_run got total_usage=None

    The contract is: "return the cached final outcome" — status, answer, error,
    iteration count, and aggregate usage. Not "return the identical object."
    """
    from dendrux.types import RunResult

    run = await state_store.get_run(run_id)
    if run is None:
        raise RuntimeError(f"Idempotent run {run_id} not found in database")

    return RunResult(
        run_id=run.id,
        status=RunStatus(run.status),
        answer=run.answer,
        steps=[],
        iteration_count=run.iteration_count,
        usage=UsageStats(
            input_tokens=run.total_input_tokens,
            output_tokens=run.total_output_tokens,
            total_tokens=run.total_input_tokens + run.total_output_tokens,
            cost_usd=run.total_cost_usd,
        ),
        error=run.error,
    )


async def retry(
    original_run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    redact: Callable[[str], str] | None = None,
    extra_notifier: LoopNotifier | None = None,
    **kwargs: Any,
) -> RunResult:
    """Retry a terminal run with approximate prior context.

    Creates a fresh run seeded with the original run's conversation history
    from persisted traces. The LLM sees the prior conversation and can
    continue from context. This is NOT resume (no pause_data, no exact
    state reconstruction) — it's a new run with approximate context.

    Works for any terminal run (error, cancelled, max_iterations, success).
    Source run must have been executed with a retry-capable loop (not
    SingleCall). The retry agent can differ from the original — different
    model, different tools — the only requirement is that the source run
    has persisted traces to seed from.

    Args:
        original_run_id: The terminal run to retry from.
        agent: Agent definition for the retry run. Can differ from the
            original agent (different model, tools, prompt).
        provider: LLM provider for the retry run.
        state_store: Persistence backend (required — retry needs traces).
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop for the retry. Defaults to ReActLoop.
        tenant_id: Optional tenant ID.
        metadata: Optional developer metadata for the retry run.
        redact: Optional string scrubber.
        extra_notifier: Optional notifier.
        **kwargs: Forwarded to the LLM provider.

    Returns:
        RunResult from the retry run.

    Raises:
        ValueError: If the source run is not terminal, if the source
            run was SingleCall, or if no traces exist.
    """
    from dendrux.types import Message, Role

    # 1. Validate source run is terminal
    source_run = await state_store.get_run(original_run_id)
    if source_run is None:
        raise ValueError(f"Run '{original_run_id}' not found.")

    terminal_statuses = {"success", "error", "cancelled", "max_iterations"}
    if source_run.status not in terminal_statuses:
        raise ValueError(
            f"Run '{original_run_id}' is in status '{source_run.status}' — "
            f"only terminal runs (error, cancelled, max_iterations, success) can be retried."
        )

    # 2. Validate source run was not SingleCall (no iterative history to seed)
    source_loop = (source_run.meta or {}).get("dendrux.loop", "")
    if source_loop == "SingleCall":
        raise ValueError(
            "SingleCall runs cannot be retried with context — "
            "they have no iterative history to seed. Call agent.run() instead."
        )

    resolved_loop = loop or agent.loop or ReActLoop()

    # 3. Read traces and reconstruct approximate history
    traces = await state_store.get_traces(original_run_id)
    initial_history: list[Message] = []
    for trace in sorted(traces, key=lambda t: t.order_index):
        try:
            role = Role(trace.role)
        except ValueError:
            continue  # skip unknown roles

        meta = trace.meta or {}
        msg_kwargs: dict[str, Any] = {"role": role, "content": trace.content}

        # Reconstruct TOOL message fields from trace meta
        if role == Role.TOOL:
            msg_kwargs["name"] = meta.get("tool_name", "unknown")
            msg_kwargs["call_id"] = meta.get("call_id", generate_ulid())

        # Reconstruct ASSISTANT tool_calls from trace meta
        if role == Role.ASSISTANT and "tool_calls" in meta:
            from dendrux.types import ToolCall

            msg_kwargs["tool_calls"] = [
                ToolCall(
                    name=tc["name"],
                    params=tc.get("params"),
                    id=tc.get("id", generate_ulid()),
                    provider_tool_call_id=tc.get("provider_tool_call_id"),
                )
                for tc in meta["tool_calls"]
            ]

        initial_history.append(Message(**msg_kwargs))

    if not initial_history:
        raise ValueError(
            f"Run '{original_run_id}' has no persisted traces. Cannot retry without prior context."
        )

    # 4. Extract the original user input from the first USER message
    user_input = next(
        (m.content for m in initial_history if m.role == Role.USER),
        "Continue from previous attempt.",
    )

    # 5. Create the retry run
    resolved_strategy = strategy or NativeToolCalling()
    provider_kwargs = dict(kwargs) if kwargs else {}

    run_id = generate_ulid()
    sequencer = EventSequencer()

    parent_ctx = get_delegation_context()
    parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)
    effective_max_depth = _resolve_max_delegation_depth(_UNSET_DEPTH, parent_ctx)

    redacted_input = redact(user_input) if redact else user_input
    run_meta = dict(metadata) if metadata else {}
    run_meta["dendrux.loop"] = type(resolved_loop).__name__
    run_meta["dendrux.retry_of"] = original_run_id

    create_result = await state_store.create_run(
        run_id,
        agent.name,
        input_data={"input": redacted_input},
        model=provider.model,
        strategy=type(resolved_strategy).__name__,
        parent_run_id=parent_run_id,
        delegation_level=delegation_level,
        tenant_id=tenant_id,
        meta=run_meta,
        retry_of_run_id=original_run_id,
    )
    run_id = create_result.run_id

    # Create persistence recorder
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.tool import get_tool_def

    target_lookup = {}
    for fn in agent.tools:
        td = get_tool_def(fn)
        target_lookup[td.name] = td.target
    recorder = PersistenceRecorder(
        state_store,
        run_id,
        model=provider.model,
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
        {"agent_name": agent.name, "retry_of": original_run_id},
    )

    this_ctx = DelegationContext(
        run_id=run_id,
        delegation_level=delegation_level,
        persisted=True,
        store_identity=get_store_identity(state_store),
        max_delegation_depth=effective_max_depth,
    )
    ctx_token = set_delegation_context(this_ctx)

    try:
        result = await resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=resolved_strategy,
            user_input=user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=extra_notifier,
            initial_history=initial_history,
            provider_kwargs=provider_kwargs or None,
        )

        if result.status in (RunStatus.WAITING_CLIENT_TOOL, RunStatus.WAITING_HUMAN_INPUT):
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
            redacted_answer = redact(result.answer) if redact and result.answer else result.answer
            finalize_won = await state_store.finalize_run(
                run_id,
                status=result.status.value,
                answer=redacted_answer,
                iteration_count=result.iteration_count,
                total_usage=result.usage,
                expected_current_status="running",
            )
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
            logger.error("Failed to persist ERROR status for retry run %s", run_id, exc_info=True)
        if error_won:
            await _emit_event(
                state_store, run_id, "run.error", sequencer, {"error": str(exc)[:500]}
            )
        raise

    finally:
        reset_delegation_context(ctx_token)


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None = ...,
    metadata: dict[str, Any] | None = ...,
    redact: Callable[[str], str] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    **kwargs: Any,
) -> RunStream: ...


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None = ...,
    metadata: dict[str, Any] | None = ...,
    redact: Callable[[str], str] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    **kwargs: Any,
) -> RunStream: ...


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
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    **kwargs: Any,
) -> RunStream:
    """Stream an agent run as RunEvents, returning a RunStream.

    Synchronous — returns immediately with run_id available. All async
    setup (DB row, notifiers) runs lazily on first iteration.

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
    # Validate depth early — synchronous, before generator starts.
    # Full resolution (parent context, clamping) happens inside _generate()
    # because parent_ctx requires async state_store resolution first.
    # But validation of the raw input can fail fast here.
    if (
        not isinstance(max_delegation_depth, _UnsetType)
        and max_delegation_depth is not None
        and (not isinstance(max_delegation_depth, int) or max_delegation_depth < 0)
    ):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, "
            f"got {max_delegation_depth!r}"
        )

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
        (state_store_resolver, create_run, notifier init) are also caught
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
            eff_max_depth = _resolve_max_delegation_depth(max_delegation_depth, parent_ctx)

            recorder: LoopRecorder | None = None

            if store is not None:
                redacted_input = redact(user_input) if redact else user_input
                run_meta = dict(metadata) if metadata else {}
                run_meta["dendrux.loop"] = type(resolved_loop).__name__
                if eff_max_depth is not None:
                    run_meta["dendrux.max_delegation_depth"] = eff_max_depth
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

                from dendrux.runtime.persistence import PersistenceRecorder
                from dendrux.tool import get_tool_def

                target_lookup = {}
                for fn in agent.tools:
                    td = get_tool_def(fn)
                    target_lookup[td.name] = td.target
                recorder = PersistenceRecorder(
                    store,
                    run_id,
                    model=provider.model,
                    provider_name=type(provider).__name__,
                    target_lookup=target_lookup,
                    redact=redact,
                    event_sequencer=sequencer,
                )

            # 2. Set delegation context for the generator lifetime
            this_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=delegation_level,
                persisted=store is not None,
                store_identity=get_store_identity(store),
                max_delegation_depth=eff_max_depth,
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
                recorder=recorder,
                notifier=extra_notifier,
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
                    logger.error("Failed to persist ERROR status for run %s", run_id, exc_info=True)
                if error_won:
                    await _emit_event_safe(
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
                    await _emit_event(store, run_id, "run.cancelled", sequencer, {})
            except Exception:
                logger.error("Failed to cancel run %s during stream cleanup", run_id, exc_info=True)

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
    extra_notifier: LoopNotifier | None = None,
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
        extra_notifier: Optional additional notifier for SSE streaming.
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
        extra_notifier=extra_notifier,
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
    extra_notifier: LoopNotifier | None = None,
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
        extra_notifier: Optional additional notifier for SSE streaming.
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
        extra_notifier=extra_notifier,
    )


class _ResumeContext:
    """Prepared state for re-entering the loop after a resume.

    Built by _prepare_resume(), consumed by both _resume_core() (batch)
    and resume_stream() (streaming). Does NOT own claiming — that happens
    before prepare is called.
    """

    __slots__ = (
        "history",
        "recorder",
        "notifier",
        "sequencer",
        "pause_state",
        "resolved_loop",
        "resolved_strategy",
    )

    def __init__(
        self,
        *,
        history: list[Message],
        recorder: LoopRecorder,
        notifier: LoopNotifier | None,
        sequencer: EventSequencer,
        pause_state: PauseState,
        resolved_loop: Loop,
        resolved_strategy: Strategy,
    ) -> None:
        self.history = history
        self.recorder = recorder
        self.notifier = notifier
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
    extra_notifier: LoopNotifier | None = None,
) -> _ResumeContext:
    """Build everything needed to re-enter the loop after a resume.

    Receives an already-loaded PauseState — does NOT load pause data or
    claim the run. Those steps are owned by the caller (_resume_core for
    direct resume, resume_stream for streaming, resume_claimed for bridge).

    Steps performed:
      1. Initialize sequencer from DB (continues across pause boundaries)
      2. Record durable run.resumed event
      3. Build resume history (inject tool results or user input)
      4. Create recorder with correct trace offset
      5. Notify recorder of injected messages
    """
    from dendrux.runtime.persistence import PersistenceRecorder
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

    # 4. Create recorder with correct trace offset
    traces = await state_store.get_traces(run_id)
    trace_order_offset = max((t.order_index for t in traces), default=-1) + 1

    target_lookup = {}
    for fn in agent.tools:
        td = get_tool_def(fn)
        target_lookup[td.name] = td.target
    recorder = PersistenceRecorder(
        state_store,
        run_id,
        model=provider.model,
        provider_name=type(provider).__name__,
        target_lookup=target_lookup,
        redact=redact,
        initial_order_index=trace_order_offset,
        event_sequencer=sequencer,
    )

    # 5. Record + notify injected messages and tool completions
    #    Recorder is fail-closed (exceptions propagate).
    #    Notifier is best-effort (swallowed via notify helpers).
    from dendrux.loops._helpers import notify_message as _notify_msg
    from dendrux.loops._helpers import notify_tool as _notify_tool

    if tool_results is not None:
        pending_by_id = {tc.id: tc for tc in pause_state.pending_tool_calls}
        injected_start = len(pause_state.history)
        for i, tr in enumerate(tool_results):
            msg = history[injected_start + i]
            tc = pending_by_id[tr.call_id]
            await recorder.on_message_appended(msg, pause_state.iteration)
            await recorder.on_tool_completed(tc, tr, pause_state.iteration)
            await _notify_msg(extra_notifier, msg, pause_state.iteration)
            await _notify_tool(extra_notifier, tc, tr, pause_state.iteration)
    elif user_input is not None:
        msg = history[-1]
        await recorder.on_message_appended(msg, pause_state.iteration)
        await _notify_msg(extra_notifier, msg, pause_state.iteration)

    return _ResumeContext(
        history=history,
        recorder=recorder,
        notifier=extra_notifier,
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
    extra_notifier: LoopNotifier | None = None,
    _skip_claim: bool = False,
) -> RunResult:
    """Shared resume logic for tool results and clarification input.

    Args:
        extra_notifier: Optional additional LoopNotifier (e.g. TransportNotifier)
            for SSE streaming during resume. Passed separately from the
            PersistenceRecorder — recorder handles persistence, notifier handles
            notifications.
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
            await _raise_resume_claim_failure(state_store, run_id, expected_status)

    # 4-8. Prepare history, notifier, sequencer (shared with resume_stream)
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
        extra_notifier=extra_notifier,
    )

    # 9. Set delegation context — resumed run is the active parent for any
    #    nested agent.run() calls spawned during this resume cycle.
    #    Read max_delegation_depth from persisted metadata so the depth
    #    guard survives pause/resume cycles.
    run_record = await state_store.get_run(run_id)
    resume_delegation_level = run_record.delegation_level if run_record else 0
    resume_max_depth: int | None = None
    if run_record and run_record.meta:
        raw = run_record.meta.get("dendrux.max_delegation_depth")
        if isinstance(raw, int):
            resume_max_depth = raw
    resume_ctx = DelegationContext(
        run_id=run_id,
        delegation_level=resume_delegation_level,
        persisted=True,
        store_identity=get_store_identity(state_store),
        max_delegation_depth=resume_max_depth,
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
            recorder=ctx.recorder,
            notifier=ctx.notifier,
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
                    state_store,
                    run_id,
                    "run.completed",
                    ctx.sequencer,
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
            logger.error("Failed to persist ERROR status for run %s", run_id, exc_info=True)
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
    extra_notifier: LoopNotifier | None = None,
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
                await _raise_resume_claim_failure(store, run_id, expected)

            # 5. Prepare (shared helper — history, notifier, sequencer)
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
                extra_notifier=extra_notifier,
            )
            _shared["sequencer"] = ctx.sequencer

            # 6. Set delegation context for the generator lifetime.
            #    Read max_delegation_depth from persisted metadata so the
            #    depth guard survives pause/resume cycles.
            run_record = await store.get_run(run_id)
            resume_level = run_record.delegation_level if run_record else 0
            resume_max_depth: int | None = None
            if run_record and run_record.meta:
                raw = run_record.meta.get("dendrux.max_delegation_depth")
                if isinstance(raw, int):
                    resume_max_depth = raw
            resume_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=resume_level,
                persisted=True,
                store_identity=get_store_identity(store),
                max_delegation_depth=resume_max_depth,
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
                recorder=ctx.recorder,
                notifier=ctx.notifier,
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
                            store,
                            run_id,
                            "run.completed",
                            ctx.sequencer,
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
                    logger.error("Failed to persist ERROR status for run %s", run_id, exc_info=True)
                if error_won:
                    await _emit_event_safe(
                        store,
                        run_id,
                        "run.error",
                        sequencer,
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
                    await _emit_event(store, run_id, "run.cancelled", sequencer, {})
            except Exception:
                logger.error(
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
    extra_notifier: LoopNotifier | None = None,
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
        extra_notifier: Optional TransportNotifier for SSE streaming.

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
            extra_notifier=extra_notifier,
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
            extra_notifier=extra_notifier,
            _skip_claim=True,
        )

    raise ValueError(
        f"Run '{run_id}' has pause_data but no submitted_tool_results "
        "or submitted_user_input — nothing to resume with."
    )
