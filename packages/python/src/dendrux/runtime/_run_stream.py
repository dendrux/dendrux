"""Drive fresh-run streaming in the caller task through the shared outcome adapters."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, overload

import dendrux.chat as _chat
import dendrux.loops._helpers as _hooks
import dendrux.runtime.context as _context
import dendrux.runtime.state as _state
import dendrux.strategies.native as _native
from dendrux._lifecycle import (
    CancelReason,
)
from dendrux._meta_keys import (
    CANCEL_PERSISTED_META_KEY,
)
from dendrux._sentinel import _OMITTED, _OmittedType, _UnsetType
from dendrux.chat import (
    history_has_images,
    normalize_chat_history,
    validate_chat_history,
)
from dendrux.context_blocks import (
    ContextBlock,
    fold_context,
    validate_context,
)
from dendrux.errors import (
    RunFencedError,
    safe_exception_info,
)
from dendrux.runtime import (
    _managed,
    _run_contracts,
    _run_entry,
    _run_errors,
    _run_events,
    _run_images,
    _run_outcome,
    _run_pending,
    _run_start,
    _run_tools,
)
from dendrux.runtime._entry_checks import resolve_runner_tenant, runner_entry_checks
from dendrux.runtime._managed_cancel import (
    ManagedRootStoppedError,
)
from dendrux.runtime._run_contracts import (
    _UNSET_DEPTH,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.runtime._tenant_checks import _tenant_scope_chosen
from dendrux.runtime.context import (
    DelegationContext,
    StreamContext,
    get_store_identity,
    resolve_child_tenant,
)
from dendrux.runtime.replay import (
    ReplayLast,
    ReplayPlan,
    ReplayRequest,
    validate_replay_args,
)
from dendrux.runtime.state import ClaimResult
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.skills._session import SkillSession
from dendrux.types import (
    IdempotencyConflictError,
    RunAlreadyActiveError,
    RunEventType,
    RunStatus,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Callable,
        Sequence,
    )

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.media._image import UserContent
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import ClaimResult, RunRecord, StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        CreateRunResult,
        Limits,
        RunEvent,
        RunResult,
        RunStream,
    )


def _cached_stream_event(cached: RunResult, run_id: str) -> RunEvent:
    from dendrux.types import RunEvent

    event_type = {
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
    }.get(cached.status, RunEventType.RUN_COMPLETED)
    return RunEvent(
        type=event_type,
        run_id=run_id,
        run_result=cached,
        error=cached.error if event_type == RunEventType.RUN_ERROR else None,
        phase="final_answer" if cached.status == RunStatus.SUCCESS else None,
    )


def _stream_claim(
    store: StateStore,
    loop: Loop,
    created: CreateRunResult,
    child_claim: ClaimResult | None,
    *,
    root_limits: Limits | None,
) -> StateStore:
    """The new run's owned store: a managed child's token comes from its bind."""
    owned = _state.claimed_store(
        store, created.run_id, child_claim or _run_start._creation_claim(loop, created)
    )
    _managed.stamp_root_header(owned, created.run_id, limits=root_limits)
    return owned


def _stream_entry_checks(
    agent: Agent, tenant_id: str | None, shared: dict[str, Any], parent_ctx: Any, **checks: Any
) -> None:
    try:
        entry_tenant = resolve_child_tenant(parent_ctx, tenant_id)
    except ValueError:
        return
    shared["existing_run"] = True
    runner_entry_checks(agent, "stream", tenant_id=entry_tenant, **checks)
    shared["existing_run"] = False


async def _resolve_stream_store(
    store: StateStore | None,
    resolver: Callable[[], Any] | None,
    agent: Agent,
    tenant_id: str | None | _OmittedType,
) -> StateStore | None:
    if store is None and resolver is not None:
        store = await resolver()
        resolve_runner_tenant(agent, "stream", tenant_id, state_store=store)
    return store


async def _stream_replay_runs(
    request: ReplayRequest | None,
    store: StateStore | None,
    tenant_id: str | None,
    loop: Loop,
    shared: dict[str, Any],
    limits: Limits,
) -> tuple[list[Any], tuple[str, ...]]:
    if request is None:
        return [], ()
    shared["existing_run"] = True
    if store is None:
        raise ValueError(
            "runs= requires persistence (database_url or state_store): "
            "replay reads the earlier runs' traces."
        )
    loaded = await _run_entry._load_new_run_replay(
        store, request, tenant_id, loop, limits=limits, method="run_stream"
    )
    shared["existing_run"] = False
    return loaded.runs, loaded.omitted


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunStream: ...


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunStream: ...


def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = None,
    context: list[ContextBlock] | None = None,
    activate_skills: Sequence[str] | None = None,
    runs: list[str] | None = None,
    replay_last: ReplayLast = 0,
    replay_reasoning: bool = True,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    _managed_invocation: Any = None,
    **kwargs: Any,
) -> RunStream:
    """Stream an agent run as RunEvents, returning a RunStream.

    Synchronous — returns immediately with run_id available. All async
    setup (DB row, notifiers) runs lazily on first iteration. With
    ``idempotency_key``, the initial run_id is provisional until that setup
    resolves the key. A cached outcome yields one terminal event, without
    replaying text or tool events. Read its answer from ``stream.result``.

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

    ``_managed_invocation`` (internal, 0.7): stream a managed child created by
    its parent's atomic bind instead of ``create_run``. Its id, parent link,
    tenant and depth come from the invocation; a failure before the bind
    raises to the managed driver instead of yielding ``RUN_ERROR``.
    """
    invocation = _managed_invocation
    # Validate depth early — synchronous, before generator starts.
    # Full resolution (parent context, clamping) happens inside _generate()
    # because parent_ctx requires async state_store resolution first.
    # But validation of the raw input can fail fast here.
    requested_tenant_id = tenant_id
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "stream", tenant_id, state_store=state_store)
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

    # Validate input + chat history synchronously so TypeError / ValueError
    # surfaces to the caller immediately, not as a RUN_ERROR event. Image
    # ingest is async: it runs on first iteration, before create_run, and
    # its errors are raised from the first ``__anext__`` (no run row).
    entry_user_input = user_input  # as passed, for the runner entry checks
    user_input, input_items = _chat.split_user_content(user_input)
    validate_chat_history(history)
    image_input = bool(input_items) or history_has_images(history)
    normalized_history = [] if image_input else normalize_chat_history(history)
    resolved_strategy = strategy or _native.NativeToolCalling()
    resolved_loop = _run_entry._new_run_loop(agent, loop)
    _run_tools._validate_loop_skill_compat(agent, resolved_loop)
    if image_input:
        _run_images._require_image_loop(resolved_loop, "run_stream")
    replay_req = validate_replay_args(
        runs, replay_last, replay_reasoning, loop=resolved_loop, context=context
    )
    # When chat history is provided, build the complete starting message list
    # (loops use initial_history as-is and do not separately append user_input).
    # The new user_input is recorded by the runner itself inside _generate()
    # so seeded prior turns stay out of react_traces. With runs=, the replayed
    # prefix is folded in after expansion.
    skill_session = SkillSession.prepare(agent, loop=resolved_loop, explicit=activate_skills)
    with skill_session.bind():
        if replay_req is None and not image_input:
            initial_seeded = fold_context(
                normalized_history, context, user_input, session=skill_session
            )
        else:
            validate_context(context)
            initial_seeded = None
    provider_kwargs = dict(kwargs)

    # A managed child's id is its parent's bind candidate, minted at the handoff.
    run_id = _managed.new_run_id(invocation)

    # Shared mutable state — generator populates on first iteration,
    # cleanup reads if the consumer abandons after setup.
    _shared: dict[str, Any] = {"state_store": state_store}

    mcp_meta: dict[str, Any] = {}

    stream_context = StreamContext(skill_session)

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        """Inner generator — lazy async setup, then loop stream with lifecycle.

        The entire body is wrapped in try/except so that setup failures
        (state_store_resolver, create_run, notifier init) are also caught
        and yielded as RUN_ERROR — honoring the "no exception to consumer" contract.
        """
        # store is captured in the closure for the error handler.
        # Starts as the pre-provided value (often None for the lazy path).
        nonlocal run_id, extra_notifier
        store = state_store
        startup = _run_start._RunLifecycle(run_id, store, None)
        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        seeded_history = initial_seeded
        replay_plan: ReplayPlan | None = None
        image_runtime: ImageRuntime | None = None
        user_parts: tuple[ContentPart, ...] | None = None
        history_messages = normalized_history
        # Entry checks after tenant inheritance, before any I/O. Like image
        # setup they raise (no RUN_ERROR event, no run row, no hooks). A child
        # whose explicit tenant differs from its parent's still fails later, as
        # RUN_ERROR, exactly as before.
        # A managed child never inherits the caller's delegation context.
        entry_parent = _managed.parent_context(invocation, _context.get_delegation_context)
        _stream_entry_checks(
            agent,
            tenant_id,
            _shared,
            entry_parent,
            metadata=metadata,
            user_input=entry_user_input,
            history=history,
            context=context,
            provider_kwargs=provider_kwargs,
            idempotency_key=idempotency_key,
            provider=provider,
        )
        if image_input:
            # Image setup raises (no RUN_ERROR event, no run row, no hooks).
            _shared["existing_run"] = True
            store = await _resolve_stream_store(
                store, state_store_resolver, agent, requested_tenant_id
            )
            await ensure_agent_limits_fit(agent, store)
            _shared["state_store"] = store
            image_runtime, user_parts, history_messages, cached = await _run_images._entry_media(
                agent,
                provider,
                resolved_loop,
                store,
                resolve_child_tenant(_context.get_delegation_context(), tenant_id),
                run_id,
                True,
                user_input,
                input_items,
                history,
                history_messages,
                provider_kwargs,
                idempotency_key,
                output_type,
                context,
                replay_req,
                skill_session,
            )
            if cached is not None:
                run_id = cached.run_id
                stream.run_id = run_id
                yield _cached_stream_event(cached, run_id)
                return
            with skill_session.bind():
                seeded_history = _run_entry._new_run_history(
                    replay_req, history_messages, context, user_input, skill_session, user_parts
                )
            _shared["existing_run"] = False
        try:
            # 1. Resolve state store lazily (may create DB engine)
            store = await _resolve_stream_store(
                store, state_store_resolver, agent, requested_tenant_id
            )
            await ensure_agent_limits_fit(agent, store)
            _shared["state_store"] = store
            # --- Delegation context ---
            parent_ctx = _managed.parent_context(invocation, _context.get_delegation_context)
            parent_run_id, delegation_level = _managed.parent_link(invocation, parent_ctx, store)
            eff_max_depth = _run_entry._resolve_max_delegation_depth(
                max_delegation_depth, parent_ctx
            )
            eff_tenant = resolve_child_tenant(parent_ctx, tenant_id)
            root_limits = await _managed.enter(agent, invocation, store, resolved_loop)
            if image_runtime is None and _run_contracts._accepts_kwarg(
                resolved_loop.run_stream, "image_runtime"
            ):
                image_runtime = _run_images._image_runtime(agent, store, eff_tenant, run_id)
            # Resolve system prompt BEFORE create_run() so skill loading
            # failures don't leave a DB row stuck in running.
            _system_prompt = agent.get_system_prompt(loop=resolved_loop)
            _run_entry._require_idempotency_store(store, idempotency_key)

            replay_runs, replay_omitted = await _stream_replay_runs(
                replay_req, store, eff_tenant, resolved_loop, _shared, agent._limits
            )

            if store is not None:
                run_meta, fingerprint = _run_entry._new_run_metadata(
                    agent,
                    resolved_loop,
                    skill_session,
                    metadata,
                    eff_max_depth,
                    user_input=user_input,
                    idempotency_key=idempotency_key,
                    output_type=output_type,
                    history=history_messages,
                    context=context,
                    replay=replay_req,
                    parts=user_parts,
                    streaming=True,
                )
                try:
                    seed_kwargs, run_guardrails = await _run_entry._runtime_seeds(
                        store,
                        agent,
                        provider,
                        resolved_loop,
                        resolved_strategy,
                        provider_kwargs,
                        seeded_history,
                        user_input=user_input,
                        run_id=run_id,
                        tenant_id=eff_tenant,
                        replay_req=replay_req,
                        replay_runs=replay_runs,
                        replay_omitted=replay_omitted,
                        normalized_history=history_messages,
                        context=context,
                        user_parts=user_parts,
                    )
                    created = await _run_entry._create_projected_run(
                        False,
                        store,
                        resolved_loop,
                        run_id,
                        agent.name,
                        input_data=_run_images._input_data(user_input, user_parts),
                        model=provider.model,
                        strategy=type(resolved_strategy).__name__,
                        parent_run_id=parent_run_id,
                        delegation_level=delegation_level,
                        tenant_id=eff_tenant,
                        meta=run_meta,
                        idempotency_key=idempotency_key,
                        idempotency_fingerprint=fingerprint,
                        **_run_images._input_edge_kwargs(image_runtime, user_parts),
                        **seed_kwargs,
                        invocation=invocation,
                    )
                    # A child bound by another caller reads as existing_active below.
                    create_result, child_claim = _run_entry._managed_creation(created)
                except IdempotencyConflictError:
                    # No row was created and no lifecycle opened; the error
                    # handler must not fire failure hooks for a run that
                    # never started (mirrors run(), which re-raises bare).
                    _shared["existing_run"] = True
                    raise
                _shared["existing_run"] = create_result.outcome != "created"
                run_id = create_result.run_id
                stream.run_id = run_id
                if create_result.outcome == "existing_terminal":
                    cached = await _run_pending._build_cached_result(store, run_id, output_type)
                    yield _cached_stream_event(cached, run_id)
                    return
                if create_result.outcome == "existing_active":
                    raise RunAlreadyActiveError(run_id, create_result.status)
                store = _stream_claim(
                    store, resolved_loop, create_result, child_claim, root_limits=root_limits
                )
                _shared["state_store"] = store
                _shared["claimed"] = True

            # 2. Set delegation context for the generator lifetime
            this_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=delegation_level,
                persisted=store is not None,
                store_identity=get_store_identity(store),
                max_delegation_depth=eff_max_depth,
                tenant_id=eff_tenant,
                tenant_explicit=tenant_explicit,
                state_store=store,
                lease_interval_s=agent.lease_interval.total_seconds(),
                loop=asyncio.get_running_loop(),
            )
            startup.run_id = run_id
            startup.store = store
            trace_link, extra_notifier = await _run_start._start_new_run(
                startup,
                agent,
                provider,
                eff_tenant,
                seed_kwargs,
                run_guardrails,
                extra_notifier,
                parent_ctx,
                {"agent_name": agent.name, "system_prompt": _system_prompt},
                before_event=lambda: stream_context.set_delegation(this_ctx),
            )
            this_ctx = replace(this_ctx, trace_link=trace_link)
            stream_context.set_delegation(this_ctx)
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = startup.recorder
            _shared["notifier"] = extra_notifier

            yield RunEvent(type=RunEventType.RUN_STARTED, run_id=run_id)

            await _run_tools._discover_runtime_tools(
                agent, startup.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
            )

            replay_plan, seeded_history = await _run_entry._apply_replay(
                agent,
                provider,
                resolved_loop,
                provider_kwargs,
                state_store=store,
                run_id=run_id,
                replay_req=replay_req,
                replay_runs=replay_runs,
                replay_omitted=replay_omitted,
                normalized_history=history_messages,
                context=context,
                user_input=user_input,
                seeded_history=seeded_history,
                user_parts=user_parts,
            )
            await _run_images._register_memory_input(image_runtime, store, run_id, user_parts)
            image_view = await _run_entry._new_run_image_view(
                image_runtime, store, run_id, seeded_history, replay_runs, replay_plan, user_parts
            )

            # When chat history was provided, the loop receives a complete
            # message list as initial_history and skips its own user-message
            # recording. Record the new user_input here so it still lands in
            # react_traces (seeded prior turns are NOT recorded).
            await _run_entry._record_seeded_user(
                startup.recorder,
                extra_notifier,
                run_id,
                user_input,
                seeded_history,
                context,
                skill_session,
            )

            # 3. Stream the loop. The interrupt signal lets cancel_run reach
            # an in-flight provider stream owned by this process; the closing
            # wrapper guarantees the loop generator is finalized in this task
            # when the consumer abandons the stream.
            interrupt = agent._task_manager.register_interrupt(run_id)
            loop_stream = resolved_loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=resolved_strategy,
                user_input=user_input,
                run_id=run_id,
                recorder=startup.recorder,
                notifier=extra_notifier,
                initial_history=seeded_history,
                provider_kwargs=provider_kwargs or None,
                output_type=output_type,
                state_store=store,
                interrupt=interrupt,
                **_run_images._image_loop_kwargs(
                    resolved_loop.run_stream, image_runtime, image_view
                ),
            )
            async with aclosing(
                _run_outcome._stream_loop_events(
                    loop_stream,
                    state_store=store,
                    recorder=startup.recorder,
                    notifier=extra_notifier,
                    run_id=run_id,
                    provider_kwargs=provider_kwargs,
                    shared=_shared,
                    mcp_meta=mcp_meta,
                    replay_plan=replay_plan,
                    new_run=True,
                )
            ) as events:
                async for event in events:
                    yield event

        except (ManagedRootStoppedError, RunFencedError) as stop:
            result = await startup._stopped(stop, startup.recorder, extra_notifier)
            yield _run_events._fenced_stream_event(run_id, result)

        except Exception as exc:
            # Before its bind a managed child is not a run: the driver decides
            # for its parent, exactly as for run() (0.7).
            _managed.raise_unbound(invocation, _shared.get("claimed"))
            # Persist error, yield RUN_ERROR, return cleanly. No re-raise.
            # Covers both setup failures and loop execution errors.
            await _run_start._stream_run_failure(
                store, _shared, startup.recorder, extra_notifier, run_id, exc
            )
            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=RunStatus.ERROR,
                    error=_run_errors._safe_error(exc),
                    error_info=_run_errors._provider_error_info(exc),
                    meta=_run_outcome._stream_error_meta(mcp_meta, exc),
                ),
                run_id=run_id,
                error=_run_errors._safe_error(exc),
            )
        finally:
            _run_images._release_memory_edges(image_runtime, store, run_id)
            if not _shared.get("existing_run"):
                agent._task_manager.release_interrupt(run_id)

    async def _cleanup() -> RunResult | None:
        """CAS-guarded cancellation + local lifecycle close.

        Two responsibilities:
          (1) If the run is still RUNNING in the store, finalize it as
              CANCELLED (CAS-guarded so we don't clobber a concurrent
              terminal status set elsewhere).
          (2) If the lifecycle pair was opened locally (recorder/notifier
              stashed by ``_generate`` after ``run_started`` fired), close
              it with a ``run_finished`` event on the LOCAL notifier so
              OTel root spans close. This fires regardless of whether
              the CAS won — if some other caller already finalized the
              run, our local notifier still needs the close event.

        Status resolution for the close event:
          - CAS won → CANCELLED.
          - CAS lost + store available → query store for actual status.
          - No store → CANCELLED (best-effort fallback).
        """
        if not _shared.get("claimed") and _shared.get("state_store") is not None:
            return None
        store = _shared.get("state_store")

        cancel_won = False
        persisted_row: RunRecord | None | _OmittedType = _OMITTED

        def mark_cancelled() -> None:
            nonlocal cancel_won
            cancel_won = True

        partial_answer = _run_outcome._partial_answer(_shared)
        if store is not None:
            try:
                cancel_won, persisted_row = await _run_events._persist_run_cancellation(
                    store,
                    run_id,
                    {
                        "reason": CancelReason.STREAM_CLOSED.value,
                        "partial_output": partial_answer is not None,
                    },
                    on_won=mark_cancelled,
                    answer=partial_answer,
                    emit_event=True,
                    read_on_loss=lambda: "notifier" in _shared,
                    safe_read=True,
                )
            except Exception:
                logger.error(  # noqa: TRY400 - Keep safe_exception_info() to exclude vendor text and exception chains.
                    "Failed to cancel run %s during stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        # Close the lifecycle pair if it was opened.
        if "notifier" in _shared:
            cleanup_recorder = _shared.get("recorder")
            terminal_status = RunStatus.CANCELLED
            cancel_persisted = cancel_won
            if store is not None and not cancel_won:
                try:
                    run_record = (
                        await store.get_run(run_id)
                        if isinstance(persisted_row, _OmittedType)
                        else persisted_row
                    )
                    if run_record is not None:
                        terminal_status = RunStatus(run_record.status)
                        cancel_persisted = terminal_status == RunStatus.CANCELLED
                except Exception:  # noqa: S110 - Cleanup retains CANCELLED fallback when persisted status is unavailable.
                    pass  # CANCELLED is a safe fallback
            cancelled_result = RunResult(
                run_id=run_id,
                status=terminal_status,
                meta={**deepcopy(mcp_meta), CANCEL_PERSISTED_META_KEY: cancel_persisted},
            )
            try:
                await _hooks.record_run_finished(cleanup_recorder, run_id, cancelled_result)
                await _hooks.notify_run_finished(_shared.get("notifier"), run_id, cancelled_result)
            except Exception:
                logger.warning(
                    "Lifecycle finish hooks failed during stream cleanup for run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )

            return cancelled_result
        return None

    stream = _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _run_events._shielded_cleanup(_cleanup, run_id=run_id),
    )
    return stream


run_stream.__module__ = "dendrux.runtime.runner"
