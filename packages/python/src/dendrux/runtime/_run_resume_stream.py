"""Drive lazy resumed streams and own their claim/iterator cleanup."""

from __future__ import annotations

from contextlib import aclosing
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import dendrux.chat as _chat
import dendrux.loops._helpers as _hooks
import dendrux.runtime.state as _state
import dendrux.runtime.submit as _submit
import dendrux.skills._session as _skills
from dendrux._lifecycle import (
    CancelReason,
)
from dendrux._meta_keys import (
    CANCEL_PERSISTED_META_KEY,
)
from dendrux._sentinel import _OMITTED, _OmittedType
from dendrux.errors import (
    RunFencedError,
    safe_exception_info,
)
from dendrux.loops._helpers import (
    record_run_started,
)
from dendrux.runtime import (
    _reclaim,
    _run_contracts,
    _run_errors,
    _run_events,
    _run_images,
    _run_observers,
    _run_outcome,
    _run_resume_prep,
    _run_resume_replay,
    _run_start,
    _run_tools,
)
from dendrux.runtime._managed_cancel import (
    ManagedRootStoppedError,
)
from dendrux.runtime._managed_cancel import (
    stopped_resume_result as _stopped_resume_result,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.runtime.context import (
    StreamContext,
)
from dendrux.runtime.state import ClaimResult
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.types import (
    PauseState,
    RunEventType,
    RunStatus,
    _usage_from_dict,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Callable,
    )

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier
    from dendrux.media._image import UserContent
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import ClaimResult, RunRecord, StateStore
    from dendrux.types import (
        ContentPart,
        RunEvent,
        RunResult,
        RunStream,
        ToolResult,
    )


from dendrux.runtime import _run_approval, _run_resume_loop


async def _stream_resume_images(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore | None,
    state_store_resolver: Callable[[], Any] | None,
    run_id: str,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    input_items: tuple[Any, ...],
) -> tuple[StateStore, list[ToolResult] | None, tuple[ContentPart, ...] | None, bool]:
    results_prepared = False
    # Image setup raises (the run stays waiting; no RUN_ERROR event).
    if store is None and state_store_resolver is not None:
        store = await state_store_resolver()
    if store is None:
        from dendrux.errors import PersistenceNotConfiguredError

        raise PersistenceNotConfiguredError()
    await ensure_agent_limits_fit(agent, store)
    raw = await store.get_pause_state(run_id)
    if raw is None:
        raise ValueError("Run has no pause state")
    _submit.require_pause_id(
        agent, expected_pause_id, pause_id=raw.get("pause_id"), store=store, run_id=run_id
    )
    media_pause = PauseState.from_dict(raw)
    if media_pause.agent_name and media_pause.agent_name != agent.name:
        raise ValueError("Agent name mismatch on resume")
    if stream_results is not None:
        _submit.validate_result_names(run_id, media_pause, stream_results)
        stream_results = await _submit.prepare_resume_results(
            agent, store, run_id, media_pause, stream_results
        )
        results_prepared = True
    stream_results, user_parts = await _run_images._ingest_resume_images(
        agent,
        provider,
        store,
        run_id,
        loop=None,
        tool_results=stream_results,
        input_items=input_items,
        loop_method="run_stream",
    )
    return store, stream_results, user_parts, results_prepared


async def _stream_resume_state(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    results_prepared: bool,
    stream_context: StreamContext,
) -> tuple[Any, _run_contracts._PauseReads, Any, PauseState, bool, list[ToolResult] | None]:
    skill_session = await _skills.continuation(agent, store, run_id)
    stream_context.set_session(skill_session)

    (
        run_record,
        reads,
        raw_pause,
        pause_state,
        recovering,
        _,
    ) = await _run_resume_prep._read_resume_state(
        store,
        agent,
        provider,
        run_id,
        False,
        expected_pause_id,
        None,
        None,
        streaming=True,
    )

    if stream_results is not None:
        _submit.validate_result_names(run_id, pause_state, stream_results)
        if not results_prepared:
            stream_results = await _submit.prepare_resume_results(
                agent, store, run_id, pause_state, stream_results
            )

    return run_record, reads, raw_pause, pause_state, recovering, stream_results


class _StreamClaim(NamedTuple):
    store: StateStore
    record: Any
    reads: _run_contracts._PauseReads
    raw_pause: Any
    pause: Any
    recovering: bool
    results: list[ToolResult] | None
    expected: str
    stopped: RunResult | None = None


async def _stream_resume_claim(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    claimed: ClaimResult | None,
    stream_context: StreamContext,
    shared: dict[str, Any],
    *,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    results_prepared: bool,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
) -> _StreamClaim:
    """Own the run for a streamed resume: claim its pause, or take an owned continuation.

    ``claimed`` (0.7): the run is already owned (a managed parent claimed for
    adoption) and continues from its checkpoint. A continuation stopped by a
    marked run carries its result in ``stopped``.
    """
    from dendrux.types import RunResult

    if claimed is not None:
        owned, record, reads, rebuilt = await _stream_claimed_state(
            agent, store, run_id, claimed, stream_context
        )
        shared["state_store"], shared["claimed"] = owned, True
        if isinstance(rebuilt, RunResult):
            return _StreamClaim(owned, record, reads, None, None, True, None, "", rebuilt)
        return _StreamClaim(
            owned, record, reads, None, rebuilt, True, None, RunStatus.RUNNING.value
        )
    record, reads, raw_pause, pause, recovering, stream_results = await _stream_resume_state(
        agent,
        provider,
        store,
        run_id,
        expected_pause_id,
        stream_results,
        results_prepared,
        stream_context,
    )
    # 4. Determine expected status and claim
    expected = _resume_expected_status(stream_results, user_input)
    store, expected, raw_pause = await _run_resume_prep._claim_resume(
        store,
        run_id,
        raw_pause,
        agent,
        expected,
        expected_pause_id=expected_pause_id,
        results=stream_results,
        user_input=user_input,
        user_parts=user_parts,
        checkpoint=reads.checkpoint,
        on_release=partial(_remember_stream_release, shared),
    )
    shared["state_store"], shared["claimed"] = store, True
    stopped = await _run_resume_loop._stop_marked_continuation(
        store, run_id, record, pause, agent=agent, claimed_here=True
    )
    return _StreamClaim(
        store, record, reads, raw_pause, pause, recovering, stream_results, expected, stopped
    )


async def _stream_claimed_state(
    agent: Agent,
    store: StateStore,
    run_id: str,
    claim: ClaimResult,
    stream_context: StreamContext,
) -> tuple[StateStore, Any, _run_contracts._PauseReads, PauseState | RunResult]:
    """An owned run's continuation state, rebuilt from its checkpoint (0.7 adoption).

    The same reads and stop checks as reclaim's continuation: a marked run stops
    before any model call (its stop result is returned instead of a pause); a
    rebuild failure is recorded like a reclaim's.
    """
    from dendrux.errors import ManagedStateError
    from dendrux.skills._session import SkillSession

    session = SkillSession.prepare(agent)
    session.run_id = run_id
    stream_context.set_session(session)
    owned = _state.claimed_store(store, run_id, claim)
    state = claim.loop_state or {}
    if state.get("submission"):
        # Adoption completes its submission atomically; anything else is not ours.
        raise ManagedStateError(reason="binding_outstanding", run_id=run_id)
    history_records: list[Any] = []
    async with _reclaim._reclaim_setup(owned, run_id):
        record = await owned.get_run(run_id)
        stopped = await _run_resume_loop._stop_marked_continuation(
            owned,
            run_id,
            record,
            agent=agent,
            iteration=state.get("iteration", 0),
            usage=_usage_from_dict(state.get("usage", {})),
        )
        if stopped is not None:
            return owned, record, _run_contracts._PauseReads(record), stopped
        pause = await _reclaim.rebuild_pause(
            agent, owned, run_id, state, history_records=history_records
        )
    return owned, record, _run_contracts._PauseReads(record, history_records=history_records), pause


def _stopped_stream_event(result: RunResult) -> RunEvent:
    from dendrux.types import RunEvent

    return RunEvent(
        type=RunEventType.RUN_CANCELLED
        if result.status == RunStatus.CANCELLED
        else RunEventType.RUN_PAUSED,
        run_result=result,
    )


async def _stream_approval_events(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    user_input: str | None,
    image_runtime: ImageRuntime | None,
    meta: dict[str, Any],
    on_work: Callable[[], None] | None = None,
) -> AsyncGenerator[RunEvent, None]:
    from dendrux.types import RunEvent

    approval, approving = await _run_approval._resume_approval_batch(
        agent,
        store,
        run_id,
        ctx,
        expected,
        results,
        user_input,
        image_runtime,
        streaming=True,
        on_work=on_work,
    )
    if approval is not None:
        for tc, tr in approval.results:
            yield RunEvent(
                type=RunEventType.TOOL_RESULT,
                tool_call=tc,
                tool_result=tr,
                iteration=ctx.pause_state.iteration,
                phase="commentary",
            )
    stop = await _run_approval._finish_approval_batch(
        run_id, ctx, expected, results, approval, approving
    )
    if stop is not None:
        _run_observers._attach_mcp_meta(stop, meta)
        yield _stopped_stream_event(
            await _run_resume_loop._finish_resumed_result(store, run_id, ctx, stop)
        )


def _resume_expected_status(results: list[ToolResult] | None, user_input: str | None) -> str:
    if results is not None:
        return RunStatus.WAITING_CLIENT_TOOL.value
    if user_input is not None:
        return RunStatus.WAITING_HUMAN_INPUT.value
    return RunStatus.WAITING_APPROVAL.value


def _mark_stream_recovery_work(shared: dict[str, Any]) -> None:
    shared["work_started"] = True
    shared["progress"]._start()


def _stream_resume_started(shared: dict[str, Any]) -> bool:
    # Work is an independent reason to finalize; it must never be made retryable.
    return bool(shared.get("work_started") or shared.get("claimed"))


async def _stream_resume_failure(
    _shared: dict[str, Any], extra_notifier: LoopNotifier | None, run_id: str, exc: Exception
) -> RunStatus:
    store = _shared.get("state_store")
    err_ctx = _shared.get("ctx")
    progress = _shared.get("progress")
    if (
        store is not None
        and progress is not None
        and progress._can_release(err_ctx, exc)
        and _run_resume_loop._resume_release_supported(store, _shared["expected_status"])
    ):
        return await _run_resume_loop._try_resume_release(
            store, run_id, _shared["expected_status"], err_ctx, exc
        )
    return await _finish_stream_resume_failure(_shared, store, err_ctx, extra_notifier, run_id, exc)


async def _finish_stream_resume_failure(
    _shared: dict[str, Any],
    store: StateStore | None,
    err_ctx: _run_contracts._ResumeContext | None,
    extra_notifier: LoopNotifier | None,
    run_id: str,
    exc: Exception,
) -> RunStatus:
    if store is not None and _stream_resume_started(_shared):
        await _run_start._persist_run_failure(store, run_id, exc, safe_event=True)
    try:
        if err_ctx is not None:
            await _hooks.record_run_failed(err_ctx.recorder, run_id, exc)
            await _hooks.notify_run_failed(err_ctx.notifier, run_id, exc)
        else:
            # Setup failed before _prepare_resume built the recorder.
            # Fire the notifier-side hook so OTel can surface the error
            # even when no DB-side recorder exists.
            await _hooks.notify_run_failed(extra_notifier, run_id, exc)
    except Exception:
        logger.warning(
            "Lifecycle hooks failed during resume_stream error handler for run %s",
            run_id,
            exc_info=safe_exception_info(),
        )
    prelude = _shared.get("released_prelude")
    if prelude is not None:
        return await _run_resume_loop._released_resume_status(prelude, run_id)
    return RunStatus.ERROR


def _remember_stream_release(shared: dict[str, Any], store: StateStore) -> None:
    shared["released_prelude"] = store


async def _finish_resume_stream_cleanup(
    store: StateStore | None,
    run_id: str,
    _shared: dict[str, Any],
    mcp_meta: dict[str, Any],
    cancel_won: bool,
    persisted_row: RunRecord | None | _OmittedType,
) -> RunResult:
    from dendrux.types import RunResult

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
            "Lifecycle finish hooks failed during resume stream cleanup for run %s",
            run_id,
            exc_info=safe_exception_info(),
        )

    return cancelled_result


async def _initial_stream_images(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore | None,
    resolver: Callable[[], Any] | None,
    run_id: str,
    expected_pause_id: str | None,
    results: list[ToolResult] | None,
    items: tuple[Any, ...],
) -> tuple[StateStore | None, list[ToolResult] | None, tuple[ContentPart, ...] | None, bool]:
    if items or any(result.images for result in results or ()):
        return await _stream_resume_images(
            agent, provider, store, resolver, run_id, expected_pause_id, results, items
        )
    return store, results, None, False


def resume_stream(
    run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tool_results: list[ToolResult] | None = None,
    user_input: UserContent | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
    _claimed: ClaimResult | None = None,
) -> RunStream:
    """Stream a resumed run as RunEvents, returning a RunStream.

    Synchronous — returns immediately. All async setup (store resolution,
    pause state load, claim, history build) runs lazily on first iteration.

    ``_claimed`` (internal, 0.7): continue a run this caller already owns
    (a managed parent claimed for adoption) from its checkpoint, exactly as
    reclaim's continuation does, instead of loading a pause and claiming it.

    Execution exceptions yield RUN_ERROR events. Before consumption or work,
    cleanup can release the claim; the result reports the observed status and
    the failed attempt's error. Otherwise the run is finalized as ERROR.

    First event is RUN_RESUMED (not RUN_STARTED) carrying the existing run_id.

    Image inputs (``tool_results[i].images`` or a multimodal ``user_input``)
    are ingested before the claim; an image error is raised from the first
    ``__anext__`` and leaves the run waiting.
    """
    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    input_items: tuple[Any, ...] = ()
    if user_input is not None:
        user_input, input_items = _chat.split_user_content(user_input)
    _shared: dict[str, Any] = {"state_store": state_store, "ctx": None}

    mcp_meta: dict[str, Any] = {}

    stream_context = StreamContext()

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        store = state_store
        stream_results = tool_results
        user_parts: tuple[ContentPart, ...] | None = None
        store, stream_results, user_parts, results_prepared = await _initial_stream_images(
            agent,
            provider,
            store,
            state_store_resolver,
            run_id,
            expected_pause_id,
            stream_results,
            input_items,
        )
        try:
            store = await _run_resume_prep._resolve_resume_stream_store(store, state_store_resolver)
            _shared["state_store"] = store

            await ensure_agent_limits_fit(agent, store)

            claim = await _stream_resume_claim(
                agent,
                provider,
                store,
                run_id,
                _claimed,
                stream_context,
                _shared,
                expected_pause_id=expected_pause_id,
                stream_results=stream_results,
                results_prepared=results_prepared,
                user_input=user_input,
                user_parts=user_parts,
            )
            if claim.stopped is not None:
                yield _run_events._fenced_stream_event(run_id, claim.stopped)
                return
            store, run_record, reads, raw_pause = (
                claim.store,
                claim.record,
                claim.reads,
                claim.raw_pause,
            )
            pause_state, recovering, stream_results, expected = (
                claim.pause,
                claim.recovering,
                claim.results,
                claim.expected,
            )
            _shared["expected_status"] = expected
            progress = _shared["progress"] = _run_contracts._ResumeProgress()

            # 5. Prepare (shared helper — history, notifier).
            #    Does NOT fire any lifecycle hooks; that's done below inside
            #    the same try block so failure or released-finish hooks pair
            #    with on_run_started.
            run_tenant = run_record.tenant_id if run_record else None
            ctx = await _run_resume_prep._prepare_resume(
                run_id,
                pause_state,
                state_store=store,
                agent=agent,
                provider=provider,
                expected_status=expected,
                tool_results=None if recovering else stream_results,
                user_input=None if recovering else user_input,
                extra_notifier=extra_notifier,
                user_parts=user_parts,
                tenant_id=run_tenant,
                history_records=reads.history_records,
                pause_data=raw_pause,
            )
            image_runtime = _run_images._image_runtime(agent, store, run_tenant, run_id)
            _shared["ctx"] = ctx
            progress.submission = getattr(ctx.recorder, "_loop_state", {}).get("submission")

            # 5a. Fire run-level lifecycle hooks. The exception handler closes
            #     the pair with failure or the observed released status.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            ctx.recorder_started = True
            trace_link, ctx.notifier = await _run_observers._start_trace(
                ctx.notifier,
                store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
                on_started=ctx._mark_notifier_started,
            )
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = ctx.recorder
            _shared["notifier"] = ctx.notifier

            await agent.get_tool_lookups(loop=ctx.resolved_loop)
            mcp_meta.update(_run_tools._snapshot_mcp_meta(agent))

            # 5b. tool/user-input history. Inside the try
            #     so a recorder failure here fires on_run_failed.
            if not recovering:
                await _run_resume_replay._replay_resume_injected_history(
                    ctx,
                    run_id,
                    tool_results=stream_results,
                    user_input=user_input,
                    on_consumed=progress._start,
                )

            stream_context.set_delegation(
                _run_resume_loop._resume_delegation(
                    agent, store, run_id, run_record, trace_link, _tenant_explicit
                )
            )

            # 7. Emit RUN_RESUMED as first event (contract guarantee)
            yield RunEvent(type=RunEventType.RUN_RESUMED, run_id=run_id)

            if recovering:
                stop = await _run_resume_replay._resolve_recovery(
                    agent,
                    store,
                    run_id,
                    ctx,
                    image_runtime=image_runtime,
                    tenant_id=run_record.tenant_id if run_record else None,
                    on_work=partial(_mark_stream_recovery_work, _shared),
                )
                if stop is not None:
                    yield _stopped_stream_event(
                        await _run_resume_loop._finish_resumed_result(store, run_id, ctx, stop)
                    )
                    return

            if not recovering:
                async with aclosing(
                    _stream_approval_events(
                        agent,
                        store,
                        run_id,
                        ctx,
                        expected,
                        stream_results,
                        user_input,
                        image_runtime,
                        mcp_meta,
                        progress._start,
                    )
                ) as approval_events:
                    async for event in approval_events:
                        yield event
                        if event.run_result is not None:
                            return

            # 9. Stream the loop (see run_stream for interrupt + closing notes)
            interrupt = agent._task_manager.register_interrupt(run_id)
            progress._start()
            loop_stream = ctx.resolved_loop.run_stream(
                **_run_resume_loop._resumed_loop_kwargs(
                    agent, provider, store, run_id, ctx, image_runtime, ctx.resolved_loop.run_stream
                ),
                interrupt=interrupt,
            )
            async with aclosing(
                _run_outcome._stream_loop_events(
                    _run_resume_loop._resume_loop_stream(ctx, loop_stream),
                    state_store=store,
                    recorder=ctx.recorder,
                    notifier=ctx.notifier,
                    run_id=run_id,
                    provider_kwargs=ctx.pause_state.provider_kwargs,
                    shared=_shared,
                    mcp_meta=mcp_meta,
                )
            ) as events:
                async for event in events:
                    yield event

        except (ManagedRootStoppedError, RunFencedError) as stop:
            store = _shared["state_store"]
            result = await _stopped_resume_result(stop, store, run_id, _shared.get("ctx"))
            yield _run_events._fenced_stream_event(run_id, result)

        except Exception as exc:
            failure_status = await _stream_resume_failure(_shared, extra_notifier, run_id, exc)

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=failure_status,
                    error=_run_errors._safe_error(exc),
                    error_info=_run_errors._provider_error_info(exc),
                    meta=_run_resume_loop._resume_warning_meta(
                        _shared.get("ctx"), _run_outcome._stream_error_meta(mcp_meta, exc)
                    ),
                ),
                error=_run_errors._safe_error(exc),
            )

        finally:
            if _shared.get("claimed"):
                agent._task_manager.release_interrupt(run_id)

    async def _cleanup() -> RunResult | None:
        """CAS-guarded cancellation + local lifecycle close.

        Mirrors run_stream._cleanup. The lifecycle close fires whether or
        not the CAS won — if some other caller already finalized the run,
        the local notifier still needs the close event so OTel root spans
        are closed. Status resolution: CAS won → CANCELLED, CAS lost →
        query store for actual status (falling back to CANCELLED).
        """
        if not _shared.get("claimed"):
            return None
        store = _shared.get("state_store")
        # run.cancelled is recorded only once the resume itself was recorded.
        resumed = _shared.get("ctx") is not None

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
                    emit_event=resumed,
                    read_on_loss=lambda: "notifier" in _shared,
                    safe_read=True,
                )
            except Exception:
                logger.error(  # noqa: TRY400 - Keep safe_exception_info() to exclude vendor text and exception chains.
                    "Failed to cancel run %s during resume stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        if "notifier" in _shared:
            return await _finish_resume_stream_cleanup(
                store, run_id, _shared, mcp_meta, cancel_won, persisted_row
            )
        return None

    return _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _run_events._shielded_cleanup(_cleanup, run_id=run_id),
    )


_StreamClaim.__module__ = "dendrux.runtime.runner"
resume_stream.__module__ = "dendrux.runtime.runner"
