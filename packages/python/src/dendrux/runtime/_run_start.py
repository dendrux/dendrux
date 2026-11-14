"""Own fresh-run claim/startup and exactly-once lifecycle unwinding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dendrux.loops._helpers as _hooks
import dendrux.runtime.state as _state
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    StoredEventType,
)
from dendrux.errors import (
    safe_exception_info,
)
from dendrux.loops._helpers import (
    record_run_started,
)
from dendrux.runtime import (
    _managed,
    _managed_budget,
    _run_entry,
    _run_errors,
    _run_events,
    _run_images,
    _run_observers,
    _run_pending,
)
from dendrux.runtime._managed import DESCEND_KEY as MANAGED_DESCEND_KEY
from dendrux.runtime._managed_cancel import (
    ManagedStopLifecycle,
)
from dendrux.runtime._run_errors import logger
from dendrux.runtime.state import (
    ClaimResult,
    _initial_loop_state,
)
from dendrux.types import (
    IdempotencyConflictError,
    RunAlreadyActiveError,
    RunStatus,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.runtime.context import DelegationContext
    from dendrux.runtime.state import StateStore
    from dendrux.runtime.tasks import RunTaskManager
    from dendrux.types import (
        CreateRunResult,
        Limits,
        RunResult,
    )


async def _persist_run_failure(
    store: StateStore,
    run_id: str,
    exc: Exception | None,
    *,
    safe_event: bool = False,
    payload: dict[str, Any] | None = None,
    finalize: dict[str, Any] | None = None,
    strict: bool = False,
) -> bool:
    """Finalize a running run as ERROR and write its ``run.error`` under one rule.

    Without ``payload``/``finalize`` both derive from ``exc``. A caller with its own
    safe event data and terminal fields (a blocked-content stop) passes them and
    ``exc=None``; ``strict`` then lets a failed finalize propagate.
    """
    try:
        # Deriving text from ``exc`` can itself raise; it fails like the write does.
        payload, finalize = _failure_fields(store, exc, payload, finalize)
        won, needs_event = await _run_events._finalize_running_event(
            store,
            run_id,
            RunStatus.ERROR.value,
            StoredEventType.RUN_ERROR.value,
            payload,
            **finalize,
        )
    except Exception:
        if strict:
            raise
        logger.error(  # noqa: TRY400 - Keep safe_exception_info() to exclude vendor text and exception chains.
            "Failed to persist ERROR status for run %s", run_id, exc_info=safe_exception_info()
        )
        return False
    if needs_event:
        emit = _run_events._emit_event_safe if safe_event else _run_events._emit_event
        await emit(store, run_id, StoredEventType.RUN_ERROR.value, payload)
    return bool(won)


def _failure_fields(
    store: StateStore,
    exc: Exception | None,
    payload: dict[str, Any] | None,
    finalize: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if exc is not None:
        payload = _run_errors._error_payload(exc) if payload is None else payload
        finalize = _failure_finalize_fields(store, exc) if finalize is None else finalize
    return payload or {}, finalize or {}


def _failure_finalize_fields(store: StateStore, exc: Exception) -> dict[str, Any]:
    return {
        "error": _run_errors._safe_error(exc),
        **_run_errors._error_store_kwargs(store, exc),
        "total_usage": None,
    }


async def _stream_run_failure(
    store: StateStore | None,
    shared: dict[str, Any],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    exc: Exception,
) -> None:
    if store is not None and shared.get("claimed"):
        await _persist_run_failure(store, run_id, exc, safe_event=True)
    if not shared.get("existing_run"):
        try:
            await _hooks.record_run_failed(recorder, run_id, exc)
            await _hooks.notify_run_failed(notifier, run_id, exc)
        except Exception:
            logger.warning(
                "Lifecycle hooks failed during stream error handler for run %s",
                run_id,
                exc_info=safe_exception_info(),
            )


def _creation_claim(loop: Loop, created: CreateRunResult) -> ClaimResult:
    """The creator's epoch-0 claim, with a fresh checkpoint for a format-2 run."""
    state = (
        _initial_loop_state(_run_entry._recovery_loop_name(loop))
        if created.storage_format == STORAGE_FORMAT_CHECKPOINTED
        else None
    )
    return ClaimResult(0, created.storage_format, state, None)


async def _finish_lifecycle(
    lifecycle: Any, recorder: LoopRecorder | None, notifier: LoopNotifier | None, result: RunResult
) -> None:
    """Close the run's sinks unless it exits at a managed handoff (0.7)."""
    if MANAGED_DESCEND_KEY in result.meta:
        return  # Not finished: the parent continues after its child is adopted.
    lifecycle.recorder_closed = True
    await _hooks.record_run_finished(recorder, lifecycle.run_id, result)
    lifecycle.notifier_closed = True
    await _hooks.notify_run_finished(notifier, lifecycle.run_id, result)


async def _record_finished(
    recorder: LoopRecorder | None, notifier: LoopNotifier | None, run_id: str, result: RunResult
) -> None:
    """Record and notify a finished run; a managed handoff is not finished (0.7)."""
    if MANAGED_DESCEND_KEY in result.meta:
        return  # The parent exits at a managed handoff (0.7 driver).
    await _hooks.record_run_finished(recorder, run_id, result)
    await _hooks.notify_run_finished(notifier, run_id, result)


@dataclass
class _RunLifecycle(ManagedStopLifecycle):
    run_id: str
    store: StateStore | None
    manager: RunTaskManager | None
    owned: bool = False
    registered: bool = False
    started: bool = False
    recorder_closed: bool = False
    notifier_closed: bool = False
    recorder: LoopRecorder | None = None
    recorder_started: bool = False
    notifier_started: bool = False

    def _mark_notifier_started(self) -> None:
        self.notifier_started = True

    def _claim(
        self,
        store: StateStore,
        loop: Loop,
        created: CreateRunResult,
        claim: ClaimResult | None = None,
        root_limits: Limits | None = None,
    ) -> StateStore:
        self.run_id = created.run_id
        # A managed child's token comes from its bind and carries its header.
        self.store = _state.claimed_store(
            store, self.run_id, claim or _creation_claim(loop, created)
        )
        _managed.stamp_root_header(self.store, self.run_id, limits=root_limits)
        self.owned = True
        if self.manager is not None:
            self.manager.register_current(self.run_id)
            self.registered = True
        return self.store

    async def _finish(
        self, recorder: LoopRecorder | None, notifier: LoopNotifier | None, result: RunResult
    ) -> None:
        await _finish_lifecycle(self, recorder, notifier, result)

    async def _failed(
        self,
        exc: Exception,
        meta: dict[str, Any],
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
    ) -> None:
        if not self.started and isinstance(exc, (RunAlreadyActiveError, IdempotencyConflictError)):
            return
        _run_observers._attach_run_error_meta(exc, self.run_id, meta)
        if self.owned and self.store is not None:
            await _persist_run_failure(self.store, self.run_id, exc)
        if self.started:
            await self._fail_sinks(exc, recorder, notifier)

    async def _fail_sinks(
        self, exc: Exception, recorder: LoopRecorder | None, notifier: LoopNotifier | None
    ) -> None:
        if not self.recorder_closed:
            self.recorder_closed = True
            await _hooks.record_run_failed(recorder, self.run_id, exc)
        if not self.notifier_closed:
            self.notifier_closed = True
            await _hooks.notify_run_failed(notifier, self.run_id, exc)

    async def _fenced(
        self, recorder: LoopRecorder | None, notifier: LoopNotifier | None
    ) -> RunResult:
        assert self.store is not None
        result = await _run_pending._build_cached_result(self.store, self.run_id)
        if self.started:
            open_recorder = None if self.recorder_closed else recorder
            open_notifier = None if self.notifier_closed else notifier
            self.recorder_closed = self.notifier_closed = True
            await _run_events._finish_fenced_lifecycle(
                self.run_id, result, open_recorder, open_notifier
            )
        return result

    async def _cancel_finish(
        self, recorder: LoopRecorder | None, notifier: LoopNotifier | None, result: RunResult
    ) -> None:
        if not self.recorder_closed:
            self.recorder_closed = True
            try:
                await _hooks.record_run_finished(recorder, self.run_id, result)
            except Exception:
                logger.warning(
                    "Recorder finish failed for run %s", self.run_id, exc_info=safe_exception_info()
                )
        if not self.notifier_closed:
            self.notifier_closed = True
            await _hooks.notify_run_finished(notifier, self.run_id, result)

    async def _cancelled(
        self,
        result: RunResult | None,
        meta: dict[str, Any],
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
    ) -> tuple[RunResult, bool]:
        manager = self.manager if self.registered else None
        interrupted, swallow = _run_events._take_task_interrupt(manager, self.run_id)
        cancelled = await _run_events._drain_task_cancellation(
            self.store if self.owned else None, self.run_id, interrupted
        )
        if result is not None and (
            self.store is None
            or (result.status == cancelled.status and result.meta.get("_finalize_won"))
        ):
            cancelled = result
        _run_observers._attach_mcp_meta(cancelled, meta)
        if self.started:
            await self._cancel_finish(recorder, notifier, cancelled)
        return cancelled, _run_events._may_swallow_cancellation(swallow)

    def _release(self) -> None:
        if self.registered and self.manager is not None:
            self.manager.release_current(self.run_id)


def _recorder_tool_defs(agent: Agent) -> list[Any]:
    """The run's tool definitions, generated subagent calls included (0.7)."""
    from dendrux.runtime._subagents import subagent_tool_def
    from dendrux.tool import get_tool_def

    definitions = [get_tool_def(fn) for fn in agent.tools]
    definitions.extend(subagent_tool_def(child) for child in agent.subagents)
    return definitions


def _runtime_recorder(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    tenant_id: str | None,
    seeds: dict[str, Any],
    guardrails: Any,
) -> LoopRecorder:
    from dendrux.runtime.persistence import PersistenceRecorder

    definitions = _recorder_tool_defs(agent)
    recorder = PersistenceRecorder(
        store,
        run_id,
        model=provider.model,
        provider_name=type(provider).__name__,
        target_lookup={td.name: td.target for td in definitions},
        seed_traces=seeds.get("seed_traces"),
        guardrail_engine=guardrails,
        **_run_images._recorder_image_kwargs(tenant_id, provider),
    )
    _managed_budget.configure_recorder(recorder, agent)
    return recorder


async def _start_new_run(
    lifecycle: _RunLifecycle,
    agent: Agent,
    provider: LLMProvider,
    tenant_id: str | None,
    seeds: dict[str, Any],
    guardrails: Any,
    notifier: LoopNotifier | None,
    parent: DelegationContext | None,
    event_data: dict[str, Any],
    *,
    before_event: Callable[[], None] | None = None,
) -> tuple[Any, LoopNotifier | None]:
    if lifecycle.store is not None:
        lifecycle.recorder = _runtime_recorder(
            lifecycle.store, agent, provider, lifecycle.run_id, tenant_id, seeds, guardrails
        )
    if before_event is not None:
        before_event()
    await _run_events._emit_event(
        lifecycle.store, lifecycle.run_id, StoredEventType.RUN_STARTED.value, event_data
    )
    lifecycle.started = True
    await record_run_started(
        lifecycle.recorder, lifecycle.run_id, agent_name=agent.name, agent_model=provider.model
    )
    lifecycle.recorder_started = True
    return await _run_observers._start_trace(
        notifier,
        lifecycle.store,
        lifecycle.run_id,
        agent_name=agent.name,
        agent_model=provider.model,
        parent=parent,
        on_started=lifecycle._mark_notifier_started,
    )


_RunLifecycle.__module__ = "dendrux.runtime.runner"
