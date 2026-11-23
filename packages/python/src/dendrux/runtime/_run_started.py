"""Deliver a committed root run ID while the caller owns its lifecycle."""

from __future__ import annotations

import asyncio
from inspect import isawaitable
from typing import TYPE_CHECKING, cast

from dendrux import _codes
from dendrux._lifecycle import CancelReason
from dendrux.errors import OnStartedError, RunFencedError
from dendrux.runtime._run_errors import logger
from dendrux.runtime._run_outcome import _finalize_loop_cancelled
from dendrux.runtime._run_start import _persist_run_failure
from dendrux.runtime.context import DelegationContext, get_store_identity, set_delegation_context
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.agent import Agent
    from dendrux.loops.base import LoopNotifier
    from dendrux.runtime._run_start import _RunLifecycle
    from dendrux.runtime.state import StateStore, _ClaimedStateStore


async def _cancel_requested(store: StateStore, run_id: str) -> bool:
    control = await store.get_run_control(run_id)
    if control.status is not None and control.claim_epoch is not None:
        # The claimed store already checked this snapshot against its epoch.
        return control.cancel_requested
    # Flag-only stores need the durable row to complete this callback checkpoint.
    # _RunLifecycle._claim always wraps a fresh run with its ClaimResult.
    row = await store.get_run(run_id)
    epoch = cast("_ClaimedStateStore", store).claim_epoch
    if row is None or row.status != RunStatus.RUNNING.value or row.claim_epoch != epoch:
        _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
        raise RunFencedError(run_id, claim_epoch=epoch)
    return control.cancel_requested or row.cancel_requested


async def notify_then_bind_context(
    lifecycle: _RunLifecycle,
    agent: Agent,
    callback: Callable[[str], Awaitable[None] | None] | None,
    notifier: LoopNotifier | None,
    *,
    trace_link: str | None,
    delegation_level: int,
    max_depth: int | None,
    tenant_id: str | None,
    tenant_explicit: bool,
) -> RunResult | None:
    """Deliver the root id in the caller context before binding the new run."""
    store = lifecycle.store
    delivered = False
    if callback is not None and store is not None and delegation_level == 0:
        result = await _notify_started(lifecycle, store, callback, notifier)
        if result is not None:
            return result
        delivered = True
    ctx = DelegationContext(
        run_id=lifecycle.run_id,
        trace_link=trace_link,
        delegation_level=delegation_level,
        persisted=store is not None,
        store_identity=get_store_identity(store),
        max_delegation_depth=max_depth,
        tenant_id=tenant_id,
        tenant_explicit=tenant_explicit,
        state_store=store,
        start_callback_delivered=delivered,
        lease_interval_s=agent.lease_interval.total_seconds(),
        loop=asyncio.get_running_loop(),
    )
    lifecycle.context_token = set_delegation_context(ctx)
    return None


async def _notify_started(
    lifecycle: _RunLifecycle,
    store: StateStore,
    callback: Callable[[str], Awaitable[None] | None],
    notifier: LoopNotifier | None,
) -> RunResult | None:
    """Return a settled cancellation if the callback boundary stops this run."""
    run_id = lifecycle.run_id
    failure = None
    try:
        pending = callback(run_id)
        if isawaitable(pending):
            await pending
    except Exception as exc:  # noqa: BLE001, RUF100 - user callback isolation; only fixed diagnostics.
        _codes.log_code(logger, _codes.ON_STARTED_FAILED, run_id=run_id, callback_outcome="raised")
        failure = exc
    if not await _cancel_requested(store, run_id):
        if failure is not None:
            return await _fail_started(lifecycle, store, notifier, failure)
        return None
    return await _cancel_started(lifecycle, store, notifier)


async def _fail_started(
    lifecycle: _RunLifecycle, store: StateStore, notifier: LoopNotifier | None, failure: Exception
) -> RunResult:
    run_id = lifecycle.run_id
    safe = OnStartedError(run_id)
    # The outer error handler must not replace or repeat this atomic transition.
    lifecycle.failure_write_started = True
    won = await _persist_run_failure(
        store,
        run_id,
        None,
        payload={"error": safe.safe_message, "code": safe.code},
        finalize={"error": safe.safe_message},
        optional_failure_reason=safe.code,
        strict=True,
    )
    if not won:
        if await _cancel_requested(store, run_id):
            return await _cancel_started(lifecycle, store, notifier)
        return await lifecycle._fenced(lifecycle.recorder, notifier)
    await lifecycle._fail_sinks(safe, lifecycle.recorder, notifier)
    # Observers can retain their error, so attach the private cause only to the caller's copy.
    raise OnStartedError(run_id) from failure


async def _cancel_started(
    lifecycle: _RunLifecycle, store: StateStore, notifier: LoopNotifier | None
) -> RunResult:
    run_id = lifecycle.run_id
    result = await _finalize_loop_cancelled(
        store,
        run_id,
        RunResult(run_id=run_id, status=RunStatus.CANCELLED),
        None,
        {},
        [RunStatus.RUNNING.value],
        {"reason": CancelReason.CANCEL_REQUESTED.value},
    )
    await lifecycle._finish(lifecycle.recorder, notifier, result)
    return result
