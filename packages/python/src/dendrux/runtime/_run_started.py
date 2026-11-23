"""Deliver a committed root run ID while the caller owns its lifecycle."""

from __future__ import annotations

import asyncio
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, cast

from dendrux import _codes
from dendrux._lifecycle import CancelReason
from dendrux.errors import RunFencedError
from dendrux.runtime._run_errors import logger
from dendrux.runtime._run_outcome import _finalize_loop_cancelled
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


async def bind_context_and_notify(
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
    """Bind the fresh run context through lifecycle cleanup, then deliver its root id."""
    store = lifecycle.store
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
        lease_interval_s=agent.lease_interval.total_seconds(),
        loop=asyncio.get_running_loop(),
    )
    lifecycle.context_token = set_delegation_context(ctx)
    return await notify_started(lifecycle, callback, delegation_level, notifier)


async def notify_started(
    lifecycle: _RunLifecycle,
    callback: Callable[[str], Awaitable[None] | None] | None,
    delegation_level: int,
    notifier: LoopNotifier | None,
) -> RunResult | None:
    """Return a settled cancellation if the callback boundary stops this run."""
    store, run_id = lifecycle.store, lifecycle.run_id
    if callback is None or store is None or delegation_level != 0:
        return None
    if not await _cancel_requested(store, run_id):
        pending = callback(run_id)
        if isawaitable(pending):
            await pending
        if not await _cancel_requested(store, run_id):
            # Only runs that delivered the callback need the extra SingleCall dispatch read.
            recorder: Any = lifecycle.recorder  # Fresh, persisted runs own this recorder.
            recorder._start_callback_delivered = True
            return None
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
