"""Deliver a committed root run ID while the caller owns its lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dendrux import _codes
from dendrux._lifecycle import CancelReason
from dendrux.errors import RunFencedError
from dendrux.runtime._run_errors import logger
from dendrux.runtime._run_outcome import _finalize_loop_cancelled
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.loops.base import LoopNotifier
    from dendrux.runtime._run_start import _RunLifecycle
    from dendrux.runtime.state import StateStore


async def _cancel_requested(store: StateStore, run_id: str) -> bool:
    control = await store.get_run_control(run_id)
    if control.status is not None and control.claim_epoch is not None:
        # The claimed store already checked this snapshot against its epoch.
        return control.cancel_requested
    # Older stores expose only flags on the control view. Check the durable row.
    row = await store.get_run(run_id)
    epoch: int | None = getattr(store, "claim_epoch", None)
    if row is None or row.status != RunStatus.RUNNING.value or row.claim_epoch != epoch:
        _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
        raise RunFencedError(run_id, claim_epoch=epoch)
    return control.cancel_requested or row.cancel_requested


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
        if pending is not None:
            await pending
        if not await _cancel_requested(store, run_id):
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
