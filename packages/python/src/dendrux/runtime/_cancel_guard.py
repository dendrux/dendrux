"""Recover cancellation lost by Python 3.11 database pool waits."""

from __future__ import annotations

import asyncio
import logging
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

from dendrux import _codes

if TYPE_CHECKING:
    from collections.abc import Awaitable

_ENABLED = sys.version_info < (3, 12)
logger = logging.getLogger(__name__)
T = TypeVar("T")


class _CancelGuard:
    def __init__(self, task: asyncio.Task[Any]) -> None:
        self._task = task
        self._lost_before: int | None = None

    def store_calls(self, store: T) -> T:
        return cast("T", _StoreCalls(store, self))

    async def watch(self, operation: Awaitable[T]) -> T:
        if asyncio.current_task() is not self._task:
            return await operation
        before = self._task.cancelling()
        delivered = False
        try:
            return await operation
        except asyncio.CancelledError:
            delivered = True
            self._lost_before = None
            raise
        finally:
            if not delivered and self._task.cancelling() > before:
                self._lost_before = (
                    before if self._lost_before is None else min(self._lost_before, before)
                )

    async def check(self, run_id: str) -> None:
        if asyncio.current_task() is not self._task or self._lost_before is None:
            return
        before, self._lost_before = self._lost_before, None
        if self._task.cancelling() <= before:
            return
        # Deliver a queued cancel normally before synthesizing a lost one.
        await asyncio.sleep(0)
        if self._task.cancelling() > before:
            _codes.log_code(logger, _codes.RUN_CANCEL_RECOVERED, run_id=run_id)
            raise asyncio.CancelledError


class _StoreCalls:
    def __init__(self, store: Any, guard: _CancelGuard) -> None:
        self._store = store
        self._guard = guard

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._store, name)

        @wraps(method)
        async def watched(*args: Any, **kwargs: Any) -> Any:
            return await self._guard.watch(method(*args, **kwargs))

        return watched


def capture() -> _CancelGuard | None:
    if not _ENABLED:
        return None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return _CancelGuard(task) if task is not None else None
