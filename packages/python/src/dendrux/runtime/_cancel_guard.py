"""Recover cancellation lost by Python 3.11 database pool waits."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from dendrux import _codes

_ENABLED = sys.version_info < (3, 12)
logger = logging.getLogger(__name__)


class _CancelGuard:
    def __init__(self, task: asyncio.Task[Any]) -> None:
        self._task = task
        self._baseline = task.cancelling()

    async def check(self, run_id: str) -> None:
        if asyncio.current_task() is not self._task:
            return
        count = self._task.cancelling()
        self._baseline = min(self._baseline, count)
        if count <= self._baseline:
            return
        # Deliver a queued cancel normally before synthesizing a lost one.
        await asyncio.sleep(0)
        count = self._task.cancelling()
        self._baseline = min(self._baseline, count)
        if count > self._baseline:
            _codes.log_code(logger, _codes.RUN_CANCEL_RECOVERED, run_id=run_id)
            raise asyncio.CancelledError


def capture() -> _CancelGuard | None:
    if not _ENABLED:
        return None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return _CancelGuard(task) if task is not None else None
