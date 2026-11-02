"""Deterministically simulate Python 3.11's lost-cancellation races.

Two helpers, one per place a cancel can go missing: ``simulate_lost_cancel``
makes ``asyncio.wait_for`` drop one cancel; ``LostCancel`` makes a store's
pool wait return after swallowing one cancel at a recorder boundary.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import TYPE_CHECKING, Any

from dendrux.runtime.persistence import PersistenceRecorder

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import pytest


def simulate_lost_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make wait_for lose one cancellation, independent of the Python version."""
    lost = False

    async def wait_for(awaitable: Awaitable[Any], timeout: float | None) -> Any:
        nonlocal lost
        try:
            async with asyncio.timeout(timeout):
                return await awaitable
        except asyncio.CancelledError:
            if lost:
                raise
            lost = True
            return True

    monkeypatch.setattr(asyncio, "wait_for", wait_for)


class LostCancel:
    """Model a pool wait returning after swallowing cancellation."""

    def __init__(self, *, swallow=True):
        self.entered = asyncio.Event()
        self.swallow = swallow
        self.dropped = False
        self.owner = None
        self.run_id = None

    def install(self, monkeypatch, store, hook, method):
        original_hook = getattr(PersistenceRecorder, hook)
        original_wait = getattr(store, method)

        @wraps(original_hook)
        async def at_boundary(recorder, run_id, *args, **kwargs):
            if not self.entered.is_set():
                self.owner = asyncio.current_task()
                self.run_id = run_id
            return await original_hook(recorder, run_id, *args, **kwargs)

        @wraps(original_wait)
        async def connection_wait(*args, **kwargs):
            if asyncio.current_task() is self.owner and not self.entered.is_set():
                self.entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    if not self.swallow:
                        raise
                    self.dropped = True
            return await original_wait(*args, **kwargs)

        monkeypatch.setattr(PersistenceRecorder, hook, at_boundary)
        monkeypatch.setattr(store, method, connection_wait)
