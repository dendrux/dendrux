"""Deterministically model a pool wait returning after swallowing cancellation."""

import asyncio
from functools import wraps

from dendrux.runtime.persistence import PersistenceRecorder


class LostCancel:
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
