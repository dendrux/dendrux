"""Unit tests for :class:`dendrux.runtime.tasks.RunTaskManager`.

Covers the in-process task lifecycle used by :meth:`Agent.cancel_run`:
spawn, cancel propagation, and auto-removal from tracking on completion.
"""

from __future__ import annotations

import asyncio
import contextlib

from dendrux.runtime.tasks import RunTaskManager


class TestSpawnAndTracking:
    async def test_spawn_returns_running_task(self) -> None:
        mgr = RunTaskManager()

        async def work() -> str:
            return "done"

        task = mgr.spawn("r1", work())
        assert mgr.is_running("r1")
        assert await task == "done"

    async def test_task_removes_itself_on_completion(self) -> None:
        mgr = RunTaskManager()

        async def work() -> None:
            return None

        task = mgr.spawn("r1", work())
        await task
        assert not mgr.is_running("r1")
        assert len(mgr) == 0

    async def test_task_removes_itself_on_exception(self) -> None:
        mgr = RunTaskManager()

        async def boom() -> None:
            raise RuntimeError("nope")

        task = mgr.spawn("r1", boom())
        with contextlib.suppress(RuntimeError):
            await task
        assert not mgr.is_running("r1")


class TestCancel:
    async def test_cancel_of_unknown_run_returns_false(self) -> None:
        mgr = RunTaskManager()
        assert mgr.cancel("missing") is False

    async def test_cancel_propagates_to_task(self) -> None:
        mgr = RunTaskManager()
        observed = asyncio.Event()

        async def long_work() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                observed.set()
                raise

        task = mgr.spawn("r1", long_work())
        # Yield once so the task enters `await asyncio.sleep`.
        await asyncio.sleep(0)

        assert mgr.cancel("r1") is True
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert observed.is_set()
        assert not mgr.is_running("r1")

    async def test_cancel_is_idempotent_after_completion(self) -> None:
        mgr = RunTaskManager()

        async def work() -> str:
            return "done"

        task = mgr.spawn("r1", work())
        await task
        # Task already removed; subsequent cancel should report False, not raise.
        assert mgr.cancel("r1") is False
