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

        task = mgr.spawn("00000000000000000000000105", work())
        assert mgr.is_running("00000000000000000000000105")
        assert await task == "done"

    async def test_task_removes_itself_on_completion(self) -> None:
        mgr = RunTaskManager()

        async def work() -> None:
            return None

        task = mgr.spawn("00000000000000000000000105", work())
        await task
        assert not mgr.is_running("00000000000000000000000105")
        assert len(mgr) == 0

    async def test_task_removes_itself_on_exception(self) -> None:
        mgr = RunTaskManager()

        async def boom() -> None:
            raise RuntimeError("nope")

        task = mgr.spawn("00000000000000000000000105", boom())
        with contextlib.suppress(RuntimeError):
            await task
        assert not mgr.is_running("00000000000000000000000105")


class TestCancel:
    async def test_cancel_of_unknown_run_returns_false(self) -> None:
        mgr = RunTaskManager()
        assert mgr.cancel("missing") is False

    async def test_cancel_propagates_to_task(self) -> None:
        mgr = RunTaskManager()
        observed = asyncio.Event()

        async def long_work() -> None:
            mgr.register_current("00000000000000000000000105")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                observed.set()
                raise

        task = mgr.spawn("00000000000000000000000105", long_work())
        # Yield once so the task enters `await asyncio.sleep`.
        await asyncio.sleep(0)

        assert mgr.cancel("00000000000000000000000105") is True
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert observed.is_set()
        assert not mgr.is_running("00000000000000000000000105")

    async def test_cancel_is_idempotent_after_completion(self) -> None:
        mgr = RunTaskManager()

        async def work() -> str:
            return "done"

        task = mgr.spawn("00000000000000000000000105", work())
        await task
        # Task already removed; subsequent cancel should report False, not raise.
        assert mgr.cancel("00000000000000000000000105") is False


class TestInterruptSignals:
    async def test_register_returns_unset_event(self) -> None:
        mgr = RunTaskManager()
        signal = mgr.register_interrupt("00000000000000000000000105")
        assert isinstance(signal, asyncio.Event)
        assert not signal.is_set()
        assert mgr.has_interrupt("00000000000000000000000105")

    async def test_register_is_idempotent_per_run(self) -> None:
        mgr = RunTaskManager()
        first = mgr.register_interrupt("00000000000000000000000105")
        second = mgr.register_interrupt("00000000000000000000000105")
        assert first is second

    async def test_cancel_sets_registered_interrupt(self) -> None:
        mgr = RunTaskManager()
        signal = mgr.register_interrupt("00000000000000000000000105")
        assert mgr.cancel("00000000000000000000000105") is True
        assert signal.is_set()

    async def test_cancel_returns_false_when_nothing_tracked(self) -> None:
        mgr = RunTaskManager()
        assert mgr.cancel("ghost") is False

    async def test_release_removes_interrupt(self) -> None:
        mgr = RunTaskManager()
        mgr.register_interrupt("00000000000000000000000105")
        mgr.release_interrupt("00000000000000000000000105")
        assert not mgr.has_interrupt("00000000000000000000000105")
        assert mgr.cancel("00000000000000000000000105") is False

    async def test_release_unknown_is_noop(self) -> None:
        mgr = RunTaskManager()
        mgr.release_interrupt("ghost")
        assert not mgr.has_interrupt("ghost")

    async def test_cancel_sets_interrupt_and_cancels_task_together(self) -> None:
        mgr = RunTaskManager()
        started = asyncio.Event()

        async def work() -> None:
            mgr.register_current("00000000000000000000000105")
            started.set()
            await asyncio.sleep(10)

        task = mgr.spawn("00000000000000000000000105", work())
        signal = mgr.register_interrupt("00000000000000000000000105")
        await started.wait()

        assert mgr.cancel("00000000000000000000000105") is True
        assert signal.is_set()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
