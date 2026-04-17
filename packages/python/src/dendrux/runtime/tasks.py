"""In-process asyncio task lifecycle for agent runs.

Used by :meth:`Agent.cancel_run` for same-process cancellation:
the task is tracked at spawn, cancel() is best-effort cooperative
cancellation against an in-flight run, and finished tasks are removed
from tracking automatically.

Cross-process cancellation is not handled here — that requires the run
to be executing in this process. Runs on other workers can still be
CAS-finalized in the DB; they just won't be preempted mid-call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RunTaskManager:
    """Tracks background asyncio.Tasks for agent runs in the current process.

    Usage::

        manager = RunTaskManager()
        manager.spawn(run_id, coro)
        manager.cancel(run_id)  # best-effort cooperative cancel
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def spawn(self, run_id: str, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Spawn a background task for ``run_id`` and track it.

        The returned task resolves to the coroutine's result, so callers
        can ``await`` it to get the typed return value (e.g.
        :class:`~dendrux.types.RunResult`). Tasks auto-remove themselves
        from tracking on completion.
        """
        task: asyncio.Task[T] = asyncio.create_task(
            self._run_wrapper(run_id, coro), name=f"run:{run_id}"
        )
        self._tasks[run_id] = task
        return task

    async def _run_wrapper(self, run_id: str, coro: Coroutine[Any, Any, T]) -> T:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Run %s failed with exception", run_id)
            raise
        finally:
            self._tasks.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        """Cancel the tracked task for ``run_id``.

        Returns True if a task was tracked and the cancel signal was
        delivered; False if no task is currently tracked for this run
        (either it was never spawned in this process or it already
        finished).
        """
        task = self._tasks.get(run_id)
        if task is None:
            return False
        task.cancel()
        return True

    def is_running(self, run_id: str) -> bool:
        """Return True if ``run_id`` has a tracked task in this process."""
        return run_id in self._tasks

    def __len__(self) -> int:
        return len(self._tasks)
