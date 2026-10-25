"""In-process asyncio task lifecycle for agent runs.

Used by :meth:`Agent.cancel_run` for same-process cancellation:
the task is tracked at spawn, cancel() is best-effort cooperative
cancellation against an in-flight run, and finished tasks are removed
from tracking automatically.

It also holds one interrupt signal per streamed run owned by this
process. :meth:`cancel` sets the signal so the loop driving the stream
can abandon an in-flight provider call between tokens.

Batch ``run()`` registers the task it executes on (:meth:`register_current`).
:meth:`cancel` cancels that task and records the run as interrupted, so the
runner can tell a ``cancel_run`` interrupt from a caller's own
``task.cancel()`` (:meth:`take_interrupt`). Several runs can share one task
(a child run awaited inside its parent's tool); the task is cancelled once
and every interrupted run on it is recorded.

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

from dendrux.errors import safe_exception_info

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
        self._interrupts: dict[str, asyncio.Event] = {}
        self._owned: dict[str, asyncio.Task[Any]] = {}
        # Runs whose owned task this manager cancelled and that have not yet
        # observed it, per task.
        self._pending: dict[asyncio.Task[Any], set[str]] = {}

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
        task.add_done_callback(_retrieve_exception)
        self._tasks[run_id] = task
        return task

    async def _run_wrapper(self, run_id: str, coro: Coroutine[Any, Any, T]) -> T:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Run %s failed: code=run_failed error_type=%s",
                run_id,
                type(exc).__name__,
                extra={"run_id": run_id, "code": "run_failed"},
                exc_info=safe_exception_info(),
            )
            raise
        finally:
            self._tasks.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        """Cancel the tracked task and/or set the interrupt signal for ``run_id``.

        Returns True if at least one signal was delivered (a background
        task was cancelled or a streamed run's interrupt was set); False
        if nothing is tracked for this run in this process.
        """
        delivered = False
        task = self._tasks.get(run_id)
        if task is not None and self._owned.get(run_id) is not task:
            task.cancel()
            delivered = True
        signal = self._interrupts.get(run_id)
        if signal is not None:
            signal.set()
            delivered = True
        owned = self._owned.get(run_id)
        if owned is not None and not owned.done():
            pending = self._pending.setdefault(owned, set())
            if run_id not in pending:
                if not pending:
                    owned.cancel()
                pending.add(run_id)
            delivered = True
        return delivered

    def register_current(self, run_id: str) -> None:
        """Track the current task as the one executing ``run_id`` (batch run())."""
        task = asyncio.current_task()
        if task is not None:
            self._owned[run_id] = task

    def take_interrupt(self, run_id: str) -> tuple[bool, bool]:
        """Consume a pending :meth:`cancel` interrupt for ``run_id``.

        Called by the runner when it catches ``CancelledError``. Returns
        ``(interrupted, others_pending)``: whether this manager cancelled the
        run's task for ``run_id``, and whether it still owes the same task's
        cancellation to another run (an ancestor sharing the task), in which
        case the runner must keep propagating it.
        """
        task = self._owned.get(run_id)
        if task is None:
            return False, False
        pending = self._pending.get(task)
        if not pending or run_id not in pending:
            return False, bool(pending)
        pending.discard(run_id)
        if not pending:
            del self._pending[task]
        return True, bool(pending)

    def release_current(self, run_id: str) -> None:
        """Forget ``run_id``'s task. No-op if unknown.

        An interrupt still pending here was never observed (the tool that
        received the ``CancelledError`` swallowed it); its cancel request is
        withdrawn so it cannot leak into the caller's later awaits.
        """
        task = self._owned.pop(run_id, None)
        if task is None:
            return
        pending = self._pending.get(task)
        if pending and run_id in pending:
            pending.discard(run_id)
            if not pending:
                del self._pending[task]
                if task.cancelling():
                    task.uncancel()

    def register_interrupt(self, run_id: str) -> asyncio.Event:
        """Return the interrupt signal for a streamed run, creating it if needed.

        The runner registers before driving the loop and releases in its
        cleanup; :meth:`cancel` sets it.
        """
        signal = self._interrupts.get(run_id)
        if signal is None:
            signal = asyncio.Event()
            self._interrupts[run_id] = signal
        return signal

    def release_interrupt(self, run_id: str) -> None:
        """Forget the interrupt signal for ``run_id``. No-op if unknown."""
        self._interrupts.pop(run_id, None)

    def has_interrupt(self, run_id: str) -> bool:
        """Return True if a streamed run's interrupt is registered here."""
        return run_id in self._interrupts

    def is_running(self, run_id: str) -> bool:
        """Return True if ``run_id`` has a tracked task in this process."""
        return run_id in self._tasks

    def __len__(self) -> int:
        return len(self._tasks)


def _retrieve_exception(task: asyncio.Task[Any]) -> None:
    """Consume failures without changing what a later awaiter receives."""
    if not task.cancelled():
        task.exception()
