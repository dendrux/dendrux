"""Minimal in-memory task manager for background agent runs (D8).

Not a full worker pool — just the minimum viable task lifecycle for a
single-process server. Ensures:
- Task exceptions are observed (not silently swallowed by asyncio)
- Terminal events buffered for late SSE subscribers (TTL + max-size bounded)
- task.cancel() provides best-effort cooperative cancellation

Sprint 4 will move terminal replay to DB-backed reconstruction.
The in-memory buffer will become an optimization, not the source of truth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Defaults for terminal event retention
_DEFAULT_TERMINAL_TTL_SECONDS = 600  # 10 minutes
_DEFAULT_MAX_TERMINAL_EVENTS = 10_000


class RunTaskManager:
    """Manages background asyncio.Tasks for agent runs.

    Terminal event buffer is bounded by both TTL and max size to prevent
    unbounded memory growth from runs that never open SSE.

    Args:
        terminal_ttl_seconds: How long to keep terminal events for late
            subscribers. Default: 10 minutes.
        max_terminal_events: Maximum number of buffered terminal events.
            When exceeded, oldest entries are evicted. Default: 10,000.

    Usage:
        manager = RunTaskManager()
        manager.spawn(run_id, coro)
        manager.cancel(run_id)
        event = manager.get_terminal_event(run_id)
    """

    def __init__(
        self,
        *,
        terminal_ttl_seconds: float = _DEFAULT_TERMINAL_TTL_SECONDS,
        max_terminal_events: int = _DEFAULT_MAX_TERMINAL_EVENTS,
    ) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._terminal_events: dict[str, dict[str, Any]] = {}
        self._terminal_timestamps: dict[str, float] = {}
        self._terminal_ttl = terminal_ttl_seconds
        self._max_terminal = max_terminal_events

    def spawn(self, run_id: str, coro: Any) -> asyncio.Task[Any]:
        """Spawn a background task for a run."""
        task = asyncio.create_task(self._run_wrapper(run_id, coro), name=f"run:{run_id}")
        self._tasks[run_id] = task
        return task

    async def _run_wrapper(self, run_id: str, coro: Any) -> Any:
        """Wrapper that observes exceptions and manages task lifecycle.

        Does NOT buffer terminal events — that is the responsibility of
        the CAS winner (_run_agent for success/error, DELETE for cancel).
        """
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
        """Cancel a running task. Returns True if a task was found and cancelled."""
        task = self._tasks.get(run_id)
        if task is None:
            return False
        task.cancel()
        return True

    def buffer_terminal_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Buffer a terminal event for late SSE subscribers.

        Enforces max-size cap by evicting oldest entries when full.
        Called by the CAS winner — only the code path that won the
        terminal DB transition should call this.
        """
        # Evict expired entries opportunistically
        self._evict_expired()

        # Evict oldest if at capacity
        while len(self._terminal_events) >= self._max_terminal and self._terminal_timestamps:
            oldest_id = min(self._terminal_timestamps, key=self._terminal_timestamps.get)  # type: ignore[arg-type]
            self._terminal_events.pop(oldest_id, None)
            self._terminal_timestamps.pop(oldest_id, None)

        self._terminal_events[run_id] = event
        self._terminal_timestamps[run_id] = time.monotonic()

    def get_terminal_event(self, run_id: str) -> dict[str, Any] | None:
        """Get the terminal event for a completed run (for late SSE subscribers).

        Returns None if the event has expired or was never buffered.
        """
        ts = self._terminal_timestamps.get(run_id)
        if ts is None:
            return None
        if time.monotonic() - ts > self._terminal_ttl:
            # Expired — clean up and return None
            self._terminal_events.pop(run_id, None)
            self._terminal_timestamps.pop(run_id, None)
            return None
        return self._terminal_events.get(run_id)

    def cleanup(self, run_id: str) -> None:
        """Remove all state for a run. Call after the SSE stream ends."""
        self._tasks.pop(run_id, None)
        self._terminal_events.pop(run_id, None)
        self._terminal_timestamps.pop(run_id, None)

    def _evict_expired(self) -> None:
        """Remove terminal events that have exceeded the TTL."""
        now = time.monotonic()
        expired = [
            rid for rid, ts in self._terminal_timestamps.items() if now - ts > self._terminal_ttl
        ]
        for rid in expired:
            self._terminal_events.pop(rid, None)
            self._terminal_timestamps.pop(rid, None)

    def is_running(self, run_id: str) -> bool:
        """Check if a task is currently running for this run_id."""
        return run_id in self._tasks

    @property
    def terminal_event_count(self) -> int:
        """Number of buffered terminal events (for monitoring)."""
        return len(self._terminal_events)

    def __len__(self) -> int:
        return len(self._tasks)
