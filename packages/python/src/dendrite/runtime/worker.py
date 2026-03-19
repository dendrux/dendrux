"""WorkerLoop — DB-polling worker for background agent execution.

A long-running async process that:
  1. Polls DB for pending runs → claims via ExecutionLease → executes
  2. Sweeps for stale running runs (heartbeat expired) → reclaims
  3. Manages concurrency via capacity-based claiming

The execution seam is execute_pending_run() and recover_run() in
runner.py — public module-level functions that future Celery/Redis
workers can call directly without using the DB polling loop.

Usage:
    worker = WorkerLoop(state_store=store, registry=registry)
    await worker.start()  # blocks until stop()

    # From another coroutine:
    await worker.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrite.registry import AgentRegistry
    from dendrite.runtime.state import StateStore

logger = logging.getLogger(__name__)


class WorkerLoop:
    """DB-polling worker that claims, executes, and recovers agent runs.

    Args:
        state_store: Persistence backend for run state and lease operations.
        registry: Agent configurations. The worker only executes runs whose
            agent_name is registered here.
        max_concurrent: Maximum parallel runs this worker will execute.
        poll_interval: Seconds between pending-run polls.
        sweep_interval: Seconds between stale-run sweeps.
        stale_threshold: Seconds before a running run is considered stale.
    """

    def __init__(
        self,
        state_store: StateStore,
        registry: AgentRegistry,
        *,
        max_concurrent: int = 5,
        poll_interval: float = 2.0,
        sweep_interval: float = 30.0,
        stale_threshold: int = 60,
    ) -> None:
        self._store = state_store
        self._registry = registry
        self._max_concurrent = max_concurrent
        self._poll_interval = poll_interval
        self._sweep_interval = sweep_interval
        self._stale_threshold = stale_threshold

        self._shutdown = asyncio.Event()
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._drain_timeout: float = 30.0

    @property
    def active_count(self) -> int:
        """Number of currently executing runs."""
        return len(self._active_tasks)

    async def start(self) -> None:
        """Run the worker until stop() is called. Blocks."""
        logger.info(
            "WorkerLoop starting (max_concurrent=%d, poll=%.1fs, sweep=%.1fs, stale=%ds)",
            self._max_concurrent,
            self._poll_interval,
            self._sweep_interval,
            self._stale_threshold,
        )

        poll_task = asyncio.create_task(self._poll_loop(), name="worker-poll")
        sweep_task = asyncio.create_task(self._sweep_loop(), name="worker-sweep")

        try:
            await self._shutdown.wait()
        finally:
            # Stop poll/sweep loops
            poll_task.cancel()
            sweep_task.cancel()
            for t in [poll_task, sweep_task]:
                with contextlib.suppress(asyncio.CancelledError):
                    await t

            # Drain active tasks with timeout
            if self._active_tasks:
                logger.info(
                    "Waiting for %d active runs to complete (timeout=%.1fs)...",
                    len(self._active_tasks),
                    self._drain_timeout,
                )
                done, pending = await asyncio.wait(
                    self._active_tasks.values(), timeout=self._drain_timeout
                )
                if pending:
                    logger.warning(
                        "%d tasks did not finish within %.1fs — cancelling",
                        len(pending),
                        self._drain_timeout,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                self._cleanup_done_tasks()

        logger.info("WorkerLoop stopped")

    async def stop(self, timeout: float = 30.0) -> None:
        """Signal the worker to stop.

        After calling stop(), start() will:
        1. Stop poll/sweep loops
        2. Wait for in-flight tasks to complete
        3. Cancel tasks that exceed the timeout
        4. NOT force-release leases — they expire naturally and the
           sweeper (on another worker) recovers them.

        The timeout is stored for start()'s finally block.
        """
        logger.info("WorkerLoop stopping...")
        self._drain_timeout = timeout
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodically poll for pending runs."""
        while not self._shutdown.is_set():
            try:
                await self._poll_once()
            except Exception:
                logger.warning("Poll cycle failed", exc_info=True)

            # Sleep but wake on shutdown
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._poll_interval)
                return  # shutdown signaled
            except TimeoutError:
                pass  # normal — keep polling

    async def _sweep_loop(self) -> None:
        """Periodically sweep for stale runs."""
        while not self._shutdown.is_set():
            try:
                await self._sweep_once()
            except Exception:
                logger.warning("Sweep cycle failed", exc_info=True)

            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._sweep_interval)
                return
            except TimeoutError:
                pass

    async def _poll_once(self) -> int:
        """Find pending runs and spawn execution tasks. Returns count spawned."""
        # Clean up completed tasks first
        self._cleanup_done_tasks()

        available = self._max_concurrent - len(self._active_tasks)
        if available <= 0:
            return 0

        # Query only as many as we have capacity for.
        # TODO: if many pending runs belong to unregistered agents, this
        # query returns them but we skip them, potentially starving valid
        # runs further back. In practice the registry covers all agents
        # on the worker, so this is unlikely. If it becomes a problem,
        # filter by agent_name in the query (requires StateStore change).
        pending_runs = await self._store.list_runs(status="pending", limit=available)

        spawned = 0
        for run_record in pending_runs:
            # Skip if we don't have a config for this agent
            try:
                self._registry.get(run_record.agent_name)
            except KeyError:
                logger.debug(
                    "Skipping run %s — agent '%s' not in registry",
                    run_record.id,
                    run_record.agent_name,
                )
                continue

            # Skip if already being handled (shouldn't happen, but defensive)
            if run_record.id in self._active_tasks:
                continue

            # Determine if this is a fresh run or a recovery
            recover = run_record.retry_count > 0

            task = asyncio.create_task(
                self._run_task(run_record.id, run_record.agent_name, recover=recover),
                name=f"worker-run-{run_record.id[:8]}",
            )
            self._active_tasks[run_record.id] = task
            spawned += 1

        if spawned:
            logger.info("Spawned %d run tasks (%d active)", spawned, len(self._active_tasks))
        return spawned

    async def _sweep_once(self) -> int:
        """Find stale runs and reclaim them. Returns count reclaimed."""
        stale_ids = await self._store.find_stale_runs(self._stale_threshold)
        reclaimed = 0
        for run_id in stale_ids:
            try:
                success = await self._store.reclaim_stale_run(run_id)
                if success:
                    reclaimed += 1
                    logger.info("Reclaimed stale run %s (will be picked up by next poll)", run_id)
                else:
                    logger.info("Stale run %s hit retry limit — marked as error", run_id)
            except Exception:
                logger.warning("Failed to reclaim stale run %s", run_id, exc_info=True)

        return reclaimed

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def _run_task(self, run_id: str, agent_name: str, *, recover: bool) -> None:
        """Execute or recover a single run. Called as an asyncio.Task."""
        try:
            config = self._registry.get(agent_name)
            provider = config.provider_factory()
            strategy = config.strategy_factory() if config.strategy_factory else None
            loop = config.loop_factory() if config.loop_factory else None
            redact = config.redact

            if recover:
                from dendrite.runtime.runner import recover_run

                result = await recover_run(
                    run_id,
                    state_store=self._store,
                    agent=config.agent,
                    provider=provider,
                    strategy=strategy,
                    loop=loop,
                    redact=redact,
                )
                logger.info("Recovered run %s → %s", run_id, result.status.value)
            else:
                from dendrite.runtime.runner import execute_pending_run

                result = await execute_pending_run(
                    run_id,
                    state_store=self._store,
                    agent=config.agent,
                    provider=provider,
                    strategy=strategy,
                    loop=loop,
                    redact=redact,
                )
                logger.info("Completed run %s → %s", run_id, result.status.value)

        except Exception:
            # execute_pending_run / recover_run already handle finalize_run(error)
            # internally. This catch is for unexpected failures only.
            logger.warning("Run task %s failed", run_id, exc_info=True)
        finally:
            self._active_tasks.pop(run_id, None)

    def _cleanup_done_tasks(self) -> None:
        """Remove completed/cancelled tasks from the active map."""
        done = [rid for rid, t in self._active_tasks.items() if t.done()]
        for rid in done:
            task = self._active_tasks.pop(rid)
            # Observe any unhandled exception to prevent asyncio warnings.
            # task.exception() raises CancelledError on cancelled tasks,
            # so check cancelled() first.
            if not task.cancelled() and task.exception() is not None:
                logger.debug("Task for run %s had exception: %s", rid, task.exception())
