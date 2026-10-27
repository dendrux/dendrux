"""The per-Agent tool limiter's waiting queue is bounded and refuses clearly."""

from __future__ import annotations

import asyncio
import threading

import pytest

from dendrux.tools._pool import (
    DEFAULT_MAX_QUEUED_PER_AGENT,
    MAX_QUEUED_CALLS,
    QUEUE_LIMIT_ENV,
    CallTracker,
    ToolCallLimiter,
    ToolPoolSaturatedError,
    ToolQueueFullError,
    run_sync_tool,
)


async def _spawn(limiter: ToolCallLimiter, n: int) -> list[asyncio.Task[None]]:
    tasks = [asyncio.create_task(limiter.acquire()) for _ in range(n)]
    await asyncio.sleep(0)
    return tasks


async def _settle(tasks: list[asyncio.Task[None]]) -> tuple[int, int]:
    """(waiting, refused) for a batch of acquire tasks."""
    await asyncio.sleep(0)
    refused = sum(1 for t in tasks if t.done() and isinstance(t.exception(), ToolQueueFullError))
    return sum(1 for t in tasks if not t.done()), refused


async def _drain(limiter: ToolCallLimiter, tasks: list[asyncio.Task[None]]) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    limiter.release()


class TestQueueIsBounded:
    async def test_default_bound_is_below_the_process_backstop(self) -> None:
        assert 1 <= DEFAULT_MAX_QUEUED_PER_AGENT <= MAX_QUEUED_CALLS
        limiter = ToolCallLimiter(1)
        assert limiter.max_waiters == DEFAULT_MAX_QUEUED_PER_AGENT

    async def test_flood_queues_up_to_the_bound_and_refuses_the_rest(self) -> None:
        limiter = ToolCallLimiter(1)
        await limiter.acquire()
        tasks = await _spawn(limiter, 2048)
        waiting, refused = await _settle(tasks)
        try:
            assert waiting == DEFAULT_MAX_QUEUED_PER_AGENT
            assert refused == 2048 - DEFAULT_MAX_QUEUED_PER_AGENT
            assert len(limiter._waiters) == DEFAULT_MAX_QUEUED_PER_AGENT
        finally:
            await _drain(limiter, tasks)

    async def test_refusal_is_typed_and_a_pool_saturation_error(self) -> None:
        limiter = ToolCallLimiter(1, max_waiters=2)
        await limiter.acquire()
        tasks = await _spawn(limiter, 3)
        try:
            error = tasks[2].exception()
            assert isinstance(error, ToolQueueFullError)
            assert isinstance(error, ToolPoolSaturatedError)
            assert "queue" in str(error)
        finally:
            await _drain(limiter, tasks)

    async def test_override_through_the_constructor(self) -> None:
        limiter = ToolCallLimiter(1, max_waiters=3)
        await limiter.acquire()
        tasks = await _spawn(limiter, 10)
        waiting, refused = await _settle(tasks)
        try:
            assert (waiting, refused) == (3, 7)
        finally:
            await _drain(limiter, tasks)

    async def test_override_through_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(QUEUE_LIMIT_ENV, "4")
        limiter = ToolCallLimiter(1)
        assert limiter.max_waiters == 4

    @pytest.mark.parametrize("raw", ["-1", "abc", "1.5"])
    def test_bad_environment_value_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv(QUEUE_LIMIT_ENV, raw)
        limiter = ToolCallLimiter(1)  # building an Agent never reads the environment
        with pytest.raises(ValueError, match=QUEUE_LIMIT_ENV):
            limiter.max_waiters  # noqa: B018

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "3"])
    def test_bad_constructor_value_is_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_waiters"):
            ToolCallLimiter(1, max_waiters=bad)  # type: ignore[arg-type]

    async def test_zero_means_no_waiting_at_all(self) -> None:
        limiter = ToolCallLimiter(1, max_waiters=0)
        await limiter.acquire()
        with pytest.raises(ToolQueueFullError):
            await limiter.acquire()
        limiter.release()
        await limiter.acquire()
        limiter.release()


class TestNothingIsLostOrCorrupted:
    async def test_queued_calls_run_in_order_and_the_queue_reopens(self) -> None:
        limiter = ToolCallLimiter(1, max_waiters=2)
        await limiter.acquire()
        first, second, third = await _spawn(limiter, 3)
        assert third.done() and isinstance(third.exception(), ToolQueueFullError)
        limiter.release()
        await first
        assert not second.done()
        limiter.release()
        await second
        limiter.release()
        assert limiter.in_use == 0
        # Space is back: a fresh call acquires at once, and a full queue is possible again.
        await limiter.acquire()
        again = await _spawn(limiter, 2)
        assert [t.done() for t in again] == [False, False]
        await _drain(limiter, again)
        assert limiter.in_use == 0

    async def test_a_cancelled_waiter_frees_its_place(self) -> None:
        limiter = ToolCallLimiter(1, max_waiters=1)
        await limiter.acquire()
        (waiter,) = await _spawn(limiter, 1)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        (next_waiter,) = await _spawn(limiter, 1)
        assert not next_waiter.done()
        limiter.release()
        await next_waiter
        limiter.release()
        assert limiter.in_use == 0

    async def test_run_sync_tool_refuses_before_using_a_pool_thread(self) -> None:
        limiter = ToolCallLimiter(1, max_waiters=0)
        await limiter.acquire()
        ran: list[int] = []
        with pytest.raises(ToolQueueFullError):
            await run_sync_tool(
                lambda: ran.append(1), [], {}, limiter=limiter, tracker=CallTracker()
            )
        assert ran == []
        assert limiter.in_use == 1
        limiter.release()
        assert limiter.in_use == 0


class TestAcrossThreadsAndLoops:
    def test_the_bound_is_exact_with_many_threads_each_on_its_own_loop(self) -> None:
        limit, threads, per_thread = 50, 8, 40
        limiter = ToolCallLimiter(1, max_waiters=limit)
        accepted: list[int] = []
        refused: list[int] = []
        lock = threading.Lock()
        queued_everywhere = threading.Barrier(threads + 1)
        release_all = threading.Event()

        async def worker() -> None:
            tasks = [asyncio.create_task(limiter.acquire()) for _ in range(per_thread)]
            await asyncio.sleep(0.05)
            done = [t for t in tasks if t.done()]
            with lock:
                refused.append(sum(isinstance(t.exception(), ToolQueueFullError) for t in done))
                accepted.append(len(tasks) - len(done))
            queued_everywhere.wait(timeout=10)
            while not release_all.is_set():
                await asyncio.sleep(0.01)
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        def run() -> None:
            asyncio.run(worker())

        asyncio.run(limiter.acquire())  # hold the only slot
        pool = [threading.Thread(target=run) for i in range(threads)]
        for t in pool:
            t.start()
        queued_everywhere.wait(timeout=10)
        try:
            assert sum(accepted) == limit
            assert sum(refused) == threads * per_thread - limit
            assert len(limiter._waiters) == limit
        finally:
            release_all.set()
            for t in pool:
                t.join(timeout=10)
        assert not any(t.is_alive() for t in pool)


class TestSurfacedToTheModel:
    async def test_a_full_agent_queue_gives_a_retryable_result_and_never_runs_the_tool(
        self,
    ) -> None:
        import json
        import threading

        from dendrux.loops.react import _execute_tool
        from dendrux.tool import tool
        from dendrux.tools import build_tool_lookups
        from dendrux.tools._lookups import make_call_limiter
        from dendrux.types import ToolCall

        release = threading.Event()
        runs: list[int] = []

        @tool(timeout_seconds=5)
        def hold(i: int) -> str:
            """Hold."""
            runs.append(i)
            release.wait(5)
            return "ok"

        limiter = make_call_limiter(1, max_waiters=1)
        lookups = build_tool_lookups([hold], call_limiter=limiter)
        first = asyncio.create_task(_execute_tool(ToolCall(name="hold", params={"i": 1}), lookups))
        for _ in range(200):
            if runs:
                break
            await asyncio.sleep(0.01)
        second = asyncio.create_task(_execute_tool(ToolCall(name="hold", params={"i": 2}), lookups))
        await asyncio.sleep(0.05)
        third = await _execute_tool(ToolCall(name="hold", params={"i": 3}), lookups)
        assert third.success is False
        assert json.loads(third.payload)["code"] == "tool_pool_saturated"
        assert "safe to retry" in (third.error or "")
        release.set()
        assert (await first).success and (await second).success
        assert sorted(runs) == [1, 2]
        assert limiter.in_use == 0
