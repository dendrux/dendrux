"""0.5.0 D-048: bounded sync-tool execution.

One process-wide pool of daemon threads, a per-Agent limiter
(``Agent(tool_threads=)``), ``timeout_not_started`` vs
``timeout_outcome_unknown``, the zombie gauge, and no thread leak after a
cancelled run.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import ProviderTimeoutError
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop, _execute_tool
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.tools import _pool as pool_mod
from dendrux.tools import build_tool_lookups, tool_pool_stats
from dendrux.tools._lookups import make_call_limiter
from dendrux.tools._pool import ToolCallLimiter, _ToolPool
from dendrux.types import LLMResponse, RunStatus, ToolCall


@pytest.fixture
def small_pool(monkeypatch: pytest.MonkeyPatch):
    """A fresh pool per test; its threads are released and shut down after."""
    created: list[_ToolPool] = []

    def make(size: int) -> _ToolPool:
        pool = _ToolPool(size)
        created.append(pool)
        monkeypatch.setattr(pool_mod, "_pool", pool)
        return pool

    yield make
    for pool in created:
        pool.shutdown()


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


class Gauge:
    """Thread-safe current / peak concurrency counter."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self) -> None:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def __exit__(self, *exc: Any) -> None:
        with self.lock:
            self.current -= 1


def _batch(name: str, n: int, *, final: str = "done") -> MockLLM:
    calls = [ToolCall(name=name, params={"i": i}) for i in range(n)]
    return MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text=final)])


async def _run(agent: Agent, llm: MockLLM):
    return await ReActLoop().run(
        agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
    )


def _payloads(llm: MockLLM) -> list[Any]:
    return [
        json.loads(m.content) for m in llm.call_history[1]["messages"] if m.role.value == "tool"
    ]


class TestCapsUnderLoad:
    async def test_per_agent_limiter_and_pool_size_hold(self, small_pool) -> None:
        small_pool(6)
        gauges = {"a": Gauge(), "b": Gauge(), "c": Gauge()}
        total = Gauge()

        def make(name: str):
            def work(i: int) -> int:
                """Work."""
                with gauges[name], total:
                    time.sleep(0.02)
                return i

            work.__name__ = f"work_{name}"
            return tool()(work)

        agents = {
            "a": Agent(prompt="p", tools=[make("a")], tool_threads=2),
            "b": Agent(prompt="p", tools=[make("b")], tool_threads=3),
            "c": Agent(prompt="p", tools=[make("c")], tool_threads=8),
        }
        llms = {name: _batch(f"work_{name}", 24) for name in agents}
        results = await asyncio.gather(*[_run(agents[n], llms[n]) for n in agents])

        assert all(r.status == RunStatus.SUCCESS for r in results)
        for name in agents:
            assert sorted(_payloads(llms[name])) == list(range(24))
        assert gauges["a"].peak <= 2
        assert gauges["b"].peak <= 3
        assert gauges["c"].peak <= 6
        assert total.peak <= 6
        stats = tool_pool_stats()
        assert stats.threads <= 6 and stats.busy == 0 and stats.queued == 0
        for agent in agents.values():
            assert agent._tool_call_limiter.in_use == 0

    async def test_sync_tools_never_use_the_default_executor(self, small_pool) -> None:
        small_pool(2)
        names: list[str] = []

        @tool()
        def who(i: int) -> str:
            """Who."""
            names.append(threading.current_thread().name)
            return "ok"

        llm = _batch("who", 3)
        await _run(Agent(prompt="p", tools=[who]), llm)
        assert names and all(n.startswith("dendrux-tool-") for n in names)
        assert all(t.daemon for t in threading.enumerate() if t.name.startswith("dendrux-tool-"))

    async def test_context_variables_reach_sync_tools(self, small_pool) -> None:
        small_pool(1)
        var: contextvars.ContextVar[str] = contextvars.ContextVar("v", default="unset")
        seen: list[str] = []

        @tool()
        def read_var() -> str:
            """Read."""
            seen.append(var.get())
            return "ok"

        var.set("from-caller")
        lookups = build_tool_lookups([read_var], call_limiter=make_call_limiter(2))
        await _execute_tool(ToolCall(name="read_var"), lookups)
        assert seen == ["from-caller"]


class TestTimeouts:
    async def test_a_queued_call_times_out_as_not_started(self, small_pool) -> None:
        small_pool(1)
        release = threading.Event()
        ran: list[str] = []

        @tool(timeout_seconds=5)
        def blocker() -> str:
            """Block."""
            release.wait(5)
            return "done"

        @tool(timeout_seconds=0.1)
        def victim() -> str:
            """Never runs."""
            ran.append("victim")
            return "ran"

        lookups = build_tool_lookups([blocker, victim], call_limiter=make_call_limiter(4))
        blocking = asyncio.create_task(_execute_tool(ToolCall(name="blocker"), lookups))
        await _until(lambda: tool_pool_stats().busy == 1)

        result = await _execute_tool(ToolCall(name="victim"), lookups)

        payload = json.loads(result.payload)
        assert payload["code"] == "timeout_not_started"
        assert "safe to retry" in payload["error"]
        assert result.meta == {"code": "timeout_not_started"}
        release.set()
        assert (await blocking).success
        await _until(lambda: tool_pool_stats().busy == 0)
        assert ran == []  # the cancelled call never ran, even after a thread freed
        assert tool_pool_stats().queued == 0
        assert lookups.call_limiter.in_use == 0

    async def test_a_started_call_times_out_as_outcome_unknown(self, small_pool) -> None:
        small_pool(2)
        release = threading.Event()
        effects: list[str] = []

        @tool(timeout_seconds=0.1)
        def slow_refund(order_id: int) -> str:
            """Refund slowly."""
            release.wait(5)
            effects.append("refunded")
            return "refunded"

        limiter = make_call_limiter(4)
        lookups = build_tool_lookups([slow_refund], call_limiter=limiter)
        result = await _execute_tool(
            ToolCall(name="slow_refund", params={"order_id": 7}), lookups, run_id="run-1"
        )

        payload = json.loads(result.payload)
        assert payload["code"] == "timeout_outcome_unknown"
        assert "do not retry blindly" in payload["error"]
        assert result.meta == {"code": "timeout_outcome_unknown", "zombie_threads": 1}
        assert tool_pool_stats().zombies == 1
        # The slot stays taken while the thread runs: hung calls hold their Agent's slots.
        assert limiter.in_use == 1

        release.set()
        await _until(lambda: tool_pool_stats().zombies == 0)
        await _until(lambda: limiter.in_use == 0)
        assert effects == ["refunded"]

    async def test_a_hung_agent_cannot_starve_another(self, small_pool) -> None:
        small_pool(4)
        release = threading.Event()

        @tool(timeout_seconds=0.1)
        def hang(i: int) -> str:
            """Hang."""
            release.wait(5)
            return "late"

        @tool(timeout_seconds=2)
        def fine(i: int) -> int:
            """Fine."""
            return i

        hung = Agent(prompt="p", tools=[hang], tool_threads=2)
        healthy = Agent(prompt="p", tools=[fine], tool_threads=2)
        hung_llm = _batch("hang", 3)
        await _run(hung, hung_llm)
        codes = sorted(p["code"] for p in _payloads(hung_llm))
        # Two started and hung; the third waited for its own Agent's slot.
        assert codes == [
            "timeout_not_started",
            "timeout_outcome_unknown",
            "timeout_outcome_unknown",
        ]
        assert tool_pool_stats().zombies == 2

        healthy_llm = _batch("fine", 4)
        result = await _run(healthy, healthy_llm)
        assert result.status == RunStatus.SUCCESS
        assert sorted(_payloads(healthy_llm)) == [0, 1, 2, 3]
        release.set()
        await _until(lambda: tool_pool_stats().zombies == 0)

    async def test_a_timeout_raised_by_the_tool_is_an_ordinary_error(self, small_pool) -> None:
        small_pool(1)

        @tool(timeout_seconds=30)
        async def delegate() -> str:
            """Runs a child agent whose provider times out."""
            raise ProviderTimeoutError(provider="anthropic", model="m", message="read timeout")

        @tool(timeout_seconds=30)
        def sync_http() -> str:
            """A sync client with its own timeout."""
            raise TimeoutError("upstream timed out")

        lookups = build_tool_lookups([delegate, sync_http], call_limiter=make_call_limiter(2))
        for name in ("delegate", "sync_http"):
            result = await _execute_tool(ToolCall(name=name), lookups)
            payload = json.loads(result.payload)
            assert result.success is False
            assert "code" not in payload
            assert "timed out after" not in payload["error"]

    async def test_async_timeout_reports_outcome_unknown(self, small_pool) -> None:
        @tool(timeout_seconds=0.05)
        async def slow() -> str:
            """Slow."""
            await asyncio.sleep(5)
            return "late"

        lookups = build_tool_lookups([slow])
        result = await _execute_tool(ToolCall(name="slow"), lookups)
        assert json.loads(result.payload)["code"] == "timeout_outcome_unknown"
        assert "zombie_threads" not in result.meta


class TestCancellation:
    async def test_no_thread_leak_after_cancel(self, small_pool) -> None:
        small_pool(3)
        release = threading.Event()
        started: list[int] = []
        lock = threading.Lock()

        @tool(timeout_seconds=30)
        def block(i: int) -> int:
            """Block."""
            with lock:
                started.append(i)
            release.wait(5)
            return i

        agent = Agent(prompt="p", tools=[block], tool_threads=2)
        limiter = agent._tool_call_limiter
        baseline = threading.active_count()
        for round_ in range(20):
            started.clear()
            task = asyncio.create_task(_run(agent, _batch("block", 6)))
            await _until(lambda: len(started) == 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            stats = tool_pool_stats()
            # Only the two started calls are left running, as zombies.
            assert stats.zombies == 2, round_
            assert stats.queued == 0
            release.set()
            await _until(lambda: tool_pool_stats().zombies == 0)
            await _until(lambda: limiter.in_use == 0)
            release.clear()
            assert len(started) == 2  # queued siblings never ran
        assert tool_pool_stats().threads <= 3
        assert threading.active_count() <= baseline + 3

    async def test_shutdown_cancels_queued_calls(self, small_pool) -> None:
        pool = small_pool(1)
        release = threading.Event()
        item = pool.submit(release.wait, 5)
        await _until(lambda: pool.stats().busy == 1)
        queued = pool.submit(lambda: "never")
        pool.shutdown()
        assert queued.future.cancelled()
        release.set()
        assert item.future.result(timeout=5) is True
        with pytest.raises(RuntimeError, match="shut down"):
            pool.submit(lambda: None)


class TestLimiter:
    def test_limiter_is_usable_from_several_event_loops(self) -> None:
        limiter = ToolCallLimiter(1)

        async def hold(seconds: float) -> None:
            await limiter.acquire()
            try:
                await asyncio.sleep(seconds)
            finally:
                limiter.release()

        asyncio.run(hold(0.01))
        errors: list[BaseException] = []

        def other_loop() -> None:
            try:
                asyncio.run(hold(0.05))
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=other_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert errors == []
        assert limiter.in_use == 0

    async def test_cancelled_waiters_never_leak_slots(self) -> None:
        limiter = ToolCallLimiter(1)
        await limiter.acquire()
        waiters = [asyncio.create_task(limiter.acquire()) for _ in range(5)]
        await asyncio.sleep(0)
        for w in waiters[:3]:
            w.cancel()
        limiter.release()
        await asyncio.sleep(0.01)
        granted = [w for w in waiters[3:] if w.done() and not w.cancelled()]
        assert len(granted) == 1
        limiter.release()
        await asyncio.sleep(0.01)
        limiter.release()
        assert limiter.in_use == 0

    async def test_saturated_pool_rejects_without_running(
        self, small_pool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small_pool(1)
        monkeypatch.setattr(pool_mod, "MAX_QUEUED_CALLS", 1)
        release = threading.Event()

        @tool(timeout_seconds=5)
        def hold(i: int) -> str:
            """Hold."""
            release.wait(5)
            return "ok"

        lookups = build_tool_lookups([hold])
        first = asyncio.create_task(_execute_tool(ToolCall(name="hold", params={"i": 1}), lookups))
        await _until(lambda: tool_pool_stats().busy == 1)
        second = asyncio.create_task(_execute_tool(ToolCall(name="hold", params={"i": 2}), lookups))
        await _until(lambda: tool_pool_stats().queued == 1)
        third = await _execute_tool(ToolCall(name="hold", params={"i": 3}), lookups)
        assert json.loads(third.payload)["code"] == "tool_pool_saturated"
        release.set()
        assert (await first).success and (await second).success


class TestConfiguration:
    def test_env_var_sets_the_pool_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pool_mod, "_pool", None)
        monkeypatch.setenv("DENDRUX_TOOL_THREADS", "5")
        assert tool_pool_stats().max_threads == 5
        assert pool_mod.get_pool().max_threads == 5
        pool_mod.get_pool().shutdown()

    @pytest.mark.parametrize("value", ["0", "-1", "many"])
    def test_invalid_env_var_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setattr(pool_mod, "_pool", None)
        monkeypatch.setenv("DENDRUX_TOOL_THREADS", value)
        with pytest.raises(ValueError, match="DENDRUX_TOOL_THREADS"):
            pool_mod.get_pool()

    def test_default_size_and_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pool_mod, "_pool", None)
        monkeypatch.delenv("DENDRUX_TOOL_THREADS", raising=False)
        assert tool_pool_stats().max_threads == 32
        agent = Agent(prompt="p")
        assert isinstance(agent._tool_call_limiter, ToolCallLimiter)
        assert agent._tool_call_limiter.slots == 8
