"""Call-scoped lease renewal never outlives the provider or tool call."""

import asyncio
from datetime import timedelta

import pytest

from dendrux import Agent
from dendrux.errors import PersistenceError, RunFencedError
from dendrux.llm.mock import MockLLM
from dendrux.runtime import context
from dendrux.runtime.state import StoreCapabilities
from tests._helpers.gates import wait_for_gate


@pytest.mark.parametrize("seconds", [0, 4.999])
def test_lease_interval_floor(seconds):
    import inspect

    assert "lease_interval" in inspect.signature(Agent).parameters, (
        "Agent needs a bounded renewal interval"
    )
    with pytest.raises(ValueError):
        Agent(prompt="p", provider=MockLLM([]), lease_interval=timedelta(seconds=seconds))


async def test_lease_tick_is_call_scoped_and_only_one_write_outstanding():
    assert hasattr(context, "lease_scope"), "in-flight calls must renew their lease"

    changed = asyncio.Event()

    class Store:
        capabilities = StoreCapabilities(fenced_writes=True)

        def __init__(self):
            self.count = self.active = self.maximum = 0

        async def touch_progress(self, run_id):
            self.active += 1
            self.maximum = max(self.active, self.maximum)
            self.count += 1
            try:
                await asyncio.sleep(0)
            finally:
                self.active -= 1
                changed.set()

    store = Store()
    ctx = context.DelegationContext(
        "00000000000000000000000199", persisted=True, state_store=store, lease_interval_s=0.02
    )
    token = context.set_delegation_context(ctx)

    tickers = set()

    async def call():
        async with context.lease_scope():
            tickers.add(ctx._lease_ticker[0])
            async with asyncio.timeout(10):
                while store.count < 3:
                    await changed.wait()
                    changed.clear()

    try:
        await asyncio.gather(call(), call(), call())
        assert ctx._lease_ticker[0] is None
        assert len(tickers) == 1 and all(ticker.done() for ticker in tickers)
        assert store.active == 0 and store.maximum == 1
        assert store.count >= 3
    finally:
        context.reset_delegation_context(token)


async def test_fenced_tick_raises_at_step_exit_without_cancelling_the_caller():
    assert hasattr(context, "lease_scope"), "a lost epoch must stop the in-flight call"

    class Store:
        capabilities = StoreCapabilities(fenced_writes=True)
        count = 0

        async def touch_progress(self, run_id):
            self.count += 1
            if self.count > 1:
                raise RunFencedError(run_id, claim_epoch=0)

    token = context.set_delegation_context(
        context.DelegationContext(
            "00000000000000000000000199", persisted=True, state_store=Store(), lease_interval_s=0.01
        )
    )
    finished_step = False
    try:
        with pytest.raises(RunFencedError):
            async with context.lease_scope():
                async with asyncio.timeout(10):
                    while context.get_delegation_context()._lease_errors[0] is None:
                        await asyncio.sleep(0.001)
                assert asyncio.current_task().cancelling() == 0
                finished_step = True
        assert finished_step
    finally:
        context.reset_delegation_context(token)


async def test_a23_stuck_renewal_has_deadline_and_cancellation_is_prompt():
    assert hasattr(context, "lease_scope"), "renewal writes need their own deadline"

    class Store:
        capabilities = StoreCapabilities(fenced_writes=True)
        active = 0

        async def touch_progress(self, run_id):
            self.active += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

    store = Store()
    token = context.set_delegation_context(
        context.DelegationContext(
            "00000000000000000000000199", persisted=True, state_store=store, lease_interval_s=0.02
        )
    )
    try:
        with pytest.raises(PersistenceError) as caught:
            async with context.lease_scope():
                await asyncio.sleep(0.1)
        assert caught.value.code == "lease_lost"
        assert store.active == 0
    finally:
        context.reset_delegation_context(token)


async def test_a23_external_cancellation_cancels_outstanding_renewal():
    class Store:
        capabilities = StoreCapabilities(fenced_writes=True)

        def __init__(self):
            self.calls = self.active = 0
            self.started = asyncio.Event()

        async def touch_progress(self, run_id):
            self.calls += 1
            if self.calls == 1:
                return
            self.active += 1
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

    store = Store()
    token = context.set_delegation_context(
        context.DelegationContext(
            "00000000000000000000000199", persisted=True, state_store=store, lease_interval_s=0.02
        )
    )

    async def call():
        async with context.lease_scope():
            await asyncio.Event().wait()

    task = asyncio.create_task(call())
    try:
        await wait_for_gate(store.started, task=task)
        before = asyncio.get_running_loop().time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.get_running_loop().time() - before < 0.2
        assert store.active == 0
    finally:
        task.cancel()
        context.reset_delegation_context(token)


async def test_heartbeat_fence_is_retrieved_and_raised_on_next_step():
    class Store:
        capabilities = StoreCapabilities(fenced_writes=True)

        async def touch_progress(self, run_id):
            raise RunFencedError(run_id, claim_epoch=0)

    loop = asyncio.get_running_loop()
    ctx = context.DelegationContext(
        "00000000000000000000000199", persisted=True, state_store=Store(), loop=loop
    )
    token = context.set_delegation_context(ctx)
    errors = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, data: errors.append(data))
    try:
        context.heartbeat()
        async with asyncio.timeout(10):
            while ctx._lease_errors[0] is None:
                await asyncio.sleep(0.001)
        assert isinstance(ctx._lease_errors[0], RunFencedError)
        with pytest.raises(RunFencedError):
            async with context.lease_scope():
                raise AssertionError("fenced step entered")
        assert not errors
    finally:
        loop.set_exception_handler(previous)
        context.reset_delegation_context(token)


async def test_parallel_scopes_share_one_renewal_clock(monkeypatch):
    from types import SimpleNamespace

    ticks = asyncio.Queue()
    renewed = asyncio.Event()
    now = [100.0]

    async def tick_sleep(seconds):
        await ticks.get()
        now[0] += seconds

    monkeypatch.setattr(context, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(
        context, "asyncio", SimpleNamespace(**(vars(asyncio) | {"sleep": tick_sleep}))
    )

    class Store:
        capabilities = StoreCapabilities(fenced_writes=True)
        count = 0

        async def touch_progress(self, run_id):
            self.count += 1
            renewed.set()

    store = Store()
    ctx = context.DelegationContext(
        "00000000000000000000000199", persisted=True, state_store=store, lease_interval_s=5
    )
    token = context.set_delegation_context(ctx)
    entered = 0
    ready, release = asyncio.Event(), asyncio.Event()

    async def call():
        nonlocal entered
        async with context.lease_scope():
            entered += 1
            if entered == 20:
                ready.set()
            await release.wait()

    tasks = [asyncio.create_task(call()) for _ in range(20)]
    try:
        await wait_for_gate(ready)
        assert store.count == 1
        assert ctx._lease_users[0] == 20
        ticker = ctx._lease_ticker[0]
        assert ticker is not None
        # Three controlled intervals, irrespective of thread scheduling or machine speed.
        for expected in range(2, 5):
            renewed.clear()
            ticks.put_nowait(None)
            await wait_for_gate(renewed, task=ticker)
            assert store.count == expected
            assert ctx._lease_ticker[0] is ticker
    finally:
        release.set()
        await asyncio.gather(*tasks)
        context.reset_delegation_context(token)
    assert ctx._lease_ticker[0] is None and ticker.done()
    assert ctx._lease_users[0] == 0
    assert store.count == 4
