"""Cancellation ownership at the public flush and close boundaries."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.unit.test_live_updates_base import Bus


@pytest.fixture(params=["flush", "close"])
def boundary(request):
    def configure(adapter, **limits):
        bus = Bus(**limits)
        if request.param == "flush":
            bus.enqueue(["run"])
            bus.send = adapter
        else:
            bus.aclose_sender = adapter
        return bus, getattr(bus, request.param)

    return configure


async def test_pending_request_precedes_transport_cleanup(boundary):
    async def replace_cancel(*args):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            raise ConnectionError("cleanup") from OSError("socket")

    adapter = AsyncMock(side_effect=replace_cancel)
    bus, operation = boundary(adapter)
    asyncio.current_task().cancel("pending")
    with pytest.raises(asyncio.CancelledError) as caught:
        await operation()
    assert caught.value.args == ("pending",)
    adapter.assert_not_called()
    assert bus.aborts == 1
    assert bus.stats().dropped_failed == (operation == bus.flush)


@pytest.mark.parametrize(
    "cleanup", ["propagate", "explicit_os_cause", "swallow", "deadline", "second_request"]
)
async def test_new_request_during_adapter(boundary, cleanup):
    entered = asyncio.Event()
    delivered = []

    async def adapter(*args):
        try:
            entered.set()
            await asyncio.Future()
        except asyncio.CancelledError as cancel:
            delivered.append(cancel)
            if cleanup == "second_request":
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel, "later")
            if cleanup == "swallow":
                return
            if cleanup == "explicit_os_cause":
                raise ConnectionError("cleanup") from OSError("socket")
            if cleanup == "deadline":
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    raise cancel from None
            raise

    budget = 0.1 if cleanup == "deadline" else 5
    bus, operation = boundary(
        adapter, close_timeout_s=budget, send_timeout_s=budget, flush_deadline_s=budget
    )
    task = asyncio.create_task(operation())
    async with asyncio.timeout(5):
        await entered.wait()
        task.cancel("during adapter")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    if cleanup != "swallow":
        assert caught.value is delivered[0]
        assert caught.value.args == ("during adapter",)
    assert bus.aborts == 1
    assert bus.stats().sent == 0


@pytest.mark.parametrize(
    "outcome,stale",
    [
        ("connection", True),
        ("future", False),
        ("future", True),
        ("adapter_timeout", True),
        ("module_timeout", True),
        ("swallowed_module_timeout", True),
    ],
)
async def test_adapter_failure_uses_ordinary_accounting(boundary, outcome, stale, caplog):
    async def adapter(*args):
        if outcome == "connection":
            raise ConnectionError
        if outcome == "future":
            future = asyncio.get_running_loop().create_future()
            future.cancel("adapter")
            await future
        if outcome == "adapter_timeout":
            async with asyncio.timeout(0):
                await asyncio.Future()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            if outcome != "swallowed_module_timeout":
                raise

    budget = 0.02 if "module_timeout" in outcome else 5
    bus, operation = boundary(
        adapter, close_timeout_s=budget, send_timeout_s=budget, flush_deadline_s=budget
    )
    if stale:
        asyncio.current_task().cancel("handled")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
    await operation()
    assert bus.aborts == 1
    if operation == bus.flush:
        assert bus.stats().dropped_failed == 1
        code = "live_updates_send_failed"
    else:
        assert bus._state == "closed"
        code = "live_updates_close_timeout" if "timeout" in outcome else "live_updates_unavailable"
    assert [record.code for record in caplog.records] == [code]


async def test_request_just_before_adapter_returns_keeps_reason(boundary):
    async def adapter(*args):
        asyncio.current_task().cancel("last moment")

    bus, operation = boundary(adapter)
    with pytest.raises(asyncio.CancelledError) as caught:
        await operation()
    assert caught.value.args == ("last moment",)
    assert bus.aborts == 1
    await asyncio.sleep(0)
