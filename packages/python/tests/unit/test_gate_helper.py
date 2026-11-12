"""Contracts for the shared test readiness gate."""

import asyncio

import pytest

from tests._helpers.gates import GATE_TIMEOUT, wait_for_gate


def test_default_timeout():
    assert GATE_TIMEOUT == 30
    assert wait_for_gate.__kwdefaults__["timeout"] == GATE_TIMEOUT


@pytest.mark.parametrize("timeout", [0, 1])
async def test_already_reached_gate(timeout):
    event = asyncio.Event()
    event.set()
    await wait_for_gate(event, timeout=timeout)


@pytest.mark.parametrize("with_task", [False, True])
async def test_waits_until_gate_is_reached(with_task):
    event = asyncio.Event()
    release = asyncio.Event()

    async def run():
        event.set()
        await release.wait()

    task = asyncio.create_task(run())
    try:
        await wait_for_gate(event, task=task if with_task else None, timeout=1)
        assert event.is_set()
        assert not task.done()
    finally:
        release.set()
        await task


@pytest.mark.parametrize("already_done", [False, True])
async def test_run_exception_is_reraised(already_done):
    error = ValueError("original run failure")

    async def run():
        raise error

    task = asyncio.create_task(run())
    if already_done:
        await asyncio.gather(task, return_exceptions=True)
    try:
        with pytest.raises(ValueError) as caught:
            await wait_for_gate(asyncio.Event(), task=task, timeout=1)
        assert caught.value is error
    finally:
        await asyncio.gather(task, return_exceptions=True)


async def test_run_success_before_gate_fails_clearly():
    async def run():
        return "finished"

    task = asyncio.create_task(run())
    try:
        with pytest.raises(AssertionError, match="run ended before gate"):
            await wait_for_gate(asyncio.Event(), task=task, timeout=1)
    finally:
        await task


async def test_cancelled_run_propagates_cancellation():
    task = asyncio.create_task(asyncio.Event().wait())
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    with pytest.raises(asyncio.CancelledError):
        await wait_for_gate(asyncio.Event(), task=task, timeout=1)


@pytest.mark.parametrize("with_task", [False, True])
async def test_timeout_fails_clearly_and_preserves_run(with_task):
    task = asyncio.create_task(asyncio.Event().wait())
    before = asyncio.all_tasks()
    try:
        with pytest.raises(AssertionError, match="timed out waiting for gate after 0"):
            await wait_for_gate(asyncio.Event(), task=task if with_task else None, timeout=0)
        assert not task.done()
        assert asyncio.all_tasks() == before
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_caller_cancellation_drains_waiter_and_preserves_run():
    event = asyncio.Event()
    waiting = asyncio.Event()
    original_wait = event.wait

    async def observed_wait():
        waiting.set()
        return await original_wait()

    event.wait = observed_wait
    task = asyncio.create_task(asyncio.Event().wait())
    before = asyncio.all_tasks()
    caller = asyncio.create_task(wait_for_gate(event, task=task))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not task.done()
        assert asyncio.all_tasks() == before
    finally:
        caller.cancel()
        task.cancel()
        await asyncio.gather(caller, task, return_exceptions=True)


async def test_reached_gate_wins_when_run_also_finishes():
    event = asyncio.Event()

    async def run():
        event.set()

    task = asyncio.create_task(run())
    await task
    await wait_for_gate(event, task=task, timeout=0)


async def test_gate_set_then_cleared_still_releases_its_waiter():
    event = asyncio.Event()
    waiting = asyncio.Event()
    original_wait = event.wait

    async def observed_wait():
        waiting.set()
        return await original_wait()

    event.wait = observed_wait

    async def pulse():
        await waiting.wait()
        event.set()
        event.clear()

    task = asyncio.create_task(pulse())
    try:
        await wait_for_gate(event, task=task, timeout=1)
        assert not event.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
