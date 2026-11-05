"""Direct tests for how a drained cleanup task resolves."""

import asyncio
import logging

import pytest

from dendrux.runtime.tasks import _drain_cleanup
from dendrux.types import generate_ulid

RUN_ID = generate_ulid()


async def test_drain_reraises_task_failure_when_no_cancel_retained(caplog):
    async def failing():
        raise ValueError("task_failure")

    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError, match="task_failure"):
        await _drain_cleanup(failing(), run_id=RUN_ID)
    assert "persistence_error" not in caplog.text


async def test_drain_retained_cancel_wins_over_task_cancellation(caplog):
    entered, release = asyncio.Event(), asyncio.Event()

    async def self_cancelled():
        entered.set()
        await release.wait()
        raise asyncio.CancelledError

    task = asyncio.create_task(_drain_cleanup(self_cancelled(), run_id=RUN_ID))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with caplog.at_level(logging.DEBUG), pytest.raises(asyncio.CancelledError):
        await task
    assert "persistence_error" not in caplog.text


async def test_drain_returns_result_and_retained_cancel():
    entered, release = asyncio.Event(), asyncio.Event()

    async def fine():
        entered.set()
        await release.wait()
        return 7

    async def outer():
        return await _drain_cleanup(fine(), run_id=RUN_ID)

    task = asyncio.create_task(outer())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    result, cancellation = await task
    assert result == 7
    assert isinstance(cancellation, asyncio.CancelledError)
