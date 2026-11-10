"""Deferred ownership operations inherit and restore their caller's context."""

import asyncio

import pytest

from dendrux.runtime import durability


@pytest.mark.parametrize("task_store", [False, True])
async def test_factory_and_operation_share_marker_without_leaking(task_store):
    outer = {"outer": True}
    token = durability._handoff_commit.set(outer)
    markers = []
    cancelled = []

    async def operation():
        markers.append(durability._handoff_commit.get())
        durability._mark_commit_issued()
        return 7

    def factory():
        markers.append(durability._handoff_commit.get())
        return asyncio.create_task(operation()) if task_store else operation()

    async def cleanup(result):
        cancelled.append(result)

    try:
        assert await durability._ownership_handoff(None, factory, cleanup, run_id="run") == 7
        assert markers[0] is markers[1]
        assert markers[0] == {"issued": True}
        assert durability._handoff_commit.get() is outer
        assert outer == {"outer": True}
        assert cancelled == []
    finally:
        durability._handoff_commit.reset(token)


async def test_factory_failure_restores_marker():
    outer = {"outer": True}
    token = durability._handoff_commit.set(outer)
    error = ValueError("store construction failed")

    def factory():
        assert durability._handoff_commit.get() is not outer
        raise error

    async def cleanup(result):
        pytest.fail("No ownership was acquired")

    try:
        with pytest.raises(ValueError) as caught:
            await durability._ownership_handoff(None, factory, cleanup, run_id="run")
        assert caught.value is error
        assert durability._handoff_commit.get() is outer
    finally:
        durability._handoff_commit.reset(token)
