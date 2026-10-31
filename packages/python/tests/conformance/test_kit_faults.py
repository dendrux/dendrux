"""The repo's own fault injector does what the kit's contracts assume it does."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.runtime.state import EventWrite, TraceWrite
from dendrux.testing.store_kit.sql import SqlFaultInjector
from dendrux.types import generate_ulid
from tests.conformance._kit_backends import KitBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[tuple[Any, SqlFaultInjector]]:
    backend = KitBackend("sqlite-file", tmp_path)
    store = await backend.fresh_store()
    injector = SqlFaultInjector()
    try:
        yield store, injector
    finally:
        injector.close()
        await backend.close()


async def _run(store: Any) -> str:
    run_id = generate_ulid()
    await store.create_run(run_id, "Faults")
    return run_id


async def test_before_commit_persists_nothing(world: tuple[Any, SqlFaultInjector]) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "before_commit")
    with pytest.raises(Exception):  # noqa: B017 - any failure, the point is what was stored
        await store.save_run_event(run_id, event_type="x", event_id=generate_ulid())
    assert await store.get_run_events(run_id) == []


async def test_after_commit_persists_and_still_raises(world: tuple[Any, SqlFaultInjector]) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "after_commit")
    with pytest.raises(Exception):  # noqa: B017
        await store.save_run_event(run_id, event_type="x", event_id=generate_ulid())
    assert len(await store.get_run_events(run_id)) == 1


async def test_a_fault_fires_once(world: tuple[Any, SqlFaultInjector]) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "after_commit")
    with pytest.raises(Exception):  # noqa: B017
        await store.save_run_event(run_id, event_type="x", event_id=generate_ulid())
    await store.save_run_event(run_id, event_type="y", event_id=generate_ulid())
    assert len(await store.get_run_events(run_id)) == 2


async def test_cancel_kind_raises_cancelled_error(world: tuple[Any, SqlFaultInjector]) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "before_commit", kind="cancel")
    with pytest.raises(asyncio.CancelledError):
        await store.save_run_event(run_id, event_type="x")
    assert await store.get_run_events(run_id) == []


async def test_mid_write_cuts_a_transaction_with_several_writes(
    world: tuple[Any, SqlFaultInjector],
) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "mid_write", kind="cancel")
    with pytest.raises(asyncio.CancelledError):
        await store.finalize_with_event(
            run_id,
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
            answer="a",
            claim_epoch=0,
            trace=TraceWrite(generate_ulid(), "assistant", "a", 0),
            event=EventWrite(generate_ulid(), "run.completed", 0, {}, None, None),
        )
    record = await store.get_run(run_id)
    assert record.status == "running"
    assert await store.get_traces(run_id) == [] and await store.get_run_events(run_id) == []


async def test_after_lets_earlier_write_transactions_through(
    world: tuple[Any, SqlFaultInjector],
) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "before_commit", kind="cancel", after=1)
    await store.save_run_event(run_id, event_type="first")
    with pytest.raises(asyncio.CancelledError):
        await store.save_run_event(run_id, event_type="second")
    assert [e.event_type for e in await store.get_run_events(run_id)] == ["first"]


async def test_disarm_removes_the_fault(world: tuple[Any, SqlFaultInjector]) -> None:
    store, injector = world
    run_id = await _run(store)
    injector.arm(store, "before_commit")
    injector.disarm(store)
    await store.save_run_event(run_id, event_type="x")
    assert len(await store.get_run_events(run_id)) == 1
