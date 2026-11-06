"""End to end on Postgres with the store-side announcing of step A2 (needs A2 merged).

Writers are ``PostgresStateStore(engine, live_updates=True)``; readers are
``RunStore(store)`` on their own store, so hints travel through real NOTIFY
and the coordinator reads through the store's batched queries.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.store import (
    EventFeed,
    FollowEnd,
    PostgresStateStore,
    PostgresUpdates,
    RunStore,
    StoredEvent,
)
from dendrux.types import generate_ulid
from tests.integration.test_follow_coordinator_pg import _spawn

if TYPE_CHECKING:
    from pathlib import Path


async def test_native_store_hint_reaches_a_follower_long_before_the_safety_poll(
    pg_test_url: str, engine: Any, store: Any
) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres LISTEN/NOTIFY")
    writer_engine, reader_engine = (
        create_async_engine(pg_test_url),
        create_async_engine(pg_test_url),
    )
    writer = PostgresStateStore(writer_engine, live_updates=True)
    reader_store = PostgresStateStore(reader_engine, live_updates=True)
    assert isinstance(writer.live_updates, PostgresUpdates)
    batched = reader_store.get_events_for_runs
    calls: list[int] = []

    async def counted(heads: Any, *, limit_per_run: int) -> Any:
        calls.append(len(heads))
        return await batched(heads, limit_per_run=limit_per_run)

    reader_store.get_events_for_runs = counted  # type: ignore[method-assign]
    reader = RunStore(reader_store)
    run = generate_ulid()
    await writer.create_run(run, "native")
    feed = EventFeed(reader, poll_healthy_s=120)  # only a hint can deliver in time
    try:
        async with feed.follow(run, tenant_id=None) as follow:
            live = reader_store.live_updates
            assert live is not None
            async with asyncio.timeout(10):
                while not live.healthy:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.6)  # past the join and gap reads
            await writer.save_run_event(run, event_type="tick")
            async with asyncio.timeout(10):
                item = await _next_event(follow)
            assert item.sequence_index == 0
            await writer.finalize_with_event(
                run,
                status="success",
                allowed_current_statuses=["running"],
                event_type="run.completed",
            )
            async with asyncio.timeout(10):
                rest = [i async for i in follow]
        assert [i.sequence_index for i in rest if isinstance(i, StoredEvent)] == [1]
        assert rest[-1] == FollowEnd("terminal", 1)
        assert calls  # the coordinator used the batched read
    finally:
        await reader.close()
        await reader_store.close()
        await writer.close()
        await writer_engine.dispose()
        await reader_engine.dispose()


async def _next_event(follow: Any) -> StoredEvent:
    while True:
        item = await anext(follow)
        if isinstance(item, StoredEvent):
            return item


READER = r"""
import asyncio, json, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux.store import FollowEnd, PostgresUpdates, RunStore, StoredEvent, store_for_engine


async def main(url, channel, run_id):
    engine = create_async_engine(url)
    store = store_for_engine(engine, live_updates=PostgresUpdates(url=url, channel=channel))
    reader = RunStore(store)

    async def one():
        seen = []
        async with reader.follow(run_id, tenant_id=None, max_duration_s=60) as follow:
            async for item in follow:
                if isinstance(item, StoredEvent):
                    seen.append(item.sequence_index)
                elif isinstance(item, FollowEnd):
                    return seen, item.reason

    tasks = [asyncio.create_task(one()) for _ in range(2)]
    while not store.live_updates.healthy:
        await asyncio.sleep(0.01)
    print("ready", flush=True)
    print(json.dumps(await asyncio.gather(*tasks)), flush=True)
    await reader.close()
    await store.close()
    await engine.dispose()


asyncio.run(main(*sys.argv[1:]))
"""

WRITER = r"""
import asyncio, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux.store import PostgresUpdates, store_for_engine


async def main(url, channel, run_id, events, finalize):
    engine = create_async_engine(url)
    store = store_for_engine(engine, live_updates=PostgresUpdates(url=url, channel=channel))
    for _ in range(int(events)):
        await store.save_run_event(run_id, event_type="tick")  # the store announces
    if finalize == "1":
        await store.finalize_with_event(
            run_id,
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
        )
    await store.close()
    await engine.dispose()


asyncio.run(main(*sys.argv[1:]))
"""


async def test_two_native_writers_two_native_readers_no_lost_terminal_no_duplicate(
    pg_test_url: str, engine: Any, store: Any, tmp_path: Path
) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres LISTEN/NOTIFY across processes")
    run_id, channel = generate_ulid(), "p6b_" + generate_ulid().lower()
    await store.create_run(run_id, "native-two-by-two")
    reader_script, writer_script = tmp_path / "reader.py", tmp_path / "writer.py"
    reader_script.write_text(READER)
    writer_script.write_text(WRITER)
    readers = [await _spawn(reader_script, pg_test_url, channel, run_id) for _ in range(2)]
    try:
        async with asyncio.timeout(60):
            for reader in readers:
                assert reader.stdout is not None
                line = await reader.stdout.readline()
                assert line.strip() == b"ready", (await reader.stderr.read()).decode()[-2000:]
            writers = [
                await _spawn(writer_script, pg_test_url, channel, run_id, "40", "0")
                for _ in range(2)
            ]
            for proc in writers:
                _, err = await proc.communicate()
                assert proc.returncode == 0, err.decode()[-2000:]
            final = await _spawn(writer_script, pg_test_url, channel, run_id, "0", "1")
            _, err = await final.communicate()
            assert final.returncode == 0, err.decode()[-2000:]
            outputs = [await reader.communicate() for reader in readers]
    finally:
        for reader in readers:
            if reader.returncode is None:
                reader.kill()
    for (out, err), reader in zip(outputs, readers, strict=True):
        assert reader.returncode == 0, err.decode()[-2000:]
        for seen, reason in json.loads(out.decode().strip().splitlines()[-1]):
            assert seen == list(range(81)) and reason == "terminal"
