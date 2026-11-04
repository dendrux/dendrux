"""Two writer processes and two reader processes on one run, on Postgres.

Writers announce through ``PostgresUpdates`` after each save (the store-side
announce hooks are step A2; this test drives the same base calls by hand).
Readers follow through ``RunStore`` on a store carrying the live updates. Every
follower must see every event exactly once, in order, then ``FollowEnd``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dendrux.types import generate_ulid

READER = r"""
import asyncio, json, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import FollowEnd, PostgresUpdates, RunStore, StoredEvent


class Announced:
    def __init__(self, inner, live):
        self._inner, self.live_updates = inner, live

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def main(url, channel, run_id):
    engine = create_async_engine(url)
    live = PostgresUpdates(url=url, channel=channel)
    reader = RunStore(Announced(store_for_engine(engine), live))

    async def one():
        seen = []
        async with reader.follow(run_id, tenant_id=None, max_duration_s=60) as follow:
            async for item in follow:
                if isinstance(item, StoredEvent):
                    seen.append(item.sequence_index)
                elif isinstance(item, FollowEnd):
                    return seen, item.reason

    tasks = [asyncio.create_task(one()) for _ in range(2)]
    while not live.healthy:
        await asyncio.sleep(0.01)
    print("ready", flush=True)
    results = await asyncio.gather(*tasks)
    print(json.dumps(results), flush=True)
    await reader.close()
    await live.close()
    await engine.dispose()


asyncio.run(main(*sys.argv[1:]))
"""

WRITER = r"""
import asyncio, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import PostgresUpdates


async def main(url, channel, run_id, events, finalize):
    engine = create_async_engine(url)
    store = store_for_engine(engine)
    live = PostgresUpdates(url=url, channel=channel)
    for _ in range(int(events)):
        await store.save_run_event(run_id, event_type="tick")
        live.enqueue([run_id])
        await live.flush()
    if finalize == "1":
        await store.finalize_with_event(
            run_id,
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
        )
        live.enqueue([run_id])
        await live.flush()
    await live.close()
    await engine.dispose()


asyncio.run(main(*sys.argv[1:]))
"""


async def _spawn(script: Path, *args: str) -> asyncio.subprocess.Process:
    root = Path(__file__).resolve().parents[2]
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(root),
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )


async def test_two_writers_two_readers_no_lost_terminal_no_duplicate(
    pg_test_url: str, engine: Any, store: Any, tmp_path: Path
) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres LISTEN/NOTIFY across processes")
    run_id = generate_ulid()
    channel = "p6b_" + generate_ulid().lower()
    await store.create_run(run_id, "two-by-two")
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
                await _spawn(writer_script, pg_test_url, channel, run_id, "40", "0"),
                await _spawn(writer_script, pg_test_url, channel, run_id, "40", "0"),
            ]
            for writer in writers:
                _, err = await writer.communicate()
                assert writer.returncode == 0, err.decode()[-2000:]
            final = await _spawn(writer_script, pg_test_url, channel, run_id, "0", "1")
            _, err = await final.communicate()
            assert final.returncode == 0, err.decode()[-2000:]
            outputs = [await reader.communicate() for reader in readers]
    finally:
        for reader in readers:
            if reader.returncode is None:
                reader.kill()
    expected = list(range(81))  # 80 ticks and the terminal event
    for (out, err), reader in zip(outputs, readers, strict=True):
        assert reader.returncode == 0, err.decode()[-2000:]
        followers = json.loads(out.decode().strip().splitlines()[-1])
        for seen, reason in followers:
            assert seen == expected  # every event once, in order: none lost, none twice
            assert reason == "terminal"
