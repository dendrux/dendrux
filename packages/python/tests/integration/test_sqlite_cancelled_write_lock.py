"""A cancelled SQLite writer must release its lock while its traceback is retained."""

from __future__ import annotations

import asyncio
import gc
import threading
import traceback
import warnings
from pathlib import Path

import aiosqlite
import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import SAWarning
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
async def test_cancelled_store_write_releases_lock_with_retained_traceback(
    engine, store, monkeypatch
) -> None:
    await _cancel_a_write_then_write_elsewhere(engine, store, monkeypatch)


@pytest.mark.parametrize(
    "database",
    [
        "{disk}?mode=memory",  # no uri=true: SQLAlchemy opens the plain path and drops the query
        "{disk}?mode=memory&uri=false",
        # SQLAlchemy decodes %25 once; SQLite cuts the option name at the NUL and applies
        # mode=rwc after mode=memory, so it opens the disk file.
        "file:{disk}?mode=memory&mode%2500x=rwc&uri=true",
    ],
)
async def test_disk_url_that_mentions_memory_keeps_the_guard(
    tmp_path, monkeypatch, database
) -> None:
    disk = tmp_path / "on-disk.db"
    engine = create_async_engine("sqlite+aiosqlite:///" + database.format(disk=disk))
    try:
        async with engine.begin() as connection:
            main = (await connection.exec_driver_sql("PRAGMA database_list")).one()
            assert Path(main[2]).resolve() == disk.resolve(), "the case must open the disk file"
            await connection.run_sync(Base.metadata.create_all)
        await _cancel_a_write_then_write_elsewhere(engine, store_for_engine(engine), monkeypatch)
    finally:
        await engine.dispose()


async def _cancel_a_write_then_write_elsewhere(engine, store, monkeypatch) -> None:
    """Cancel a store write mid-transaction, keep its traceback, then write from another engine."""
    await store.create_run("parent", "agent")
    stepped = asyncio.Event()
    never = asyncio.Event()
    fetchall = aiosqlite.Cursor.fetchall

    async def pause_stepped_read(cursor):
        # Finalization updates the run, then reads the next event sequence.
        # execute() has stepped SELECT to SQLITE_ROW; fetchall has not finalized it.
        if cursor.connection.in_transaction:
            stepped.set()
            await never.wait()
        return await fetchall(cursor)

    monkeypatch.setattr(aiosqlite.Cursor, "fetchall", pause_stepped_read)
    writer = asyncio.create_task(
        store.finalize_with_event(
            "parent",
            status="cancelled",
            allowed_current_statuses=["running"],
            event_type="run_cancelled",
        )
    )
    retained = None
    probe = create_async_engine(engine.url, poolclass=NullPool, connect_args={"timeout": 0.05})
    # A URI database names its disk file after the file: scheme.
    journal = Path(str(engine.url.database).removeprefix("file:") + "-journal")
    try:
        await asyncio.wait_for(stepped.wait(), timeout=2)
        assert journal.exists(), "the cancelled operation must already hold a write transaction"
        writer.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(writer, timeout=2)
        retained = caught.value
        assert retained.__traceback__ is not None
        async with asyncio.timeout(1), probe.connect() as connection:
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            await connection.exec_driver_sql(
                "UPDATE agent_runs SET model = ? WHERE id = ?", ("after-cancel", "parent")
            )
            await connection.commit()
            assert (
                await connection.exec_driver_sql(
                    "SELECT status FROM agent_runs WHERE id = ?", ("parent",)
                )
            ).scalar_one() == "running"
        assert not journal.exists()
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        await probe.dispose()
        # Release traceback-owned cursors only after checking the independent writer.
        if retained is not None:
            traceback.clear_frames(retained.__traceback__)


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
async def test_force_termination_finalizes_a_retained_stream_cursor(engine, store) -> None:
    await store.create_run("parent", "agent")
    # Two rows: with one, the first step already reaches SQLITE_DONE and resets the statement.
    await store.create_run("sibling", "agent")
    probe = create_async_engine(engine.url, poolclass=NullPool, connect_args={"timeout": 0.05})
    journal = Path(str(engine.url.database) + "-journal")
    async with engine.connect() as connection:
        await connection.exec_driver_sql("UPDATE agent_runs SET model = 'uncommitted'")
        retained = await connection.stream(text("SELECT id FROM agent_runs"))
        assert journal.exists()
        # Outside the adaptation greenlet SQLAlchemy must use force termination.
        connection.sync_connection.invalidate()
        try:
            async with asyncio.timeout(1), probe.connect() as other:
                await other.exec_driver_sql("BEGIN IMMEDIATE")
                await other.exec_driver_sql("UPDATE agent_runs SET model = 'committed'")
                await other.commit()
            assert not journal.exists()
            assert retained is not None
        finally:
            await probe.dispose()


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
async def test_busy_worker_does_not_extend_double_cancelled_store_write(
    engine, store, monkeypatch
) -> None:
    await store.create_run("parent", "agent")
    stepped, busy = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    fetchall = aiosqlite.Cursor.fetchall
    blocking = None

    def hold_worker():
        loop.call_soon_threadsafe(busy.set)
        assert release.wait(timeout=5), "test did not release its worker"

    async def pause_stepped_read(cursor):
        nonlocal blocking
        if cursor.connection.in_transaction:
            # cursor.connection is the sqlite3 connection; its aiosqlite proxy owns the worker.
            blocking = asyncio.create_task(cursor._conn._execute(hold_worker))
            await busy.wait()
            stepped.set()
            await asyncio.Event().wait()
        return await fetchall(cursor)

    monkeypatch.setattr(aiosqlite.Cursor, "fetchall", pause_stepped_read)
    close_entered = asyncio.Event()
    original_close = aiosqlite.Connection.close

    async def observed_close(driver):
        close_entered.set()
        await original_close(driver)

    monkeypatch.setattr(aiosqlite.Connection, "close", observed_close)
    writer = asyncio.create_task(
        store.finalize_with_event(
            "parent",
            status="cancelled",
            allowed_current_statuses=["running"],
            event_type="run_cancelled",
        )
    )
    probe = create_async_engine(engine.url, poolclass=NullPool, connect_args={"timeout": 0.1})
    retained = None
    try:
        await asyncio.wait_for(stepped.wait(), timeout=2)
        writer.cancel()
        await asyncio.wait_for(close_entered.wait(), timeout=2)
        writer.cancel()
        with pytest.raises(asyncio.CancelledError) as retained:
            await asyncio.wait_for(writer, timeout=1)
        assert not release.is_set(), "the cancelled caller must finish before its worker"
        release.set()
        await asyncio.wait_for(blocking, timeout=2)
        async with asyncio.timeout(1), probe.connect() as other:
            await other.exec_driver_sql("BEGIN IMMEDIATE")
            await other.exec_driver_sql("UPDATE agent_runs SET model = 'committed'")
            await other.commit()
        assert not Path(str(engine.url.database) + "-journal").exists()
        assert retained.value.__traceback__ is not None
    finally:
        release.set()
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        if blocking is not None:
            await asyncio.gather(blocking, return_exceptions=True)
        await probe.dispose()
        # The cancelled terminate leaves SQLAlchemy's record to the garbage collector;
        # collect it here so its SAWarning is not reported against a later test.
        retained = writer = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The garbage collector", SAWarning)
            gc.collect()


@pytest.mark.skipif(
    not hasattr(aiosqlite.Connection, "stop"),
    reason="before aiosqlite 0.22.1 SQLAlchemy drops a collected connection without closing it",
)
@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
async def test_garbage_collected_connection_finalizes_a_retained_cursor(engine, store) -> None:
    await store.create_run("parent", "agent")
    await store.create_run("sibling", "agent")
    probe = create_async_engine(engine.url, poolclass=NullPool, connect_args={"timeout": 0.05})
    journal = Path(str(engine.url.database) + "-journal")
    cursors = []

    def keep_cursor(_conn, cursor, *_args) -> None:
        cursors.append(cursor)

    event.listen(engine.sync_engine, "after_cursor_execute", keep_cursor)
    connection = await engine.connect()
    await connection.exec_driver_sql("UPDATE agent_runs SET model = 'uncommitted'")
    result = await connection.stream(text("SELECT id FROM agent_runs"))
    event.remove(engine.sync_engine, "after_cursor_execute", keep_cursor)
    assert journal.exists()
    # Keep only the DBAPI cursors, as a retained traceback does; the connection is never
    # returned, so the garbage collector makes SQLAlchemy force-terminate it.
    del result, connection
    with pytest.warns(SAWarning, match="garbage collector"):
        gc.collect()
    try:
        async with asyncio.timeout(1), probe.connect() as other:
            await other.exec_driver_sql("BEGIN IMMEDIATE")
            await other.exec_driver_sql("UPDATE agent_runs SET model = 'committed'")
            await other.commit()
        assert not journal.exists()
        assert len(cursors) == 2
    finally:
        await probe.dispose()
