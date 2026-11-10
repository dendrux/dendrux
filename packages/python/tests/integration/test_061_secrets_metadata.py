"""Metadata readers serialize with independent blocked-provenance writers."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux._meta_keys import ERROR_INFO_META_KEY, LATEST_REWRITE_META_KEY, PAUSE_REQUEST_META_KEY
from dendrux.runtime._blocked_content import _blocked_write
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.stores.sql._types import SetFields
from dendrux.types import generate_ulid
from tests._fixtures.storage import pin_storage_format

KEY = "dendrux.blocked"
SITES = ["rewrite", "finalize", "finalize_event", "pause_request"]


def _metadata_read(statement):
    if not getattr(statement, "is_select", False):
        return False
    columns = list(statement.selected_columns)
    return len(columns) == 1 and columns[0].name == "meta" and columns[0].table.name == "agent_runs"


async def _write_metadata(store, run_id, site):
    info = {"code": "safe_error"}
    if site == "rewrite":
        await store.save_trace(
            run_id, role="system", content="[]", order_index=0, trace_kind="rewrite"
        )
    elif site == "finalize":
        assert await store.finalize_run(run_id, status="error", error_info=info)
    elif site == "finalize_event":
        assert (
            await store.finalize_with_event(
                run_id,
                status="error",
                allowed_current_statuses=["running"],
                event_type="run.error",
                event_data={"error_info": info},
            )
            is not None
        )
    else:
        assert await store.request_pause(run_id, reason="safe reason")


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("mark_first", [False, True])
async def test_metadata_read_serializes_with_block_mark(
    store, engine, monkeypatch, fmt, site, mark_first
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent Postgres transactions and row-lock observation")
    await pin_storage_format(engine, fmt)
    run_id = generate_ulid()
    await store.create_run(run_id, "metadata", meta={"keep": "unchanged"})
    if site == "rewrite":
        original_patch = store.json_patch_expr

        def fallback(column, patch):
            if isinstance(patch, SetFields) and LATEST_REWRITE_META_KEY in patch.fields:
                return None
            return original_patch(column, patch)

        monkeypatch.setattr(store, "json_patch_expr", fallback)

    app_name = "r0h-mark-" + run_id
    mark_engine = create_async_engine(
        engine.url, connect_args={"server_settings": {"application_name": app_name}}
    )
    mark_store = store_for_engine(mark_engine)
    read = asyncio.Event()
    release = asyncio.Event()
    writer = None
    marker = None
    waiter = None
    original_execute, original_scalar = AsyncSession.execute, AsyncSession.scalar

    async def after_read(statement):
        if asyncio.current_task() is writer and _metadata_read(statement):
            read.set()
            await release.wait()

    async def execute(session, statement, *args, **kwargs):
        result = await original_execute(session, statement, *args, **kwargs)
        await after_read(statement)
        return result

    async def scalar(session, statement, *args, **kwargs):
        result = await original_scalar(session, statement, *args, **kwargs)
        await after_read(statement)
        return result

    async def mark():
        with _blocked_write(mark_store, run_id):
            await mark_store.save_run_event(
                run_id, event_type="guardrail.blocked", data={"count": 1}
            )

    monkeypatch.setattr(AsyncSession, "execute", execute)
    monkeypatch.setattr(AsyncSession, "scalar", scalar)
    try:
        if mark_first:
            await mark()
        writer = asyncio.create_task(_write_metadata(store, run_id, site))
        waiter = asyncio.create_task(read.wait())
        done, _ = await asyncio.wait(
            {writer, waiter}, timeout=15, return_when=asyncio.FIRST_COMPLETED
        )
        if writer in done:
            await writer
        assert waiter in done, "Expected the metadata read before its whole-document write"
        if not mark_first:
            marker = asyncio.create_task(mark())
            async with asyncio.timeout(15):
                while True:
                    if marker.done():
                        await marker
                        pytest.fail("Mark committed between the metadata read and write")
                    async with engine.connect() as observer:
                        waiting = await observer.scalar(
                            text(
                                "SELECT count(*) FROM pg_stat_activity "
                                "WHERE datname = current_database() "
                                "AND application_name = :application "
                                "AND wait_event_type = 'Lock' "
                                "AND cardinality(pg_blocking_pids(pid)) > 0"
                            ),
                            {"application": app_name},
                        )
                    if waiting:
                        break
                    await asyncio.sleep(0.01)
            assert not marker.done()
        release.set()
        await asyncio.wait_for(writer, 15)
        if marker is not None:
            await asyncio.wait_for(marker, 15)
        row = await store.get_run(run_id)
        assert row.meta[KEY] is True
        assert row.meta["keep"] == "unchanged"
        if site == "rewrite":
            assert row.meta[LATEST_REWRITE_META_KEY]["order_index"] == 0
        elif site == "pause_request":
            assert row.meta[PAUSE_REQUEST_META_KEY] == {"reason": "safe reason"}
        else:
            assert row.meta[ERROR_INFO_META_KEY] == {"code": "safe_error"}
    finally:
        release.set()
        tasks = [task for task in (writer, marker, waiter) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await mark_store.close()
        await mark_engine.dispose()
