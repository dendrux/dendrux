"""Continuation read failures on the native MySQL-family CI matrix."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._engine_options import engine_options
from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine
from tests.integration import test_resume_read_failures as contracts


@pytest.fixture
async def read_store(mysql_url):
    engine = create_async_engine(mysql_url, **engine_options(mysql_url))
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda sync: seed_settings(sync, active=True))
        store = store_for_engine(engine)
        try:
            yield store
        finally:
            await store.close()
    finally:
        await engine.dispose()


@pytest.mark.parametrize("entry", contracts.ENTRIES)
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_setup_release(read_store, monkeypatch, entry, storage_format):
    await contracts.test_setup_read_failure_releases_exact_claim(
        read_store, read_store._engine, monkeypatch, entry, storage_format
    )


@pytest.mark.parametrize("entry", contracts.ENTRIES)
async def test_stale_release(read_store, monkeypatch, entry):
    await contracts.test_stale_setup_release_cannot_change_new_claim(
        read_store, read_store._engine, monkeypatch, entry
    )


@pytest.mark.parametrize("entry", contracts.ENTRIES)
async def test_cancel_during_release(read_store, monkeypatch, entry):
    await contracts.test_cancel_request_wins_setup_release(
        read_store, read_store._engine, monkeypatch, entry
    )


@pytest.mark.parametrize("entry", contracts.ENTRIES)
async def test_unknown_write_outcome(read_store, monkeypatch, entry):
    await contracts.test_unknown_write_outcome_is_not_released(
        read_store, read_store._engine, monkeypatch, entry
    )


async def test_committed_recovery(read_store, monkeypatch, caplog):
    await contracts.test_reclaim_returns_committed_success_when_recovery_read_fails(
        read_store, read_store._engine, monkeypatch, caplog
    )


@pytest.mark.parametrize("entry", ["run", "run_stream", "retry", *contracts.ENTRIES])
@pytest.mark.parametrize("read", ["pending", "pause_refresh"])
async def test_committed_pause(read_store, monkeypatch, caplog, entry, read):
    await contracts.test_committed_pause_survives_projection_failure(
        read_store, read_store._engine, monkeypatch, caplog, entry, read
    )
