"""Continuation read failures on the native MySQL-family CI matrix."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._engine_options import engine_options
from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine
from tests.integration import test_cancel_refresh_signaling as cancellation
from tests.integration import test_resume_progress_boundaries as boundaries
from tests.integration import test_resume_read_failures as contracts
from tests.integration import test_resume_release_outcomes as outcomes


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


@pytest.mark.parametrize("entry", outcomes.ENTRIES)
@pytest.mark.parametrize("unsupported", ["method", "epoch"])
async def test_unavailable_release(read_store, monkeypatch, entry, unsupported):
    await outcomes.test_unavailable_release_handles_setup_failure(
        read_store, read_store._engine, monkeypatch, entry, unsupported
    )


@pytest.mark.parametrize("entry", contracts.ENTRIES)
async def test_unavailable_release_fencing(read_store, monkeypatch, entry):
    await outcomes.test_unavailable_release_cannot_finalize_new_claim(
        read_store, read_store._engine, monkeypatch, entry
    )


@pytest.mark.parametrize("entry", outcomes.ENTRIES[:6])
@pytest.mark.parametrize("unsupported", ["method", "epoch"])
async def test_submit_governance_cleanup(read_store, monkeypatch, entry, unsupported):
    await outcomes.test_submit_governance_without_release_finalizes(
        read_store, read_store._engine, monkeypatch, entry, unsupported
    )


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
@pytest.mark.parametrize("release_result", ["success", "failure", "missing", "lost"])
async def test_release_observers(read_store, monkeypatch, entry, release_result):
    await outcomes.test_release_observers_report_outcome(
        read_store, read_store._engine, monkeypatch, entry, release_result
    )


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
async def test_consumed_recovery(read_store, monkeypatch, entry):
    await boundaries.test_recovery_consumption_failure_is_not_released(
        read_store, read_store._engine, monkeypatch, entry
    )


async def test_recovery_dispatch(read_store, monkeypatch):
    await boundaries.test_stream_recovery_marks_work_independently_of_submission(
        read_store, read_store._engine, monkeypatch
    )


async def test_strict_recovery_read(read_store, monkeypatch):
    await outcomes.test_get_result_keeps_recovery_read_strict(
        read_store, read_store._engine, monkeypatch
    )


@pytest.mark.parametrize("failure", ["refresh", "transition", "frontier"])
async def test_cancel_signal_after_read_failure(read_store, monkeypatch, failure):
    await cancellation.test_cancel_response_failure_still_interrupts_tool(
        read_store, monkeypatch, failure
    )


async def test_legacy_cascade_signals(read_store, monkeypatch):
    await cancellation.test_legacy_cascade_signals_completed_children_when_later_child_fails(
        read_store, monkeypatch
    )


@pytest.mark.parametrize("read", ["get_next_trace_index", "get_pause_state"])
async def test_reclaim_without_release(read_store, monkeypatch, read):
    await outcomes.test_reclaim_without_release_can_retry_setup_failure(
        read_store, read_store._engine, monkeypatch, read
    )


@pytest.mark.parametrize("kind", ["int", "view", "claim"])
async def test_unsupported_cleanup_epoch(read_store, monkeypatch, kind):
    await outcomes.test_unsupported_cleanup_keeps_original_epoch(
        read_store, read_store._engine, monkeypatch, kind
    )
