"""The Postgres store's native live updates (needs step A2's ``live_updates=`` hook).

Supersedes A2's temporary ``test_postgres_native_hook_is_lazy_and_typed_until_b``.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._postgres_updates import PostgresUpdates, asyncpg_dsn
from dendrux.runtime.stores._factory import store_for_engine

SECRET = "pw-s3cr3t"


async def test_live_updates_true_binds_postgres_updates_to_the_engine_url() -> None:
    engine = create_async_engine(f"postgresql+asyncpg://app:{SECRET}@db.internal/orbit")
    try:
        assert store_for_engine(engine).live_updates is None  # off unless asked
        store = store_for_engine(engine, live_updates=True)
        updates = store.live_updates
        assert isinstance(updates, PostgresUpdates)
        assert updates._send_dsn == updates._listen_dsn == asyncpg_dsn(engine.url)
        await store.close()
    finally:
        await engine.dispose()


async def test_driver_only_engine_option_refuses_live_updates_without_echo() -> None:
    url = f"postgresql+asyncpg://app:{SECRET}@db/orbit?prepared_statement_cache_size=0"
    engine = create_async_engine(url)
    try:
        with pytest.raises(ValueError) as caught:
            store_for_engine(engine, live_updates=True)
        assert "prepared_statement_cache_size" in str(caught.value)
        assert SECRET not in str(caught.value)
        explicit = PostgresUpdates(url="postgresql://app@direct/orbit")
        store = store_for_engine(engine, live_updates=explicit)  # the documented way out
        assert store.live_updates is explicit
        await store.close()
    finally:
        await engine.dispose()


@pytest.mark.parametrize("key", ["live_updates", "notify_channel"])
async def test_run_store_from_database_url_refuses_live_update_options(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlalchemy.ext.asyncio

    from dendrux.store import RunStore

    def no_engine(*args: object, **kwargs: object) -> None:
        raise AssertionError("refused before any engine is built")

    monkeypatch.setattr(sqlalchemy.ext.asyncio, "create_async_engine", no_engine)
    with pytest.raises(ValueError, match=r"pass it to RunStore\(store\)") as caught:
        RunStore.from_database_url("sqlite+aiosqlite://", database_options={key: True})
    assert "Live updates require an explicit store" in str(caught.value)
