"""A standard engine remains inspectable through real connection and pool lifecycle changes."""

from types import SimpleNamespace

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._postgres_updates import PostgresUpdates


async def _assert_binding(engine):
    bus = PostgresUpdates()
    try:
        bus.bind(SimpleNamespace(_engine=engine))
        assert bus._settings.send_dsn is not None
    finally:
        await bus.close()


async def _pid(engine):
    async with engine.connect() as connection:
        return await connection.scalar(text("SELECT pg_backend_pid()"))


@pytest.mark.parametrize(
    "state",
    ["connected", "missing_table", "unique_violation", "recycled", "disposed", "failed_connect"],
)
async def test_used_standard_engine_remains_bindable(pg_test_url, state):
    url = make_url(pg_test_url)
    if state == "failed_connect":
        url = url.set(password="RF2_INTENTIONALLY_WRONG")
    engine = create_async_engine(url, pool_recycle=0 if state == "recycled" else -1)
    try:
        await _assert_binding(engine)
        if state == "failed_connect":
            with pytest.raises((asyncpg.InvalidPasswordError, DBAPIError)):
                await _pid(engine)
        elif state == "missing_table":
            with pytest.raises(ProgrammingError):
                async with engine.connect() as connection:
                    await connection.execute(text("SELECT * FROM rf2_fix2_missing_table"))
        elif state == "unique_violation":
            with pytest.raises(IntegrityError):
                async with engine.begin() as connection:
                    await connection.execute(
                        text(
                            "CREATE TEMP TABLE rf2_fix2_unique (id int PRIMARY KEY) ON COMMIT DROP"
                        )
                    )
                    await connection.execute(text("INSERT INTO rf2_fix2_unique VALUES (1), (1)"))
        else:
            before = await _pid(engine)
            if state == "recycled":
                assert await _pid(engine) != before
            if state == "disposed":
                await engine.dispose()
                await _assert_binding(engine)
                assert await _pid(engine) != before
        await _assert_binding(engine)
        await engine.dispose()
        await _assert_binding(engine)
    finally:
        await engine.dispose()
