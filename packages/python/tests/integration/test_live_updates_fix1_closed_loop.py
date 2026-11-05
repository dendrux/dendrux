"""Detached close must release real server connections before garbage collection."""

import asyncio
import gc
import os
import uuid
import warnings

import asyncpg
import pytest
import redis.asyncio as redis
from sqlalchemy.engine import make_url

from dendrux.store import PostgresUpdates, RedisUpdates


async def _count(kind, url, name):
    if kind == "postgres":
        connection = await asyncpg.connect(url)
        try:
            return await connection.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name=$1", name
            )
        finally:
            await connection.close()
    connection = redis.from_url(url)
    try:
        return sum(row["name"] == name for row in await connection.client_list())
    finally:
        await connection.aclose()


def _assert_closed_loop_close(pg_test_url, kind, cancel_listener):
    name = "rf2fix-" + uuid.uuid4().hex
    if kind == "postgres":
        url = (
            make_url(pg_test_url).set(drivername="postgresql").render_as_string(hide_password=False)
        )
        bus = PostgresUpdates(url=url, connect_args={"server_settings": {"application_name": name}})
    else:
        url = os.environ["DENDRUX_TEST_REDIS_URL"]
        bus = RedisUpdates(url, channel=name, client_options={"client_name": name})
    loop = asyncio.new_event_loop()

    async def open_connections():
        bus.enqueue(["r"])
        await bus.flush()
        await bus._acquire_listener()
        async with asyncio.timeout(5):
            while not bus.healthy:
                await asyncio.sleep(0.01)
        assert await _count(kind, url, name) == 2

    loop.run_until_complete(open_connections())
    pending = list(asyncio.all_tasks(loop))
    if cancel_listener:
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()
    asyncio.run(bus.close())
    asyncio.run(bus.close())

    async def verify():
        async with asyncio.timeout(2):
            while await _count(kind, url, name):
                await asyncio.sleep(0.01)

    asyncio.run(verify())
    assert bus._state == "closed"
    if kind == "postgres":
        assert bus._sender is None and bus._listener is None
    else:
        assert bus._wire.pool is None and bus._wire.subscription is None


@pytest.mark.parametrize("kind", ["postgres", "redis"])
@pytest.mark.parametrize("cancel_listener", [True, False])
def test_closed_loop_close_disconnects_without_gc(pg_test_url, kind, cancel_listener):
    try:
        _assert_closed_loop_close(pg_test_url, kind, cancel_listener)
    finally:
        # The helper's local references are gone. Collect the deliberately abandoned
        # loop objects here so their ResourceWarnings cannot escape into another test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()
