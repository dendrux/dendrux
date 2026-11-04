"""Fix round 1 for RedisUpdates: URL options, close ownership, bounded input, one detection budget.

Tests named ``real_*`` use the installed redis-py (no server, no network): its URL parser, pool
and protocol parser are the things under test. The rest use the scripted stand-in.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dendrux.db import _redis_updates as ru
from tests.unit.test_redis_updates_fakes import FakeWorld

HOST = "redis.sentinel-host.example"
FAST: dict[str, Any] = dict(
    send_timeout_s=0.05,
    flush_deadline_s=0.2,
    close_timeout_s=1.0,
    backoff_initial_s=0.01,
    backoff_max_s=0.02,
    health_check_s=0.05,
)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    fake = FakeWorld()
    monkeypatch.setattr(ru, "_load_api", fake.api)
    return fake


def make(url: str = f"redis://{HOST}:6380/0", **overrides: Any) -> ru.RedisUpdates:
    return ru.RedisUpdates(url, **{**FAST, **overrides})


# -- R1: URL query options may not change the fixed transport settings -------------------------

PROTECTED_QUERIES = [
    "socket_timeout=900",
    "socket_connect_timeout=900",
    "max_connections=100",
    "health_check_interval=1",
    "retry_on_timeout=true",
    "retry_on_error=1",
    "retry=1",
    "protocol=3",
    "parser_class=x",
    "connection_class=x",
    "single_connection_client=true",
    "socket%5Ftimeout=900",  # percent-encoded key: redis-py decodes it, so we must too
    "db=1&max_connections=5",
]


@pytest.mark.parametrize("query", PROTECTED_QUERIES)
def test_real_a_url_query_may_not_change_a_protected_setting(query: str) -> None:
    url = f"redis://:s3cret-sentinel@{HOST}:6380/0?{query}"
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(url, **FAST)
    assert "sentinel" not in str(caught.value) and "6380" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_real_harmless_url_query_options_still_work_and_the_fixed_ones_win() -> None:
    updates = ru.RedisUpdates(f"redis://{HOST}/0?client_name=dx", **FAST)
    pool = updates._new_pool(0.2)
    connection = pool.make_connection()
    assert pool.max_connections == 1
    assert (connection.socket_timeout, connection.socket_connect_timeout) == (0.2, 0.05)
    assert connection.health_check_interval == 0


# -- R2: closing the sender never loses the handle that can abort its socket -------------------


async def eventually(predicate: Any) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def stall_cleanup(world: FakeWorld) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the sender's client and pool cleanup hang; returns (entered, never_released)."""
    entered, gate = asyncio.Event(), asyncio.Event()

    async def hang(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await gate.wait()

    world.pools[0].aclose = hang  # type: ignore[method-assign]
    world.clients[0].aclose = hang  # type: ignore[method-assign]
    return entered, gate


async def test_a_stalled_close_deadline_still_aborts_the_sender_socket(world: FakeWorld) -> None:
    updates = make(close_timeout_s=0.2)
    await updates.send(['["a"]'])
    stall_cleanup(world)
    async with asyncio.timeout(5):
        await updates.close()  # the total close deadline passes while cleanup hangs
    assert world.connections[0].aborted


async def test_a_cancelled_close_aborts_the_sender_socket_and_a_retry_finishes(
    world: FakeWorld,
) -> None:
    updates = make(close_timeout_s=30.0)
    await updates.send(['["a"]'])
    entered, _ = stall_cleanup(world)
    task = asyncio.ensure_future(updates.close())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.connections[0].aborted  # before any retry
    async with asyncio.timeout(5):
        await updates.close()  # a later close finishes the job
    assert updates._state == "closed"


async def test_a_cancelled_aclose_sender_aborts_the_socket_itself(world: FakeWorld) -> None:
    updates = make()
    await updates.send(['["a"]'])
    entered, _ = stall_cleanup(world)
    task = asyncio.ensure_future(updates.aclose_sender())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.connections[0].aborted


async def test_real_pool_cleanup_that_never_returns_cannot_strand_a_socket() -> None:
    from types import SimpleNamespace

    updates = ru.RedisUpdates(
        f"redis://{HOST}/0", send_timeout_s=0.01, flush_deadline_s=0.05, close_timeout_s=0.2
    )
    pool = updates._sender().connection_pool  # redis-py's own pool, no connection made
    aborted: list[bool] = []
    transport = SimpleNamespace(abort=lambda: aborted.append(True))
    pool._available_connections.append(
        SimpleNamespace(_writer=SimpleNamespace(transport=transport))
    )

    async def never() -> None:
        await asyncio.Event().wait()

    pool.aclose = never  # redis-py 8 awaits maintenance work here before it disconnects
    async with asyncio.timeout(5):
        await updates.close()
    assert aborted == [True]
    pool._available_connections.clear()


# -- R3: an incoming message is bounded before it is read into memory --------------------------


def listen_parser(updates: ru.RedisUpdates) -> tuple[Any, Any]:
    """The protocol parser redis-py would use on the listen connection, over an in-memory stream."""
    subscription = ru._Subscription(updates)
    connection = subscription.pool.make_connection()
    connection._reader = asyncio.StreamReader()
    connection._parser.on_connect(connection)
    return connection._parser, connection._reader


def message(size: int) -> bytes:
    return (
        b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$"
        + str(size).encode()
        + b"\r\n"
        + b"x" * size
        + b"\r\n"
    )


async def test_real_a_message_at_the_bound_is_read_and_one_byte_more_is_refused() -> None:
    updates = ru.RedisUpdates(f"redis://{HOST}/0", **FAST)  # max_message_bytes 7,900
    parser, reader = listen_parser(updates)
    reader.feed_data(message(7_900))
    assert len((await parser.read_response())[2]) == 7_900
    reader.feed_data(message(7_901))
    with pytest.raises(Exception) as caught:
        await parser.read_response()
    assert type(updates._plain_error(caught.value)) is ConnectionError
    assert "xxxx" not in str(caught.value)


async def test_real_a_realistic_message_ten_times_over_is_read() -> None:
    parser, reader = listen_parser(ru.RedisUpdates(f"redis://{HOST}/0", **FAST))
    reader.feed_data(message(1_000))  # a normal hint is about 100 bytes
    assert len((await parser.read_response())[2]) == 1_000


async def test_real_a_huge_declared_length_is_refused_before_any_body_arrives() -> None:
    parser, reader = listen_parser(ru.RedisUpdates(f"redis://{HOST}/0", **FAST))
    reader.feed_data(b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$1000000000\r\n")  # and no body at all
    with pytest.raises(Exception) as caught:  # noqa: B017
        async with asyncio.timeout(1):  # waiting for the body would end in this timeout instead
            await parser.read_response()
    assert not isinstance(caught.value, TimeoutError)


async def test_real_the_bound_follows_max_message_bytes() -> None:
    parser, reader = listen_parser(ru.RedisUpdates(f"redis://{HOST}/0", max_message_bytes=3_000))
    reader.feed_data(message(3_000))
    assert len((await parser.read_response())[2]) == 3_000
    reader.feed_data(message(3_001))
    with pytest.raises(Exception):  # noqa: B017
        await parser.read_response()


def test_real_a_redis_client_without_a_boundable_parser_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "redis._parsers", None)
    with pytest.raises(ImportError, match=r"dendrux\[redis\]"):
        ru._load_api()
