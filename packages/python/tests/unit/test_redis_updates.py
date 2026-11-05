"""RedisUpdates (spec 3.5) against a scripted stand-in for redis.asyncio.

The real server is exercised in tests/integration/test_redis_updates.py.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import dendrux
from dendrux.db import _redis_updates as ru
from dendrux.db._live_updates import LiveUpdates
from tests.unit.test_redis_updates_fakes import (
    FakeConnectionError,
    FakeRedisError,
    FakeTimeoutError,
    FakeWorld,
)

URL = "redis://:s3cret-sentinel@redis.sentinel-host.example:6380/2"
CHANNEL = "dendrux:events"
FAST: dict[str, Any] = dict(
    send_timeout_s=0.05,
    flush_deadline_s=0.2,
    close_timeout_s=1.0,
    backoff_initial_s=0.01,
    backoff_max_s=0.02,
    health_check_s=0.1,
)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    fake = FakeWorld()
    monkeypatch.setattr(ru, "_load_api", fake.api)
    return fake


def make(**overrides: Any) -> ru.RedisUpdates:
    options: dict[str, Any] = {**FAST, **overrides}
    return ru.RedisUpdates(options.pop("url", URL), **options)


async def eventually(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.005)


class Consumer:
    def __init__(self) -> None:
        self.hints: list[tuple[str, ...]] = []
        self.gaps = 0

    def on_hints(self, run_ids: Any) -> None:
        self.hints.append(tuple(run_ids))

    def on_gap(self) -> None:
        self.gaps += 1


# -- construction ----------------------------------------------------------------------------


def test_constructing_without_the_redis_client_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setitem(sys.modules, "redis.asyncio", None)
    with pytest.raises(ImportError, match=r"pip install dendrux\[redis\]"):
        ru.RedisUpdates(URL)


def test_importing_the_module_without_the_client_works_and_only_use_fails() -> None:
    source = str(Path(dendrux.__file__).resolve().parent.parent)
    code = (
        "import sys\n"
        "sys.modules['redis'] = None\n"
        "sys.modules['redis.asyncio'] = None\n"
        "from dendrux.db import _redis_updates as m\n"
        "try:\n"
        "    m.RedisUpdates('redis://localhost')\n"
        "except ImportError as error:\n"
        "    print('ImportError', 'dendrux[redis]' in str(error))\n"
    )
    env = {**os.environ, "PYTHONPATH": source}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["ImportError", "True"]


@pytest.mark.parametrize(
    "bad", ["", "http://secret-host", "localhost:6379", "redis", 5, None, b"redis://x"]
)
def test_a_url_that_is_not_a_redis_url_is_refused_without_echoing_it(
    world: FakeWorld, bad: Any
) -> None:
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(bad)
    assert "secret-host" not in str(caught.value)


@pytest.mark.parametrize("url", ["redis://h", "rediss://h:1/0", "unix:///var/run/redis.sock"])
def test_the_three_redis_url_schemes_are_accepted(world: FakeWorld, url: str) -> None:
    assert ru.RedisUpdates(url).healthy is False


@pytest.mark.parametrize("bad", ["", "x" * 513, "has space", "new\nline", "nul\x00", 5, None])
def test_a_channel_outside_its_limits_is_refused(world: FakeWorld, bad: Any) -> None:
    with pytest.raises(ValueError):
        make(channel=bad)


def test_a_channel_ten_times_a_realistic_name_is_accepted(world: FakeWorld) -> None:
    name = "app:" + "e" * 496  # 500 characters; realistic names are about 50
    assert make(channel=name)._channel == name


def test_limits_reach_the_base_and_bad_limits_are_refused(world: FakeWorld) -> None:
    assert make(max_pending=7, max_batch=3).max_pending == 7
    with pytest.raises(ValueError):
        make(max_batch=0)


def test_repr_and_attributes_do_not_show_the_url(world: FakeWorld) -> None:
    updates = make()
    assert "s3cret" not in repr(updates) and "sentinel-host" not in repr(updates)
    assert "s3cret" not in repr(vars(updates)) and "sentinel-host" not in repr(vars(updates))
    assert CHANNEL in repr(updates)


@pytest.mark.parametrize(
    "key",
    [
        "socket_timeout",
        "socket_connect_timeout",
        "retry",
        "retry_on_timeout",
        "retry_on_error",
        "max_connections",
        "connection_pool",
        "connection_class",
        "single_connection_client",
        "health_check_interval",
    ],
)
def test_options_that_would_defeat_the_transport_are_rejected(world: FakeWorld, key: str) -> None:
    with pytest.raises(ValueError) as caught:
        make(client_options={key: 1})
    assert key in str(caught.value)
    assert "sentinel" not in str(caught.value)


def test_an_option_the_client_refuses_is_a_value_error_without_the_url(world: FakeWorld) -> None:
    with pytest.raises(ValueError) as caught:
        make(client_options={"bogus_option": 1})
    assert "sentinel" not in str(caught.value) and caught.value.__cause__ is None


def test_allowed_options_reach_the_pool_next_to_the_forced_ones(world: FakeWorld) -> None:
    updates = make(client_options={"password": "pw", "db": 3, "client_name": "dx"})
    asyncio.run(updates.send(['["a"]']))
    kwargs = world.pools[-1].kwargs
    assert (kwargs["password"], kwargs["db"], kwargs["client_name"]) == ("pw", 3, "dx")
    assert kwargs["max_connections"] == 1
    assert kwargs["socket_timeout"] == kwargs["socket_connect_timeout"] == 0.05
    assert kwargs["retry"].retries == 0


# -- sender ----------------------------------------------------------------------------------


async def test_one_send_is_one_pipelined_round_trip_without_a_transaction(
    world: FakeWorld,
) -> None:
    updates = make()
    await updates.send(['["a"]', '["b"]'])
    assert len(world.pipelines) == 1
    assert world.pipelines[0].transaction is False
    assert world.published == [(CHANNEL, '["a"]'), (CHANNEL, '["b"]')]


async def test_sends_share_one_pool_one_connection_and_one_client(world: FakeWorld) -> None:
    updates = make()
    for _ in range(3):
        await updates.send(['["a"]'])
    assert (len(world.pools), len(world.connections), len(world.clients)) == (1, 1, 1)
    assert world.pools[0].max_connections == 1


async def test_flush_publishes_the_base_wire_format_on_the_channel(world: FakeWorld) -> None:
    updates = make(channel="my:chan")
    updates.enqueue(["R1", "R2"])
    await updates.flush()
    assert world.published == [("my:chan", '["R1","R2"]')]
    assert updates.stats().sent == 2


async def test_the_maximum_batch_and_ten_times_it_are_delivered_within_the_deadline(
    world: FakeWorld,
) -> None:
    updates = make(flush_deadline_s=5.0, send_timeout_s=1.0)
    updates.enqueue(f"run-{i:06d}" for i in range(500))
    await updates.flush()
    assert updates.stats().sent == 500
    updates.enqueue(f"more-{i:06d}" for i in range(5_000))
    await updates.flush()
    assert updates.stats().sent == 5_500
    assert len(world.pipelines) <= 1 + 10 + 1  # a few large pipelines, never one per id


async def test_a_stalled_publish_is_aborted_and_the_next_send_uses_a_fresh_connection(
    world: FakeWorld,
) -> None:
    release = asyncio.Event()

    async def stall(pipeline: Any) -> None:
        await release.wait()

    world.execute_hook = stall
    updates = make()
    updates.enqueue(["R1"])
    async with asyncio.timeout(5):  # a hang, not a speed, is what this guards against
        await updates.flush()
    assert updates.stats().dropped_failed == 1
    assert world.connections[0].aborted
    world.execute_hook = None
    await asyncio.sleep(0.05)  # leave the sender's backoff
    updates.enqueue(["R2"])
    await updates.flush()
    assert world.published == [(CHANNEL, '["R2"]')]
    assert len(world.pools) == 2 and len(world.connections) == 2
    assert not world.connections[1].aborted


async def test_abort_sender_drops_the_connection_that_is_in_use_right_now(
    world: FakeWorld,
) -> None:
    gate = asyncio.Event()

    async def stall(pipeline: Any) -> None:
        await gate.wait()

    world.execute_hook = stall
    updates = make()
    task = asyncio.ensure_future(updates.send(['["a"]']))
    await eventually(lambda: bool(world.pipelines))
    updates.abort_sender()  # synchronous
    assert world.connections[0].aborted
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_abort_sender_is_safe_with_no_connection_and_when_repeated(
    world: FakeWorld,
) -> None:
    updates = make()
    updates.abort_sender()
    await updates.send(['["a"]'])
    updates.abort_sender()
    updates.abort_sender()
    assert world.connections[0].aborted


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (
            FakeConnectionError("Error 111 connecting to redis.sentinel-host.example:6380"),
            ConnectionError,
        ),
        (FakeTimeoutError("Timeout reading from redis.sentinel-host.example:6380"), TimeoutError),
        (FakeRedisError("WRONGPASS s3cret-sentinel"), RuntimeError),
        (OSError("host redis.sentinel-host.example unreachable"), ConnectionError),
    ],
)
async def test_publish_errors_become_plain_errors_without_host_or_secrets(
    world: FakeWorld, raised: Exception, expected: type[Exception]
) -> None:
    async def fail(pipeline: Any) -> None:
        raise raised

    world.execute_hook = fail
    updates = make()
    with pytest.raises(expected) as caught:
        await updates.send(['["a"]'])
    error = caught.value
    assert type(error) is expected
    assert "sentinel" not in str(error) and "sentinel" not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


async def test_caller_cancellation_of_a_send_is_not_converted(world: FakeWorld) -> None:
    gate = asyncio.Event()

    async def stall(pipeline: Any) -> None:
        await gate.wait()

    world.execute_hook = stall
    updates = make()
    task = asyncio.ensure_future(updates.send(['["a"]']))
    await eventually(lambda: bool(world.pipelines))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_aclose_sender_closes_the_client_and_pool_and_tolerates_failure(
    world: FakeWorld,
) -> None:
    updates = make()
    await updates.aclose_sender()  # nothing open yet
    await updates.send(['["a"]'])
    await updates.aclose_sender()
    assert world.pools[0].closed and world.clients[0].closed
    await updates.send(['["b"]'])  # a later send starts again
    assert len(world.pools) == 2

    async def broken() -> None:
        raise FakeConnectionError("boom")

    world.pools[1].aclose = broken  # type: ignore[method-assign]
    await updates.aclose_sender()


async def test_close_through_the_base_closes_the_sender(world: FakeWorld) -> None:
    updates = make()
    updates.enqueue(["R1"])
    await updates.flush()
    await updates.close()
    assert world.pools[0].closed


# -- listener --------------------------------------------------------------------------------


async def test_the_listener_is_healthy_only_after_the_server_acknowledged_the_subscription(
    world: FakeWorld,
) -> None:
    world.ack_delay = 0.1
    updates = make(send_timeout_s=0.2, flush_deadline_s=0.4)
    iterator = updates.receive()
    pending = asyncio.ensure_future(anext(iterator))
    await asyncio.sleep(0.04)
    assert not pending.done()  # subscribed, but not acknowledged yet
    assert world.pubsubs[0].subscribed == [CHANNEL]
    assert await pending is None
    await iterator.aclose()


async def test_an_acknowledgement_for_another_channel_is_not_accepted(world: FakeWorld) -> None:
    world.ack_mode = "wrong_channel"
    updates = make()
    with pytest.raises(TimeoutError):
        await anext(updates.receive())


async def test_a_missing_acknowledgement_times_out_and_cleans_up(world: FakeWorld) -> None:
    world.ack_mode = "never"
    updates = make()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(5):
            await anext(updates.receive())
    assert world.pubsubs[0].closed and world.pools[0].closed


async def test_messages_are_yielded_as_text_in_order(world: FakeWorld) -> None:
    updates = make(health_check_s=30.0)
    iterator = updates.receive()
    assert await anext(iterator) is None
    pubsub = world.pubsubs[0]
    pubsub.push('["R1"]')
    pubsub.push(b'["R2"]')
    pubsub.push('["R3"]')
    assert [await anext(iterator) for _ in range(3)] == ['["R1"]', '["R2"]', '["R3"]']
    await iterator.aclose()


async def test_undecodable_bytes_become_a_message_the_base_drops(world: FakeWorld) -> None:
    updates = make(health_check_s=30.0)
    iterator = updates.receive()
    await anext(iterator)
    world.pubsubs[0].push(b"\xff\xfe[")
    message = await anext(iterator)
    assert isinstance(message, str) and updates._decode(message) is None
    await iterator.aclose()


async def test_a_hint_flood_of_ten_times_the_inbound_bound_is_delivered_in_order(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=30.0, max_inbound=100)
    iterator = updates.receive()
    await anext(iterator)
    for i in range(1_000):
        world.pubsubs[0].push(f'["r{i}"]')
    seen = [await anext(iterator) for _ in range(1_000)]
    assert seen[0] == '["r0"]' and seen[-1] == '["r999"]'
    await iterator.aclose()


async def test_an_idle_connection_is_pinged_with_a_fresh_token_each_time(
    world: FakeWorld,
) -> None:
    updates = make()
    iterator = updates.receive()
    await anext(iterator)
    waiting = asyncio.ensure_future(anext(iterator))
    await eventually(lambda: len(world.pubsubs[0].pings) >= 2)
    pings = world.pubsubs[0].pings
    assert len(set(pings)) == len(pings)  # tokens are not reused
    assert not waiting.done()  # matching pongs keep it alive
    world.pubsubs[0].push('["R1"]')
    assert await waiting == '["R1"]'
    await iterator.aclose()


@pytest.mark.parametrize("mode", ["wrong", "silent"])
async def test_a_missing_or_mismatched_pong_ends_the_listener(world: FakeWorld, mode: str) -> None:
    world.pong_mode = mode
    updates = make()
    iterator = updates.receive()
    await anext(iterator)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(5):  # ends by itself after health_check_s + 4 x send_timeout_s
            await anext(iterator)
    assert world.pubsubs[0].closed and world.pools[0].closed


async def test_messages_arriving_while_a_pong_is_awaited_are_still_delivered(
    world: FakeWorld,
) -> None:
    world.pong_mode = "silent"
    updates = make(send_timeout_s=0.5, flush_deadline_s=1.0)
    iterator = updates.receive()
    await anext(iterator)
    waiting = asyncio.ensure_future(anext(iterator))
    await eventually(lambda: bool(world.pubsubs[0].pings))
    world.pubsubs[0].push('["R1"]')
    assert await waiting == '["R1"]'
    await iterator.aclose()


async def test_a_connection_error_ends_the_listener_with_a_plain_error_and_closes_it(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=30.0)
    iterator = updates.receive()
    await anext(iterator)
    world.pubsubs[0].fail(FakeConnectionError("closed by redis.sentinel-host.example:6380"))
    with pytest.raises(ConnectionError) as caught:
        await anext(iterator)
    assert "sentinel" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert world.pubsubs[0].closed and world.pools[0].closed


async def test_a_second_subscribe_acknowledgement_is_surfaced_as_the_gap_object(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=30.0)
    iterator = updates.receive()
    await anext(iterator)
    pubsub = world.pubsubs[0]
    pubsub.queue.put_nowait(
        {"type": "subscribe", "pattern": None, "channel": CHANNEL.encode(), "data": 1}
    )
    pubsub.push('["R1"]')
    assert await anext(iterator) is LiveUpdates.GAP  # a hidden resubscribe is a gap
    assert await anext(iterator) == '["R1"]'
    await iterator.aclose()


async def test_other_control_replies_are_ignored(world: FakeWorld) -> None:
    updates = make(health_check_s=30.0)
    iterator = updates.receive()
    await anext(iterator)
    pubsub = world.pubsubs[0]
    pubsub.queue.put_nowait({"type": "unsubscribe", "channel": b"x", "data": 0})
    pubsub.queue.put_nowait({"type": "pmessage", "channel": b"x", "data": b"[]"})
    pubsub.push('["R1"]')
    assert await anext(iterator) == '["R1"]'
    await iterator.aclose()


async def test_cancelling_the_listener_closes_its_connection(world: FakeWorld) -> None:
    updates = make(health_check_s=30.0)
    iterator = updates.receive()
    await anext(iterator)
    waiting = asyncio.ensure_future(anext(iterator))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await iterator.aclose()
    assert world.pubsubs[0].closed and world.pools[0].closed
    assert world.connections[0].aborted


async def test_the_listener_owns_its_own_single_connection_pool(world: FakeWorld) -> None:
    updates = make(health_check_s=30.0)
    await updates.send(['["a"]'])
    iterator = updates.receive()
    await anext(iterator)
    sender_pool, listener_pool = world.pools
    assert listener_pool is not sender_pool
    kwargs = listener_pool.kwargs
    assert kwargs["max_connections"] == 1 and kwargs["retry"].retries == 0
    assert kwargs["socket_connect_timeout"] == 0.05 and kwargs["socket_timeout"] == 0.2
    await iterator.aclose()


async def test_abort_listener_and_abort_sender_touch_only_their_own_connection(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=30.0)
    updates.abort_listener()  # nothing open: harmless
    await updates.send(['["a"]'])
    iterator = updates.receive()
    await anext(iterator)
    sender_connection = world.pools[0].connections[0]
    listener_connection = world.pubsubs[0].connection
    assert listener_connection is not sender_connection
    updates.abort_listener()  # synchronous, repeatable
    updates.abort_listener()
    assert listener_connection.aborted and not sender_connection.aborted
    updates.abort_sender()
    assert sender_connection.aborted
    await iterator.aclose()


# -- with the base: health, gaps and reconnects ----------------------------------------------


async def test_a_lost_connection_surfaces_a_gap_then_recovers_with_another(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=30.0)
    consumer = Consumer()
    updates._register(consumer, stream_events=False)
    lease = await updates._acquire_listener()
    await eventually(lambda: updates.healthy)
    assert consumer.gaps == 1  # became healthy
    world.pubsubs[0].push('["R1"]')
    await eventually(lambda: consumer.hints == [("R1",)])

    world.pubsubs[0].fail(FakeConnectionError("closed"))
    # degraded (hints may have been missed) then healthy again (everything is re-read once):
    # two more gaps, and the second subscription is a new connection
    await eventually(lambda: consumer.gaps == 3 and updates.healthy and len(world.pubsubs) == 2)
    world.pubsubs[1].push('["R2"]')
    await eventually(lambda: consumer.hints[-1] == ("R2",))
    await updates._release_listener(lease)
    await updates.close()
    assert all(pubsub.closed for pubsub in world.pubsubs)


async def test_the_base_drops_a_malformed_hint_without_ending_the_listener(
    world: FakeWorld,
) -> None:
    updates = make(health_check_s=30.0)
    consumer = Consumer()
    updates._register(consumer, stream_events=False)
    lease = await updates._acquire_listener()
    await eventually(lambda: updates.healthy)
    world.pubsubs[0].push("not json")
    world.pubsubs[0].push('["R1"]')
    await eventually(lambda: consumer.hints == [("R1",)])
    assert updates.stats().bad_messages == 1 and updates.healthy
    await updates._release_listener(lease)
    await updates.close()
