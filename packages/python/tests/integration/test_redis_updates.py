"""RedisUpdates against a real Redis (DENDRUX_TEST_REDIS_URL; skipped when it is unset).

Every test uses a private channel and a private client name, so tests can share one server and
run in parallel. A small TCP proxy sits between a RedisUpdates and the server where a test needs
to see the bytes on the wire or to stall one direction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest

from dendrux.db import _redis_updates as ru
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import EventFeed, RunStore, _follow
from tests._helpers.log_records import record_text

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

REDIS_URL = os.environ.get("DENDRUX_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="DENDRUX_TEST_REDIS_URL not set")

FAST: dict[str, Any] = dict(
    send_timeout_s=0.3,
    flush_deadline_s=1.0,
    close_timeout_s=3.0,
    backoff_initial_s=0.05,
    backoff_max_s=0.1,
    health_check_s=0.2,
)


async def eventually(predicate: Callable[[], Any], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


class Consumer:
    def __init__(self) -> None:
        self.hints: list[tuple[str, ...]] = []
        self.gaps = 0

    def on_hints(self, run_ids: Any) -> None:
        self.hints.append(tuple(run_ids))

    def on_gap(self) -> None:
        self.gaps += 1

    @property
    def ids(self) -> list[str]:
        return [run_id for batch in self.hints for run_id in batch]


@pytest.fixture
def channel() -> str:
    return f"dxtest:{uuid.uuid4().hex}"


@pytest.fixture
def name() -> str:
    return f"dx-{uuid.uuid4().hex[:12]}"


@pytest.fixture
async def admin() -> AsyncIterator[Any]:
    import redis.asyncio as aioredis

    client = aioredis.Redis.from_url(REDIS_URL)
    yield client
    await client.aclose()


def make(url: str | None = None, **overrides: Any) -> ru.RedisUpdates:
    options: dict[str, Any] = {**FAST, **overrides}
    return ru.RedisUpdates(url or REDIS_URL, **options)


async def clients(admin: Any, name: str) -> list[dict[str, str]]:
    """Server-side view of the connections whose client name is ``name``."""
    raw = await admin.execute_command("CLIENT", "LIST")
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    found = []
    for line in text.splitlines():
        fields = dict(part.split("=", 1) for part in line.split(" ") if "=" in part)
        if fields.get("name") == name:
            found.append(fields)
    return found


async def numsub(admin: Any, channel: str) -> int:
    reply = await admin.pubsub_numsub(channel)
    return int(reply[0][1])


async def listening(updates: ru.RedisUpdates, consumer: Consumer) -> object:
    updates._register(consumer, stream_events=False)
    lease = await updates._acquire_listener()
    await eventually(lambda: updates.healthy)
    return lease


# -- a TCP proxy to see and stall the wire ----------------------------------------------------


class Link:
    def __init__(self) -> None:
        self.up = bytearray()  # client -> server, as the proxy saw it
        self.client_closed = asyncio.Event()


class Proxy:
    def __init__(self, host: str, port: int) -> None:
        self.target = (host, port)
        self.port = 0
        self.links: list[Link] = []
        self.stall_up = False  # swallow client -> server bytes
        self.hold_down = False  # delay server -> client bytes
        self.hold_on_subscribe = False  # start holding server bytes as SUBSCRIBE passes
        self._release = asyncio.Event()
        self._release.set()
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        self.port = int(self._server.sockets[0].getsockname()[1])

    def hold(self) -> None:
        self.hold_down = True
        self._release.clear()

    def release(self) -> None:
        self.hold_down = False
        self._release.set()

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.ensure_future(self._serve(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        link = Link()
        self.links.append(link)
        up_reader, up_writer = await asyncio.open_connection(*self.target)
        pumps = [
            asyncio.ensure_future(self._pump_up(reader, up_writer, link)),
            asyncio.ensure_future(self._pump_down(up_reader, writer)),
        ]
        await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for pump in pumps:
            pump.cancel()
        for stream in (writer, up_writer):
            stream.close()

    async def _pump_up(
        self, reader: asyncio.StreamReader, up_writer: asyncio.StreamWriter, link: Link
    ) -> None:
        try:
            while data := await reader.read(65536):
                link.up += data
                if self.hold_on_subscribe and b"SUBSCRIBE" in data:
                    self.hold()
                if not self.stall_up:
                    up_writer.write(data)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        link.client_closed.set()

    async def _pump_down(
        self, up_reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(ConnectionError, asyncio.IncompleteReadError):
            while data := await up_reader.read(65536):
                await self._release.wait()
                writer.write(data)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


@pytest.fixture
async def proxy() -> AsyncIterator[Proxy]:
    parts = urlsplit(REDIS_URL)
    server = Proxy(parts.hostname or "127.0.0.1", parts.port or 6379)
    await server.start()
    yield server
    await server.close()


def proxied(server: Proxy) -> str:
    return f"redis://127.0.0.1:{server.port}{urlsplit(REDIS_URL).path}"


# -- the wire ---------------------------------------------------------------------------------


async def test_a_hint_sent_by_one_instance_reaches_the_consumer_of_another(channel: str) -> None:
    writer, reader = make(channel=channel), make(channel=channel)
    consumer = Consumer()
    lease = await listening(reader, consumer)
    writer.enqueue(["R1", "R2"])
    await writer.flush()
    await eventually(lambda: consumer.hints == [("R1", "R2")])
    assert writer.stats().sent == 2
    await reader._release_listener(lease)
    await reader.close()
    await writer.close()


async def test_a_hint_burst_ten_times_the_inbound_bound_arrives_in_full(channel: str) -> None:
    ids = [f"run-{i:05d}" for i in range(10_000)]
    ready, received = asyncio.Event(), asyncio.Event()

    class BurstConsumer(Consumer):
        def on_gap(self) -> None:
            super().on_gap()
            ready.set()

        def on_hints(self, run_ids: Any) -> None:
            super().on_hints(run_ids)
            if len(self.ids) >= len(ids):
                received.set()

    # This is a delivery test. The watchdog must fire before sender/liveness deadlines;
    # FAST's 200 ms health checks can disconnect a healthy subscriber on a busy runner.
    options = dict(send_timeout_s=30.0, flush_deadline_s=30.0, health_check_s=30.0)
    writer, reader = make(channel=channel, **options), make(channel=channel, **options)
    consumer = BurstConsumer()
    reader._register(consumer, stream_events=False)
    lease = None
    try:
        async with asyncio.timeout(30):
            lease = await reader._acquire_listener()
            await ready.wait()
            assert reader.healthy
            for start in range(0, len(ids), 500):
                writer.enqueue(ids[start : start + 500])
                await writer.flush()
            await received.wait()
        assert consumer.ids == ids  # one sender, one channel: order is kept
        assert writer.stats().sent == len(ids)
        assert writer.stats().dropped_failed == 0
        assert reader.stats().gaps == 1  # the first healthy transition only
        assert reader.stats().dropped_inbound == 0
    finally:
        if lease is not None:
            await reader._release_listener(lease)
        await reader.close()
        await writer.close()


async def test_the_subscription_is_in_place_when_the_listener_reports_ready(
    admin: Any, channel: str
) -> None:
    reader = make(channel=channel, health_check_s=30.0)
    iterator = reader.receive()
    assert await anext(iterator) is None
    assert await numsub(admin, channel) == 1  # the server had already registered us
    await iterator.aclose()
    await eventually_async(lambda: numsub(admin, channel), 0)


async def eventually_async(read: Callable[[], Any], expected: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while (value := await read()) != expected:
        assert time.monotonic() < deadline, f"{value!r} != {expected!r}"
        await asyncio.sleep(0.01)


async def test_healthy_is_not_reported_before_the_acknowledgement_arrives(
    proxy: Proxy, channel: str
) -> None:
    proxy.hold_on_subscribe = True  # the server's reply to SUBSCRIBE is held back
    reader = make(proxied(proxy), channel=channel, send_timeout_s=3.0, flush_deadline_s=5.0)
    consumer = Consumer()
    reader._register(consumer, stream_events=False)
    lease = await reader._acquire_listener()
    await eventually(lambda: any(b"SUBSCRIBE" in bytes(link.up) for link in proxy.links))
    await asyncio.sleep(0.3)  # a negative claim: slowness can only make it pass vacuously
    assert reader.healthy is False
    assert consumer.gaps == 0
    proxy.release()
    await eventually(lambda: reader.healthy)
    await reader._release_listener(lease)
    await reader.close()


async def test_an_idle_listener_is_pinged_on_its_own_connection_with_fresh_tokens(
    proxy: Proxy, channel: str
) -> None:
    reader = make(proxied(proxy), channel=channel)
    consumer = Consumer()
    lease = await listening(reader, consumer)

    def tokens() -> list[bytes]:
        return re.findall(rb"PING\r\n\$16\r\n([0-9a-f]{16})\r\n", bytes(proxy.links[0].up))

    await eventually(lambda: len(tokens()) >= 3)  # idle: the listener keeps asking
    assert len(set(tokens())) == len(tokens())  # a new token each time
    assert reader.healthy  # and the server answered every one in time
    await reader._release_listener(lease)
    await reader.close()


async def test_a_listener_whose_pings_go_unanswered_degrades_then_recovers(
    proxy: Proxy, channel: str
) -> None:
    reader = make(proxied(proxy), channel=channel)
    consumer = Consumer()
    lease = await listening(reader, consumer)
    gaps = consumer.gaps
    proxy.stall_up = True  # a half-open connection: nothing reaches the server any more
    await eventually(lambda: consumer.gaps > gaps, timeout=15)  # went unhealthy
    proxy.stall_up = False
    await eventually(lambda: reader.healthy and consumer.gaps >= gaps + 2, timeout=15)
    writer = make(channel=channel)
    writer.enqueue(["R1"])
    await writer.flush()
    await eventually(lambda: consumer.ids == ["R1"])
    await reader._release_listener(lease)
    await reader.close()
    await writer.close()


async def test_a_killed_listener_connection_surfaces_a_gap_and_resubscribes_once(
    admin: Any, channel: str, name: str
) -> None:
    reader = make(channel=channel, client_options={"client_name": name})
    consumer = Consumer()
    lease = await listening(reader, consumer)
    assert consumer.gaps == 1
    (victim,) = await clients(admin, name)
    await admin.execute_command("CLIENT", "KILL", "ID", victim["id"])
    # degraded (a gap) and healthy again (a gap): redis-py did not resubscribe behind our back
    await eventually(lambda: consumer.gaps == 3 and reader.healthy)
    await eventually_async(lambda: _count(admin, name), 1)  # redis-py's own reconnect is dropped
    (survivor,) = await clients(admin, name)
    assert survivor["id"] != victim["id"]
    assert await numsub(admin, channel) == 1
    writer = make(channel=channel)
    writer.enqueue(["R9"])
    await writer.flush()
    await eventually(lambda: consumer.ids == ["R9"])
    await reader._release_listener(lease)
    await reader.close()
    await writer.close()


async def test_an_oversized_message_is_dropped_without_a_connection_gap(
    admin: Any, channel: str, name: str
) -> None:
    reader = make(channel=channel, client_options={"client_name": name}, health_check_s=30.0)
    consumer = Consumer()
    lease = await listening(reader, consumer)
    gaps = consumer.gaps
    generation = reader._generation
    (connection,) = await clients(admin, name)
    await admin.publish(channel, "x" * 1_000_000)  # 127x the 7,900-byte bound
    await eventually(lambda: reader.stats().dropped_oversize == 1)
    assert consumer.gaps == gaps and reader.healthy
    assert consumer.hints == [] and reader.stats().bad_messages == 0  # it never reached the decoder
    await eventually_async(lambda: _count(admin, name), 1)
    writer = make(channel=channel)
    writer.enqueue(["R1"])
    await writer.flush()
    await eventually(lambda: consumer.ids == ["R1"])
    assert reader.healthy and consumer.gaps == gaps
    assert reader._generation == generation
    (survivor,) = await clients(admin, name)
    assert survivor["id"] == connection["id"]
    await reader._release_listener(lease)
    await reader.close()
    await writer.close()


@pytest.mark.parametrize("engine", ["sqlite"], indirect=True)
async def test_a_reconnect_gap_resyncs_events_whose_hint_was_lost(
    admin, channel, engine, monkeypatch
):
    # The resync is a Redis behaviour; one store is enough, and the Redis-only
    # CI job has no Postgres.
    updates = make(channel=channel, health_check_s=30.0, send_timeout_s=3.0, flush_deadline_s=3.0)
    original_receive = updates.receive
    reconnecting = asyncio.Event()
    reconnect = asyncio.Event()
    attempts = 0

    async def receive():
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            reconnecting.set()
            await reconnect.wait()
        iterator = original_receive()
        try:
            async for message in iterator:
                yield message
        finally:
            await iterator.aclose()

    monkeypatch.setattr(updates, "receive", receive)
    # Polling is deliberately far away; gap-triggered resync has no random delay.
    monkeypatch.setattr(_follow.random, "uniform", lambda low, high: 0.0)
    state = store_for_engine(engine, live_updates=updates)
    reader = RunStore(state)
    follower = EventFeed(
        reader, poll_interval_s=60.0, poll_max_s=60.0, poll_healthy_s=3600.0
    ).follow("reconnect-run", tenant_id=None)
    try:
        await state.create_run("reconnect-run", "reconnect")
        await state.save_run_event("reconnect-run", event_type="before")
        await follower.start()
        async with asyncio.timeout(10):
            assert (await anext(follower)).event_type == "before"
        await eventually(lambda: updates.healthy)
        gaps = updates.stats().gaps
        updates.abort_listener()
        async with asyncio.timeout(10):
            await reconnecting.wait()
        await eventually_async(lambda: numsub(admin, channel), 0)
        assert not updates.healthy and updates.stats().gaps > gaps
        await state.save_run_event("reconnect-run", event_type="during-gap")
        # The database commit survives; Pub/Sub has no subscriber and cannot replay this hint.
        assert await admin.publish(channel, '["reconnect-run"]') == 0
        reconnect.set()
        await eventually(lambda: updates.healthy and updates.stats().gaps >= gaps + 2)
        async with asyncio.timeout(10):
            assert (await anext(follower)).event_type == "during-gap"
    finally:
        reconnect.set()
        await follower.aclose()
        await reader.close()
        await state.close()


async def test_a_stalled_publish_is_aborted_and_the_next_send_uses_a_fresh_connection(
    proxy: Proxy, channel: str
) -> None:
    reader = make(channel=channel)
    consumer = Consumer()
    lease = await listening(reader, consumer)
    writer = make(proxied(proxy), channel=channel)
    writer.enqueue(["R0"])
    await writer.flush()
    await eventually(lambda: consumer.ids == ["R0"])
    assert len(proxy.links) == 1

    proxy.stall_up = True
    writer.enqueue(["R1"])
    async with asyncio.timeout(10):
        await writer.flush()  # returns by the deadline, not when the server answers
    assert writer.stats().dropped_failed == 1
    async with asyncio.timeout(5):
        await proxy.links[0].client_closed.wait()  # the socket was aborted, not left open

    proxy.stall_up = False
    await asyncio.sleep(0.3)  # leave the sender's backoff
    writer.enqueue(["R2"])
    await writer.flush()
    await eventually(lambda: consumer.ids == ["R0", "R2"])
    assert len(proxy.links) == 2  # a fresh connection
    await reader._release_listener(lease)
    await reader.close()
    await writer.close()


async def test_abort_sender_drops_the_real_socket_that_is_in_use_right_now(
    proxy: Proxy, channel: str
) -> None:
    writer = make(proxied(proxy), channel=channel, send_timeout_s=30.0, flush_deadline_s=60.0)
    await writer.send(['["R0"]'])  # connected through the proxy
    proxy.stall_up = True
    pending = asyncio.ensure_future(writer.send(['["R1"]']))
    await eventually(lambda: b"R1" in bytes(proxy.links[0].up))  # the publish is on the wire
    assert not proxy.links[0].client_closed.is_set() and not pending.done()
    writer.abort_sender()  # synchronous: no await, no timeout
    async with asyncio.timeout(5):
        await proxy.links[0].client_closed.wait()
    with pytest.raises(ConnectionError):
        await pending
    proxy.stall_up = False
    await writer.send(['["R2"]'])  # the next send starts from a fresh connection
    assert len(proxy.links) == 2
    await writer.close()


async def test_abort_listener_drops_the_real_listen_socket_and_the_base_reconnects(
    proxy: Proxy, channel: str
) -> None:
    reader = make(proxied(proxy), channel=channel, health_check_s=30.0)
    consumer = Consumer()
    lease = await listening(reader, consumer)
    reader.abort_listener()  # synchronous
    async with asyncio.timeout(5):
        await proxy.links[0].client_closed.wait()
    # one connection aborted, one new one made by the base: redis-py did not reconnect by itself
    await eventually(lambda: reader.healthy and consumer.gaps == 3)
    assert len(proxy.links) == 2
    await reader._release_listener(lease)
    await reader.close()


async def test_a_stalled_connect_is_bounded_by_the_flush_deadline(
    proxy: Proxy, channel: str
) -> None:
    proxy.stall_up = True  # even the first bytes (the handshake) go nowhere
    writer = make(proxied(proxy), channel=channel)
    writer.enqueue(["R1"])
    async with asyncio.timeout(10):
        await writer.flush()
    assert writer.stats().sent == 0 and writer.stats().dropped_failed == 1
    await writer.close()


# -- connections are not leaked ---------------------------------------------------------------


async def test_every_connection_is_closed_with_its_lease_and_with_the_instance(
    admin: Any, channel: str, name: str
) -> None:
    updates = make(channel=channel, client_options={"client_name": name})
    consumer = Consumer()
    lease = await listening(updates, consumer)
    updates.enqueue(["R1"])
    await updates.flush()
    await eventually(lambda: consumer.ids == ["R1"])
    assert len(await clients(admin, name)) == 2  # the sender and the listener

    await updates._release_listener(lease)
    await eventually_async(lambda: _count(admin, name), 1)  # the sender stays

    await updates.close()
    await eventually_async(lambda: _count(admin, name), 0)


async def _count(admin: Any, name: str) -> int:
    return len(await clients(admin, name))


# -- secrets and errors on the real client ----------------------------------------------------


async def test_a_wrong_password_and_an_unreachable_host_leak_nothing(
    caplog: pytest.LogCaptureFixture, channel: str
) -> None:
    sentinels = ("s3cret-sentinel", "6399")
    parts = urlsplit(REDIS_URL)
    cases = [
        # Server has no auth. Use the configured isolated database.
        f"redis://user:s3cret-sentinel@{parts.hostname}:{parts.port}{parts.path}",
        "redis://:s3cret-sentinel@127.0.0.1:6399/5",  # nothing listens there
    ]
    for url in cases:
        updates = make(url, channel=channel)
        updates.enqueue(["R1"])
        with caplog.at_level(logging.DEBUG):
            await updates.flush()
            iterator = updates.receive()
            with pytest.raises(Exception) as caught:
                await anext(iterator)
            await updates.close()
        error: BaseException | None = caught.value
        while error is not None:
            for sentinel in sentinels:
                assert sentinel not in f"{error!s} {error!r}"
            error = error.__cause__ or error.__context__
        for record in caplog.records:
            text = record_text(record)
            for sentinel in sentinels:
                assert sentinel not in text
        assert sentinels[0] not in repr(updates)
        caplog.clear()


def test_the_real_client_refuses_a_bad_option_without_a_server_and_without_a_loop() -> None:
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates("redis://:s3cret-sentinel@127.0.0.1:6399/5", client_options={"bogus": 1})
    assert "sentinel" not in str(caught.value)
    assert ru.RedisUpdates("rediss://127.0.0.1:6399/5", client_options={"client_name": "x"})
    assert ru.RedisUpdates("unix:///tmp/none.sock")


async def test_oversized_publications_keep_listener_and_count_whole_drops(admin, channel):
    updates = make(channel=channel, health_check_s=30.0)
    consumer = Consumer()
    updates._register(consumer)
    await updates._acquire_listener()
    try:
        await eventually(lambda: updates.healthy)
        generation = updates._generation
        connection = updates._wire.subscription.pubsub.connection
        for size in (7901, 9000, 1_000_000):
            await admin.publish(channel, b"x" * size)
        await admin.publish(channel, b'["good"]')
        await eventually(lambda: "good" in consumer.ids)
        assert updates.stats().dropped_oversize == 3
        assert updates._generation == generation
        assert updates._wire.subscription.pubsub.connection is connection
        assert updates.healthy
    finally:
        await updates.close()
