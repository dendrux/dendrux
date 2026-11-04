"""PostgresUpdates transport with a fake asyncpg: sender, listener, liveness, cleanup."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from dendrux.db._live_updates import LiveUpdates
from dendrux.db._postgres_updates import PostgresUpdates

asyncpg = pytest.importorskip("asyncpg")

SEND_URL = "postgresql://app:pw@pooler:6432/orbit?sslmode=require"
LISTEN_URL = "postgresql://app:pw@direct:5432/orbit?sslmode=require"
NOTIFY_SQL = "SELECT pg_notify($1, m) FROM unnest($2::text[]) AS m"


class Connection:
    def __init__(self, dsn: str, options: dict[str, Any], hang: str | None) -> None:
        self.dsn, self.options, self.hang = dsn, options, hang
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.listener: Any = None
        self.channel: str | None = None
        self.termination: Any = None
        self.closed = False
        self.terminated = False
        self.ping_fails = False

    async def add_listener(self, channel: str, callback: Any) -> None:
        if self.hang == "listen":
            await asyncio.Event().wait()
        if self.hang == "listen_error":
            raise OSError("driver text must not leak")
        self.channel, self.listener = channel, callback

    def add_termination_listener(self, callback: Any) -> None:
        self.termination = callback

    async def execute(self, sql: str, *args: Any) -> str:
        self.statements.append((sql, args))
        if self.hang == "send":
            await asyncio.Event().wait()
        if sql == "SELECT 1":
            if self.hang == "ping":
                await asyncio.Event().wait()
            if self.ping_fails:
                raise OSError("gone")
        return "SELECT 1"

    async def close(self, **kwargs: Any) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = self.closed = True
        if self.termination is not None:
            self.termination(self)

    def is_closed(self) -> bool:
        return self.closed

    def notify(self, payload: str) -> None:
        self.listener(self, 1, self.channel, payload)


class Driver:
    def __init__(self) -> None:
        self.connections: list[Connection] = []
        self.hang: str | None = None
        self.fail_connect = False

    async def connect(self, dsn: str, **options: Any) -> Connection:
        if self.fail_connect:
            raise OSError("refused")
        connection = Connection(dsn, options, self.hang)
        self.connections.append(connection)
        return connection


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> Driver:
    fake = Driver()
    monkeypatch.setattr(asyncpg, "connect", fake.connect)
    return fake


def _updates(**kw: Any) -> PostgresUpdates:
    kw.setdefault("url", SEND_URL)
    return PostgresUpdates(**kw)


async def test_sender_is_one_lazy_dedicated_connection_with_one_statement(driver: Driver) -> None:
    updates = _updates(listen_url=LISTEN_URL, connect_args={"command_timeout": 3})
    assert driver.connections == []  # lazy
    await updates.send(['["r1"]', '["r2"]'])
    await updates.send(['["r3"]'])
    (sender,) = driver.connections
    assert "pooler:6432" in sender.dsn  # the send URL, never the listen URL
    assert sender.options["statement_cache_size"] == 0
    assert sender.options["command_timeout"] == 3
    assert 0 < sender.options["timeout"] <= updates.send_timeout_s
    assert sender.options["server_settings"]["application_name"] == "dendrux-live-updates"
    assert sender.statements == [
        (NOTIFY_SQL, ("dendrux_events", ['["r1"]', '["r2"]'])),
        (NOTIFY_SQL, ("dendrux_events", ['["r3"]'])),
    ]


async def test_abort_sender_terminates_and_next_send_reconnects(driver: Driver) -> None:
    updates = _updates()
    await updates.send(['["r1"]'])
    updates.abort_sender()
    updates.abort_sender()  # idempotent
    assert driver.connections[0].terminated
    await updates.send(['["r2"]'])
    assert len(driver.connections) == 2 and not driver.connections[1].closed
    await updates.aclose_sender()
    assert driver.connections[1].closed


async def test_stalled_send_is_bounded_by_the_flush_deadline(driver: Driver) -> None:
    driver.hang = "send"
    updates = _updates(send_timeout_s=0.05, flush_deadline_s=0.1)
    updates.enqueue(["r1"])
    async with asyncio.timeout(5):
        await updates.flush()
    assert driver.connections[0].terminated
    assert updates.stats().dropped_failed == 1


async def _first(iterator: Any) -> Any:
    async with asyncio.timeout(5):
        return await anext(iterator)


async def test_listener_yields_none_after_listen_then_raw_payloads(driver: Driver) -> None:
    updates = _updates(listen_url=LISTEN_URL)
    stream = updates.receive()
    assert await _first(stream) is None
    (listener,) = driver.connections
    assert "direct:5432" in listener.dsn and listener.channel == "dendrux_events"
    assert listener.options["statement_cache_size"] == 0
    listener.notify('["r1"]')
    listener.notify("not json")  # raw: the base decodes, the transport never parses
    assert await _first(stream) == '["r1"]'
    assert await _first(stream) == "not json"
    await stream.aclose()
    assert listener.terminated  # receive() closes its own connection


async def test_termination_ends_receive_and_closes(driver: Driver) -> None:
    updates = _updates()
    stream = updates.receive()
    await _first(stream)
    driver.connections[0].terminate()
    with pytest.raises(ConnectionError):
        await _first(stream)


async def test_liveness_ping_runs_only_after_silence_on_the_listen_connection(
    driver: Driver,
) -> None:
    updates = _updates(health_check_s=0.05)
    stream = updates.receive()
    await _first(stream)
    listener = driver.connections[0]
    listener.notify('["r1"]')
    assert await _first(stream) == '["r1"]'
    assert not [s for s, _ in listener.statements if s == "SELECT 1"]
    reader = asyncio.ensure_future(anext(stream))
    async with asyncio.timeout(5):
        while not listener.statements:
            await asyncio.sleep(0.01)
    assert listener.statements[0][0] == "SELECT 1"
    assert len(driver.connections) == 1  # never on another connection
    listener.ping_fails = True
    with pytest.raises(OSError):
        async with asyncio.timeout(5):
            await reader
    assert listener.terminated


async def test_stalled_ping_ends_receive_within_its_deadline(driver: Driver) -> None:
    updates = _updates(health_check_s=0.05, send_timeout_s=0.05, flush_deadline_s=0.05)
    stream = updates.receive()
    await _first(stream)
    driver.connections[0].hang = "ping"
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(5):
            await anext(stream)
    assert driver.connections[0].terminated


@pytest.mark.parametrize("hang", ["listen", "listen_error"])
async def test_setup_failure_terminates_the_half_open_connection(
    driver: Driver, hang: str, caplog: Any
) -> None:
    caplog.set_level(logging.DEBUG)
    driver.hang = hang
    updates = _updates(send_timeout_s=0.05, flush_deadline_s=0.05)
    lease = await updates.acquire_listener()
    async with asyncio.timeout(5):
        while not (driver.connections and driver.connections[0].terminated):
            await asyncio.sleep(0.01)
    assert not updates.healthy
    await updates.release_listener(lease)
    assert all(c.terminated for c in driver.connections)
    assert "driver text" not in caplog.text and "pooler" not in caplog.text


async def test_inbound_overflow_yields_the_exact_gap_object(driver: Driver) -> None:
    updates = _updates(max_inbound=2)
    stream = updates.receive()
    await _first(stream)
    for index in range(5):
        driver.connections[0].notify(f'["r{index}"]')
    assert await _first(stream) is LiveUpdates.GAP
    assert [await _first(stream), await _first(stream)] == ['["r3"]', '["r4"]']
    assert updates.stats().dropped_inbound == 3
    await stream.aclose()


async def test_wire_text_equal_to_a_gap_name_is_never_a_gap(driver: Driver) -> None:
    updates = _updates()
    stream = updates.receive()
    await _first(stream)
    driver.connections[0].notify("GAP")
    assert await _first(stream) == "GAP"
    await stream.aclose()


async def test_abort_listener_drops_the_listen_connection_only(driver: Driver) -> None:
    updates = _updates()
    await updates.send(['["r1"]'])
    stream = updates.receive()
    await _first(stream)
    sender, listener = driver.connections
    updates.abort_listener()
    assert listener.terminated and not sender.closed
    with pytest.raises(ConnectionError):
        await _first(stream)


async def test_listener_lease_delivers_hints_to_consumers_and_reconnects(driver: Driver) -> None:
    updates = _updates(backoff_initial_s=0.01, backoff_max_s=0.02)
    hints: list[tuple[str, ...]] = []

    class Consumer:
        gaps = 0

        def on_hints(self, run_ids: Any) -> None:
            hints.append(tuple(run_ids))

        def on_gap(self) -> None:
            Consumer.gaps += 1

    updates.register(Consumer(), stream_events=False)
    lease = await updates.acquire_listener()
    async with asyncio.timeout(5):
        while not updates.healthy:
            await asyncio.sleep(0.01)
        driver.connections[0].notify('["r1","r2"]')
        while not hints:
            await asyncio.sleep(0.01)
        driver.connections[0].terminate()
        while len(driver.connections) < 2 or not updates.healthy:
            await asyncio.sleep(0.01)
    assert hints == [("r1", "r2")]
    assert Consumer.gaps >= 3  # healthy, degraded, healthy again
    await updates.release_listener(lease)
    assert all(c.terminated for c in driver.connections)


async def test_connect_failure_never_logs_driver_text_or_destination(
    driver: Driver, caplog: Any
) -> None:
    caplog.set_level(logging.DEBUG)
    driver.fail_connect = True
    updates = _updates(backoff_initial_s=0.01, backoff_max_s=0.02)
    updates.enqueue(["r1"])
    await updates.flush()
    lease = await updates.acquire_listener()
    await asyncio.sleep(0.05)
    await updates.release_listener(lease)
    await updates.close()
    text = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    assert "refused" not in text and "pooler" not in text and "pw" not in text
