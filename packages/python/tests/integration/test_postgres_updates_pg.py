"""PostgresUpdates on the private Postgres test database: real NOTIFY and LISTEN.

Each test uses its own channel (channels are scoped to one database, and every
xdist worker has its own database), and names its connections so it can find
and kill exactly its own backends.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import pytest
from sqlalchemy.engine import make_url

from dendrux.db._postgres_updates import PostgresUpdates
from dendrux.types import generate_ulid
from tests.integration.test_060_notifier_liveness import Proxy

asyncpg = pytest.importorskip("asyncpg")


async def _until(predicate: Any, seconds: float = 10) -> None:
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.005)


def _dsn(url: str) -> str:
    return make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


class Consumer:
    def __init__(self) -> None:
        self.hints: list[str] = []
        self.gaps = 0

    def on_hints(self, run_ids: Any) -> None:
        self.hints.extend(run_ids)

    def on_gap(self) -> None:
        self.gaps += 1


def _pair(url: str, name: str, **kw: Any) -> tuple[PostgresUpdates, PostgresUpdates]:
    channel = "p6b_" + generate_ulid().lower()
    writer = PostgresUpdates(
        url=url, channel=channel, connect_args={"server_settings": {"application_name": name + "w"}}
    )
    reader = PostgresUpdates(
        url=url,
        channel=channel,
        connect_args={"server_settings": {"application_name": name + "r"}},
        **kw,
    )
    return writer, reader


async def _backends(url: str, name: str) -> int:
    connection = await asyncpg.connect(_dsn(url))
    try:
        return int(
            await connection.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1", name
            )
        )
    finally:
        await connection.close()


async def _kill(url: str, name: str) -> int:
    connection = await asyncpg.connect(_dsn(url))
    try:
        return int(
            await connection.fetchval(
                "SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                "WHERE application_name = $1",
                name,
            )
        )
    finally:
        await connection.close()


async def test_real_notify_reaches_a_leased_listener_on_its_own_connections(
    pg_test_url: str,
) -> None:
    name = "p6b" + generate_ulid()[-8:]
    writer, reader = _pair(pg_test_url, name)
    consumer = Consumer()
    reader.register(consumer, stream_events=False)
    lease = await reader.acquire_listener()
    try:
        await _until(lambda: reader.healthy)
        runs = [generate_ulid() for _ in range(600)]  # two messages, one statement
        writer.enqueue(runs)
        await writer.flush()
        assert writer.stats().sent == 600 and writer.stats().pending == 0
        await _until(lambda: set(consumer.hints) == set(runs))
        assert await _backends(pg_test_url, name + "w") == 1  # the sender, nothing pooled
        assert await _backends(pg_test_url, name + "r") == 1  # the listener only
    finally:
        await reader.release_listener(lease)
        await writer.close()
        await reader.close()
    async with asyncio.timeout(10):
        while await _backends(pg_test_url, name + "w") + await _backends(pg_test_url, name + "r"):
            await asyncio.sleep(0.05)


async def test_killed_listener_backend_degrades_reconnects_and_delivers_again(
    pg_test_url: str, caplog: Any
) -> None:
    caplog.set_level(logging.INFO)
    name = "p6b" + generate_ulid()[-8:]
    writer, reader = _pair(pg_test_url, name, backoff_initial_s=0.05, backoff_max_s=0.1)
    consumer = Consumer()
    reader.register(consumer, stream_events=False)
    lease = await reader.acquire_listener()
    try:
        await _until(lambda: reader.healthy)
        generation = reader.generation
        assert await _kill(pg_test_url, name + "r") == 1
        await _until(lambda: reader.generation >= generation + 2 and reader.healthy)
        run = generate_ulid()
        writer.enqueue([run])
        await writer.flush()
        await _until(lambda: run in consumer.hints)
        codes = [getattr(r, "code", None) for r in caplog.records]
        assert codes.count("live_updates_degraded") == 1
        assert codes.count("live_updates_recovered") == 2
        assert pg_test_url.split("@")[-1] not in caplog.text
    finally:
        await reader.release_listener(lease)
        await writer.close()
        await reader.close()


async def test_killed_sender_backend_costs_at_most_one_batch_then_recovers(
    pg_test_url: str,
) -> None:
    name = "p6b" + generate_ulid()[-8:]
    writer, reader = _pair(pg_test_url, name, backoff_initial_s=0.01, backoff_max_s=0.02)
    writer._backoff = writer._listener_backoff = 0.01
    consumer = Consumer()
    reader.register(consumer, stream_events=False)
    lease = await reader.acquire_listener()
    try:
        await _until(lambda: reader.healthy)
        writer.enqueue([generate_ulid()])
        await writer.flush()
        assert await _kill(pg_test_url, name + "w") == 1
        sent = [generate_ulid() for _ in range(3)]
        for run in sent:
            writer.enqueue([run])
            await writer.flush()
            await asyncio.sleep(0.03)  # past any backoff
        async with asyncio.timeout(10):  # stranded ids go out with a later flush
            while writer.stats().pending:
                await asyncio.sleep(0.03)
                await writer.flush()
        assert writer.stats().dropped_failed <= 1
        await _until(lambda: sent[-1] in consumer.hints)
        assert await _backends(pg_test_url, name + "w") == 1  # one fresh sender connection
    finally:
        await reader.release_listener(lease)
        await writer.close()
        await reader.close()


async def test_unreachable_listener_never_turns_healthy_and_leaves_no_backend(
    pg_test_url: str, caplog: Any
) -> None:
    caplog.set_level(logging.DEBUG)
    missing = make_url(pg_test_url).set(database="p6b_missing_database_test")
    reader = PostgresUpdates(
        url=pg_test_url, listen_url=missing, backoff_initial_s=0.05, backoff_max_s=0.1
    )
    lease = await reader.acquire_listener()
    await asyncio.sleep(0.3)
    assert not reader.healthy
    await reader.release_listener(lease)
    await reader.close()
    text = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    assert "p6b_missing_database_test" not in text and "root" not in text


async def test_frozen_tcp_path_is_detected_by_the_listen_ping_and_recovers(
    pg_test_url: str, caplog: Any
) -> None:
    caplog.set_level(logging.INFO)
    parsed = make_url(pg_test_url)
    async with Proxy(parsed.host, parsed.port or 5432) as proxy:
        through = parsed.set(
            host="127.0.0.1", port=proxy.port_number, query={**parsed.query, "sslmode": "disable"}
        )
        reader = PostgresUpdates(
            url=pg_test_url,
            listen_url=through,
            health_check_s=0.2,
            send_timeout_s=0.25,
            flush_deadline_s=0.25,
            backoff_initial_s=0.05,
            backoff_max_s=0.1,
        )
        lease = await reader.acquire_listener()
        try:
            await _until(lambda: reader.healthy)
            generation = reader.generation
            proxy.gate.clear()  # bytes stop flowing; no socket closes
            await _until(lambda: not reader.healthy)
            proxy.gate.set()
            await _until(lambda: reader.healthy and reader.generation >= generation + 2)
        finally:
            proxy.gate.set()
            await reader.release_listener(lease)
            await reader.close()


async def test_pgbouncer_sends_through_the_pool_and_listens_direct(pg_test_url: str) -> None:
    pooled = os.environ.get("DENDRUX_TEST_PGBOUNCER_URL")
    if not pooled:
        pytest.skip("DENDRUX_TEST_PGBOUNCER_URL not set; PgBouncer is not installed locally")
    assert make_url(pooled).database == make_url(pg_test_url).database
    channel = "p6b_" + generate_ulid().lower()
    writer = PostgresUpdates(url=pooled, channel=channel)
    reader = PostgresUpdates(url=pooled, listen_url=pg_test_url, channel=channel)
    consumer = Consumer()
    reader.register(consumer, stream_events=False)
    lease = await reader.acquire_listener()
    try:
        await _until(lambda: reader.healthy)
        for _ in range(5):  # several sends may land on different pooled backends
            run = generate_ulid()
            writer.enqueue([run])
            await writer.flush()
            await _until(lambda run=run: run in consumer.hints)
        assert writer.stats().dropped_failed == 0
    finally:
        await reader.release_listener(lease)
        await writer.close()
        await reader.close()
