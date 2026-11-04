"""No URL, host, port or password reaches a log, an exception or a repr (AGENTS.md section 5).

Sentinels sit in the URL and in the text the stand-in client puts into its own errors, the way
redis-py puts the host and port into connection errors. They are scanned for in log text, log
record fields, exception text and the exception chain (cause and context).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from dendrux.db import _redis_updates as ru
from tests.unit.test_redis_updates_fakes import FakeConnectionError, FakeTimeoutError, FakeWorld

SENTINELS = ("s3cret-sentinel", "sentinel-host", "6380")
URL = "redis://user:s3cret-sentinel@redis.sentinel-host.example:6380/2"
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


def scan(text: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in text, f"{sentinel!r} leaked into {text[:200]!r}"


def scan_records(records: list[logging.LogRecord]) -> None:
    for record in records:
        scan(record.getMessage())
        scan(repr(record.__dict__))


def chain(error: BaseException | None) -> list[BaseException]:
    found: list[BaseException] = []
    while error is not None and error not in found:
        found.append(error)
        error = error.__cause__ or error.__context__
    return found


async def test_failed_sends_through_the_base_log_kinds_only(
    world: FakeWorld, caplog: pytest.LogCaptureFixture
) -> None:
    async def fail(pipeline: Any) -> None:
        raise FakeConnectionError(f"Error 111 connecting to redis.sentinel-host.example:6380 {URL}")

    world.execute_hook = fail
    updates = ru.RedisUpdates(URL, **FAST)
    updates.enqueue(["R1"])
    with caplog.at_level(logging.DEBUG):
        await updates.flush()
        await updates.close()
    assert any(getattr(r, "code", "") == "live_updates_send_failed" for r in caplog.records)
    scan_records(caplog.records)
    scan(repr(updates.stats()))


async def test_a_dying_listener_through_the_base_logs_kinds_only(
    world: FakeWorld, caplog: pytest.LogCaptureFixture
) -> None:
    updates = ru.RedisUpdates(URL, **{**FAST, "health_check_s": 30.0})
    with caplog.at_level(logging.DEBUG):
        lease = await updates.acquire_listener()
        await _until(lambda: updates.healthy)
        world.pubsubs[0].fail(FakeConnectionError(f"closed {URL}"))
        await _until(lambda: len(world.pubsubs) == 2 and updates.healthy)
        await updates.release_listener(lease)
        await updates.close()
    assert any(getattr(r, "code", "") == "live_updates_degraded" for r in caplog.records)
    scan_records(caplog.records)


async def _until(predicate: Any) -> None:
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


@pytest.mark.parametrize(
    "raised",
    [
        FakeConnectionError(f"Error 111 connecting to redis.sentinel-host.example:6380 {URL}"),
        FakeTimeoutError("Timeout reading from redis.sentinel-host.example:6380"),
        OSError(f"[Errno 61] Connect call failed ('10.0.0.1', 6380) {URL}"),
        ValueError(f"bad reply from {URL}"),
    ],
)
async def test_errors_leaving_the_sender_carry_no_chain_to_the_original(
    world: FakeWorld, raised: Exception
) -> None:
    async def fail(pipeline: Any) -> None:
        raise raised

    world.execute_hook = fail
    updates = ru.RedisUpdates(URL, **FAST)
    with pytest.raises(Exception) as caught:
        await updates.send(['["a"]'])
    for error in chain(caught.value):
        scan(f"{error!s} {error!r}")
    assert len(chain(caught.value)) == 1


async def test_errors_leaving_the_listener_carry_no_chain_to_the_original(
    world: FakeWorld,
) -> None:
    updates = ru.RedisUpdates(URL, **{**FAST, "health_check_s": 30.0})
    iterator = updates.receive()
    await anext(iterator)
    world.pubsubs[0].fail(FakeConnectionError(f"closed by redis.sentinel-host.example:6380 {URL}"))
    with pytest.raises(Exception) as caught:
        await anext(iterator)
    for error in chain(caught.value):
        scan(f"{error!s} {error!r}")
    assert len(chain(caught.value)) == 1


@pytest.mark.parametrize("url", ["http://user:s3cret-sentinel@redis.sentinel-host.example:6380/2"])
def test_construction_errors_do_not_echo_the_url(world: FakeWorld, url: str) -> None:
    with pytest.raises(ValueError) as caught:
        ru.RedisUpdates(url)
    for error in chain(caught.value):
        scan(f"{error!s} {error!r}")


def test_the_object_shows_neither_url_nor_password(world: FakeWorld) -> None:
    updates = ru.RedisUpdates(URL, **FAST)
    scan(repr(updates))
    scan(str(updates))
    scan(repr(vars(updates)))
