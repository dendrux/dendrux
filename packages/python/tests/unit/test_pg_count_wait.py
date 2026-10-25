"""Connection observers wait for a condition and fail with the last count."""

import asyncio
from types import SimpleNamespace

import pytest

from tests._helpers import pg_wait


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, sleeps=[])

    async def sleep(delay):
        state.sleeps.append(delay)
        state.now += delay

    monkeypatch.setattr(pg_wait, "time", SimpleNamespace(monotonic=lambda: state.now))
    monkeypatch.setattr(pg_wait, "asyncio", SimpleNamespace(sleep=sleep, wait_for=asyncio.wait_for))
    return state


async def test_waits_until_pg_reports_retirement(clock):
    values = iter([2, 1, 0])

    async def count():
        return next(values)

    assert await pg_wait.wait_pg_count(count, 0) == 0
    assert clock.sleeps == [0.05, 0.05]


async def test_leaked_connections_fail_at_deadline_with_last_observation(clock):
    async def count():
        return 3

    with pytest.raises(AssertionError, match="observed 3, expected 0"):
        await pg_wait.wait_pg_count(count, 0, timeout=0.1)
    assert clock.now == pytest.approx(0.1)
