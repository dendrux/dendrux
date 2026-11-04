"""Fix round 1 for RedisUpdates: URL options, close ownership, bounded input, one detection budget.

Tests named ``real_*`` use the installed redis-py (no server, no network): its URL parser, pool
and protocol parser are the things under test. The rest use the scripted stand-in.
"""

from __future__ import annotations

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
