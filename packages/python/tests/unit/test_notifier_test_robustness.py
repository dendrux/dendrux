"""P2-9: controlled CI worker URLs and delayed server-side socket cleanup."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dendrux.errors import NotifierUnavailableError
from tests.integration import test_060_notifier_pg as pg_tests
from tests.integration import test_060_notifier_pgbouncer as pool_tests


@pytest.mark.parametrize("session_pool", [False, True])
async def test_p2_9_pgbouncer_gate_accepts_matching_worker_database(monkeypatch, session_pool):
    url = "postgresql+asyncpg://postgres:root@localhost:5432/unit_worker_test"
    key = "DENDRUX_TEST_PGBOUNCER_SESSION_URL" if session_pool else "DENDRUX_TEST_PGBOUNCER_URL"
    monkeypatch.setenv(key, url)

    class ValidatedError(Exception):
        pass

    def validated(*args, **kwargs):
        raise ValidatedError

    monkeypatch.setattr(pool_tests, "PostgresEventNotifier", validated)
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    with pytest.raises(ValidatedError):
        await pool_tests.test_n4_controlled_delivery_with_backend_reassignment(
            url, engine, None, session_pool, monkeypatch
        )


async def test_p2_9_failed_start_waits_for_backend_disappearance(monkeypatch):
    class Notifier:
        _connection = _task = None

        async def start(self):
            raise NotifierUnavailableError("self_test")

    class Engine:
        dialect = SimpleNamespace(name="postgresql")
        reads = 0

        @asynccontextmanager
        async def connect(self):
            yield self

        async def execute(self, statement):
            self.reads += 1
            return SimpleNamespace(scalar_one=lambda: int(self.reads < 3))

    engine = Engine()
    monkeypatch.setattr(pg_tests, "PostgresEventNotifier", lambda *a, **kw: Notifier())
    await pg_tests.test_n7_failed_self_test_releases_backend("unused", engine, monkeypatch)
    assert engine.reads == 3
