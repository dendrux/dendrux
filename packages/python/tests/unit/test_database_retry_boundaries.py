"""Only guarded SQL retry scopes may consume raw driver failures."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime import durability
from dendrux.runtime.runner import _emit_event
from dendrux.store import MySQLStateStore
from dendrux.types import generate_ulid

asyncpg = pytest.importorskip("asyncpg")


class DriverResetError(ConnectionResetError):
    __module__ = "asyncpg"


@pytest.mark.parametrize("failure_type", [DriverResetError, asyncpg.ConnectionDoesNotExistError])
@pytest.mark.parametrize("scope", ["runner", "critical", "retry"])
async def test_custom_store_raw_failure_is_not_replayed(monkeypatch, failure_type, scope):
    raw = failure_type("PRIVATE_CUSTOM_STORE_SENTINEL")
    committed = []

    class RawStore:
        async def save_run_event(self, run_id, *, event_type, correlation_id=None, data=None):
            committed.append(event_type)
            raise raw

    store = RawStore()
    delay = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", delay)
    run_id = generate_ulid()

    async def write():
        await store.save_run_event(run_id, event_type="tick")

    with pytest.raises(failure_type) as caught:
        if scope == "runner":
            await _emit_event(store, run_id, "tick")
        elif scope == "critical":
            await durability.retry_critical(write, label="custom-store", run_id=run_id)
        else:
            await durability.retry_transient_db(write, label="custom-store", run_id=run_id)
    assert caught.value is raw
    assert committed == ["tick"]
    delay.assert_not_called()


@pytest.mark.parametrize("method", ["create_run", "finalize_run", "get_run"])
async def test_mysql_dependency_hint_survives_store_retry_scope(monkeypatch, method):
    asyncmy = pytest.importorskip("asyncmy")
    namespace = {"__name__": "asyncmy.connection"}
    exec(
        "async def connect(*args, **kwargs):\n"
        "    raise RuntimeError(\"'cryptography' package is required PRIVATE_AUTH_SENTINEL\")\n",
        namespace,
    )
    monkeypatch.setattr(asyncmy, "connect", namespace["connect"])
    engine = create_async_engine("mysql+asyncmy://u:p@127.0.0.1:1/unused_test")
    store = MySQLStateStore(engine)
    run_id = generate_ulid()
    calls = {
        "create_run": lambda: store.create_run(run_id, "dependency"),
        "finalize_run": lambda: store.finalize_run(run_id, status="success"),
        "get_run": lambda: store.get_run(run_id),
    }
    try:
        with pytest.raises(ImportError, match=r"dendrux\[mysql\]") as caught:
            await calls[method]()
        assert "PRIVATE_AUTH_SENTINEL" not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__context__ is None
    finally:
        await store.close()
        await engine.dispose()
