"""Sentinels cover diagnostics without exposing connection settings."""

import asyncio
import logging

import pytest

from dendrux.db._postgres_updates import PostgresUpdates

SECRET = "RF2_SECRET_SENTINEL"
HOST = "rf2-host-sentinel.invalid"


@pytest.mark.parametrize(
    "url",
    [
        f"postgresql://u:{SECRET}@{HOST}/db?{SECRET}=1",
        f"postgresql://u@localhost/db?password=first&{SECRET}=second",
        f"postgresql://u@localhost/db?password=first&{SECRET}",
    ],
)
def test_unknown_query_keys_do_not_leak(url):
    with pytest.raises(ValueError) as caught:
        PostgresUpdates(url=url)
    assert SECRET not in repr(caught.value)
    assert HOST not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "unsupported option" in str(caught.value)


def test_connection_configuration_is_redacted_in_vars():
    updates = PostgresUpdates(
        url=f"postgresql://u:{SECRET}@{HOST}/db",
        listen_url=f"postgresql://u:{SECRET}@{HOST}/db",
        connect_args={"password": SECRET, "host": HOST},
    )
    for value in [repr(updates), str(updates), repr(vars(updates)), repr(updates.stats())]:
        assert SECRET not in value
        assert HOST not in value


async def test_driver_failure_and_cancellation_diagnostics_are_secret_free(monkeypatch, caplog):
    import asyncpg

    async def fail(*args, **kwargs):
        raise OSError(SECRET + HOST)

    caplog.set_level(logging.DEBUG, logger="dendrux")
    monkeypatch.setattr(asyncpg, "connect", fail)
    updates = PostgresUpdates(url=f"postgresql://u:{SECRET}@{HOST}/db")
    updates.enqueue(["run"])
    await updates.flush()
    lease = await updates.acquire_listener()
    await asyncio.sleep(0.01)
    await updates.release_listener(lease)
    await updates.close()
    for value in [
        repr(vars(updates)),
        repr(updates.stats()),
        repr([vars(r) for r in caplog.records]),
    ]:
        assert SECRET not in value
        assert HOST not in value


async def test_cancelled_connect_does_not_retain_secrets(monkeypatch):
    import asyncpg

    entered = asyncio.Event()

    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncpg, "connect", wait)
    updates = PostgresUpdates(url=f"postgresql://u:{SECRET}@{HOST}/db")
    updates.enqueue(["run"])
    task = asyncio.create_task(updates.flush())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await updates.close()
    assert SECRET not in repr(vars(updates))
    assert HOST not in repr(vars(updates))
