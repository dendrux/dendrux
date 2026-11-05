"""Direct transport hooks redact failures and retain cancellation/cleanup semantics."""

import asyncio
import logging
import traceback
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux.db._postgres_updates import PostgresUpdates

asyncpg = pytest.importorskip("asyncpg")
SENTINEL = "PRIVATE_USER_HOST_PASSWORD_SENTINEL"
STAGES = [
    "send_connect",
    "send_execute",
    "receive_connect",
    "receive_register",
    "receive_listen",
    "receive_probe",
]


def setup(monkeypatch, stage, fail):
    connection = Mock()
    connection.is_closed.return_value = False
    connection.execute = AsyncMock()
    connection.add_listener = AsyncMock()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(asyncpg, "connect", connect)
    targets = {
        "send_connect": connect,
        "send_execute": connection.execute,
        "receive_connect": connect,
        "receive_register": connection.add_termination_listener,
        "receive_listen": connection.add_listener,
        "receive_probe": connection.execute,
    }
    targets[stage].side_effect = fail
    return PostgresUpdates(url="postgresql://u@localhost/test", health_check_s=0.1), connection


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "error_type,expected",
    [
        (asyncpg.InvalidPasswordError, RuntimeError),
        (ConnectionError, ConnectionError),
        (TimeoutError, TimeoutError),
        (ValueError, ValueError),
    ],
)
async def test_hook_failures_have_no_driver_text_or_chain(
    monkeypatch, caplog, stage, error_type, expected
):
    caplog.set_level(logging.DEBUG, logger="dendrux")
    raw = error_type(SENTINEL)
    updates, connection = setup(monkeypatch, stage, raw)
    stream = updates.receive()
    with pytest.raises(expected) as caught:
        if stage.startswith("send"):
            await updates.send(["[]"])
        else:
            if stage == "receive_probe":
                assert await anext(stream) is None
            await anext(stream)
    error = caught.value
    assert error is not raw
    assert error.__cause__ is None
    assert error.__context__ is None
    diagnostics = [
        str(error),
        repr(error),
        "".join(traceback.format_exception(error)),
        repr(vars(error)),
        repr(vars(updates)),
        repr(updates.stats()),
        repr([vars(r) for r in caplog.records]),
    ]
    assert all(SENTINEL not in item for item in diagnostics)
    if not stage.endswith("connect"):
        connection.terminate.assert_called_once()
    assert updates._sender is None
    assert updates._listener is None
    await stream.aclose()


@pytest.mark.parametrize("stage", STAGES)
async def test_hook_cancellation_is_preserved_and_cleans_up(monkeypatch, stage):
    cancelled = asyncio.CancelledError()
    updates, connection = setup(monkeypatch, stage, cancelled)
    stream = updates.receive()
    with pytest.raises(asyncio.CancelledError) as caught:
        if stage.startswith("send"):
            await updates.send(["[]"])
        else:
            if stage == "receive_probe":
                await anext(stream)
            await anext(stream)
    assert caught.value is cancelled
    if not stage.endswith("connect"):
        connection.terminate.assert_called_once()
    assert updates._sender is None
    assert updates._listener is None
    await stream.aclose()


async def test_cleanup_error_cannot_replace_sanitized_failure(monkeypatch):
    updates, connection = setup(
        monkeypatch, "receive_listen", asyncpg.InvalidPasswordError(SENTINEL)
    )
    connection.terminate.side_effect = RuntimeError(SENTINEL)
    with pytest.raises(RuntimeError) as caught:
        await anext(updates.receive())
    assert SENTINEL not in str(caught.value)
    assert caught.value.__context__ is None
