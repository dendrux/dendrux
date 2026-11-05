"""Direct sender close errors are sanitized after cleanup without masking cancellation."""

import asyncio
import logging
import traceback
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux.db._postgres_updates import PostgresUpdates

SECRET = "PRIVATE_CLOSE_SENTINEL"


@pytest.mark.parametrize("stage", ["is_closed", "close"])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, ConnectionError, ValueError])
async def test_direct_close_sanitizes_every_failure(monkeypatch, caplog, stage, error_type):
    caplog.set_level(logging.DEBUG, logger="dendrux")
    updates = PostgresUpdates(url="postgresql://u@localhost/test")
    connection = Mock()
    connection.is_closed.return_value = False
    connection.close = AsyncMock()
    raw = error_type(SECRET)
    getattr(connection, stage).side_effect = raw
    connection.terminate.side_effect = RuntimeError(SECRET)
    updates._sender = connection
    with pytest.raises(error_type) as caught:
        await updates.aclose_sender()
    error = caught.value
    assert error is not raw
    assert error.__cause__ is None and error.__context__ is None
    assert SECRET not in "".join(traceback.format_exception(error))
    assert SECRET not in repr(vars(error))
    assert SECRET not in repr(getattr(error, "transport_detail", None))
    assert SECRET not in repr([vars(record) for record in caplog.records])
    assert SECRET not in repr(vars(updates))
    assert SECRET not in repr(updates.stats())
    assert updates._sender is None
    connection.terminate.assert_called_once()
    await updates.aclose_sender()


@pytest.mark.parametrize("stage", ["is_closed", "close"])
async def test_close_cancellation_keeps_identity_and_releases_sender(stage):
    updates = PostgresUpdates(url="postgresql://u@localhost/test")
    connection = Mock()
    connection.is_closed.return_value = False
    connection.close = AsyncMock()
    cancelled = asyncio.CancelledError()
    getattr(connection, stage).side_effect = cancelled
    updates._sender = connection
    with pytest.raises(asyncio.CancelledError) as caught:
        await updates.aclose_sender()
    assert caught.value is cancelled
    assert updates._sender is None
    connection.terminate.assert_called_once()


@pytest.mark.parametrize("closed", [False, True])
async def test_successful_or_already_closed_sender_is_released(closed):
    updates = PostgresUpdates(url="postgresql://u@localhost/test")
    connection = Mock()
    connection.is_closed.return_value = closed
    connection.close = AsyncMock()
    updates._sender = connection
    await updates.aclose_sender()
    assert connection.close.await_count == (0 if closed else 1)
    assert updates._sender is None
    connection.terminate.assert_not_called()
