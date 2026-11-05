"""Transport diagnostics keep safe class names, TLS kind and dependency guidance."""

import logging
import ssl
import sys
import traceback
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux.db._live_updates import _error_kind, _error_type
from dendrux.db._postgres_updates import PostgresUpdates

asyncpg = pytest.importorskip("asyncpg")
SECRET = "PRIVATE_DIAGNOSTIC_SENTINEL"
FAULTS = [
    (ssl.SSLCertVerificationError, "tls"),
    (asyncpg.InvalidPasswordError, "other"),
    (asyncpg.TooManyConnectionsError, "other"),
    (ImportError, "other"),
    (TimeoutError, "timeout"),
    (ConnectionResetError, "connection"),
]


@pytest.mark.parametrize("fault,kind", FAULTS)
@pytest.mark.parametrize("hook", ["send", "receive", "aclose_sender"])
async def test_safe_type_and_kind_survive_normalization(monkeypatch, caplog, fault, kind, hook):
    caplog.set_level(logging.WARNING, logger="dendrux")
    raw = fault(SECRET)
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(side_effect=raw))
    updates = PostgresUpdates(url="postgresql://u@localhost/test")
    connection = Mock(is_closed=Mock(return_value=False), close=AsyncMock(side_effect=raw))
    if hook == "aclose_sender":
        updates._sender = connection
    stream = updates.receive()
    with pytest.raises(Exception) as caught:
        if hook == "receive":
            await anext(stream)
        elif hook == "send":
            await updates.send(["[]"])
        else:
            await updates.aclose_sender()
    error = caught.value
    assert _error_type(error) == fault.__name__
    assert _error_kind(error) == kind
    assert error.__cause__ is None and error.__context__ is None
    assert SECRET not in "".join(traceback.format_exception(error))
    assert SECRET not in repr(vars(error))
    assert SECRET not in repr(getattr(error, "transport_detail", None))
    updates._send_failure(error, 1)
    assert updates.stats().last_error_type == fault.__name__
    assert any(getattr(record, "error_kind", None) == kind for record in caplog.records)
    assert SECRET not in repr([vars(record) for record in caplog.records])
    assert SECRET not in repr(updates.stats())
    if fault is ImportError:
        assert isinstance(error, ImportError)
        assert "asyncpg" in str(error) and "install" in str(error).lower()
    await stream.aclose()


async def test_absent_asyncpg_has_fixed_install_hint(monkeypatch):
    updates = PostgresUpdates(url="postgresql://u@localhost/test")
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    with pytest.raises(ImportError, match="asyncpg") as caught:
        await updates.send(["[]"])
    assert "install" in str(caught.value).lower()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


async def test_early_listener_close_closes_retained_inner_generator(monkeypatch):
    connection = Mock()
    connection.add_listener = AsyncMock()
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=connection))
    updates = PostgresUpdates(url="postgresql://u@localhost/test")
    inner = updates._listen_messages()
    monkeypatch.setattr(updates, "_listen_messages", lambda: inner)
    stream = updates.receive()
    try:
        assert await anext(stream) is None
        await stream.aclose()
        assert updates._listener is None
        connection.terminate.assert_called_once()
        assert inner.ag_frame is None
    finally:
        await inner.aclose()
