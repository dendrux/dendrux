"""Driver metadata classification never retains vendor values."""

import errno

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime.stores._errors import classify_error
from dendrux.runtime.stores._factory import store_for_engine


@pytest.mark.parametrize(
    "code,kind",
    [
        (1062, "unique_violation"),
        (1213, "deadlock_or_serialization"),
        (1205, "lock_timeout"),
        (2006, "connection_lost"),
        (2013, "connection_lost"),
        (2055, "connection_lost"),
        (9999, "other"),
    ],
)
def test_mysql_error_codes(code, kind):
    result = classify_error(Exception(code, "PRIVATE_SENTINEL"))
    assert result.kind == kind
    assert result.code == code
    assert "PRIVATE_SENTINEL" not in repr(result)


@pytest.mark.parametrize(
    "key",
    [
        "agent_runs.uq_agent_runs_tenant_idempotency_key",
        "uq_agent_runs_tenant_idempotency_key",
        "PRIMARY",
    ],
)
def test_mysql_duplicate_key_names(key):
    result = classify_error(Exception(1062, f"Duplicate entry 'PRIVATE_SENTINEL' for key '{key}'"))
    assert result.constraint == key.rsplit(".", 1)[-1]
    assert "PRIVATE_SENTINEL" not in repr(result)


@pytest.mark.parametrize(
    "message",
    [
        "PRIVATE_SENTINEL",
        "Duplicate entry 'x' for key 'unsafe key'",
        "Duplicate entry 'x' for key 'safe' PRIVATE_SENTINEL",
    ],
)
def test_mysql_unknown_key_fails_closed(message):
    assert classify_error(Exception(1062, message)).constraint is None


def test_mysql_factory_selects_adapter():
    engine = create_async_engine("mysql+asyncmy://root@127.0.0.1/private_test")
    store = store_for_engine(engine)
    assert type(store).__name__ == "MySQLStateStore"
    assert store.capabilities.deployment == "multi_server"
    assert store.capabilities.notify is False


@pytest.mark.parametrize(
    "code,expected",
    [(1062, "unique_violation"), (1213, "deadlock_or_serialization"), (1205, "lock_timeout")],
)
def test_mysql_sqlstate_does_not_shadow_errno(code, expected):
    class DriverError(Exception):
        sqlstate = "23000"

    assert classify_error(DriverError(code, "PRIVATE_SENTINEL")).kind == expected


@pytest.mark.parametrize("shape", ["interface", "os_chain", "invalidated"])
def test_unknown_driver_codes_reach_connection_loss_fallback(shape):
    from asyncmy.errors import InterfaceError
    from sqlalchemy.exc import DBAPIError

    from dendrux.runtime.durability import is_transient_db_error

    if shape == "interface":
        original = InterfaceError(0, "Not connected")
        wrapped = DBAPIError(None, None, original)
    elif shape == "os_chain":
        original = Exception(9999, "PRIVATE_SENTINEL")
        original.__cause__ = OSError(errno.ECONNRESET)
        wrapped = DBAPIError(None, None, original)
    else:
        wrapped = DBAPIError(
            None, None, Exception(9999, "PRIVATE_SENTINEL"), connection_invalidated=True
        )
    assert classify_error(wrapped).kind == "connection_lost"
    assert is_transient_db_error(wrapped)
    assert "PRIVATE_SENTINEL" not in repr(classify_error(wrapped))
