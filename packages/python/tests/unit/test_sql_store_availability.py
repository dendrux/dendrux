"""Database availability metadata and the retry/commit boundary."""

import errno
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import DBAPIError

from dendrux.errors import StoreOutcomeUnknown
from dendrux.runtime.durability import is_transient_db_error, retry_transient_db
from dendrux.runtime.stores._errors import classify_error, error_chain
from dendrux.runtime.stores.sql._commit import _CommitReadback

OS_CODES = [
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EPIPE,
    errno.ECONNABORTED,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.ENETDOWN,
    errno.ENOTCONN,
    errno.ETIMEDOUT,
]


def wrap(error, depth):
    for _ in range(depth):
        error = DBAPIError(None, None, error)
    return error


@pytest.mark.parametrize("code", OS_CODES)
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_transport_errno_is_transient(code, depth):
    error = wrap(OSError(code, "PRIVATE_DB_SENTINEL"), depth)
    assert classify_error(error).kind == "connection_lost"
    assert is_transient_db_error(error)
    assert "PRIVATE_DB_SENTINEL" not in repr(classify_error(error))


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_bare_timeout_is_not_transient(depth):
    assert not is_transient_db_error(wrap(TimeoutError(), depth))


@pytest.mark.parametrize(
    "code", ["08000", "08001", "08003", "08006", "08007", "08P01", "57P01", "57P02", "57P03"]
)
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_postgres_shutdown_states(code, depth):
    asyncpg = pytest.importorskip("asyncpg")
    error = asyncpg.PostgresError.new({"C": code, "M": "PRIVATE_DB_SENTINEL"})
    assert is_transient_db_error(wrap(error, depth))


def test_implicit_context_is_ignored_even_with_cycles():
    inner = ConnectionRefusedError(errno.ECONNREFUSED, "PRIVATE_DB_SENTINEL")
    outer = Exception("wrapper")
    outer.__context__ = inner
    inner.__context__ = outer
    assert error_chain(outer) == [outer]
    assert not is_transient_db_error(wrap(outer, 2))
    outer.__suppress_context__ = True
    assert error_chain(outer) == [outer]


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_sqlalchemy_disconnect_flag(depth):
    error = DBAPIError(None, None, Exception(), connection_invalidated=True)
    assert is_transient_db_error(wrap(error, depth))


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EACCES, "private"),
        sqlite3.OperationalError("unable to open database file"),
        ValueError("no active connection"),
    ],
)
def test_unproven_os_and_sqlite_errors_are_not_transient(error):
    assert not is_transient_db_error(error)


@pytest.mark.parametrize("code", OS_CODES)
async def test_commit_transport_failure_without_proof_never_retries(code):
    session = SimpleNamespace(info={}, rollback=AsyncMock())
    store = SimpleNamespace(classify_db_error=classify_error)
    proof = _CommitReadback(store, session, None)
    attempt = AsyncMock()

    async def commit_attempt():
        await attempt()
        await proof.resolve(DBAPIError(None, None, OSError(code, "private")))

    with pytest.raises(StoreOutcomeUnknown) as caught:
        await retry_transient_db(commit_attempt, label="commit", run_id="r1")
    assert not caught.value.retryable
    assert attempt.await_count == 1


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("kind", ["operational", "pool_timeout", "disconnect"])
def test_wrapped_sqlalchemy_availability_errors(depth, kind):
    from sqlalchemy.exc import DisconnectionError, OperationalError
    from sqlalchemy.exc import TimeoutError as PoolTimeout

    errors = {
        "operational": OperationalError(None, None, Exception()),
        "pool_timeout": PoolTimeout(),
        "disconnect": DisconnectionError(),
    }
    assert is_transient_db_error(wrap(errors[kind], depth))


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("code", [2003, 2006, 2013])
def test_mysql_driver_transport_wrapping(depth, code):
    errors = pytest.importorskip("asyncmy.errors")
    error = errors.OperationalError(code, "PRIVATE_DB_SENTINEL")
    if code == 2003:
        error.__cause__ = ConnectionRefusedError(errno.ECONNREFUSED, "private")
    else:
        error.__context__ = OSError(errno.ECONNRESET, "private")
    assert is_transient_db_error(wrap(error, depth))
    assert classify_error(wrap(error, depth)).kind == "connection_lost"


async def test_refused_write_has_bounded_backoff_and_sanitized_exhaustion(monkeypatch):
    from dendrux.errors import PersistenceError
    from dendrux.runtime import durability

    sleep = AsyncMock()
    monkeypatch.setattr(durability.asyncio, "sleep", sleep)
    failure = DBAPIError(None, None, ConnectionRefusedError(errno.ECONNREFUSED, "private"))
    attempt = AsyncMock(side_effect=failure)
    with pytest.raises(PersistenceError) as caught:
        await retry_transient_db(attempt, label="availability", run_id="r1")
    assert caught.value.retryable
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert attempt.await_count == 4
    assert [call.args[0] for call in sleep.await_args_list] == [0.1, 0.2, 0.4]


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize(
    "kind",
    [
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
    ],
)
def test_connection_exception_types_without_errno(depth, kind):
    assert is_transient_db_error(wrap(kind("private"), depth))


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_exact_errno_less_oserror_is_availability(depth):
    assert is_transient_db_error(wrap(OSError("PRIVATE_DB_SENTINEL"), depth))


def test_implicit_operational_error_does_not_make_logic_failure_transient():
    from sqlalchemy.exc import NoResultFound, OperationalError

    error = NoResultFound()
    error.__context__ = OperationalError(None, None, Exception())
    assert not is_transient_db_error(error)
