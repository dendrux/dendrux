"""Connection-loss SQLSTATEs and retry wrapper precedence stay explicit."""

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from dendrux.runtime.durability import is_transient_db_error
from dendrux.runtime.stores._errors import classify_error


class DriverError(Exception):
    def __init__(self, code):
        self.sqlstate = code
        super().__init__("private-driver-message")


@pytest.mark.parametrize("code", ["57P04", "57P05"])
@pytest.mark.parametrize(
    "wrapper,expected", [(None, False), (DBAPIError, False), (OperationalError, True)]
)
def test_nonconnection_shutdown_codes_keep_wrapper_precedence(code, wrapper, expected):
    original = DriverError(code)
    error = original if wrapper is None else wrapper(None, None, original)
    assert classify_error(error).kind == "other"
    assert is_transient_db_error(error) is expected


@pytest.mark.parametrize(
    "code,kind,generic_retry,operational_retry",
    [
        ("08000", "connection_lost", True, True),
        ("08006", "connection_lost", True, True),
        ("08P01", "connection_lost", True, True),
        ("57P01", "connection_lost", True, True),
        ("57P02", "connection_lost", True, True),
        ("57P03", "connection_lost", True, True),
        ("53300", "other", True, True),
        ("57014", "other", False, False),
        ("23505", "unique_violation", False, True),
        ("42P01", "missing_table", False, False),
        ("42703", "missing_column", False, False),
        ("40001", "deadlock_or_serialization", True, True),
        ("55P03", "lock_timeout", True, True),
    ],
)
def test_retry_allowances_and_exclusions(code, kind, generic_retry, operational_retry):
    original = DriverError(code)
    assert classify_error(original).kind == kind
    assert is_transient_db_error(DBAPIError(None, None, original)) is generic_retry
    assert is_transient_db_error(OperationalError(None, None, original)) is operational_retry


@pytest.mark.parametrize("retryable", [False, True])
def test_persistence_wrapper_decides_before_classification(retryable):
    from dendrux.errors import PersistenceError

    def unexpected(error):
        raise AssertionError("typed persistence wrapper must decide first")

    assert (
        is_transient_db_error(PersistenceError(retryable=retryable), classify=unexpected)
        is retryable
    )


def test_outer_sqlstate_takes_precedence_over_nested_connection_loss():
    error = DBAPIError(None, None, DriverError("08006"))
    error.sqlstate = "57014"
    assert classify_error(error).code == "57014"
    assert not is_transient_db_error(error)
