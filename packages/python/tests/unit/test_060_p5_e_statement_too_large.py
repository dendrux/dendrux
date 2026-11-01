"""The server's own 'packet too large' answers are classified, never retried, and typed."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from dendrux.errors import PersistenceError, StatementTooLargeError
from dendrux.runtime.durability import is_transient_db_error
from dendrux.runtime.stores._errors import classify_error

TOO_LARGE = (1153, "Got a packet bigger than 'max_allowed_packet' bytes")
OVERFLOW = (1301, "Result of json_set() was larger than max_allowed_packet (16777216)")


class DriverError(Exception):
    """Stands in for a genuine MySQL driver diagnostic (asyncmy is optional in this suite)."""

    __module__ = "asyncmy.errors"


def wrapped(code: tuple[int, str]) -> OperationalError:
    return OperationalError("INSERT ...", (), DriverError(*code))


@pytest.mark.parametrize("code", [TOO_LARGE, OVERFLOW])
def test_the_servers_packet_errors_are_their_own_kind(code: tuple[int, str]) -> None:
    error = classify_error(wrapped(code))
    assert error.kind == "statement_too_large"
    assert error.code == code[0]


def test_it_is_found_anywhere_in_the_chain_and_beats_connection_loss() -> None:
    class InterfaceError(Exception):
        pass

    inner = DriverError(*TOO_LARGE)
    outer = InterfaceError(0, "Not connected")  # would classify as connection_lost alone
    outer.__cause__ = inner
    assert classify_error(outer).kind == "statement_too_large"
    assert classify_error(OperationalError("x", (), outer)).kind == "statement_too_large"


def test_it_is_never_transient_even_though_it_is_an_operational_error() -> None:
    error = wrapped(TOO_LARGE)
    assert isinstance(error, OperationalError)
    assert is_transient_db_error(error) is False


def test_the_typed_error_names_the_limit_and_never_a_value() -> None:
    error = StatementTooLargeError(limit=16 * 1024 * 1024)
    assert isinstance(error, PersistenceError)
    assert error.retryable is False
    assert error.code == "statement_too_large"
    assert error.size is None
    assert error.limit == 16 * 1024 * 1024
    assert "16777216" in str(error)
    assert "None" not in str(error)


def test_the_limit_may_be_unknown_and_the_size_best_effort() -> None:
    assert "max_allowed_packet" in str(StatementTooLargeError())
    sized = StatementTooLargeError(limit=100, size=250)
    assert sized.size == 250
    assert "250" in str(sized)


def test_a_real_asyncmy_diagnostic_is_recognised() -> None:
    errors = pytest.importorskip("asyncmy.errors")
    assert classify_error(errors.OperationalError(*TOO_LARGE)).kind == "statement_too_large"
    assert (
        classify_error(OperationalError("x", (), errors.OperationalError(*OVERFLOW))).kind
        == "statement_too_large"
    )


@pytest.mark.parametrize(
    "impostor",
    [OSError(1153, "not a diagnostic"), Exception(1153, "arbitrary"), ValueError(1301)],
)
def test_only_a_driver_diagnostic_counts_not_any_exception_with_that_number(
    impostor: BaseException,
) -> None:
    assert classify_error(impostor).kind != "statement_too_large"
    assert classify_error(OperationalError("x", (), impostor)).kind != "statement_too_large"
