"""The MySQL statement preflight measures the encoded statement before anything is sent."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

pytest.importorskip("asyncmy")

from asyncmy.converters import escape_item  # noqa: E402

from dendrux.errors import PersistenceError, StatementTooLargeError  # noqa: E402
from dendrux.runtime.stores.sql import _preflight  # noqa: E402

CHARSET = "utf8mb4"
MIB = 1024 * 1024


def real_bytes(value: Any) -> int:
    return len(escape_item(value, CHARSET).encode("utf-8", "surrogateescape"))


@pytest.mark.parametrize(
    "value",
    [
        "plain ascii",
        "quote ' and \" and backslash \\ and newline \n carriage \r nul \x00 sub \x1a",
        "café € \U0001f600",
        "\x00" * 50,
        "\\" * 50,
        "",
    ],
)
def test_a_string_is_measured_exactly_as_the_driver_encodes_it(value: str) -> None:
    assert _preflight.param_bytes(value) == real_bytes(value)


def test_random_strings_are_measured_exactly() -> None:
    rng = random.Random(7)
    alphabet = ["a", "'", '"', "\\", "\n", "\r", "\x00", "\x1a", "é", "€", "\U0001f600"]
    for _ in range(200):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 80)))
        assert _preflight.param_bytes(value) == real_bytes(value)


def test_bytes_are_never_underestimated() -> None:
    rng = random.Random(11)
    for _ in range(200):
        value = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 80)))
        measured = _preflight.param_bytes(value)
        assert real_bytes(value) <= measured <= real_bytes(value) + 9


def test_other_values_cost_a_small_constant() -> None:
    import datetime

    for value in (None, True, 12, 1.5, datetime.datetime.now(datetime.UTC)):
        assert real_bytes(value) <= _preflight.param_bytes(value) <= 64


def test_json_metadata_with_unicode_is_measured_after_ascii_escaping() -> None:
    # The metadata limit counts the JSON as UTF-8 (two bytes per letter). The JSON column
    # then escapes it to ASCII (six bytes per letter) and the driver doubles the backslash:
    # about 3.5 times the limit, which a flat 2x estimate undercounts.
    document = {"k": "\u00e9" * 1000}
    counted = len(json.dumps(document, ensure_ascii=False).encode())
    value = json.dumps(document)
    assert _preflight.param_bytes(value) == real_bytes(value)
    assert _preflight.param_bytes(value) > 3.4 * counted


def test_a_statement_is_its_text_plus_every_parameter() -> None:
    statement = "UPDATE agent_runs SET meta = %s WHERE id = %s"
    parameters = ("it's", "run-1")
    expected = len(statement) + real_bytes("it's") + real_bytes("run-1")
    assert _preflight.statement_bytes(statement, parameters, False) == expected


def test_many_rows_count_as_the_largest_single_statement() -> None:
    statement = "INSERT INTO t (a) VALUES (%s)"
    rows = [("x" * 10,), ("y" * 5000,), ("z" * 20,)]
    assert _preflight.statement_bytes(statement, rows, True) == len(statement) + real_bytes(
        "y" * 5000
    )


class _Engine:
    __slots__ = ("__weakref__",)


class _Conn:
    def __init__(self, capacity: int | None) -> None:
        self.engine = _Engine()
        if capacity is not None:
            _preflight.set_capacity(self.engine, capacity)


def call(capacity: int | None, statement: str, parameters: Any, many: bool = False) -> None:
    _preflight.before_cursor_execute(_Conn(capacity), None, statement, parameters, None, many)


def test_a_statement_that_fits_is_allowed() -> None:
    call(1000, "SELECT %s", ("x" * 900,))


def test_a_statement_over_the_server_limit_is_refused_with_both_sizes() -> None:
    with pytest.raises(StatementTooLargeError) as caught:
        call(1000, "SELECT %s", ("x" * 2000,))
    error = caught.value
    assert isinstance(error, PersistenceError)
    assert error.retryable is False
    assert error.code == "statement_too_large"
    assert error.size == len("SELECT %s") + 2002
    assert error.limit == 1000
    assert str(error.size) in str(error) and "1000" in str(error)
    assert "xxxx" not in str(error)  # never the value


def test_an_engine_without_a_discovered_limit_is_not_checked() -> None:
    call(None, "SELECT %s", ("x" * 10**6,))


def test_the_quick_bound_skips_the_exact_count_for_small_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_: Any) -> int:
        raise AssertionError("exact measurement ran")

    monkeypatch.setattr(_preflight, "param_bytes", forbidden)
    call(16 * MIB, "SELECT %s, %s", ("a" * 1000, 5))
