"""``MySQLStateStore.classify_db_error`` pinned row by row, with no driver and no server.

The override classifies first, then upgrades only a ``connection_lost`` result to
``statement_too_large`` when a bound parameter exceeds the server's packet limit.
The store is built with ``object.__new__`` and a stand-in for the capacity facts.
"""

from __future__ import annotations

import array
import errno
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.exc import OperationalError

from dendrux.runtime.stores._errors import DbError
from dendrux.runtime.stores.sql import mysql as mysql_module
from dendrux.runtime.stores.sql.mysql import MySQLStateStore

if TYPE_CHECKING:
    from collections.abc import Callable

LIMIT = 10
LOST = DbError("connection_lost", None, 2013)
TOO_LARGE = DbError("statement_too_large", None, 2013)
FLAG_TOO_LARGE = DbError("statement_too_large", None, None)
BIG = "x" * (LIMIT + 1)


class DriverError(Exception):
    __module__ = "asyncmy.errors"


def store(limit: int | None | str = LIMIT) -> MySQLStateStore:
    built = object.__new__(MySQLStateStore)
    if limit == "no-facts":
        built._dialect_facts = None  # type: ignore[assignment]
    else:
        built._dialect_facts = SimpleNamespace(max_allowed_packet=limit)  # type: ignore[assignment]
    return built


def lost(params: Any = None) -> OperationalError:
    """A dropped connection (errno 2013) carrying bound parameters."""
    return OperationalError("INSERT ...", params, Exception(2013, "gone"))


def holder(params: Any, **fields: Any) -> Exception:
    error = Exception("holder")
    error.params = params  # type: ignore[attr-defined]
    for key, value in fields.items():
        setattr(error, key, value)
    return error


def invalidated(**fields: Any) -> Exception:
    error = Exception("flagged")
    error.connection_invalidated = True  # type: ignore[attr-defined]
    for key, value in fields.items():
        setattr(error, key, value)
    return error


def cause_chain(length: int, oversize_at: int | None = None) -> Exception:
    """A ``__cause__`` chain of ``length`` errors, the root flagged as a lost connection."""
    errors = [holder(BIG if i == oversize_at else None) for i in range(length)]
    errors[0].connection_invalidated = True  # type: ignore[attr-defined]
    for outer, inner in zip(errors, errors[1:], strict=False):
        outer.__cause__ = inner
    return errors[0]


def context_beats_cause() -> Exception:
    root = invalidated()
    root.__cause__ = holder(BIG)  # popped last
    context_chain = [holder(None) for _ in range(80)]
    for outer, inner in zip(context_chain, context_chain[1:], strict=False):
        outer.__cause__ = inner
    root.__context__ = context_chain[0]  # popped first and eats the bound
    return root


def context_first() -> Exception:
    root = invalidated()
    root.__cause__ = cause_chain(80)
    root.__context__ = holder(BIG)
    return root


def cyclic(oversize: bool) -> Exception:
    a = invalidated()
    b = holder(BIG if oversize else None)
    a.__cause__ = b
    b.__context__ = a
    return a


def spent_budget_then_oversize() -> Exception:
    outer = holder(list(range(mysql_module._JSON_WALK_LIMIT)), connection_invalidated=True)
    outer.__cause__ = holder(BIG)
    return outer


def values(count: int, oversize_index: int | None) -> Exception:
    items: list[Any] = [0] * count
    if oversize_index is not None:
        items[oversize_index] = BIG
    return invalidated(params=items)


ROWS: list[tuple[str, int | None | str, Callable[[], BaseException], DbError]] = [
    # M2 limit lookup and M3 guard
    ("m2-no-facts-unchanged", "no-facts", lambda: lost((BIG,)), LOST),
    ("m2-limit-none-unchanged", None, lambda: lost((BIG,)), LOST),
    (
        "m3-other-kind-never-upgraded",
        LIMIT,
        lambda: OperationalError("x", (BIG,), Exception(9999, "x")),
        DbError("other", None, 9999),
    ),
    (
        "m3-lock-timeout-never-upgraded",
        LIMIT,
        lambda: OperationalError("x", (BIG,), Exception(1205, "x")),
        DbError("lock_timeout", None, 1205),
    ),
    (
        "m3-base-too-large-kept-with-its-own-code",
        LIMIT,
        lambda: OperationalError("x", (BIG,), DriverError(1153, "x")),
        DbError("statement_too_large", None, 1153),
    ),
    ("m3-lost-without-params-unchanged", LIMIT, lambda: lost(None), LOST),
    # M6 size test
    ("m6-str-over-limit", LIMIT, lambda: lost((BIG,)), TOO_LARGE),
    ("m6-str-at-limit-unchanged", LIMIT, lambda: lost(("x" * LIMIT,)), LOST),
    ("m6-str-over-by-utf8-bytes-only", LIMIT, lambda: lost(("é" * 6,)), TOO_LARGE),
    ("m6-str-at-limit-in-utf8-bytes-unchanged", LIMIT, lambda: lost(("é" * 5,)), LOST),
    ("m6-lone-surrogate-bytes", LIMIT, lambda: lost(("\ud800" * 4,)), TOO_LARGE),
    ("m6-bytes-over", LIMIT, lambda: lost((bytes(LIMIT + 1),)), TOO_LARGE),
    ("m6-bytes-at-limit-unchanged", LIMIT, lambda: lost((bytes(LIMIT),)), LOST),
    ("m6-bytearray-over", LIMIT, lambda: lost((bytearray(LIMIT + 1),)), TOO_LARGE),
    ("m6-memoryview-over", LIMIT, lambda: lost((memoryview(bytearray(LIMIT + 1)),)), TOO_LARGE),
    (
        "m6-memoryview-counts-nbytes-not-items",
        LIMIT,
        lambda: lost((memoryview(array.array("I", range(3))),)),
        TOO_LARGE,
    ),
    ("m6-memoryview-at-limit-unchanged", LIMIT, lambda: lost((memoryview(bytes(LIMIT)),)), LOST),
    ("m6-number-and-none-ignored", LIMIT, lambda: lost((10**30, None, 1.5, True)), LOST),
    ("m6-str-params-itself", LIMIT, lambda: lost(BIG), TOO_LARGE),
    (
        "m6-code-of-base-result-kept-none",
        LIMIT,
        lambda: invalidated(params=(BIG,)),
        DbError("statement_too_large", None, None),
    ),
    (
        "m6-code-of-base-result-kept-os-error",
        LIMIT,
        lambda: holder((BIG,), orig=OSError(errno.ECONNRESET, "reset")),
        DbError("statement_too_large", None, errno.ECONNRESET),
    ),
    # M5 parameter walk: what it descends into
    ("m5-dict-value", LIMIT, lambda: lost({"a": BIG}), TOO_LARGE),
    ("m5-nested-dict-list-tuple", LIMIT, lambda: lost({"a": [({"b": (BIG,)},)]}), TOO_LARGE),
    ("m5-executemany-list-of-dicts", LIMIT, lambda: lost([{"a": 1}, {"a": BIG}]), TOO_LARGE),
    ("m5-tuple-of-tuples", LIMIT, lambda: lost(((1,), (BIG,))), TOO_LARGE),
    ("m5-dict-key-not-inspected", LIMIT, lambda: lost({BIG: 1}), LOST),
    ("m5-set-not-descended", LIMIT, lambda: lost({frozenset({BIG})}), LOST),
    ("m5-set-member-not-descended", LIMIT, lambda: lost(({BIG},)), LOST),
    ("m5-iterator-not-descended", LIMIT, lambda: lost((iter([BIG]),)), LOST),
    # M4 error-graph walk: where it looks
    (
        "m4-params-on-orig",
        LIMIT,
        lambda: holder(None, orig=holder((BIG,)), connection_invalidated=True),
        FLAG_TOO_LARGE,
    ),
    (
        "m4-params-on-cause",
        LIMIT,
        lambda: cause_with_params(),
        FLAG_TOO_LARGE,
    ),
    (
        "m4-params-on-context-not-in-base-chain",
        LIMIT,
        lambda: context_with_params(),
        FLAG_TOO_LARGE,
    ),
    ("m4-cycle-with-oversize", LIMIT, lambda: cyclic(True), FLAG_TOO_LARGE),
    (
        "m4-cycle-without-oversize-terminates",
        LIMIT,
        lambda: cyclic(False),
        DbError("connection_lost", None, None),
    ),
    (
        "m4-error-bound-last-visited",
        LIMIT,
        lambda: cause_chain(64, oversize_at=63),
        DbError("statement_too_large", None, None),
    ),
    (
        "m4-error-bound-first-unvisited",
        LIMIT,
        lambda: cause_chain(65, oversize_at=64),
        DbError("connection_lost", None, None),
    ),
    (
        "m4-context-popped-before-cause",
        LIMIT,
        lambda: context_first(),
        DbError("statement_too_large", None, None),
    ),
    (
        "m4-context-chain-can-exhaust-the-bound-before-cause",
        LIMIT,
        lambda: context_beats_cause(),
        DbError("connection_lost", None, None),
    ),
    # M5 value bound (per error)
    (
        "m5-value-bound-last-inspected",
        LIMIT,
        lambda: values(mysql_module._JSON_WALK_LIMIT - 1, mysql_module._JSON_WALK_LIMIT - 2),
        DbError("statement_too_large", None, None),
    ),
    (
        "m5-value-bound-first-uninspected",
        LIMIT,
        lambda: values(mysql_module._JSON_WALK_LIMIT, mysql_module._JSON_WALK_LIMIT - 1),
        DbError("connection_lost", None, None),
    ),
    (
        "m5-value-budget-resets-per-error",
        LIMIT,
        lambda: spent_budget_then_oversize(),
        DbError("statement_too_large", None, None),
    ),
]


def cause_with_params() -> Exception:
    root = invalidated()
    root.__cause__ = holder((BIG,))
    return root


def context_with_params() -> Exception:
    root = invalidated()
    root.__context__ = holder((BIG,))
    return root


@pytest.mark.parametrize(
    ("limit", "build", "expected"), [row[1:] for row in ROWS], ids=[row[0] for row in ROWS]
)
def test_classify_db_error_table(
    limit: int | None | str, build: Callable[[], BaseException], expected: DbError
) -> None:
    assert store(limit).classify_db_error(build()) == expected


def test_walk_bounds_are_the_documented_ones() -> None:
    assert mysql_module._ERROR_WALK_LIMIT == 64
    assert mysql_module._JSON_WALK_LIMIT == 524288
