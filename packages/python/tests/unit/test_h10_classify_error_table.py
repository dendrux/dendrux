"""Precedence of ``classify_error`` pinned row by row (characterization of the staged passes).

Four ordered passes run over the whole error chain; the first pass that matches
wins for every element of the chain. Each row is a synthetic chain and the exact
``DbError`` it must produce. Pure: no driver, no database.
"""

from __future__ import annotations

import errno
import sqlite3
from typing import Any

import pytest

from dendrux.runtime.stores._errors import DbError, classify_error, error_chain


def _driver(module: str, name: str = "DriverError") -> type[Exception]:
    return type(name, (Exception,), {"__module__": module})


ASYNCMY = _driver("asyncmy.errors")
INTERFACE_ERROR = type("InterfaceError", (Exception,), {"__module__": "asyncmy.errors"})
PLAIN_INTERFACE_ERROR = type("InterfaceError", (Exception,), {"__module__": "app.errors"})
NOT_INTERFACE_ERROR = type("OperationalError", (Exception,), {"__module__": "app.errors"})


def link(*errors: BaseException) -> BaseException:
    """Chain errors through ``__cause__``, first is the outermost."""
    for outer, inner in zip(errors, errors[1:], strict=False):
        outer.__cause__ = inner
    return errors[0]


def attrs(error: BaseException, **fields: Any) -> BaseException:
    for key, value in fields.items():
        setattr(error, key, value)
    return error


def pg(sqlstate: Any = None, pgcode: Any = None, **fields: Any) -> BaseException:
    error = Exception("driver text")
    if sqlstate is not None:
        error.sqlstate = sqlstate  # type: ignore[attr-defined]
    if pgcode is not None:
        error.pgcode = pgcode  # type: ignore[attr-defined]
    return attrs(error, **fields)


def diag(name: Any) -> Any:
    return type("Diag", (), {"constraint_name": name})()


def lite(code: Any, message: str = "x", cls: type[Exception] = sqlite3.OperationalError) -> Any:
    error = cls(message)
    error.sqlite_errorcode = code  # type: ignore[attr-defined]
    return error


def my(code: Any, text: Any = "x", cls: type[Exception] = Exception) -> Any:
    return cls(code, text)


def os_error(number: int) -> OSError:
    return OSError(number, "socket")


def classified_os_error(args_number: int, errno_attr: int | None) -> OSError:
    error = OSError(args_number, "socket")
    error.errno = errno_attr
    return error


def sqlite_rows() -> list[tuple[str, Any, DbError]]:
    unique = sqlite3.SQLITE_CONSTRAINT_UNIQUE
    pk = sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY
    return [
        ("s3-missing-table", lite(1, "no such table: foo"), DbError("missing_table", None, 1)),
        ("s3-missing-column", lite(1, "no such column: foo"), DbError("missing_column", None, 1)),
        (
            "s3-has-no-column-named",
            lite(1, "table t has no column named c"),
            DbError("missing_column", None, 1),
        ),
        (
            "s3-message-before-busy",
            lite(5, "no such table: foo"),
            DbError("missing_table", None, 5),
        ),
        (
            "s3-message-before-unique",
            lite(unique, "no such column: foo"),
            DbError("missing_column", None, unique),
        ),
        ("s3-busy", lite(5), DbError("lock_timeout", None, 5)),
        ("s3-locked", lite(6), DbError("lock_timeout", None, 6)),
        ("s3-busy-snapshot-extended", lite(517), DbError("lock_timeout", None, 517)),
        ("s3-locked-sharedcache-extended", lite(262), DbError("lock_timeout", None, 262)),
        (
            "s3-unique-one-column",
            lite(unique, "UNIQUE constraint failed: runs.id"),
            DbError("unique_violation", "runs.id", unique),
        ),
        (
            "s3-unique-columns",
            lite(unique, "UNIQUE constraint failed: a.b, c.d"),
            DbError("unique_violation", "a.b, c.d", unique),
        ),
        (
            "s3-unique-index-form",
            lite(unique, "UNIQUE constraint failed: index 'ix_runs'"),
            DbError("unique_violation", "ix_runs", unique),
        ),
        (
            "s3-primarykey",
            lite(pk, "UNIQUE constraint failed: runs.id"),
            DbError("unique_violation", "runs.id", pk),
        ),
        (
            "s3-unique-unmatched-message",
            lite(unique, "UNIQUE constraint failed: a.b trailing"),
            DbError("unique_violation", None, unique),
        ),
        (
            "s3-unique-index-unmatched",
            lite(unique, "UNIQUE constraint failed: index 'a.b'"),
            DbError("unique_violation", None, unique),
        ),
        ("s3-constraint-other", lite(19), DbError("other", None, 19)),
        ("s3-foreignkey-other", lite(787), DbError("other", None, 787)),
        ("s3-ioerr-other", lite(10), DbError("other", None, 10)),
        (
            "s3-fake-skips-message-checks-missing-table",
            lite(1, "no such table: foo", cls=Exception),
            DbError("other", None, 1),
        ),
        (
            "s3-fake-skips-message-checks-column",
            lite(1, "table t has no column named c", cls=Exception),
            DbError("other", None, 1),
        ),
        ("s3-fake-busy", lite(5, "x", cls=Exception), DbError("lock_timeout", None, 5)),
        (
            "s3-fake-unique-keeps-message-regex",
            lite(unique, "UNIQUE constraint failed: a.b", cls=Exception),
            DbError("unique_violation", "a.b", unique),
        ),
        ("s3-non-int-code-skipped", lite("5"), DbError("other", None, None)),
    ]


def packet_rows() -> list[tuple[str, Any, DbError]]:
    rows: list[tuple[str, Any, DbError]] = []
    for package in ("asyncmy", "aiomysql", "pymysql", "MySQLdb", "mysql"):
        for number in (1153, 1301):
            rows.append(
                (
                    f"s1-{package}-{number}",
                    _driver(f"{package}.err")(number, "x"),
                    DbError("statement_too_large", None, number),
                )
            )
    base = _driver("asyncmy.errors")
    by_mro = type("AppError", (base,), {"__module__": "myapp.errors"})
    rows.append(
        ("s1-driver-found-by-mro", by_mro(1153, "x"), DbError("statement_too_large", None, 1153))
    )
    rows.append(
        (
            "s1-other-package-is-not-a-driver",
            _driver("myapp.errors")(1153, "x"),
            DbError("other", None, 1153),
        )
    )
    rows.append(("s1-impostor-exception", Exception(1153, "x"), DbError("other", None, 1153)))
    rows.append(
        (
            "s1-os-error-excluded-falls-to-s4-other",
            OSError(1153, "x"),
            DbError("other", None, 1153),
        )
    )
    rows.append(
        (
            "s1-only-first-arg-counts",
            base("x", 1153),
            DbError("other", None, None),
        )
    )
    rows.append(
        (
            "s1-found-deep-beats-outer-interface-error",
            link(INTERFACE_ERROR(0, "gone"), base(1153, "x")),
            DbError("statement_too_large", None, 1153),
        )
    )
    rows.append(
        (
            "o1-s1-beats-socket-reset",
            link(os_error(errno.ECONNRESET), base(1301, "x")),
            DbError("statement_too_large", None, 1301),
        )
    )
    rows.append(
        (
            "o2-s1-beats-sqlstate-element",
            link(pg("23505", constraint_name="c"), base(1153, "x")),
            DbError("statement_too_large", None, 1153),
        )
    )
    rows.append(
        (
            "o2-s1-beats-sqlstate-element-inner",
            link(base(1153, "x"), pg("23505")),
            DbError("statement_too_large", None, 1153),
        )
    )
    rows.append(
        (
            "o1-s1-beats-sqlite",
            link(lite(19), base(1153, "x")),
            DbError("statement_too_large", None, 1153),
        )
    )
    return rows


def sqlstate_rows() -> list[tuple[str, Any, DbError]]:
    rows: list[tuple[str, Any, DbError]] = [
        (
            "s2-unique-with-constraint",
            pg("23505", constraint_name="uq_x"),
            DbError("unique_violation", "uq_x", "23505"),
        ),
        (
            "s2-unique-diag-constraint",
            pg("23505", diag=diag("uq_diag")),
            DbError("unique_violation", "uq_diag", "23505"),
        ),
        (
            "s2-constraint-name-before-diag",
            pg("23505", constraint_name="uq_a", diag=diag("uq_b")),
            DbError("unique_violation", "uq_a", "23505"),
        ),
        (
            "s2-unique-no-constraint",
            pg("23505"),
            DbError("unique_violation", None, "23505"),
        ),
        (
            "s2-nonstr-constraint-is-kept-quirk",
            pg("23505", constraint_name=7),
            DbError("unique_violation", 7, "23505"),  # type: ignore[arg-type]
        ),
        (
            "s2-nonstr-constraint-shadows-diag-quirk",
            pg("23505", constraint_name=7, diag=diag("uq_d")),
            DbError("unique_violation", 7, "23505"),  # type: ignore[arg-type]
        ),
        (
            "s2-constraint-from-other-element",
            link(pg("23505"), pg(constraint_name="uq_inner")),
            DbError("unique_violation", "uq_inner", "23505"),
        ),
        (
            "s2-constraint-from-earlier-element",
            link(pg(constraint_name="uq_outer"), pg("23505")),
            DbError("unique_violation", "uq_outer", "23505"),
        ),
        (
            "s2-constraint-dropped-for-non-unique",
            pg("40001", constraint_name="uq_x"),
            DbError("deadlock_or_serialization", None, "40001"),
        ),
        ("s2-pgcode-source", pg(pgcode="23505"), DbError("unique_violation", None, "23505")),
        (
            "s2-sqlstate-before-pgcode",
            pg("40001", pgcode="23505"),
            DbError("deadlock_or_serialization", None, "40001"),
        ),
        (
            "s2-empty-sqlstate-falls-to-pgcode",
            pg("", pgcode="42P01"),
            DbError("missing_table", None, "42P01"),
        ),
        (
            "s2-nonstr-sqlstate-skipped",
            pg(23505),
            DbError("other", None, None),
        ),
        (
            "s2-nonstr-sqlstate-skipped-then-inner",
            link(pg(23505), pg("55P03")),
            DbError("lock_timeout", None, "55P03"),
        ),
        (
            "s2-int-first-arg-skips-sqlstate-element",
            link(attrs(Exception(9999, "x"), sqlstate="40001"), pg("55P03")),
            DbError("lock_timeout", None, "55P03"),
        ),
        (
            "s2-int-first-arg-sqlstate-only-then-other",
            attrs(Exception(9999, "x"), sqlstate="40001"),
            DbError("other", None, 9999),
        ),
        (
            "o3-errno-1062-beats-sqlstate",
            attrs(Exception(1062, "Duplicate entry 'a' for key 'uq_k'"), sqlstate="23000"),
            DbError("unique_violation", "uq_k", 1062),
        ),
        (
            "o3-errno-1213-beats-sqlstate",
            attrs(Exception(1213, "x"), sqlstate="40001"),
            DbError("deadlock_or_serialization", None, 1213),
        ),
        (
            "o3-errno-1205-beats-sqlstate",
            attrs(Exception(1205, "x"), sqlstate="HY000"),
            DbError("lock_timeout", None, 1205),
        ),
        (
            "o3-outer-sqlstate-beats-nested-connection-loss",
            link(pg("57014"), pg("08006")),
            DbError("other", None, "57014"),
        ),
        (
            "o3-outer-connection-loss-beats-nested-unique",
            link(pg("08006"), pg("23505")),
            DbError("connection_lost", None, "08006"),
        ),
        (
            "o4-s2-beats-sqlite-inner",
            link(pg("40001"), lite(5)),
            DbError("deadlock_or_serialization", None, "40001"),
        ),
        (
            "o4-s2-beats-sqlite-outer",
            link(lite(5), pg("40001")),
            DbError("deadlock_or_serialization", None, "40001"),
        ),
    ]
    simple = {
        "42P01": "missing_table",
        "42703": "missing_column",
        "40P01": "deadlock_or_serialization",
        "40001": "deadlock_or_serialization",
        "55P03": "lock_timeout",
        "08000": "connection_lost",
        "08006": "connection_lost",
        "08P01": "connection_lost",
        "57P01": "connection_lost",
        "57P02": "connection_lost",
        "57P03": "connection_lost",
        "57P04": "other",
        "57P05": "other",
        "57014": "other",
        "53300": "other",
        "42601": "other",
        "22P02": "other",
        "22P05": "other",
        "23503": "other",
    }
    for code, kind in simple.items():
        rows.append((f"s2-code-{code}", pg(code), DbError(kind, None, code)))  # type: ignore[arg-type]
    return rows


def mysql_rows() -> list[tuple[str, Any, DbError]]:
    rows: list[tuple[str, Any, DbError]] = []
    for name, number in (
        ("econnreset", errno.ECONNRESET),
        ("econnaborted", errno.ECONNABORTED),
        ("epipe", errno.EPIPE),
        ("enotconn", errno.ENOTCONN),
        ("etimedout", errno.ETIMEDOUT),
    ):
        rows.append((f"s4-os-{name}", os_error(number), DbError("connection_lost", None, number)))
    rows += [
        (
            "s4-os-errno-attr-only",
            attrs(OSError("x"), errno=errno.EPIPE),
            DbError("connection_lost", None, None),
        ),
        (
            "s4-os-errno-attr-beats-args",
            classified_os_error(5, errno.ECONNRESET),
            DbError("connection_lost", None, 5),
        ),
        (
            "s4-os-args-number-ignored-when-errno-attr-differs",
            classified_os_error(errno.ECONNRESET, errno.ENOENT),
            DbError("other", None, errno.ECONNRESET),
        ),
        (
            "s4-os-unlisted-errno",
            os_error(errno.ENOENT),
            DbError("other", None, errno.ENOENT),
        ),
        (
            "s4-os-without-number-skipped",
            OSError("socket"),
            DbError("other", None, None),
        ),
        (
            "s4-non-os-error-with-reset-number-is-unknown",
            my(errno.ECONNRESET, "x"),
            DbError("other", None, errno.ECONNRESET),
        ),
        (
            "s4-interface-error-zero",
            my(0, "Not connected", INTERFACE_ERROR),
            DbError("connection_lost", None, 0),
        ),
        (
            "s4-interface-error-zero-any-module",
            my(0, "Not connected", PLAIN_INTERFACE_ERROR),
            DbError("connection_lost", None, 0),
        ),
        (
            "s4-zero-on-other-class-is-unknown",
            my(0, "x", NOT_INTERFACE_ERROR),
            DbError("other", None, 0),
        ),
        (
            "s4-interface-error-nonzero-is-unknown",
            my(5, "x", INTERFACE_ERROR),
            DbError("other", None, 5),
        ),
        (
            "s4-1062-primary",
            my(1062, "Duplicate entry '1' for key 'runs.PRIMARY'"),
            DbError("unique_violation", "PRIMARY", 1062),
        ),
        (
            "s4-1062-bare-key",
            my(1062, "Duplicate entry '1' for key 'uq_runs'"),
            DbError("unique_violation", "uq_runs", 1062),
        ),
        (
            "s4-1062-backticks",
            my(1062, "Duplicate entry '1' for key `db.uq_runs`"),
            DbError("unique_violation", "uq_runs", 1062),
        ),
        (
            "s4-1062-triple-dotted-key-unmatched",
            my(1062, "Duplicate entry '1' for key 'db.runs.uq_runs'"),
            DbError("unique_violation", None, 1062),
        ),
        (
            "s4-1062-no-key",
            my(1062, "Duplicate entry '1'"),
            DbError("unique_violation", None, 1062),
        ),
        (
            "s4-1062-key-not-terminal",
            my(1062, "Duplicate entry '1' for key 'uq_runs' extra"),
            DbError("unique_violation", None, 1062),
        ),
        ("s4-1062-single-arg", Exception(1062), DbError("unique_violation", None, 1062)),
        (
            "s4-1062-non-str-text",
            my(1062, b"for key 'uq'"),
            DbError("unique_violation", None, 1062),
        ),
        ("s4-1146", my(1146), DbError("missing_table", None, 1146)),
        ("s4-1054", my(1054), DbError("missing_column", None, 1054)),
        ("s4-1213", my(1213), DbError("deadlock_or_serialization", None, 1213)),
        ("s4-1205", my(1205), DbError("lock_timeout", None, 1205)),
    ]
    for number in (1927, 2006, 2013, 2014, 2045, 2055, 4031):
        rows.append((f"s4-loss-{number}", my(number), DbError("connection_lost", None, number)))
    rows += [
        ("s4-unknown-code", my(9999), DbError("other", None, 9999)),
        ("s4-no-args", Exception(), DbError("other", None, None)),
        ("s4-text-only", Exception("boom"), DbError("other", None, None)),
        (
            "s4-connection-invalidated-flag",
            attrs(Exception("x"), connection_invalidated=True),
            DbError("connection_lost", None, None),
        ),
        (
            "s4-connection-invalidated-falsy",
            attrs(Exception("x"), connection_invalidated=False),
            DbError("other", None, None),
        ),
        (
            "s4-invalidated-carries-unknown-code",
            link(my(9999), attrs(Exception("x"), connection_invalidated=True)),
            DbError("connection_lost", None, 9999),
        ),
        (
            "s4-invalidated-carries-zero-code",
            link(
                my(0, "x", NOT_INTERFACE_ERROR), attrs(Exception("x"), connection_invalidated=True)
            ),
            DbError("connection_lost", None, 0),
        ),
        (
            "s4-match-beats-invalidated-flag",
            attrs(my(1213), connection_invalidated=True),
            DbError("deadlock_or_serialization", None, 1213),
        ),
        (
            "s4-flag-on-outer-unknown-inner",
            link(attrs(Exception("x"), connection_invalidated=True), my(7777)),
            DbError("connection_lost", None, 7777),
        ),
        (
            "o7-outer-1062-beats-inner-reset",
            link(my(1062, "dup for key 'k'"), os_error(errno.ECONNRESET)),
            DbError("unique_violation", "k", 1062),
        ),
        (
            "o7-outer-reset-beats-inner-1062",
            link(os_error(errno.ECONNRESET), my(1062, "dup for key 'k'")),
            DbError("connection_lost", None, errno.ECONNRESET),
        ),
        (
            "o7-matched-inner-beats-unknown-outer",
            link(my(9999), my(2013)),
            DbError("connection_lost", None, 2013),
        ),
        (
            "o7-first-unknown-code-wins",
            link(my(9999), my(8888)),
            DbError("other", None, 9999),
        ),
        (
            "o7-first-unknown-zero-is-kept",
            link(my(0, "x", NOT_INTERFACE_ERROR), my(7)),
            DbError("other", None, 0),
        ),
        (
            "o7-text-element-skipped-for-code",
            link(Exception("x"), my(9999)),
            DbError("other", None, 9999),
        ),
        (
            "o5-sqlite-beats-os-reset-inner",
            link(lite(19), os_error(errno.ECONNRESET)),
            DbError("other", None, 19),
        ),
        (
            "o5-sqlite-beats-os-reset-outer",
            link(os_error(errno.ECONNRESET), lite(19)),
            DbError("other", None, 19),
        ),
        (
            "o5-sqlite-beats-mysql-errno",
            link(my(1062, "dup for key 'k'"), lite(5)),
            DbError("lock_timeout", None, 5),
        ),
        (
            "o5-sqlite-beats-invalidated",
            link(attrs(Exception("x"), connection_invalidated=True), lite(10)),
            DbError("other", None, 10),
        ),
        (
            "o6-first-sqlite-element-wins-even-other",
            link(lite(10), lite(5)),
            DbError("other", None, 10),
        ),
    ]
    return rows


def chain_rows() -> list[tuple[str, Any, DbError]]:
    orig_wins = attrs(Exception("w"), orig=my(1213))
    orig_wins.__cause__ = my(1205)
    non_exception_orig = attrs(Exception("w"), orig="not an exception")
    non_exception_orig.__cause__ = my(1205)
    context_only = Exception("w")
    context_only.__context__ = my(1213)
    cycle_a, cycle_b = Exception("a"), my(1146)
    cycle_a.__cause__ = cycle_b
    cycle_b.__cause__ = cycle_a
    deep = link(Exception("a"), Exception("b"), Exception("c"), pg("40001"))
    return [
        ("chain-orig-before-cause", orig_wins, DbError("deadlock_or_serialization", None, 1213)),
        (
            "chain-non-exception-orig-uses-cause",
            non_exception_orig,
            DbError("lock_timeout", None, 1205),
        ),
        ("chain-context-ignored", context_only, DbError("other", None, None)),
        ("chain-cycle-terminates", cycle_a, DbError("missing_table", None, 1146)),
        ("chain-deep-cause", deep, DbError("deadlock_or_serialization", None, "40001")),
    ]


ROWS = packet_rows() + sqlstate_rows() + sqlite_rows() + mysql_rows() + chain_rows()


@pytest.mark.parametrize(("exc", "expected"), [row[1:] for row in ROWS], ids=[r[0] for r in ROWS])
def test_classify_error_precedence_table(exc: BaseException, expected: DbError) -> None:
    assert classify_error(exc) == expected


def test_table_covers_every_stage_and_overlap() -> None:
    ids = {row[0] for row in ROWS}
    for prefix in ("s1-", "s2-", "s3-", "s4-", "o1-", "o2-", "o3-", "o4-", "o5-", "o6-", "o7-"):
        assert any(i.startswith(prefix) for i in ids), prefix
    assert len(ids) == len(ROWS)


def test_error_chain_follows_orig_then_cause_once() -> None:
    inner = Exception("inner")
    outer = attrs(Exception("outer"), orig=inner)
    outer.__cause__ = Exception("ignored")
    assert error_chain(outer) == [outer, inner]
    looping = Exception("loop")
    looping.__cause__ = looping
    assert error_chain(looping) == [looping]
