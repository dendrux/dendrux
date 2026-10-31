"""The one Postgres database the conformance suite may run destructive DDL on.

The suite drops and recreates the schema in its database. It therefore accepts only the
exact database authorised for this work, ``dendrux_060_0c_test``, or a single other name the
coordinator passes explicitly in ``DENDRUX_CONFORMANCE_PG_DB``. A URL naming anything else,
including the shared ``dendrux_test`` and any other ``*_test`` name, is refused before any
connection is opened, and the ``_test`` suffix is never appended for it. Postgres cells run
serially (``-n 0``): xdist workers rewrite the name per worker, which is not authorised.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.engine import make_url

from tests._helpers.pg_url import validate_test_pg_url

URL_ENV = "DENDRUX_TEST_PG_URL"
EXTRA_NAME_ENV = "DENDRUX_CONFORMANCE_PG_DB"
AUTHORIZED_DEFAULT = "dendrux_060_0c_test"
REFUSED_ALWAYS = frozenset({"dendrux", "dendrux_test", "postgres", "template0", "template1"})


def authorized_names() -> frozenset[str]:
    extra = os.environ.get(EXTRA_NAME_ENV)
    names = {AUTHORIZED_DEFAULT}
    if extra:
        names.add(extra)
    return frozenset(names)


def authorize(raw_url: str) -> str:
    """Return the database name of ``raw_url`` if it is authorised, else raise.

    Raises ``pytest.UsageError`` (a loud, early failure) for any other database.
    """
    name = make_url(raw_url).database or ""
    allowed = authorized_names()
    if name in REFUSED_ALWAYS or name not in allowed:
        raise pytest.UsageError(
            f"{URL_ENV} names database {name!r}; the conformance suite runs destructive DDL "
            f"only on {sorted(allowed)}. Refusing before any connection."
        )
    # same hardening as every other destructive test entry point (no overrides, valid name)
    validate_test_pg_url(raw_url, append_suffix=False)
    return name
