"""Per-worker Postgres database names for ``pytest -n``."""

from __future__ import annotations

import re

from tests._helpers.pg_url import (
    MAX_IDENTIFIER_LENGTH,
    TEST_SUFFIX,
    UnsafeTestDatabaseUrl,
    render,
    validate_test_pg_url,
)

_WORKER_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


def worker_scoped_pg_url(url: str, worker_id: str | None) -> str:
    """Return ``url`` pointing at a database private to the xdist worker.

    ``dendrux_test`` becomes ``dendrux_gw0_test``. With no worker id (a serial run) the
    URL is unchanged, and so is a URL already scoped to this worker. The URL is validated
    first (no database-overriding query option, an identifier PostgreSQL accepts), and the
    result always ends in ``_test``, as the integration conftest requires before it will
    run destructive tests.
    """
    validated = validate_test_pg_url(url)
    if not worker_id:
        return url
    if not _WORKER_ID_RE.match(worker_id):
        raise UnsafeTestDatabaseUrl("xdist worker id must be alphanumeric")
    base = (validated.database or "").removesuffix(TEST_SUFFIX)
    if base.endswith(f"_{worker_id}"):
        return url  # already scoped, e.g. a nested pytest session re-ran pytest_configure
    name = f"{base}_{worker_id}{TEST_SUFFIX}"
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise UnsafeTestDatabaseUrl(
            f"worker database name would be longer than {MAX_IDENTIFIER_LENGTH} characters"
        )
    return render(validated.set(database=name))
