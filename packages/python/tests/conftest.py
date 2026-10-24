"""Shared test fixtures for Dendrux tests."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

from tests._helpers.xdist_pg import worker_scoped_pg_url

if TYPE_CHECKING:
    from collections.abc import Iterator

_PG_URL_ENV = "DENDRUX_TEST_PG_URL"


def pytest_configure(config: pytest.Config) -> None:
    """Give every xdist worker its own Postgres database.

    The integration fixtures drop and truncate tables, so workers sharing one database
    would wipe each other's rows. Runs in each worker process before any test module
    reads the variable.
    """
    url = os.environ.get(_PG_URL_ENV)
    worker = getattr(config, "workerinput", {}).get("workerid") or os.environ.get(
        "PYTEST_XDIST_WORKER"
    )
    if url and worker:
        os.environ[_PG_URL_ENV] = worker_scoped_pg_url(url, worker)


def _reenable_dendrux_loggers() -> None:
    for name, candidate in list(logging.root.manager.loggerDict.items()):
        if name.startswith("dendrux") and isinstance(candidate, logging.Logger):
            candidate.disabled = False


@pytest.fixture(autouse=True)
def _dendrux_loggers_stay_enabled() -> Iterator[None]:
    """Undo what an Alembic run in an earlier test did to the logging state.

    The migration environment script calls ``logging.config.fileConfig`` with its default
    ``disable_existing_loggers=True``, which silently disables every ``dendrux.*`` logger
    that already exists in the process. Any later test that reads ``caplog`` then sees
    nothing, depending on test order. The source fix is tracked as a contract gap (see
    ``test_logging_survives_migration.py``).
    """
    _reenable_dendrux_loggers()
    yield
    _reenable_dendrux_loggers()
