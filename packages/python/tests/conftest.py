"""Shared test fixtures for Dendrux tests."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

from tests._helpers.nlp_models import configure_models
from tests._helpers.xdist_pg import worker_scoped_pg_url

if TYPE_CHECKING:
    from collections.abc import Iterator

os.environ["DENDRUX_STRICT_LOG_FIELDS"] = "1"

_PG_URL_ENV = "DENDRUX_TEST_PG_URL"


def pytest_configure(config: pytest.Config) -> None:
    """Give every xdist worker its own Postgres database.

    The integration fixtures drop and truncate tables, so workers sharing one database
    would wipe each other's rows. Runs in each worker process before any test module
    reads the variable.
    """
    configure_models(config)
    url = os.environ.get(_PG_URL_ENV)
    worker = getattr(config, "workerinput", {}).get("workerid") or os.environ.get(
        "PYTEST_XDIST_WORKER"
    )
    if url and worker:
        os.environ[_PG_URL_ENV] = worker_scoped_pg_url(url, worker)


@pytest.fixture(autouse=True)
def _no_test_disables_a_dendrux_logger() -> Iterator[None]:
    """Fail the test that switches off a ``dendrux`` logger, not a distant caplog test.

    ``logging.config.fileConfig`` (Alembic's env script, import-linter) disables every
    existing logger by default, and a disabled logger makes later ``caplog`` assertions fail
    depending on test order. Names the culprit and repairs the state so one leak is one
    failure.
    """
    yield
    disabled = [
        name
        for name, candidate in list(logging.root.manager.loggerDict.items())
        if name.startswith("dendrux")
        and isinstance(candidate, logging.Logger)
        and candidate.disabled
    ]
    for name in disabled:
        logging.getLogger(name).disabled = False
    assert not disabled, f"this test disabled the logger(s) {sorted(disabled)[:5]}"
