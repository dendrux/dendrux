"""Shared test fixtures for Dendrux tests."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

from tests._helpers import tiers
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
    for area in tiers.AREAS:
        config.addinivalue_line("markers", f"area_{area}: tests of the {area} area (by path)")
    config.addinivalue_line(
        "markers", "slow: over the time threshold in tests/_helpers/tier_lists/slow.txt"
    )
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


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Run in parallel by default once a Postgres URL is set.

    A serial Postgres run is the slow path (and the one where test order can matter), so with
    ``DENDRUX_TEST_PG_URL`` set and no ``-n`` given, use ``DENDRUX_TEST_WORKERS`` workers
    (default: the core count, at most 6). ``-n 0`` still means serial; a run that names
    individual tests (``path::test``) or the conformance suite (whose Postgres cells run only
    in one process) stays serial.
    """
    option = config.option
    if not os.environ.get(_PG_URL_ENV) or getattr(option, "numprocesses", 1) is not None:
        return
    if getattr(option, "collectonly", False):
        return
    # Single tests stay in one process; the conformance suite runs its Postgres cells serially.
    if any("::" in arg or "conformance" in arg for arg in config.args):
        return
    if getattr(option, "dist", "no") != "no" or "PYTEST_XDIST_WORKER" in os.environ:
        return
    option.numprocesses = tiers.default_workers()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag every test with its area, and the slow and flaky ones from the committed lists."""
    slow = tiers.listed(tiers.load_list("slow.txt"))
    flaky = tiers.listed(tiers.load_list("flaky.txt"))
    root = tiers.TESTS_ROOT
    for item in items:
        path = getattr(item, "path", None)
        if path is None or not path.is_relative_to(root):
            continue
        item.add_marker(
            getattr(pytest.mark, f"area_{tiers.area_for(path.relative_to(root).as_posix())}")
        )
        nodeid = item.nodeid
        if tiers.in_list(*slow, nodeid):
            item.add_marker(pytest.mark.slow)
        if tiers.in_list(*flaky, nodeid):
            item.add_marker(pytest.mark.flaky(reruns=1))
