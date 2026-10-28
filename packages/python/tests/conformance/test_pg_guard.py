"""The destructive Postgres fixture accepts exactly the authorised database."""

from __future__ import annotations

import pytest

from tests.conformance import pg_guard

BASE = "postgresql+asyncpg://postgres:root@localhost:5432/"


def test_the_authorised_database_is_accepted() -> None:
    assert pg_guard.authorize(BASE + "dendrux_060_0c_test") == "dendrux_060_0c_test"


@pytest.mark.parametrize(
    "name",
    [
        "dendrux_test",  # the shared test database
        "dendrux",  # the owner's dev database (never rewritten to *_test here)
        "postgres",
        "dendrux_060_0b_test",  # another builder's private database
        "dendrux_060_0c_gw0_test",  # an xdist worker name
        "dendrux_060_0c",
        "dendrux_060_0c_test_",
        "other_test",
    ],
)
def test_every_other_database_is_refused_before_any_connection(name) -> None:
    with pytest.raises(pytest.UsageError):
        pg_guard.authorize(BASE + name)


def test_a_name_passed_explicitly_is_the_only_extra_accepted(monkeypatch) -> None:
    monkeypatch.setenv(pg_guard.EXTRA_NAME_ENV, "dendrux_060_0c_alt_test")
    assert pg_guard.authorize(BASE + "dendrux_060_0c_alt_test") == "dendrux_060_0c_alt_test"
    assert pg_guard.authorize(BASE + "dendrux_060_0c_test") == "dendrux_060_0c_test"
    with pytest.raises(pytest.UsageError):
        pg_guard.authorize(BASE + "dendrux_test")


def test_the_shared_database_cannot_be_authorised_even_when_named(monkeypatch) -> None:
    monkeypatch.setenv(pg_guard.EXTRA_NAME_ENV, "dendrux_test")
    with pytest.raises(pytest.UsageError):
        pg_guard.authorize(BASE + "dendrux_test")


def test_overrides_in_the_url_are_refused() -> None:
    with pytest.raises(pytest.UsageError):
        pg_guard.authorize(BASE + "dendrux_060_0c_test?host=elsewhere")


def test_the_fixture_refuses_a_shared_database_url_end_to_end() -> None:
    """A real pytest run with the shared URL fails on the guard, not on a connection."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    package = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        pg_guard.URL_ENV: BASE + "dendrux_test",
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONPATH": "src",
    }
    env.pop("PYTEST_XDIST_WORKER", None)
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/conformance/test_conformance.py",
            "-k",
            "plain_answer__run and postgres",
            "-p",
            "no:randomly",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=package,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode != 0
    assert "runs destructive DDL only on" in proc.stdout + proc.stderr
