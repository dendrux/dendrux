"""The load targets run the harness against private databases, one per server family."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-n", "-C", str(PACKAGE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_load_pg_runs_the_thousand_run_scenario_on_the_env_database() -> None:
    result = dry_run("load-pg")
    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "tools/load.py" in command
    assert "--scale full" in command
    assert "--scenarios runs" in command
    assert "--baseline" not in command  # the quick-scale baseline does not apply
    assert "--url" not in command  # DENDRUX_LOAD_PG_URL, never a URL in the Makefile


@pytest.mark.parametrize(("target", "port"), [("load-mysql", "53306"), ("load-mariadb", "53307")])
def test_mysql_family_targets_use_a_private_load_database_on_the_test_server(
    target: str, port: str
) -> None:
    result = dry_run(target, "LOAD_STATEMENT_BUDGETS=budgets.json")
    assert result.returncode == 0, result.stderr
    command = result.stdout
    url = re.search(r"--url (\S+)", command)
    assert url is not None
    assert url[1].startswith("mysql+asyncmy://")
    assert f"@127.0.0.1:{port}/" in url[1]
    database = url[1].rsplit("/", 1)[1]
    assert "_load_" in database
    assert database.endswith("_test")
    assert "--statement-budgets budgets.json" in command
    assert "--scenarios runs" in command
    assert "--baseline" not in command


@pytest.mark.parametrize("target", ["load-mysql", "load-mariadb"])
def test_a_mysql_family_target_refuses_to_run_without_declared_budgets(target: str) -> None:
    result = subprocess.run(
        ["make", "-C", str(PACKAGE), target],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode != 0
    assert "LOAD_STATEMENT_BUDGETS" in result.stdout + result.stderr
    assert "tools/load.py" not in result.stdout.replace("LOAD_STATEMENT_BUDGETS", "")
