"""CI runs the kit's live-update contract on Postgres, MySQL and MariaDB, and fails on a skip.

The contract needs the stores to take ``live_updates=`` (track A2). These steps therefore fail
until A2 is in the branch: without the fixture every live contract skips, and that is exactly
what the guard turns into a failure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

CI = Path(__file__).resolve().parents[4] / ".github" / "workflows" / "ci.yml"
GOOD = "\n".join(
    f"tests/conformance/test_store_kit.py::test_live_{name} [{{db}}] PASSED [ {n}%]"
    for n, name in enumerate("abcdefg")
)


def jobs() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text())["jobs"]


def step(job: str, prefix: str) -> dict[str, Any]:
    found = [s for s in jobs()[job]["steps"] if str(s.get("name", "")).startswith(prefix)]
    assert len(found) == 1, f"{job}: expected one step starting {prefix!r}, got {len(found)}"
    return found[0]


def guard_of(run: str) -> str:
    """The part of a step after the pytest command: the check on its log."""
    lines = run.splitlines()
    index = next(i for i, line in enumerate(lines) if "tee kit-live.log" in line)
    # GitHub expands ${{ }} before bash sees the step; do the same for the one that appears
    return "\n".join(lines[index + 1 :]).replace("${{ matrix.server }}", "mysql")


def run_guard(run: str, log: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    (tmp_path / "kit-live.log").write_text(log)
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", guard_of(run)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_postgres_runs_the_live_contract_serially_on_a_private_database() -> None:
    postgres = step("python", "Store kit live-update contract on Postgres")
    assert "matrix.shard == 1" in postgres["if"] and "'3.12'" in postgres["if"]
    env = postgres["env"]
    database = env["DENDRUX_TEST_PG_URL"].rsplit("/", 1)[1]
    assert database.endswith("_test") and database != "dendrux_test"
    assert env["DENDRUX_CONFORMANCE_PG_DB"] == database  # the harness authorises exactly this one
    run = postgres["run"]
    assert "tests/conformance/test_store_kit.py" in run and "-n 0" in run
    assert "test_live_ and postgres" in run and "--no-cov" in run


@pytest.mark.parametrize("server", ["mysql", "mariadb"])
def test_the_mysql_family_jobs_run_the_live_contract_on_their_server(server: str) -> None:
    job = jobs()["mysql-family"]
    entry = next(e for e in job["strategy"]["matrix"]["include"] if e["server"] == server)
    live = step("mysql-family", "Store kit live-update contract on")
    run = live["run"]
    assert entry["env_name"] in run or "matrix.env_name" in run
    assert "SERVER_URL" in live["env"] and "tests/conformance/test_store_kit.py" in run
    assert "test_live_ and ${{ matrix.server }}" in run


@pytest.mark.parametrize(
    ("job", "prefix", "db"),
    [
        ("python", "Store kit live-update contract on Postgres", "postgres"),
        ("mysql-family", "Store kit live-update contract on", "mysql"),
    ],
)
class TestTheSkipGuard:
    def test_seven_passes_and_nothing_else_is_accepted(
        self, job: str, prefix: str, db: str, tmp_path: Path
    ) -> None:
        run = step(job, prefix)["run"]
        result = run_guard(run, GOOD.format(db=db) + "\n7 passed, 205 deselected\n", tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize(
        "log",
        [
            "",  # nothing ran
            "7 skipped, 205 deselected\n",
            GOOD + "\ntests/conformance/test_store_kit.py::test_live_x [db] SKIPPED (why)\n",
            GOOD + "\n  live updates: not tested (no dendrux_live_store_factory fixture)\n",
            "\n".join(GOOD.splitlines()[:6]),  # one contract short
        ],
    )
    def test_a_skip_a_missing_fixture_or_a_short_run_fails(
        self, job: str, prefix: str, db: str, log: str, tmp_path: Path
    ) -> None:
        run = step(job, prefix)["run"]
        result = run_guard(run, log.replace("{db}", db), tmp_path)
        assert result.returncode != 0
        assert "::error::" in result.stdout
