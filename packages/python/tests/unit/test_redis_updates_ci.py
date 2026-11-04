"""CI runs the Redis suites on the older redis-py releases the extra allows (no network in tests).

The 8.1.0 pin of the ``dev`` extra is already exercised by every ordinary job.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

CI = Path(__file__).resolve().parents[4] / ".github" / "workflows" / "ci.yml"
OLDER = ["5.0.1", "5.3.1", "6.4.0", "7.4.0"]


def job() -> dict[str, Any]:
    jobs = yaml.safe_load(CI.read_text())["jobs"]
    assert "redis-compat" in jobs, "no job runs the Redis suites on older redis-py releases"
    return jobs["redis-compat"]  # type: ignore[no-any-return]


def test_the_matrix_covers_the_older_releases_of_the_extra() -> None:
    assert job()["strategy"]["matrix"]["redis"] == OLDER
    assert job()["strategy"]["fail-fast"] is False


def test_each_leg_installs_its_release_after_the_dev_extra_and_checks_it() -> None:
    steps = job()["steps"]
    runs = [step.get("run", "") for step in steps]
    install = next(i for i, run in enumerate(runs) if 'pip install -e ".[dev]"' in run)
    pin = next(i for i, run in enumerate(runs) if 'pip install "redis==${{ matrix.redis }}"' in run)
    assert install < pin
    assert any("redis.__version__ == '${{ matrix.redis }}'" in run for run in runs)


def test_the_suites_run_against_a_real_redis_service_and_fail_on_a_skip() -> None:
    spec = job()
    assert spec["services"]["redis"]["image"].startswith("redis:7")
    assert spec["env"]["DENDRUX_TEST_REDIS_URL"].startswith("redis://")
    run = next(s["run"] for s in spec["steps"] if "pytest" in s.get("run", ""))
    assert (
        "tests/unit/test_redis_updates" in run and "tests/integration/test_redis_updates.py" in run
    )
    assert "-n 0" in run and "skipped" in run and "exit 1" in run


@pytest.mark.parametrize(
    ("output", "pytest_status", "expected"),
    [
        ("===== 10 passed in 1.0s =====", 0, 0),
        (
            "tests/unit/test_redis_updates_ci.py::"
            "test_redis_ci_shell_rejects_any_skip_or_pytest_failure"
            "[9 passed, 1 skipped-0-1] PASSED [100%]\n"
            "===== 10 passed in 1.0s =====",
            0,
            0,
        ),
        ("skipped [1] diagnostic only\n===== 10 passed in 1.0s =====", 0, 0),
        ("===== 9 passed, 1 skipped in 1.0s =====", 0, 1),
        ("===== 1 skipped in 1.0s =====", 0, 1),
        ("SKIPPED [1] unrelated reason", 0, 1),
        ("SKIPPED [1] DENDRUX_TEST_REDIS_URL not set", 0, 1),
        ("===== 1 failed, 9 passed in 1.0s =====", 1, 1),
    ],
)
def test_redis_ci_shell_rejects_any_skip_or_pytest_failure(
    tmp_path: Path, output: str, pytest_status: int, expected: int
) -> None:
    run = next(s["run"] for s in job()["steps"] if "pytest" in s.get("run", ""))
    fake = 'pytest() { printf "%s\\n" "$TEST_OUTPUT"; return "$TEST_STATUS"; }\n'
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", fake + run],
        cwd=tmp_path,
        env={**os.environ, "TEST_OUTPUT": output, "TEST_STATUS": str(pytest_status)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("===== 10 passed, 2 skipped in 1.0s =====", 0),
        ("tests/unit/test_ci.py::test_guard[SKIPPED [1] DENDRUX_TEST_REDIS_URL not set] PASSED", 0),
        ("[gw0] [ 9%] PASSED tests/unit/test_ci.py::test[DENDRUX_TEST_PG_URL not set]", 0),
        (
            "SKIPPED [1] tests/integration/test_redis_updates.py:9: DENDRUX_TEST_REDIS_URL not set",
            1,
        ),
        ("SKIPPED [12] tests/integration/conftest.py:40: DENDRUX_TEST_PG_URL not set", 1),
    ],
)
def test_the_suite_guard_fails_on_a_real_backend_skip_only(
    tmp_path: Path, output: str, expected: int
) -> None:
    jobs = yaml.safe_load(CI.read_text())["jobs"]
    run = next(
        step["run"]
        for spec in jobs.values()
        for step in spec.get("steps", [])
        if step.get("name") == "Test"
    ).replace("${{ matrix.shard }}", "1")
    fake = 'python() { :; }\npytest() { printf "%s\\n" "$TEST_OUTPUT"; }\n'
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", fake + run],
        cwd=tmp_path,
        env={**os.environ, "TEST_OUTPUT": output},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
