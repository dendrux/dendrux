"""CI runs the Redis suites on the older redis-py releases the extra allows (no network in tests).

The 8.1.0 pin of the ``dev`` extra is already exercised by every ordinary job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    assert "-n 0" in run and "not set" in run and "exit 1" in run
