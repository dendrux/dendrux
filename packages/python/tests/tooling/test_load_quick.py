"""tools/load.py: URL safety, baseline comparison, workflow wiring and a tiny run on Postgres."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

from tools import load

PKG = Path(__file__).resolve().parents[2]
REPO = PKG.parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "bench.yml"
BASE = "postgresql+asyncpg://u:p@localhost:5432/"


@pytest.mark.parametrize(
    "url",
    [
        BASE + "dendrux",
        BASE + "dendrux_test",
        BASE + "postgres",
        BASE + "orbit_prod",
        BASE + "dendrux_load_test?database=dendrux",
        BASE + "dendrux_load_test?host=elsewhere",
        "sqlite+aiosqlite:///x_test.db",
    ],
)
def test_the_script_refuses_shared_and_unsafe_databases(url):
    with pytest.raises(ValueError):
        load.check_private_url(url)


def test_the_script_accepts_a_private_test_database():
    parsed = load.check_private_url(BASE + "dendrux_060_load_test")
    assert parsed.database == "dendrux_060_load_test"
    assert make_url(load.dsn_of(BASE + "dendrux_060_load_test")).drivername == "postgresql"


def test_main_exits_before_touching_anything_on_a_shared_database(capsys):
    assert load.main(["--url", BASE + "dendrux_test", "--scale", "tiny"]) == 2
    assert "refusing" in capsys.readouterr().err


def test_every_preset_defines_the_same_sizes():
    assert set(load.PRESETS["full"]) == set(load.PRESETS["quick"]) == set(load.PRESETS["tiny"])
    full = load.PRESETS["full"]
    assert (full["a"], full["runs"], full["followers_runs"] * full["followers_per_run"]) == (
        200,
        1000,
        10_000,
    )
    assert (full["cancel_nodes"], full["pool_runs"], full["memory_cycles"]) == (2000, 200, 50)
    assert full["kill_fraction"] == 0.05


def test_tree_has_the_requested_number_of_nodes_and_one_root():
    links = load.tree_links(2000)
    assert len(links) == 2000
    assert [n for n, parent in links if parent is None] == [0]
    assert all(parent < n for n, parent in links if parent is not None)


def test_baseline_comparison_fails_only_on_a_regression_over_twenty_percent():
    baseline = {"gates": {"a.statements_per_run": 100, "iteration.notify_on_batch": 20}}
    rows, failed = load.compare_gates(
        {"a.statements_per_run": 119, "iteration.notify_on_batch": 20}, baseline
    )
    assert not failed
    assert [r[-1] for r in rows] == ["ok", "ok"]
    rows, failed = load.compare_gates(
        {"a.statements_per_run": 121, "iteration.notify_on_batch": 20}, baseline
    )
    assert failed
    assert dict((r[0], r[-1]) for r in rows)["a.statements_per_run"] == "FAIL"
    # an improvement or an unknown metric never fails
    assert not load.compare_gates({"a.statements_per_run": 50, "new.metric": 9}, baseline)[1]


def test_committed_baseline_carries_the_gates_and_the_machine():
    baseline = json.loads((PKG / "tools" / "load_baseline.json").read_text())
    assert baseline["scale"] == "quick"
    assert "iteration.notify_on_batch" in baseline["gates"]
    assert baseline["gates"]["iteration.notify_on_batch"] <= 25
    assert baseline["recorded"]["cpu"]
    assert baseline["v050_reference"]["scenario_a_statements_per_run"] > 0


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repository checkout only")
def test_the_bench_workflow_is_weekly_pinned_and_runs_load_quick():
    import yaml

    doc = yaml.safe_load(WORKFLOW.read_text())
    triggers = doc.get("on", doc.get(True))
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"][0]["cron"]
    job = next(iter(doc["jobs"].values()))
    assert job["runs-on"] == "ubuntu-24.04"
    service = next(iter(job["services"].values()))
    assert service["image"].startswith("postgres:16-alpine")
    text = WORKFLOW.read_text()
    assert "shared_buffers = '256MB'" in text
    assert "make load-quick" in text
    assert "load_baseline.json" in text
    makefile = (PKG / "Makefile").read_text()
    assert "load-quick:" in makefile and "tools/load_baseline.json" in makefile


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repository checkout only")
def test_ci_does_not_run_the_load_script():
    assert "tools/load.py" not in (REPO / ".github" / "workflows" / "ci.yml").read_text()


def _tiny_url() -> str:
    base = os.environ.get("DENDRUX_TEST_PG_URL")
    if not base:
        pytest.skip("DENDRUX_TEST_PG_URL not set")
    parsed = make_url(base)
    # The shared test database is refused by the script; a private *_test name is used as given.
    shared = parsed.database in load._FORBIDDEN or not (parsed.database or "").endswith("_test")
    name = "dendrux_load_tooling_test" if shared else parsed.database
    return parsed.set(database=name).render_as_string(False)


def test_a_tiny_load_run_passes_its_checks_on_postgres(tmp_path):
    out = tmp_path / "load.json"
    scenarios = "a,iteration,kill_checkpoint,followers_on,cancel"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/load.py",
            "--url",
            _tiny_url(),
            "--scale",
            "tiny",
            "--scenarios",
            scenarios,
            "--json",
            str(out),
        ],
        cwd=PKG,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.exists(), proc.stderr[-2000:]
    results = json.loads(out.read_text())
    assert not [n for n, s in results["scenarios"].items() if "error" in s], proc.stderr[-2000:]
    failed = [
        c["name"]
        for c in results["checks"]
        if not c["ok"] and "0.5 scenario A within 10%" not in c["name"]
    ]
    assert not failed, failed
    gates = results["gates"]
    assert gates["iteration.notify_on_batch"] <= 25
    kill = results["scenarios"]["kill_checkpoint"]["metrics"]
    assert kill["extra_billed_calls"] == 0 and kill["runs_in_killed_process"] >= 1
    assert results["scenarios"]["a_notify_off"]["metrics"]["statements_per_run"] > 0
    assert proc.returncode in (0, 1)
