"""tools/load.py: URL safety, baseline comparison, workflow wiring and a tiny run on Postgres."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox RSS measurement")
def test_rss_measurement_does_not_require_permission_to_spawn_ps(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("process inspection denied")

    monkeypatch.setattr(load.subprocess, "run", denied)
    assert load.rss_kb() > 0


def test_baseline_comparison_fails_only_on_a_regression_over_twenty_percent():
    baseline = {
        "scale": "quick",
        "gates": {"a.statements_per_run": 100, "iteration.notify_on_batch": 20},
    }
    rows, failed = load.compare_gates(
        {"a.statements_per_run": 119, "iteration.notify_on_batch": 20}, baseline, scale="quick"
    )
    assert not failed
    assert [r[-1] for r in rows] == ["ok", "ok"]
    rows, failed = load.compare_gates(
        {"a.statements_per_run": 121, "iteration.notify_on_batch": 20}, baseline, scale="quick"
    )
    assert failed
    assert dict((r[0], r[-1]) for r in rows)["a.statements_per_run"] == "FAIL"
    # an improvement or an unknown metric never fails
    assert not load.compare_gates(
        {"a.statements_per_run": 50, "new.metric": 9}, baseline, scale="quick"
    )[1]


def test_committed_baseline_carries_the_gates_and_the_machine():
    baseline = json.loads((PKG / "tools" / "load_baseline.json").read_text())
    assert baseline["scale"] == "quick"
    assert "iteration.notify_on_batch" in baseline["gates"]
    assert baseline["gates"]["iteration.notify_on_batch"] <= 25
    recorded = baseline["recorded"]
    assert recorded["machine"] and recorded["cores"] > 0
    # Sandboxed sysctl may be unavailable; keep the raw empty/null values honest.
    assert "cpu" in recorded and "memory_gb" in recorded
    assert "v050_reference" not in baseline
    assert load.V050_REFERENCE_A == 59.0


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
    assert service["image"].startswith("postgres:16-alpine@sha256:")
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
    if os.environ.get("DENDRUX_LOAD_TESTS") != "1":
        pytest.skip("DENDRUX_LOAD_TESTS=1 is required to opt into load tests")
    base = os.environ.get("DENDRUX_TEST_PG_URL")
    if not base:
        pytest.skip("DENDRUX_TEST_PG_URL not set")
    parsed = make_url(base)
    name = f"dendrux_load_{uuid.uuid4().hex}_test"
    return parsed.set(database=name).render_as_string(False)


async def _drop_tiny_database(url: str) -> None:
    import asyncpg

    parsed = load.check_private_url(url)
    conn = await asyncpg.connect(
        parsed.set(database="postgres", drivername="postgresql").render_as_string(False)
    )
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{parsed.database}"')
    finally:
        await conn.close()


@pytest.mark.benchmark
def test_a_tiny_load_run_passes_its_checks_on_postgres(tmp_path):
    url = _tiny_url()
    out = tmp_path / "load.json"
    scenarios = "a,iteration,kill_checkpoint,followers_on,cancel"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/load.py",
            "--url",
            url,
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
    asyncio.run(_drop_tiny_database(url))
    assert out.exists(), proc.stderr[-2000:]
    results = json.loads(out.read_text())
    assert not [n for n, s in results["scenarios"].items() if "error" in s], proc.stderr[-2000:]
    failed = [c["name"] for c in results["checks"] if not c["ok"]]
    assert not failed, failed
    gates = results["gates"]
    assert gates["iteration.notify_on_batch"] <= 25
    kill = results["scenarios"]["kill_checkpoint"]["metrics"]
    assert kill["extra_billed_calls"] == 0 and kill["runs_in_killed_process"] >= 1
    assert kill["killed_worker_statements"] > 0
    assert kill["statements_including_recovery"] == (
        kill["statements"] + kill["recovery_statements"]
    )
    assert results["scenarios"]["a_notify_off"]["metrics"]["statements_per_run"] > 0
    assert proc.returncode == int(any(not c["ok"] for c in results["checks"]))


@pytest.mark.parametrize("host", ["db.example.com", "", "localhost.evil", "127.0.0.2"])
def test_remote_hosts_need_an_explicit_opt_in(host):
    url = make_url(BASE + "private_test").set(host=host).render_as_string(False)
    with pytest.raises(ValueError, match="host"):
        load.check_private_url(url)
    if host:
        assert load.check_private_url(url, allow_remote_host=True).host == host


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_loopback_hosts_are_accepted(host):
    url = make_url(BASE + "private_test").set(host=host).render_as_string(False)
    assert load.check_private_url(url).host == host


@pytest.mark.parametrize("option", ["dsn", "target_session_attrs", "user", "passfile", "unknown"])
def test_unknown_url_query_options_are_refused_even_with_remote_opt_in(option):
    with pytest.raises(ValueError, match="query"):
        load.check_private_url(BASE + f"private_test?{option}=value")
    with pytest.raises(ValueError, match="query"):
        load.check_private_url(BASE + f"private_test?{option}=value", allow_remote_host=True)


def test_sslmode_survives_both_driver_url_forms():
    url = BASE + "private_test?sslmode=require"
    assert load.check_private_url(url).query["sslmode"] == "require"
    assert make_url(load.dsn_of(url)).query["sslmode"] == "require"
    engine = load.make_engine(url, "ssl-test")
    assert not engine.url.query  # sslmode is translated to asyncpg's ssl connect argument


def test_baseline_scale_is_checked_before_comparing_counts():
    with pytest.raises(ValueError, match="scale"):
        load.compare_gates({"x": 10}, {"scale": "quick", "gates": {"x": 10}}, scale="full")


def test_scenario_a_uses_absolute_budgets_and_reports_the_release_comparison(monkeypatch):
    monkeypatch.setattr(load, "SCENARIO_A_BUDGETS", {"a_notify_off": 74, "a_notify_on": 83})
    outcomes = {
        "a_notify_off": load.Outcome(metrics={"statements_per_run": 74}),
        "a_notify_on": load.Outcome(metrics={"statements_per_run": 84}),
    }
    checks = load.scenario_a_checks(outcomes)
    assert [(c["ok"], c["limit"]) for c in checks] == [(True, 74), (False, 83)]
    assert all("0.6" in c["name"] for c in checks)
    assert outcomes["a_notify_off"].metrics["vs_v050_percent"] == 25.4


def test_post_trim_a_cost_above_budget_fails_before_twenty_percent_regression():
    outcomes = {
        "a_notify_off": load.Outcome(metrics={"statements_per_run": 75}),
        "a_notify_on": load.Outcome(metrics={"statements_per_run": 84}),
    }
    assert not any(c["ok"] for c in load.scenario_a_checks(outcomes))


def test_tiny_database_is_unique_and_the_pg_variable_alone_does_not_opt_in(monkeypatch):
    monkeypatch.setenv("DENDRUX_TEST_PG_URL", BASE + "original_private_test")
    monkeypatch.delenv("DENDRUX_LOAD_TESTS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="DENDRUX_LOAD_TESTS"):
        _tiny_url()
    monkeypatch.setenv("DENDRUX_LOAD_TESTS", "1")
    first, second = make_url(_tiny_url()), make_url(_tiny_url())
    assert first.database != second.database != "original_private_test"
    assert first.database.endswith("_test") and second.database.endswith("_test")


async def test_sampler_uses_lock_relation_identity_instead_of_query_text(monkeypatch):
    sampler = load.Sampler(BASE + "private_test")

    async def fetch(query):
        assert "pg_locks" in query and "pg_class" in query
        assert "left(query" not in query
        return [{"app": "worker", "wt": "Lock", "we": "transactionid", "run_row_wait": True}]

    async def end_sample(_):
        raise asyncio.CancelledError

    sampler.conn = SimpleNamespace(fetch=fetch)
    monkeypatch.setattr(load.asyncio, "sleep", end_sample)
    with contextlib.suppress(asyncio.CancelledError):
        await sampler._loop()
    assert sampler.max_waiters_runrow == 1


async def test_workload_timer_excludes_worker_startup(monkeypatch):
    clock = {"now": 0.0}
    measured = load.Measured(BASE + "private_test")
    handle = SimpleNamespace(pipe=SimpleNamespace(send=lambda _: None))

    async def pump(handles, *, until, **kwargs):
        clock["now"] += 100 if until == "ready" else 4

    monkeypatch.setattr(load.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(load, "spawn", lambda *args: [handle])
    monkeypatch.setattr(load, "release", lambda _: None)
    monkeypatch.setattr(load, "pump", pump)
    await load.run_workers(BASE + "private_test", "runs", "main", 1, {}, measured=measured)
    assert measured.workload_s == 4


async def test_doomed_run_errors_abort_without_waiting_for_other_runs_to_park(monkeypatch):
    from dendrux import Agent

    async def fail_run(*args, **kwargs):
        raise ValueError("before park")

    async def dispose():
        pass

    monkeypatch.setattr(Agent, "run", fail_run)
    monkeypatch.setattr(load, "make_engine", lambda *args: SimpleNamespace(dispose=dispose))
    monkeypatch.setattr(load, "Meter", lambda _: SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(load, "make_store", lambda *args, **kwargs: None)
    ctx = load.Ctx(
        "runs",
        "doomed",
        0,
        1,
        BASE + "private_test",
        {"keys": [0], "kill": "checkpoint"},
        SimpleNamespace(send=lambda *args: None, recv=lambda: "go"),
    )
    with pytest.raises(RuntimeError, match="before reaching the kill boundary"):
        await load.w_runs(ctx)


async def test_memory_admin_uses_the_guarded_engine_builder(monkeypatch):
    apps = []

    async def dispose():
        pass

    def engine(url, app, **kwargs):
        apps.append(app)
        return SimpleNamespace(dispose=dispose)

    monkeypatch.setattr(load, "make_engine", engine)
    monkeypatch.setattr(load, "Meter", lambda _: SimpleNamespace(snapshot=lambda: {}))
    ctx = load.Ctx(
        "memory",
        "main",
        0,
        1,
        BASE + "private_test?sslmode=require",
        {"cycles": 0},
        SimpleNamespace(send=lambda *args: None, recv=lambda: "go"),
    )
    await load.w_memory(ctx)
    assert apps == ["dendrux_load_w0", "dendrux_load_admin"]
