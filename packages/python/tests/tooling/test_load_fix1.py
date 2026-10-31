"""Fix-round regressions for observer isolation, leak power and probe failures."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import NullPool

from tools import load
from tools import load_monitor as monitor

MYSQL = "mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_p5_l_load_mysql_test"


async def test_observer_docker_and_its_children_are_excluded(monkeypatch):
    spawned = []
    create = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        proc = await create(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(monitor.asyncio, "create_subprocess_exec", spawn)
    sleeper = asyncio.create_task(
        monitor.command(sys.executable, "-c", "import time; time.sleep(30)")
    )
    while not spawned:
        await asyncio.sleep(0)
    docker = spawned[0].pid
    app = os.getpid()

    async def snapshot(*args):
        return (
            f"{app} 1 100 2 1 python\n{docker} {app} 9000 80 9 docker\n"
            f"{docker + 1} {docker} 8000 70 8 helper\n{docker + 2} {app} 50 1 2 python\n"
        )

    monkeypatch.setattr(monitor, "command", snapshot)
    try:
        sampler = monitor.ResourceMonitor(None)
        await sampler.sample()
        sample = sampler.buffer.samples[-1]
        assert sample["app.rss_kb"] == 150
        assert sample["app.cpu_percent"] == 3
        assert sample["app.threads"] == 3
    finally:
        sleeper.cancel()
        await asyncio.gather(sleeper, return_exceptions=True)


async def test_finished_ps_pid_is_still_excluded_from_its_snapshot(monkeypatch):
    create = asyncio.create_subprocess_exec
    spawned = []

    async def spawn(*args, **kwargs):
        proc = await create(*args, **kwargs)
        spawned.append(proc.pid)
        return proc

    monkeypatch.setattr(monitor.asyncio, "create_subprocess_exec", spawn)
    await monitor.command(sys.executable, "-c", "pass")

    async def snapshot(*args):
        return f"{os.getpid()} 1 100 2 1 python\n{spawned[0]} {os.getpid()} 9000 80 9 ps\n"

    monkeypatch.setattr(monitor, "command", snapshot)
    sampler = monitor.ResourceMonitor(None)
    await sampler.sample()
    assert sampler.buffer.samples[-1]["app.rss_kb"] == 100


async def test_mysql_connections_are_private_processlist_sum():
    async def rows(query):
        if query == "SHOW GLOBAL STATUS":
            return [{"Variable_name": "Threads_connected", "Value": "99"}]
        assert "DB = DATABASE()" in query
        return [{"state": "Sleep", "n": 2}, {"state": "Query", "n": 1}]

    data = await monitor.database_sample("mysql", rows)
    assert data["db.connections"] == 3
    assert data["server.Threads_connected"] == 99


def test_leak_fit_skips_first_cycle():
    checks = monitor.leak_checks([{"rss_kb": n} for n in [0, 1000, 1001, 1002]], {"rss_kb": 2})
    assert checks[0]["ok"] and checks[0]["value"] == 1


@pytest.mark.parametrize("metric", ["fds", "connections"])
def test_unit_growth_at_threshold_fails(metric):
    check = monitor.leak_checks([{metric: n} for n in [0, 1, 2, 3]], {metric: 1})[0]
    assert not check["ok"] and check["value"] == 1


async def test_gc_precedes_controller_endpoint_and_memory_warmup(monkeypatch):
    events = []
    monkeypatch.setattr(load, "make_engine", lambda *a, **kw: SimpleNamespace(dispose=dispose))

    async def dispose():
        pass

    async def start(self):
        pass

    async def stop(self, runs):
        return {"summary": {}, "endpoint": {"db.connections": 1}}

    monkeypatch.setattr(monitor.ResourceMonitor, "start", start)
    monkeypatch.setattr(monitor.ResourceMonitor, "stop", stop)
    monkeypatch.setattr(load.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(load, "rss_kb", lambda: events.append("rss") or 100)

    async def job():
        return load.Outcome(
            metrics={
                "resource_runs": 8,
                "warmup_cycles": 2,
                "memory_cycle_resources": [{"fds": v} for v in [0, 100, 100, 100]],
            }
        )

    outcome, _, endpoint = await load.monitored_job(
        MYSQL,
        job,
        SimpleNamespace(monitor_interval=1, monitor_capacity=10, leak_thresholds={"fds": 1}),
    )
    assert events[:2] == ["gc", "rss"]
    assert endpoint["connections"] == 1
    assert all(c["ok"] for c in outcome.checks)


def test_cli_defaults_to_four_cycles(monkeypatch):
    seen = []

    async def run(args):
        seen.append(args.leak_cycles)
        return 0

    monkeypatch.setattr(load, "run_all", run)
    assert load.main(["--url", "postgresql://u:p@localhost/task_test"]) == 0
    assert seen == [4]


async def test_full_cycle_samples_are_spooled_not_retained(tmp_path):
    assert hasattr(load, "retain_cycle"), "compact cycle retention missing"
    report = {"samples": [{"rss_kb": 1}], "summary": {"rss_kb": {"peak": 1}}}
    outcome = load.Outcome(metrics={"runs": 1})
    compact = load.retain_cycle(outcome, report, tmp_path / "cycle.json")
    assert "samples" not in compact["monitoring"]
    assert json.loads((tmp_path / "cycle.json").read_text())["monitoring"]["samples"]


async def test_missing_fd_and_pool_probes_are_not_zero(monkeypatch):
    monkeypatch.setattr(
        monitor.Path, "iterdir", lambda self: (_ for _ in ()).throw(PermissionError())
    )
    data = monitor.local_sample([SimpleNamespace(pool=NullPool(lambda: None))])
    assert "fds" not in data
    assert not any(k.startswith("pool.") for k in data)
    sampler = monitor.ResourceMonitor(None)

    async def denied(*args):
        raise PermissionError()

    monkeypatch.setattr(monitor, "command", denied)
    await sampler.sample()
    assert "app.fds" not in sampler.buffer.samples[-1]
    assert "app.fds" in sampler.unavailable


async def test_child_cpu_is_omitted_until_reaped_usage_is_observed(monkeypatch):
    real = monitor.resource.getrusage
    monkeypatch.setattr(
        monitor.resource,
        "getrusage",
        lambda who: (
            SimpleNamespace(ru_utime=0, ru_stime=0)
            if who == monitor.resource.RUSAGE_CHILDREN
            else real(who)
        ),
    )
    assert not any("children" in k for k in monitor.local_sample())


async def test_failed_docker_probe_clears_previous_values(monkeypatch):
    sampler = monitor.ResourceMonitor(None)
    sampler.docker_values = {"docker.old.memory_bytes": 1}

    async def denied(*args):
        raise PermissionError()

    monkeypatch.setattr(monitor, "command", denied)
    await sampler._docker_sample()
    assert sampler.docker_values == {}


async def test_loop_survives_unexpected_probe_exception(monkeypatch):
    sampler = monitor.ResourceMonitor(None, interval=0.001)
    calls = []
    reached = asyncio.Event()

    async def sample():
        calls.append(1)
        if len(calls) == 1:
            raise LookupError("probe failed")
        reached.set()

    monkeypatch.setattr(sampler, "sample", sample)
    task = asyncio.create_task(sampler._loop())
    try:
        await asyncio.wait_for(reached.wait(), 0.2)
        assert "sampling" in sampler.unavailable
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("error", [RuntimeError, asyncio.TimeoutError])
async def test_final_probe_error_returns_unavailable_endpoint(monkeypatch, error):
    sampler = monitor.ResourceMonitor(None)
    sampler.buffer.add({"db.connections": 5, "db.wal_bytes": 100})

    async def sample():
        raise error()

    monkeypatch.setattr(sampler, "sample", sample)
    report = await sampler.stop(1)
    assert "db.connections" not in report["endpoint"]
    assert "wal_bytes_delta" not in report
    assert "final_sample" in report["unavailable"]


@pytest.mark.parametrize("operation", ["connect", "close"])
async def test_observer_connection_operations_have_deadlines(monkeypatch, operation):
    monkeypatch.setattr(monitor, "PROBE_TIMEOUT_S", 0.02, raising=False)
    blocked = asyncio.Event()

    async def hang():
        await blocked.wait()

    sampler = monitor.ResourceMonitor(None)
    if operation == "connect":
        sampler.engine = SimpleNamespace(connect=hang)
        await asyncio.wait_for(sampler.start(), 0.2)
        assert "connect" in sampler.unavailable
    else:

        async def sample():
            pass

        monkeypatch.setattr(sampler, "sample", sample)
        sampler.conn = SimpleNamespace(close=hang)
        await asyncio.wait_for(sampler.stop(1), 0.2)
        assert "close" in sampler.unavailable
    if sampler.task:
        await sampler.stop(0)


async def test_ps_nlwp_failure_is_cached_and_pcpu_is_labelled(monkeypatch):
    attempts = []

    async def probe(*args):
        attempts.append(args)
        if "nlwp" in args[-1]:
            raise RuntimeError()
        return f"{os.getpid()} 1 100 2 python\n"

    monkeypatch.setattr(monitor, "command", probe)
    sampler = monitor.ResourceMonitor(None)
    await sampler.sample()
    await sampler.sample()
    assert sum("nlwp" in a[-1] for a in attempts) == 1
    report = await sampler.stop(1)
    assert "non-instantaneous" in report["cpu_percent_semantics"]


async def test_hung_command_is_killed_and_reaped_at_its_deadline(monkeypatch):
    create = asyncio.create_subprocess_exec
    procs = []

    async def spawn(*a, **kw):
        proc = await create(*a, **kw)
        procs.append(proc)
        return proc

    monkeypatch.setattr(monitor.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(monitor, "PROBE_TIMEOUT_S", 0.02, raising=False)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            monitor.command(sys.executable, "-c", "import time; time.sleep(30)"), 0.5
        )
    assert time.monotonic() - started < 0.4
    assert procs[0].returncode is not None


@pytest.mark.parametrize(
    "url",
    [
        MYSQL.replace("53306", "3306"),
        MYSQL.replace("dendrux_p5_l_load_mysql_test", "DENDRUX_test"),
        MYSQL.replace("127.0.0.1", "remote.example"),
    ],
)
def test_strict_mysql_url_rejects_wrong_port_shared_case_and_remote(url):
    with pytest.raises(ValueError):
        load.check_private_url(url)


def test_explicit_port_opt_in_keeps_other_mysql_guards():
    assert load.check_private_url(MYSQL.replace("53306", "3306"), allow_port=True).port == 3306
    with pytest.raises(ValueError):
        load.check_private_url(MYSQL.replace("127.0.0.1", "remote.example"), allow_port=True)


def test_unknown_workload_metric_shape_is_an_error():
    with pytest.raises(ValueError, match="workload"):
        load.workload_runs({"unrecognized_count": 5})


async def test_cancelled_sql_observer_is_rolled_back_before_final_sample(monkeypatch):
    sampler = monitor.ResourceMonitor(None)
    pending = {"rollback": True}

    async def rollback():
        pending["rollback"] = False

    async def close():
        pass

    async def sample():
        if pending["rollback"]:
            raise RuntimeError("pending rollback after SQL cancellation")
        sampler.buffer.add({"db.connections": 1})

    sampler.conn = SimpleNamespace(rollback=rollback, close=close)
    monkeypatch.setattr(sampler, "sample", sample)
    report = await sampler.stop(1)
    assert report["endpoint"]["db.connections"] == 1


async def test_periodic_probe_failure_rolls_back_before_retry(monkeypatch):
    sampler = monitor.ResourceMonitor(None)
    events = []

    async def rollback():
        events.append("rollback")

    async def sample():
        raise RuntimeError("cancelled read transaction")

    sampler.conn = SimpleNamespace(rollback=rollback)
    monkeypatch.setattr(sampler, "sample", sample)
    await sampler._safe_sample("sampling")
    assert events == ["rollback"]
