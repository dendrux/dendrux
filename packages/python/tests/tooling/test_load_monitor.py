"""Resource monitoring and cross-dialect load safety contracts."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import load

MYSQL = "mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_p5_l_load_test"


def monitor_module():
    assert (Path(load.__file__).parent / "load_monitor.py").exists(), "resource sampler missing"
    return importlib.import_module("tools.load_monitor")


@pytest.mark.parametrize("driver", ["mysql", "mysql+asyncmy", "mysql+aiomysql", "mariadb"])
def test_mysql_private_url_normalizes(driver):
    parsed = load.check_private_url(MYSQL.replace("mysql+asyncmy", driver))
    assert parsed.drivername == "mysql+asyncmy"
    assert parsed.database == "dendrux_p5_l_load_test"


@pytest.mark.parametrize("suffix", ["?host=elsewhere", "?sslmode=disable", "?database=dendrux"])
def test_mysql_url_rejects_connection_overrides(suffix):
    with pytest.raises(ValueError):
        load.check_private_url(MYSQL + suffix)


@pytest.mark.parametrize("name", ["dendrux", "dendrux_test", "live", "../x_test"])
def test_mysql_refuses_unsafe_names(name):
    with pytest.raises(ValueError):
        load.check_private_url(MYSQL.rsplit("/", 1)[0] + "/" + name)


def test_mysql_never_allows_remote_even_with_postgres_opt_in():
    with pytest.raises(ValueError):
        load.check_private_url(MYSQL.replace("127.0.0.1", "elsewhere"), allow_remote_host=True)


def test_store_uses_factory_and_only_postgres_notify(monkeypatch):
    from dendrux.runtime.stores import _factory

    seen = []
    monkeypatch.setattr(_factory, "store_for_engine", lambda e, **kw: seen.append(kw) or e)
    for backend, options in [("mysql", {}), ("postgresql", {"notify_channel": load.CHANNEL})]:
        engine = SimpleNamespace(dialect=SimpleNamespace(name=backend))
        assert load.make_store(engine, notify=True) is engine
        assert seen[-1] == options


def test_mysql_engine_has_no_asyncpg_connection_args(monkeypatch):
    import sqlalchemy.ext.asyncio

    seen = {}
    monkeypatch.setattr(
        sqlalchemy.ext.asyncio, "create_async_engine", lambda u, **kw: seen.update(kw)
    )
    load.make_engine(MYSQL, "worker")
    assert seen["connect_args"] == {}


def test_mysql_plan_skips_notify_followers_with_notes():
    assert hasattr(load, "scenario_jobs"), "dialect-aware scenario planning missing"
    jobs, notes = load.scenario_jobs(MYSQL, load.PRESETS["tiny"], 2, load.ALL_SCENARIOS, False)
    names = [name for name, _ in jobs]
    assert "runs" in names and "a_notify_off" in names and "iteration" in names
    assert "a_notify_on" not in names and "followers_on" not in names
    assert "followers_off" not in names
    assert any("NOTIFY" in note for note in notes)


def test_summary_ring_is_bounded_and_uses_last_half_median():
    m = monitor_module()
    samples = m.SampleBuffer(4)
    for n in range(6):
        samples.add({"rss_kb": n * 10, "tasks": n})
    out = samples.report(20)
    assert len(out["samples"]) == 4 and out["total_samples"] == 6
    rss = out["summary"]["rss_kb"]
    assert rss == {"peak": 50, "steady": 45, "growth_per_1000_runs": 2500}


@pytest.mark.parametrize("metric", ["rss_kb", "fds", "tasks", "connections"])
def test_leak_check_fails_each_resource_and_retains_negative_growth(metric):
    m = monitor_module()
    cycles = [{metric: v} for v in [1, 4, 7]]
    checks = m.leak_checks(cycles, {metric: 2})
    assert len(checks) == 1 and not checks[0]["ok"]
    assert checks[0]["value"] == 3
    assert m.leak_checks(list(reversed(cycles)), {metric: 2})[0]["ok"]


def test_missing_leak_probe_is_not_a_passing_zero():
    m = monitor_module()
    check = m.leak_checks([{}, {}, {}], {"fds": 1})[0]
    assert check["ok"] is False and check["value"] is None


async def test_sampler_starts_stops_and_still_records_final_sample(monkeypatch):
    m = monitor_module()
    sampler = m.ResourceMonitor(None, interval=0.01, capacity=3)
    count = 0

    async def sample():
        nonlocal count
        count += 1
        sampler.buffer.add({"rss_kb": count})

    monkeypatch.setattr(sampler, "sample", sample)
    await sampler.start()
    await asyncio.sleep(0.035)
    out = await sampler.stop(10)
    assert out["total_samples"] >= 3 and len(out["samples"]) == 3
    assert sampler.task.done()


def test_ps_descendants_and_postgres_server_processes():
    m = monitor_module()
    rows = m.parse_ps(
        "10 1 100 2.0 3 python\n11 10 50 1.0 2 python\n12 11 20 0 1 python\n20 1 300 4 5 postgres\n"
    )
    assert [p["pid"] for p in m.descendants(rows, 10)] == [10, 11, 12]
    assert sum(p["rss_kb"] for p in rows if p["command"] == "postgres") == 300


async def test_unavailable_system_probes_are_recorded_without_stopping_sampler(monkeypatch):
    m = monitor_module()

    async def denied(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(m, "command", denied)
    sampler = m.ResourceMonitor(None)
    await sampler.sample()
    out = sampler.buffer.report(1)
    assert out["summary"]["resource.user_cpu_s"]["peak"] >= 0
    assert "ps" in sampler.unavailable and "docker" in sampler.unavailable


def test_markdown_contains_peak_steady_and_growth():
    m = monitor_module()
    text = "\n".join(
        m.markdown({"summary": {"rss_kb": {"peak": 3, "steady": 2, "growth_per_1000_runs": 1}}})
    )
    assert "peak" in text and "steady" in text and "1,000" in text


async def test_mysql_status_and_processlist_sampling():
    m = monitor_module()
    queries = []

    async def rows(query):
        queries.append(query)
        if query == "SHOW GLOBAL STATUS":
            return [
                {"Variable_name": "Threads_connected", "Value": "4"},
                {"Variable_name": "Bytes_sent", "Value": "123"},
            ]
        return [{"state": "Sleep", "n": 3}]

    values = await m.database_sample("mysql", rows)
    assert values["db.connections"] == 3
    assert values["server.Threads_connected"] == 4
    assert values["db.Bytes_sent"] == 123
    assert values["db.processlist.Sleep"] == 3
    assert any("information_schema.PROCESSLIST" in q for q in queries)


async def test_postgres_sample_has_activity_waits_queue_and_wal():
    m = monitor_module()
    queries = []

    async def rows(query):
        queries.append(query)
        if "GROUP BY" in query:
            return [{"state": "idle", "n": 2, "waits": 1, "listeners": 1}]
        return [
            {
                "queue_usage": 0.1,
                "wal_bytes": 123,
                "shared_buffers_bytes": 100,
                "work_mem_bytes": 50,
            }
        ]

    values = await m.database_sample("postgresql", rows)
    assert values["db.connections"] == 2 and values["db.lock_waits"] == 1
    assert values["db.wal_bytes"] == 123
    assert any("pg_notification_queue_usage()" in q for q in queries)


def test_mysql_sql_translation_handles_bind_list_aliases_and_interval():
    assert hasattr(load, "mysql_control_sql"), "portable controller queries missing"
    query, params = load.mysql_control_sql(
        "select count(*) filter (where status <> all($1::text[])) open from agent_runs",
        (["success", "error"],),
    )
    assert "filter" not in query.lower() and "not in" in query.lower()
    assert "`open`" in query and params == {"terminal0": "success", "terminal1": "error"}
    query, _ = load.mysql_control_sql(
        "update agent_runs set last_progress_at = now() - interval '10 minutes'", ()
    )
    assert "INTERVAL 10 MINUTE" in query


async def test_failed_workload_preserves_monitoring_and_closes_engine(monkeypatch):
    m = monitor_module()
    engine = SimpleNamespace(dispose=None)
    disposed = []

    async def dispose():
        disposed.append(True)

    engine.dispose = dispose
    monkeypatch.setattr(load, "make_engine", lambda *a, **kw: engine)

    async def start(self):
        pass

    async def stop(self, runs):
        return {"summary": {"app.rss_kb": {"peak": 42}}, "runs": runs}

    monkeypatch.setattr(m.ResourceMonitor, "start", start)
    monkeypatch.setattr(m.ResourceMonitor, "stop", stop)

    async def fail():
        raise RuntimeError("workload failed")

    args = SimpleNamespace(monitor_interval=1, monitor_capacity=10)
    with pytest.raises(RuntimeError) as exc:
        await load.monitored_job(MYSQL, fail, args)
    assert getattr(exc.value, "resource_report", None) is not None
    assert disposed
    assert load.ACTIVE_MONITOR.get() is None


def test_pool_checkout_latency_and_waits_are_measured():
    assert hasattr(load, "LoadPool"), "pool checkout timing missing"
    pool = load.LoadPool(lambda: object(), pool_size=1, max_overflow=0)
    assert hasattr(pool, "checkout_latencies_ms")


def test_cli_validates_sampling_and_leak_options_before_db_access():
    for options in [
        ["--monitor-interval", "nan"],
        ["--monitor-capacity", "1"],
        ["--leak-cycles", "0"],
        ["--leak-thresholds", "[]"],
    ]:
        with pytest.raises(SystemExit) as exc:
            load.main(["--url", MYSQL, *options])
        assert exc.value.code == 2


async def test_slow_docker_does_not_delay_app_and_database_sampling(monkeypatch):
    m = monitor_module()
    gate = asyncio.Event()

    async def probe(*args):
        if args[0] == "docker":
            await gate.wait()
        return ""

    monkeypatch.setattr(m, "command", probe)
    sampler = m.ResourceMonitor(None, interval=0.01)
    await asyncio.wait_for(sampler.start(), 0.2)
    await asyncio.sleep(0.04)
    assert sampler.buffer.total >= 3
    await asyncio.wait_for(sampler.stop(1), 0.2)


async def test_docker_probe_targets_only_named_test_containers_and_parses_io(monkeypatch):
    import json

    m = monitor_module()
    calls = []

    async def probe(*args):
        calls.append(args)
        if args[1] == "ps":
            return "dendrux-test-mysql\nowner-postgres\ndendrux-test-mariadb\n"
        return json.dumps(
            {
                "Name": "dendrux-test-mysql",
                "CPUPerc": "2.5%",
                "MemUsage": "10MiB / 1GiB",
                "PIDs": "5",
                "NetIO": "1kB / 2kB",
                "BlockIO": "3MB / 4MB",
            }
        )

    monkeypatch.setattr(m, "command", probe)
    sampler = m.ResourceMonitor(None)
    await sampler._docker_sample()
    assert calls[1] == (
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        "dendrux-test-mysql",
        "dendrux-test-mariadb",
    )
    assert sampler.docker_values["docker.dendrux-test-mysql.memory_bytes"] == 10 * 2**20
    assert sampler.docker_values["docker.dendrux-test-mysql.network_out_bytes"] == 2000
    assert sampler.docker_values["docker.dendrux-test-mysql.disk_in_bytes"] == 3_000_000


async def test_process_samples_include_children_threads_and_host_postgres(monkeypatch):
    import os
    import time

    m = monitor_module()
    pid = os.getpid()

    async def probe(*args):
        return (
            f"{pid} 1 100 2.0 3 python\n{pid + 1} {pid} 50 1.0 2 python\n1 0 300 4.0 5 postgres\n"
        )

    monkeypatch.setattr(m, "command", probe)
    sampler = m.ResourceMonitor(None)
    sampler.workers[pid + 1] = (time.monotonic(), {"tasks": 5, "fds": 8, "pool.checked_out": 2})
    await sampler.sample()
    values = sampler.buffer.samples[-1]
    assert values["app.rss_kb"] == 150 and values["app.cpu_percent"] == 3
    assert values["app.threads"] == 5 and values["postgres.rss_kb"] == 300
    assert values["workers.pool.checked_out"] == 2
    assert f"pid.{pid}.rss_kb" in values
    assert not any(k.startswith("pid.") for k in sampler.buffer.peaks)


async def test_ps_falls_back_when_nlwp_is_unsupported(monkeypatch):
    import os

    m = monitor_module()

    async def probe(*args):
        if "nlwp" in args[-1]:
            raise RuntimeError("unsupported")
        return f"{os.getpid()} 1 100 2 python\n"

    monkeypatch.setattr(m, "command", probe)
    sampler = m.ResourceMonitor(None)
    await sampler.sample()
    assert sampler.buffer.samples[-1]["app.rss_kb"] == 100
    assert "threads" in sampler.unavailable


async def test_system_command_success_failure_and_cancellation_reap_child():
    import sys

    m = monitor_module()
    assert (await m.command(sys.executable, "-c", "print(42)")).strip() == "42"
    with pytest.raises(RuntimeError):
        await m.command(sys.executable, "-c", "raise SystemExit(2)")
    task = asyncio.create_task(m.command(sys.executable, "-c", "import time; time.sleep(30)"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_database_observer_closes_and_reports_wal_delta(monkeypatch):
    m = monitor_module()
    closed = []

    async def close():
        closed.append(True)

    async def execute(query):
        if "GROUP BY" in str(query):
            records = [{"state": "idle", "n": 1, "waits": 0, "listeners": 0}]
        else:
            records = [
                {
                    "queue_usage": 0.0,
                    "wal_bytes": 100,
                    "shared_buffers_bytes": 1000,
                    "work_mem_bytes": 50,
                }
            ]
        return SimpleNamespace(mappings=lambda: records)

    conn = SimpleNamespace(execute=execute, close=close)

    async def connect():
        return conn

    async def probe(*args):
        return ""

    monkeypatch.setattr(m, "command", probe)
    sampler = m.ResourceMonitor(
        SimpleNamespace(connect=connect, dialect=SimpleNamespace(name="postgresql"))
    )
    await sampler.start()
    report = await sampler.stop(1)
    assert closed and report["wal_bytes_delta"] == 0
    assert "database" not in report["unavailable"]


async def test_database_probe_failure_is_explicit_and_observer_is_closed(monkeypatch):
    m = monitor_module()
    closed = []

    async def close():
        closed.append(True)

    async def execute(query):
        raise PermissionError("unavailable")

    async def probe(*args):
        return ""

    monkeypatch.setattr(m, "command", probe)
    sampler = m.ResourceMonitor(SimpleNamespace(dialect=SimpleNamespace(name="mysql")))
    sampler.conn = SimpleNamespace(execute=execute, close=close)
    report = await sampler.stop(1)
    assert report["unavailable"]["database"] == "PermissionError"
    assert "db.connections" not in report["summary"] and closed


@pytest.mark.parametrize("value, expected", [("1 KiB", 1024), ("42B", 42), ("2.5MB", 2_500_000)])
def test_docker_size_units(value, expected):
    m = monitor_module()
    assert m._size(value) == expected


async def test_invalid_probe_configuration_and_sample_values():
    m = monitor_module()
    with pytest.raises(ValueError):
        m.SampleBuffer(1)
    with pytest.raises(ValueError):
        m.ResourceMonitor(None, interval=float("nan"))
    samples = m.SampleBuffer(2)
    samples.add({"rss": float("nan")})
    assert samples.report(0)["summary"] == {}
    with pytest.raises(ValueError):
        m._size("invalid")


def test_mysql_measurement_requires_predeclared_statement_budgets(capsys, monkeypatch):
    monkeypatch.setattr(load.asyncio, "run", lambda coro: (coro.close(), 99)[1])
    assert load.main(["--url", MYSQL, "--scenarios", "runs"]) == 2
    assert "budget" in capsys.readouterr().err.lower()


def test_cycle_budget_checks_include_every_cycle_not_only_the_last():
    assert hasattr(load, "cycle_gates"), "worst-cycle statement gate aggregation missing"
    assert load.cycle_gates([{"gates": {"runs": 50}}, {"gates": {"runs": 40}}]) == {"runs": 50}


def test_driver_counters_are_separate_from_application_meter():
    m = monitor_module()
    assert hasattr(m, "driver_counters"), "server statement counters missing"


async def test_mysql_worker_probe_has_pool_timings_and_connections():
    m = monitor_module()
    pool = SimpleNamespace(
        checkedout=lambda: 2,
        checkedin=lambda: 3,
        overflow=lambda: -5,
        checkout_peak_ms=9.0,
        checkout_waits=4,
        checkout_attempts=6,
    )
    values = m.local_sample([SimpleNamespace(pool=pool)])
    assert values["pool.connections"] == 5
    assert values["pool.checkout_peak_ms"] == 9
    assert values["pool.checkout_waits"] == 4


def test_growth_denominator_counts_pool_variants_and_explicit_run_totals():
    assert load.workload_runs({"runs_per_variant": 50, "queued": {}, "short_timeout": {}}) == 100
    assert load.workload_runs({"resource_runs": 12, "runs_per_cycle": 2, "cycles": 3}) == 12


async def test_postgres_driver_counters_use_private_database_scope():
    m = monitor_module()
    assert hasattr(m, "driver_counters"), "server statement counters missing"
    queries = []

    async def rows(query):
        queries.append(query)
        return [{"calls": 123}]

    values = await m.driver_counters("postgresql", rows)
    assert values["server_statements.calls"] == 123
    assert "current_database()" in queries[0]


async def test_mysql_driver_counters_count_protocol_and_sql_events():
    m = monitor_module()
    assert hasattr(m, "driver_counters"), "server statement counters missing"
    queries = []

    async def rows(query):
        queries.append(query)
        return [{"calls": 456}]

    values = await m.driver_counters("mysql", rows)
    assert values["server_statements.calls"] == 456
    assert "performance_schema" in queries[0] and "statement/%" in queries[0]


def test_iteration_render_supports_mysql_without_notify_on_variants():
    variant = {
        "startup": 1,
        "iteration_1": 18,
        "iteration_2": 18,
        "terminal": 1,
        "max_non_terminal": 18,
        "by_operation_iteration_2": {"checkpoint_step": 5},
        "by_operation_startup": {},
        "by_operation_terminal": {},
        "model_calls": 3,
        "latency_ms": {"n": 1},
    }
    out = load.Outcome(metrics={"variants": {"notify_off_batch": variant}, "server": {}})
    assert "checkpoint_step" in "\n".join(load.render("iteration", out))


def test_postgres_dsn_adapter_refuses_mysql():
    with pytest.raises(ValueError):
        load.dsn_of(MYSQL)


def test_iteration_budgets_are_fixed_at_21_on_and_18_off():
    assert hasattr(load, "iteration_budget_checks"), "fixed iteration budgets missing"
    variants = {
        "notify_on_batch": {"max_non_terminal": 22},
        "notify_off_batch": {"max_non_terminal": 19},
    }
    checks = load.iteration_budget_checks(variants)
    assert [c["ok"] for c in checks] == [False, False]
    assert [c["limit"] for c in checks] == [21, 18]


async def test_pool_records_real_wait_latency(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}",
        poolclass=load.LoadPool,
        pool_size=1,
        max_overflow=0,
    )
    try:
        held = await engine.connect()
        waiter = asyncio.ensure_future(engine.connect())
        await asyncio.sleep(0.04)
        assert not waiter.done()
        await held.close()
        acquired = await waiter
        await acquired.close()
        assert engine.pool.checkout_waits >= 1
        assert engine.pool.checkout_peak_ms >= 30
    finally:
        await engine.dispose()


def test_incomplete_mysql_budget_file_is_refused_before_starting_workload(
    tmp_path, capsys, monkeypatch
):
    budget = tmp_path / "budgets.json"
    budget.write_text('{"other.gate": 1}')
    monkeypatch.setattr(load.asyncio, "run", lambda coro: (coro.close(), 99)[1])
    assert (
        load.main(["--url", MYSQL, "--scenarios", "runs", "--statement-budgets", str(budget)]) == 2
    )
    assert "runs.statements_per_run" in capsys.readouterr().err
