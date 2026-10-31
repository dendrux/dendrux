"""Bounded, best-effort hardware probes for the load harness; no optional dependencies.

Database counters and Docker I/O are server-wide. RSS sums include shared pages in
multiple processes. PostgreSQL shared_buffers/work_mem are configuration limits,
not measurements of resident shared pages or allocated per-query work memory.
Missing probes are explicit; they never become synthetic zero measurements.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROBE_TIMEOUT_S = 5.0
PROBE_PIDS: set[int] = set()
RECENT_PROBE_PIDS: collections.deque[tuple[int, float]] = collections.deque(maxlen=256)

DEFAULT_THRESHOLDS = {"rss_kb": 16384.0, "fds": 1.0, "tasks": 1.0, "connections": 1.0}


class SampleBuffer:
    """Keep bounded raw samples and streaming peaks/endpoints across the whole run."""

    def __init__(self, capacity: int = 3600) -> None:
        if capacity < 2:
            raise ValueError("sample capacity must be at least 2")
        self.samples: collections.deque[dict[str, float]] = collections.deque(maxlen=capacity)
        self.total = 0
        self.first: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.peaks: dict[str, float] = {}

    def add(self, values: dict[str, float]) -> None:
        """Record one observation; metric names are bounded by probe scope."""
        self.total += 1
        self.samples.append(values)
        self.last = {}
        for key, value in values.items():
            if math.isfinite(value):
                self.first.setdefault(key, value)
                self.last[key] = value
                self.peaks[key] = max(self.peaks.get(key, value), value)

    def report(self, runs: int) -> dict[str, Any]:
        """Summarize retained tail samples and whole-window peaks and growth."""
        retained = list(self.samples)
        tail = retained[len(retained) // 2 :]
        summary = {}
        for key, peak in self.peaks.items():
            values = [s[key] for s in tail if key in s]
            summary[key] = {
                "peak": peak,
                "steady": statistics.median(values) if values else None,
                "growth_per_1000_runs": (
                    (self.last[key] - self.first[key]) * 1000 / runs
                    if runs and key in self.last
                    else None
                ),
            }
        return {
            "total_samples": self.total,
            "samples": retained,
            "summary": summary,
            "steady_window": "median of last 50% of retained samples",
            "truncated": self.total > len(retained),
            "runs": runs,
        }


def leak_checks(
    cycles: list[dict[str, float]], thresholds: dict[str, float], *, warmup: int = 1
) -> list[dict[str, Any]]:
    """Fail missing probes or positive linear growth above absolute units per cycle."""
    checks = []
    for metric, limit in thresholds.items():
        values = [c.get(metric) for c in cycles[warmup:]]
        slope = None
        if len(values) >= 2 and all(v is not None and v >= 0 for v in values):
            mean_x = (len(values) - 1) / 2
            mean_y = statistics.fmean(values)
            slope = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / sum(
                (i - mean_x) ** 2 for i in range(len(values))
            )
        checks.append(
            {
                "name": f"leak {metric} per cycle",
                "ok": slope is not None
                and (
                    slope <= 0
                    if limit == 0
                    else slope < limit
                    if metric in {"fds", "connections"}
                    else slope <= limit
                ),
                "value": slope,
                "limit": limit,
            }
        )
    return checks


async def command(*args: str) -> str:
    """Run a read-only system probe with a deadline and reap it on cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    PROBE_PIDS.add(proc.pid)
    RECENT_PROBE_PIDS.append((proc.pid, time.monotonic()))
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), PROBE_TIMEOUT_S)
        if proc.returncode:
            raise RuntimeError("probe exited unsuccessfully")
        return stdout.decode(errors="replace")
    finally:
        try:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            await proc.wait()
        finally:
            PROBE_PIDS.discard(proc.pid)


def parse_ps(output: str, *, threads: bool = True) -> list[dict[str, Any]]:
    """Parse numeric ps columns without storing process arguments."""
    rows = []
    for line in output.splitlines():
        fields = line.split(None, 5 if threads else 4)
        try:
            row = {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "rss_kb": int(fields[2]),
                "cpu_percent": float(fields[3]),
                "command": fields[-1],
            }
            if threads:
                row["threads"] = int(fields[4])
            rows.append(row)
        except (ValueError, IndexError):
            continue
    return rows


def descendants(rows: list[dict[str, Any]], pid: int) -> list[dict[str, Any]]:
    """Select the application process tree, including grandchildren."""
    selected = {pid}
    while True:
        more = {r["pid"] for r in rows if r["ppid"] in selected} - selected
        if not more:
            return [r for r in rows if r["pid"] in selected]
        selected.update(more)


def local_sample(engines: Any = ()) -> dict[str, float]:
    """Read current-process counters and pool occupancy, including idle connections."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    values = {
        "tasks": float(len(asyncio.all_tasks())),
        "resource.user_cpu_s": usage.ru_utime,
        "resource.system_cpu_s": usage.ru_stime,
        "resource.maxrss_kb": usage.ru_maxrss / (1024 if sys.platform == "darwin" else 1),
        "resource.input_blocks": float(usage.ru_inblock),
        "resource.output_blocks": float(usage.ru_oublock),
    }
    for path in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            values["fds"] = float(len(list(path.iterdir())))
            break
        except OSError:
            continue
    if children.ru_utime or children.ru_stime:
        values["resource.reaped_children_user_cpu_s"] = children.ru_utime
        values["resource.reaped_children_system_cpu_s"] = children.ru_stime
    pools = [
        e.pool
        for e in engines
        if all(hasattr(e.pool, name) for name in ("checkedout", "checkedin", "overflow"))
    ]
    if pools:
        values["pool.checked_out"] = float(sum(p.checkedout() for p in pools))
        values["pool.overflow"] = float(sum(max(0, p.overflow()) for p in pools))
        values["pool.connections"] = values["pool.checked_out"] + sum(p.checkedin() for p in pools)
        for key in ("checkout_peak_ms", "checkout_waits", "checkout_attempts"):
            if all(hasattr(pool, key) for pool in pools):
                values[f"pool.{key}"] = sum(getattr(pool, key) for pool in pools)
    io = Path("/proc/self/io")
    if io.exists():
        with contextlib.suppress(OSError, ValueError):
            for line in io.read_text().splitlines():
                key, value = line.split(":")
                if key in {"read_bytes", "write_bytes"}:
                    values[f"process_io.{key}"] = float(value)
    return values


async def database_sample(backend: str, rows: Any) -> dict[str, float]:
    """Read activity and monotonic server counters using a caller-owned connection."""
    values: dict[str, float] = {}
    if backend == "postgresql":
        states = await rows(
            "SELECT coalesce(state, 'unknown') state, count(*) n, "
            "count(*) FILTER (WHERE wait_event_type = 'Lock') waits, "
            "count(*) FILTER (WHERE application_name = '' OR application_name LIKE '%listen%') "
            "listeners FROM pg_stat_activity WHERE datname = current_database() GROUP BY state"
        )
        values["db.connections"] = float(sum(r["n"] for r in states))
        values["db.lock_waits"] = float(sum(r["waits"] for r in states))
        for row in states:
            values[f"db.activity.{row['state']}"] = float(row["n"])
        row = (
            await rows(
                "SELECT pg_notification_queue_usage() queue_usage, "
                "pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0')::float8 wal_bytes, "
                "pg_size_bytes(current_setting('shared_buffers')) shared_buffers_bytes, "
                "pg_size_bytes(current_setting('work_mem')) work_mem_bytes"
            )
        )[0]
        values.update({f"db.{k}": float(v) for k, v in row.items()})
    else:
        names = {
            "Threads_connected",
            "Innodb_row_lock_waits",
            "Innodb_data_written",
            "Bytes_sent",
            "Bytes_received",
        }
        for row in await rows("SHOW GLOBAL STATUS"):
            if row["Variable_name"] in names:
                values[f"db.{row['Variable_name']}"] = float(row["Value"])
        if "db.Threads_connected" in values:
            values["server.Threads_connected"] = values.pop("db.Threads_connected")
        processlist = await rows(
            "SELECT coalesce(COMMAND, 'unknown') state, count(*) n "
            "FROM information_schema.PROCESSLIST WHERE DB = DATABASE() GROUP BY COMMAND"
        )
        values["db.connections"] = float(sum(row["n"] for row in processlist))
        for row in processlist:
            values[f"db.processlist.{row['state']}"] = float(row["n"])
    return values


async def driver_counters(backend: str, rows: Any) -> dict[str, float]:
    """Read server-observed statement totals separately from application cursor events.

    Postgres totals are scoped to the private DB. MySQL performance_schema totals
    are server-wide and include SQL/protocol statement events when instrumentation
    is enabled. Neither is an exact wire-byte count or isolated application cost.
    """
    query = (
        "SELECT sum(calls) calls FROM pg_stat_statements WHERE dbid = "
        "(SELECT oid FROM pg_database WHERE datname = current_database())"
        if backend == "postgresql"
        else "SELECT sum(COUNT_STAR) calls "
        "FROM performance_schema.events_statements_summary_global_by_event_name "
        "WHERE EVENT_NAME LIKE 'statement/%'"
    )
    row = (await rows(query))[0]
    if row["calls"] is None:
        raise ValueError("statement instrumentation unavailable")
    return {"server_statements.calls": float(row["calls"])}


def _size(value: str) -> float:
    number, unit = value.strip().split() if " " in value.strip() else (None, None)
    if number is None:
        import re

        match = re.fullmatch(r"([\d.]+)([A-Za-z]+)?", value.strip())
        if not match:
            raise ValueError("unrecognized size")
        number, unit = match.groups()
    return (
        float(number)
        * {
            None: 1,
            "B": 1,
            "kB": 1000,
            "KB": 1000,
            "KiB": 1024,
            "MB": 10**6,
            "MiB": 2**20,
            "GB": 10**9,
            "GiB": 2**30,
        }[unit]
    )


class ResourceMonitor:
    """Sample resources around a workload; unavailable probes carry typed reasons."""

    def __init__(self, engine: Any, *, interval: float = 1, capacity: int = 3600) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("sample interval must be positive and finite")
        self.engine = engine
        self.interval = interval
        self.buffer = SampleBuffer(capacity)
        self.task: asyncio.Task | None = None
        self.unavailable: dict[str, str] = {
            "postgres_memory": (
                "shared_buffers/work_mem are configured limits; allocation unavailable"
            ),
            "database_fds": "not available from SQL; host /proc probes used when accessible",
            "driver_traffic": (
                "application SQL counted separately; server statement counters require "
                "extension/permissions"
            ),
            "app_network": "per-process network bytes unavailable without privileged OS tracing",
            "pgbouncer": "not configured for this direct database run",
            "redis": "not used by these workloads",
            "dedicated_connections": (
                "activity is sampled; listener/update roles not reliably identifiable"
            ),
        }
        self.workers: dict[int, tuple[float, dict[str, float]]] = {}
        self.conn: Any = None
        self.docker_task: asyncio.Task | None = None
        self.docker_values: dict[str, float] = {}
        self.rss_reader: Any = None
        self.driver_probe_failed = False
        self.ps_threads_supported = True
        self.unavailable["docker"] = "not yet sampled"

    async def start(self) -> None:
        """Start observers with bounded connect/sample latency and explicit failures."""
        if self.engine is not None:
            try:
                self.conn = await asyncio.wait_for(self.engine.connect(), PROBE_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                self.unavailable["connect"] = type(exc).__name__
        self.docker_task = asyncio.create_task(self._docker_loop())
        try:
            await self._safe_sample("sampling")
        except BaseException:
            await self.stop(0)
            raise
        self.task = asyncio.create_task(self._loop())

    async def _safe_sample(self, label: str) -> None:
        try:
            await asyncio.wait_for(self.sample(), PROBE_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            self.unavailable[label] = type(exc).__name__
            await self._rollback()

    async def _rollback(self) -> None:
        if self.conn is not None and hasattr(self.conn, "rollback"):
            try:
                await asyncio.wait_for(self.conn.rollback(), PROBE_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                self.unavailable["rollback"] = type(exc).__name__

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            await self._safe_sample("sampling")

    async def sample(self) -> None:
        """Collect independent probes; one missing capability does not hide the others."""
        values = local_sample()
        values["elapsed_s"] = time.monotonic()
        for key, value in list(values.items()):
            values[f"pid.{os.getpid()}.{key}"] = value
        workers = [
            v
            for ts, v in self.workers.values()
            if time.monotonic() - ts < max(3, self.interval * 3)
        ]
        for pid, (ts, data) in self.workers.items():
            if time.monotonic() - ts < max(3, self.interval * 3):
                for key, value in data.items():
                    values[f"pid.{pid}.{key}"] = value
        for key in {k for w in workers for k in w}:
            if all(key in w for w in workers):
                values[f"workers.{key}"] = sum(w[key] for w in workers)
            else:
                self.unavailable[f"workers.{key}"] = "incomplete worker telemetry"
        values["app.tasks"] = values["tasks"] + sum(w.get("tasks", 0) for w in workers)
        if "fds" in values and all("fds" in w for w in workers):
            values["app.fds"] = values["fds"] + sum(w["fds"] for w in workers)
        else:
            self.unavailable["app.fds"] = "one or more process fd probes unavailable"
        if not any(k.startswith("pool.") for k in values):
            self.unavailable["pool"] = "no observable local pool (no engines or NullPool)"
        if "resource.reaped_children_user_cpu_s" not in values:
            self.unavailable["child_cpu"] = "no reaped-child CPU usage observed"
        try:
            if self.ps_threads_supported:
                try:
                    output = await command("ps", "-axo", "pid=,ppid=,rss=,pcpu=,nlwp=,comm=")
                    rows = parse_ps(output)
                except RuntimeError:
                    self.ps_threads_supported = False
                    self.unavailable["threads"] = "ps nlwp unsupported on this host"
            if not self.ps_threads_supported:
                output = await command("ps", "-axo", "pid=,ppid=,rss=,pcpu=,comm=")
                rows = parse_ps(output, threads=False)
            excluded = PROBE_PIDS | {
                pid for pid, ts in RECENT_PROBE_PIDS if time.monotonic() - ts < PROBE_TIMEOUT_S * 2
            }
            while True:
                more = {r["pid"] for r in rows if r["ppid"] in excluded} - excluded
                if not more:
                    break
                excluded.update(more)
            app = [r for r in descendants(rows, os.getpid()) if r["pid"] not in excluded]
            server = [r for r in rows if Path(r["command"]).name in {"postgres", "postmaster"}]
            for prefix, processes in (("app", app), ("postgres", server)):
                if not processes:
                    self.unavailable[prefix + "_processes"] = "no visible matching processes"
                    continue
                for key in ("rss_kb", "cpu_percent", "threads"):
                    if all(key in p for p in processes):
                        values[f"{prefix}.{key}"] = sum(p[key] for p in processes)
                for p in processes:
                    proc_dir = Path(f"/proc/{p['pid']}")
                    with contextlib.suppress(OSError):
                        values[f"pid.{p['pid']}.fds"] = float(
                            len(list((proc_dir / "fd").iterdir()))
                        )
                    with contextlib.suppress(OSError, ValueError):
                        for line in (proc_dir / "io").read_text().splitlines():
                            key, value = line.split(":")
                            if key in {"read_bytes", "write_bytes"}:
                                values[f"{prefix}.disk.{key}"] = values.get(
                                    f"{prefix}.disk.{key}", 0
                                ) + float(value)
                # Bounded by the ring; retain per-PID samples without unbounded summary keys.
                for p in processes:
                    for key in ("rss_kb", "cpu_percent", "threads"):
                        if key in p:
                            values[f"pid.{p['pid']}.{key}"] = p[key]
        except (OSError, RuntimeError, TimeoutError) as exc:
            self.unavailable["ps"] = type(exc).__name__
        values.update(self.docker_values)
        if self.conn is not None:
            from sqlalchemy import text

            async def rows(query: str) -> list[dict[str, Any]]:
                result = await self.conn.execute(text(query))
                return [dict(r) for r in result.mappings()]

            try:
                values.update(
                    await asyncio.wait_for(
                        database_sample(self.engine.dialect.name, rows), PROBE_TIMEOUT_S
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.unavailable["database"] = type(exc).__name__
            if not self.driver_probe_failed:
                try:
                    values.update(
                        await asyncio.wait_for(
                            driver_counters(self.engine.dialect.name, rows), PROBE_TIMEOUT_S
                        )
                    )
                    self.unavailable.pop("driver_traffic", None)
                except Exception as exc:  # noqa: BLE001
                    self.driver_probe_failed = True
                    self.unavailable["driver_traffic"] = type(exc).__name__
        if "app.rss_kb" not in values and all("rss_kb" in w for w in workers):
            # Worker RSS is current RSS from the harness's platform-specific probe.
            own_rss = self.rss_reader() if self.rss_reader else values["resource.maxrss_kb"]
            values["app.rss_kb"] = own_rss + sum(w.get("rss_kb", 0) for w in workers)
            self.unavailable["app_rss_scope"] = (
                "ps unavailable; platform RSS probe (ru_maxrss fallback) plus worker current RSS"
            )
        self.buffer.add(values)
        self.workers = {
            pid: (ts, data)
            for pid, (ts, data) in self.workers.items()
            if time.monotonic() - ts < max(3, self.interval * 3)
        }
        # Per-PID summaries would otherwise accumulate across an unbounded number of workers.
        for mapping in (self.buffer.first, self.buffer.last, self.buffer.peaks):
            for key in list(mapping):
                if key.startswith("pid."):
                    del mapping[key]

    async def _docker_loop(self) -> None:
        while True:
            try:
                await self._docker_sample()
            except Exception as exc:  # noqa: BLE001
                self.docker_values = {}
                self.unavailable["docker"] = type(exc).__name__
            await asyncio.sleep(self.interval)

    async def _docker_sample(self) -> None:
        values = {}
        try:
            ids = (
                await command(
                    "docker", "ps", "--filter", "name=^/dendrux-test-", "--format", "{{.Names}}"
                )
            ).splitlines()
            ids = [n for n in ids if n.startswith("dendrux-test-")]
            if ids:
                output = await command(
                    "docker", "stats", "--no-stream", "--format", "{{json .}}", *ids
                )
                for line in output.splitlines():
                    row = json.loads(line)
                    prefix = f"docker.{row['Name']}."
                    values[prefix + "cpu_percent"] = float(row["CPUPerc"].rstrip("%"))
                    values[prefix + "memory_bytes"] = _size(row["MemUsage"].split("/")[0])
                    values[prefix + "pids"] = float(row["PIDs"])
                    for field, label in (("NetIO", "network"), ("BlockIO", "disk")):
                        incoming, outgoing = row[field].split("/")
                        values[prefix + label + "_in_bytes"] = _size(incoming)
                        values[prefix + label + "_out_bytes"] = _size(outgoing)
            else:
                self.unavailable["docker"] = "no running dendrux-test-* containers"
        except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as exc:
            self.unavailable["docker"] = type(exc).__name__
            values = {}
        except asyncio.CancelledError:
            self.docker_values = {}
            raise
        self.docker_values = values
        if values:
            self.unavailable.pop("docker", None)

    async def stop(self, runs: int) -> dict[str, Any]:
        """Stop, take an endpoint sample, close the observer, and return JSON-safe data."""
        tasks = [task for task in (self.task, self.docker_task) if task is not None]
        for task in tasks:
            task.cancel()
        self.buffer.last = {}
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._rollback()
            try:
                await asyncio.wait_for(self.sample(), PROBE_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                self.unavailable["final_sample"] = type(exc).__name__
                self.buffer.last = {}
        finally:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                if self.conn is not None:
                    conn = self.conn

                    async def close_connection() -> None:
                        try:
                            await asyncio.wait_for(conn.close(), PROBE_TIMEOUT_S)
                        except Exception as exc:  # noqa: BLE001
                            self.unavailable["close"] = type(exc).__name__
                        finally:
                            self.conn = None

                    closing = asyncio.create_task(close_connection())
                    try:
                        await asyncio.shield(closing)
                    finally:
                        # Complete bounded cleanup before propagating caller cancellation.
                        await closing
        report = self.buffer.report(runs)
        report.update(
            interval_s=self.interval,
            unavailable=dict(self.unavailable),
            endpoint=dict(self.buffer.last),
            cpu_percent_semantics="ps pcpu is a non-instantaneous lifetime CPU average",
        )
        if "db.wal_bytes" in self.buffer.last:
            report["wal_bytes_delta"] = (
                self.buffer.last["db.wal_bytes"] - self.buffer.first["db.wal_bytes"]
            )
        return report


def markdown(report: dict[str, Any]) -> list[str]:
    """Render a compact resource table alongside the full JSON samples."""
    lines = ["| resource | peak | steady | growth / 1,000 runs |", "|---|---:|---:|---:|"]
    for key in (
        "app.rss_kb",
        "app.cpu_percent",
        "app.fds",
        "app.tasks",
        "db.connections",
        "db.lock_waits",
        "db.wal_bytes",
        "rss_kb",
    ):
        if key in report["summary"]:
            row = report["summary"][key]
            lines.append(
                f"| {key} | {row['peak']} | {row['steady']} | {row['growth_per_1000_runs']} |"
            )
    return lines
