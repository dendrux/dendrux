"""Load script: scale evidence for the seven checks in AGENTS.md section 4.

Usage (from packages/python)::

    python tools/load.py --url postgresql+asyncpg://u:p@localhost:5432/dendrux_load_test \\
        --scale full --json load-results.json
    python tools/load.py --url ... --scale quick --baseline tools/load_baseline.json

The database must end in ``_test``: the script drops and recreates every table in it.
The provider is scripted (no network). ``--workers N`` OS processes each own a pool
(``pool_size=10``, ``max_overflow=5``), so the aggregate pool ceiling is ``15 * N``.
The harness adds a few connections of its own (a sampler and an admin connection in the
controller, a billing connection per worker); the report lists every connection by role.

Scenarios: ``a`` (the 0.5 client-tool pause/submit run), ``iteration`` (statements per
one-tool iteration), ``runs`` (concurrent runs), ``kill_checkpoint`` and ``kill_inflight``
(worker processes killed with SIGKILL, then recovered), ``followers_on`` and
``followers_off`` (notifier on and off), ``cancel`` (a large delegation tree),
``pool`` (pool exhaustion), ``history`` (a near-16 MiB history resumed in a new process)
and ``memory`` (repeated cycles in one process).

The script also runs against a v0.5.0 checkout (``a``, ``runs``, ``pool``, ``history``):
point PYTHONPATH at that checkout's ``src``.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import functools
import gc
import inspect
import json
import math
import multiprocessing
import os
import platform
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "load_baseline.json"

POOL = {"pool_size": 10, "max_overflow": 5, "pool_timeout": 120, "pool_pre_ping": True}
POOL_CEILING = POOL["pool_size"] + POOL["max_overflow"]
CHANNEL = "dendrux_events"
SAMPLE_INTERVAL_S = 0.01
BIG_CHARS = 985_000  # 17 results put the full history just below the 16 MiB rebuild bound
V050_REFERENCE_A = 59.0  # recorded 2026-10-27, commit dca4f05e (owner-docs load-test-results.md)

LABEL: ContextVar[str] = ContextVar("load_label", default="other")
OP: ContextVar[str] = ContextVar("load_op", default="-")

PRESETS: dict[str, dict[str, Any]] = {
    "full": {
        "a": 200,
        "iteration_runs": 3,
        "runs": 1000,
        "kill_runs": 1000,
        "kill_fraction": 0.05,
        "followers_runs": 1000,
        "followers_per_run": 10,
        "cancel_nodes": 2000,
        "pool_runs": 200,
        "history_calls": 17,
        "memory_cycles": 50,
        "memory_runs": 100,
        "idle_s": 10.0,
    },
    "quick": {
        "a": 200,
        "iteration_runs": 2,
        "runs": 200,
        "kill_runs": 200,
        "kill_fraction": 0.05,
        "followers_runs": 100,
        "followers_per_run": 10,
        "cancel_nodes": 500,
        "pool_runs": 50,
        "history_calls": 5,
        "memory_cycles": 8,
        "memory_runs": 30,
        "idle_s": 5.0,
    },
    "tiny": {
        "a": 20,
        "iteration_runs": 1,
        "runs": 20,
        "kill_runs": 20,
        "kill_fraction": 0.1,
        "followers_runs": 10,
        "followers_per_run": 3,
        "cancel_nodes": 60,
        "pool_runs": 20,
        "history_calls": 2,
        "memory_cycles": 3,
        "memory_runs": 10,
        "idle_s": 2.0,
    },
}
ALL_SCENARIOS = [
    "a",
    "iteration",
    "runs",
    "kill_checkpoint",
    "kill_inflight",
    "followers_on",
    "followers_off",
    "cancel",
    "pool",
    "history",
    "memory",
]
LEGACY_SCENARIOS = ["a", "runs", "pool", "history"]  # runnable on a v0.5.0 checkout


# ---------------------------------------------------------------------------
# Safety: private databases only
# ---------------------------------------------------------------------------

_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_FORBIDDEN = {"dendrux", "dendrux_test", "postgres", "template0", "template1"}
_OVERRIDES = {"database", "dbname", "host", "hostaddr", "port", "service", "unix_sock"}


def check_private_url(url: str) -> Any:
    """Return the parsed URL, or raise ValueError unless it names a private ``*_test`` database."""
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("load.py runs on PostgreSQL only")
    name = parsed.database or ""
    if not _DB_NAME.match(name) or len(name) > 63:
        raise ValueError("the database name must be 1-63 characters of [A-Za-z0-9_]")
    if not name.endswith("_test"):
        raise ValueError("refusing: the database name must end in _test (the script drops tables)")
    if name in _FORBIDDEN:
        raise ValueError(f"refusing: {name} is a shared database; use a private *_test name")
    if _OVERRIDES & set(parsed.query):
        raise ValueError("refusing: the URL carries query options that override the target")
    return parsed.set(drivername="postgresql+asyncpg")


def dsn_of(url: str) -> str:
    return check_private_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


# ---------------------------------------------------------------------------
# Instrumentation (runs inside every worker)
# ---------------------------------------------------------------------------


def _norm(statement: str) -> str:
    return re.sub(r"\s+", " ", statement).strip()


def _param_bytes(parameters: Any) -> int:
    total = 0
    values = parameters.values() if isinstance(parameters, dict) else parameters
    if not isinstance(values, (list, tuple, type({}.values()))):
        return 0
    for value in values:
        if isinstance(value, (list, tuple)):  # executemany rows
            total += _param_bytes(value)
        elif isinstance(value, (str, bytes)):
            total += len(value)
        elif value is not None:
            total += 8
    return total


class Meter:
    """App-side statement counts per label and operation, bytes sent, pool checkouts."""

    SAMPLE_CAP = 400

    def __init__(self, engine: Any) -> None:
        from sqlalchemy import event

        self.engine = engine
        self.n = 0
        self.by_label: collections.Counter[str] = collections.Counter()
        self.by_label_op: collections.Counter[str] = collections.Counter()
        self.ops: collections.Counter[str] = collections.Counter()
        self.kinds: collections.Counter[str] = collections.Counter()
        self.bytes_sent = 0
        self.samples: dict[str, tuple[str, Any, str, bool]] = {}
        self.override: str | None = None
        self.active = 0
        self.peak_active = 0
        self.checkouts = 0
        sync = engine.sync_engine
        event.listen(sync, "before_cursor_execute", self._execute)
        event.listen(sync, "checkout", self._checkout)
        event.listen(sync, "checkin", self._checkin)

    def _execute(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        self.n += 1
        label = self.override or LABEL.get()
        self.by_label[label] += 1
        self.by_label_op[f"{label}|{OP.get()}"] += 1
        flat = _norm(statement)
        head = flat[:30]
        if head.startswith(("INSERT", "UPDATE")):
            self.bytes_sent += _param_bytes(parameters)
        kind = flat[:220]
        self.kinds[kind] += 1
        small = _param_bytes(parameters) < 65536
        if small and kind not in self.samples and len(self.samples) < self.SAMPLE_CAP:
            self.samples[kind] = (statement, _plain(parameters), label, bool(executemany))

    def _checkout(self, *a: Any) -> None:
        self.active += 1
        self.checkouts += 1
        self.peak_active = max(self.peak_active, self.active)

    def _checkin(self, *a: Any) -> None:
        self.active -= 1

    def op(self, label: str, count: int = 1) -> None:
        self.ops[label] += count

    def snapshot(self) -> dict[str, Any]:
        return {
            "statements": self.n,
            "by_label": dict(self.by_label),
            "by_label_op": dict(self.by_label_op),
            "ops": dict(self.ops),
            "top_statements": self.kinds.most_common(30),
            "bytes_sent": self.bytes_sent,
            "peak_checked_out": self.peak_active,
            "checkouts": self.checkouts,
            "samples": self.samples,
        }


def _plain(parameters: Any) -> Any:
    if isinstance(parameters, dict):
        return {k: _plain(v) for k, v in parameters.items()}
    if isinstance(parameters, (list, tuple)):
        return [_plain(v) for v in parameters]
    if isinstance(parameters, (str, int, float, bool, bytes, type(None))):
        return parameters
    return parameters  # datetimes and the like pickle fine


def merge_meters(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "statements": 0,
        "by_label": collections.Counter(),
        "by_label_op": collections.Counter(),
        "ops": collections.Counter(),
        "top": collections.Counter(),
        "bytes_sent": 0,
        "peak_checked_out_sum": 0,
        "checkouts": 0,
        "samples": {},
    }
    for p in parts:
        out["statements"] += p["statements"]
        out["by_label"].update(p["by_label"])
        out["by_label_op"].update(p["by_label_op"])
        out["ops"].update(p["ops"])
        out["top"].update(dict(p["top_statements"]))
        out["bytes_sent"] += p["bytes_sent"]
        out["peak_checked_out_sum"] += p["peak_checked_out"]
        out["checkouts"] += p["checkouts"]
        for k, v in p["samples"].items():
            out["samples"].setdefault(k, v)
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def lat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50_ms": round(pct(values, 0.5), 1),
        "p95_ms": round(pct(values, 0.95), 1),
        "max_ms": round(max(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
    }


def rss_kb() -> int:
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    if sys.platform == "darwin":
        import ctypes

        class TaskInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", ctypes.c_int * 2),
                ("system_time", ctypes.c_int * 2),
                ("policy", ctypes.c_int),
                ("suspend_count", ctypes.c_int),
            ]

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.mach_task_self.restype = ctypes.c_uint
        info = TaskInfo()
        count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
        # MACH_TASK_BASIC_INFO reports current RSS; ru_maxrss is a lifetime peak.
        result = lib.task_info(lib.mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count))
        if result:
            raise OSError(f"task_info failed with Mach status {result}")
        return info.resident_size // 1024
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return int(out.stdout.strip() or 0)


def fd_count() -> int:
    for path in ("/proc/self/fd", "/dev/fd"):
        with contextlib.suppress(OSError):
            return len(os.listdir(path))
    return -1


# ---------------------------------------------------------------------------
# Scripted provider and tools (self-contained, stateless: the reply depends on the history)
# ---------------------------------------------------------------------------


def make_llm(
    mode: str,
    *,
    hook: Any = None,
    stream_chunks: int = 0,
    usage: bool = False,
    big_calls: int = 0,
    server_calls: int = 1,
) -> Any:
    from dendrux.llm.base import LLMProvider
    from dendrux.types import LLMResponse, ProviderCapabilities, ToolCall, UsageStats

    class LoadLLM(LLMProvider):
        capabilities = ProviderCapabilities(supports_native_tools=True, supports_tool_call_ids=True)

        def __init__(self) -> None:
            self.calls = 0

        @property
        def model(self) -> str:
            return "scripted"

        def _reply(self, key: str, n_tool: int) -> Any:
            name = None
            if mode == "client":
                name = "client_read" if n_tool == 0 else None
            elif mode == "server":
                name = "work" if n_tool < server_calls else None
            elif mode == "big":
                if n_tool < big_calls:
                    name = "big_work"
                elif n_tool == big_calls:
                    name = "client_read"
            kwargs: dict[str, Any] = {}
            if usage:
                kwargs["usage"] = UsageStats(
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                    cost_usd=0.001,
                    cost_source="provider",
                )
            if name is None:
                response = LLMResponse(text="done", **kwargs)
            else:
                response = LLMResponse(
                    tool_calls=[
                        ToolCall(name=name, params={}, provider_tool_call_id=f"{key}-{n_tool}")
                    ],
                    **kwargs,
                )
            response.model = "scripted"
            return response

        async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
            first = next(m for m in messages if m.role.value == "user")
            key = str(first.content).split(" ", 1)[-1]
            n_tool = sum(1 for m in messages if m.role.value == "tool")
            self.calls += 1
            if hook is not None:
                await hook(key, n_tool)
            return self._reply(key, n_tool)

        async def complete_stream(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
            from dendrux.types import StreamEvent, StreamEventType

            response = await self.complete(messages, tools, **kwargs)
            for _ in range(stream_chunks):
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            yield StreamEvent(type=StreamEventType.DONE, raw=response)

    return LoadLLM()


def make_tools(state: dict[str, Any]) -> dict[str, Any]:
    from dendrux import tool

    big = _big_payload()

    @tool(target="client")
    async def client_read() -> str:
        """Read on the client."""
        raise AssertionError("client tool executed on the server")

    # `idempotent=` arrived in 0.6; the same tools run on a 0.5 checkout without it.
    kwargs: dict[str, Any] = (
        {"idempotent": True} if "idempotent" in inspect.signature(tool).parameters else {}
    )

    @tool(**kwargs)
    async def work() -> str:
        """Perform one bounded server operation."""
        state["effects"] = state.get("effects", 0) + 1
        gate = state.get("work_gate")
        if gate is not None:
            await gate()
        return "ok"

    @tool(**kwargs)
    async def big_work() -> str:
        """Return one large payload."""
        return big

    return {"client_read": client_read, "work": work, "big_work": big_work}


_BIG: str | None = None


def _big_payload() -> str:
    global _BIG
    if _BIG is None:
        import base64
        import random

        rng = random.Random(7)
        _BIG = base64.b64encode(rng.randbytes(BIG_CHARS * 3 // 4)).decode()[:BIG_CHARS]
    return _BIG


def agent_kwargs(**extra: Any) -> dict[str, Any]:
    from dendrux import Agent

    names = inspect.signature(Agent.__init__).parameters
    kwargs: dict[str, Any] = {"name": "load", "prompt": "p", "max_iterations": 30}
    if "strict_tenancy" in names:
        kwargs["strict_tenancy"] = False
    if "on_governance_change" in names:
        kwargs["on_governance_change"] = "warn"
    kwargs.update({k: v for k, v in extra.items() if k in names})
    return kwargs


def make_store(engine: Any, *, notify: bool) -> Any:
    from dendrux.runtime.state import SQLAlchemyStateStore

    if notify and "notify_channel" in inspect.signature(SQLAlchemyStateStore.__init__).parameters:
        return SQLAlchemyStateStore(engine, notify_channel=CHANNEL)
    return SQLAlchemyStateStore(engine)


def make_engine(url: str, app: str, *, pool: dict[str, Any] | None = None) -> Any:
    from sqlalchemy.ext.asyncio import create_async_engine

    return create_async_engine(
        url, **(pool or POOL), connect_args={"server_settings": {"application_name": app}}
    )


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    scenario: str
    role: str
    index: int
    workers: int
    url: str
    p: dict[str, Any]
    pipe: Any
    state: dict[str, Any] = field(default_factory=dict)
    errors: collections.Counter = field(default_factory=collections.Counter)

    def shard(self, items: list[Any]) -> list[Any]:
        return items[self.index :: self.workers]

    def send(self, kind: str, payload: Any = None) -> None:
        self.pipe.send((kind, payload))


def key_of(i: int) -> str:
    return f"r{i:05d}"


def status_value(result: Any) -> str:
    status = getattr(result, "status", None)
    return str(getattr(status, "value", status))


async def billing_hook(ctx: Ctx, engine_name: str = "dendrux_load_billing") -> Any:
    """A hook that records every provider call durably, outside the instrumented engine."""
    from sqlalchemy import text

    bill = make_engine(ctx.url, engine_name, pool={"pool_size": 3, "max_overflow": 0})
    ctx.state["bill_engine"] = bill

    async def record(key: str, n_tool: int) -> None:
        async with bill.begin() as conn:
            await conn.execute(
                text("INSERT INTO load_billed (run_key, n_tool) VALUES (:k, :n)"),
                {"k": key, "n": n_tool},
            )

    return record


async def w_a(ctx: Ctx) -> dict[str, Any]:
    """The 0.5 scenario A: each run asks for a client tool, pauses, then is resumed."""
    from dendrux import Agent
    from dendrux.types import RunStatus, ToolResult

    engine = make_engine(ctx.url, f"dendrux_load_w{ctx.index}")
    meter = Meter(engine)
    store = make_store(engine, notify=ctx.p.get("notify", False))
    tools = make_tools(ctx.state)
    keys = ctx.shard([key_of(i) for i in range(ctx.p["runs"])])
    lat_total: list[float] = []
    lat_run: list[float] = []
    lat_submit: list[float] = []
    ok = 0
    shared_llm = make_llm("client")
    shared = (
        Agent(**agent_kwargs(tools=[tools["client_read"]], provider=shared_llm, state_store=store))
        if ctx.p.get("shared")
        else None
    )
    ctx.send("ready")
    ctx.pipe.recv()

    async def one(key: str) -> None:
        nonlocal ok
        try:
            LABEL.set("init")
            llm = shared_llm if shared is not None else make_llm("client")
            agent = shared or Agent(
                **agent_kwargs(tools=[tools["client_read"]], provider=llm, state_store=store)
            )
            t0 = time.perf_counter()
            LABEL.set("run")
            paused = await agent.run(f"go {key}")
            t1 = time.perf_counter()
            assert paused.status != RunStatus.SUCCESS and paused.pending is not None, paused.status
            call = paused.pending.tool_calls[0]
            LABEL.set("submit")
            done = await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name=call.name, call_id=call.id, payload='"ok"')],
                expected_pause_id=paused.pause_id,
            )
            t2 = time.perf_counter()
            LABEL.set("close")
            if shared is None:
                await agent.close()
            meter.op("run")
            meter.op("submit")
            if done.status == RunStatus.SUCCESS and (shared is not None or llm.calls == 2):
                ok += 1
            lat_run.append((t1 - t0) * 1000)
            lat_submit.append((t2 - t1) * 1000)
            lat_total.append((t2 - t0) * 1000)
        except Exception as exc:  # noqa: BLE001 - counted and reported, never hidden
            ctx.errors[type(exc).__name__] += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(k) for k in keys))
    wall = time.perf_counter() - t0
    out = {
        "runs": len(keys),
        "ok": ok,
        "wall_s": wall,
        "lat": {"total": lat_total, "run": lat_run, "submit": lat_submit},
        "meter": meter.snapshot(),
        "errors": dict(ctx.errors),
        "provider_calls": shared_llm.calls if shared is not None else None,
    }
    if shared is not None:
        await shared.close()
    await engine.dispose()
    return out


async def w_runs(ctx: Ctx) -> dict[str, Any]:
    """Concurrent server-tool runs. Roles: ``main`` (normal), ``doomed`` (parks, then is killed)."""
    from dendrux import Agent
    from dendrux.types import RunStatus

    doomed = ctx.role == "doomed"
    engine = make_engine(ctx.url, "dendrux_load_doomed" if doomed else f"dendrux_load_w{ctx.index}")
    meter = Meter(engine)
    store = make_store(engine, notify=ctx.p.get("notify", False))
    kill = ctx.p.get("kill")  # None | "checkpoint" | "inflight"
    all_keys = [key_of(i) for i in ctx.p["keys"]]
    keys = all_keys if doomed else ctx.shard(all_keys)
    billing = await billing_hook(ctx) if ctx.p.get("billing") else None
    parked = 0
    forever = asyncio.Event()

    async def park() -> None:
        nonlocal parked
        parked += 1
        if parked == len(keys):
            ctx.send("parked", {"runs": len(keys), "meter": meter.snapshot()})
        await forever.wait()

    async def hook(key: str, n_tool: int) -> None:
        if billing is not None:
            await billing(key, n_tool)
        if doomed and kill == "inflight" and n_tool == 0:
            await park()

    if doomed and kill == "checkpoint":
        ctx.state["work_gate"] = park
    tools = make_tools(ctx.state)
    llm = make_llm("server", hook=hook, usage=True)
    agent = Agent(**agent_kwargs(tools=[tools["work"]], provider=llm, state_store=store))
    lat: list[float] = []
    ok = 0
    ctx.send("ready")
    ctx.pipe.recv()

    async def one(key: str) -> None:
        nonlocal ok
        try:
            LABEL.set("run")
            t0 = time.perf_counter()
            result = await agent.run(f"go {key}")
            lat.append((time.perf_counter() - t0) * 1000)
            meter.op("run")
            if result.status == RunStatus.SUCCESS:
                ok += 1
            else:
                ctx.errors[f"status_{status_value(result)}"] += 1
        except Exception as exc:  # noqa: BLE001
            ctx.errors[type(exc).__name__] += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(k) for k in keys))
    wall = time.perf_counter() - t0
    out = {
        "runs": len(keys),
        "ok": ok,
        "wall_s": wall,
        "lat": {"total": lat},
        "meter": meter.snapshot(),
        "errors": dict(ctx.errors),
        "provider_calls": llm.calls,
        "effects": ctx.state.get("effects", 0),
    }
    await agent.close()
    if billing is not None:
        await ctx.state["bill_engine"].dispose()
    await engine.dispose()
    return out


async def w_recover(ctx: Ctx) -> dict[str, Any]:
    """Sweep for expired leases, then reclaim this worker's share of the marked runs."""
    from dendrux import Agent, sweep
    from dendrux.types import RunStatus

    engine = make_engine(ctx.url, f"dendrux_load_w{ctx.index}")
    meter = Meter(engine)
    store = make_store(engine, notify=False)
    billing = await billing_hook(ctx)
    tools = make_tools(ctx.state)
    llm = make_llm("server", hook=billing, usage=True)
    agent = Agent(**agent_kwargs(tools=[tools["work"]], provider=llm, state_store=store))
    ctx.send("ready")
    ctx.pipe.recv()
    t0 = time.perf_counter()
    LABEL.set("sweep")
    page = await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    meter.op("sweep")
    items = sorted(page.reclaimable, key=lambda r: r.run_id)
    mine = items[ctx.index :: ctx.workers]
    ok = 0
    lat: list[float] = []

    async def one(item: Any) -> None:
        nonlocal ok
        try:
            LABEL.set("reclaim")
            t = time.perf_counter()
            result = await agent.reclaim(item.run_id, tenant_id=None, item=item)
            lat.append((time.perf_counter() - t) * 1000)
            meter.op("reclaim")
            if result.status == RunStatus.SUCCESS:
                ok += 1
            else:
                ctx.errors[f"status_{status_value(result)}"] += 1
        except Exception as exc:  # noqa: BLE001
            ctx.errors[type(exc).__name__] += 1

    await asyncio.gather(*(one(i) for i in mine))
    wall = time.perf_counter() - t0
    out = {
        "swept": len(items),
        "reclaimed": len(mine),
        "ok": ok,
        "wall_s": wall,
        "lat": {"reclaim": lat},
        "meter": meter.snapshot(),
        "errors": dict(ctx.errors),
        "provider_calls": llm.calls,
    }
    await ctx.state["bill_engine"].dispose()
    await engine.dispose()
    return out


# --- iteration: statements per one-tool iteration -------------------------------------------


async def w_iteration(ctx: Ctx) -> dict[str, Any]:
    """Statements per non-terminal iteration (model call + one server tool), by store operation."""
    from dendrux import Agent
    from dendrux.loops.base import BaseNotifier
    from dendrux.types import RunStatus

    engine = make_engine(ctx.url, "dendrux_load_w0")
    meter = Meter(engine)
    out: dict[str, Any] = {"variants": {}}
    methods = [
        "create_run",
        "set_trace_link",
        "save_trace",
        "save_run_event",
        "get_run_control",
        "checkpoint_step",
        "touch_progress",
        "save_llm_interaction",
        "save_usage",
        "start_tool_call",
        "complete_tool_call",
        "finalize_with_event",
        "get_run",
        "update_run_meta",
    ]
    ctx.send("ready")
    ctx.pipe.recv()
    for notify in (True, False):
        for streaming in (False, True):
            name = f"notify_{'on' if notify else 'off'}_{'stream' if streaming else 'batch'}"
            store = make_store(engine, notify=notify)
            phase = {"name": "startup", "iteration": 0}

            def wrap(method_name: str, method: Any, phase: dict = phase) -> Any:
                @functools.wraps(method)
                async def wrapped(*args: Any, **kwargs: Any) -> Any:
                    if method_name == "get_run_control" and phase["name"] in ("startup", "between"):
                        phase["iteration"] += 1
                        phase["name"] = (
                            f"iteration_{phase['iteration']}"
                            if phase["iteration"] <= 2
                            else "terminal"
                        )
                        meter.override = phase["name"]
                    label = method_name
                    if method_name == "save_run_event":
                        label += ":" + str(kwargs.get("event_type"))
                    if method_name == "checkpoint_step":
                        trace = kwargs.get("trace")
                        label += ":" + (trace.role if trace else "state")
                    token = OP.set(label)
                    try:
                        return await method(*args, **kwargs)
                    finally:
                        OP.reset(token)

                return wrapped

            for m in methods:
                if hasattr(store, m):
                    setattr(store, m, wrap(m, getattr(store, m)))

            class Observer(BaseNotifier):
                def __init__(self, phase: dict) -> None:
                    self.phase = phase

                def trace_link(self, run_id: str) -> str | None:
                    return "00-" + "1" * 32 + "-" + "2" * 16 + "-01"

                async def on_message_appended(
                    self, run_id: str, message: Any, iteration: int
                ) -> None:
                    if message.role.value == "tool":
                        self.phase["name"] = "between"

            tools = make_tools(ctx.state)
            results = []
            for r in range(ctx.p["runs"]):
                phase.update(name="startup", iteration=0)
                meter.override = "startup"
                before_op = dict(meter.by_label_op)
                before_label = dict(meter.by_label)
                llm = make_llm(
                    "server", usage=True, stream_chunks=50 if streaming else 0, server_calls=2
                )
                agent = Agent(
                    **agent_kwargs(tools=[tools["work"]], provider=llm, state_store=store)
                )
                t0 = time.perf_counter()
                if streaming:
                    async with agent.stream(f"go {name}-{r}", notifier=Observer(phase)) as stream:
                        events = [e async for e in stream]
                    result = events[-1].run_result
                else:
                    result = await agent.run(f"go {name}-{r}", notifier=Observer(phase))
                meter.override = None
                assert result.status == RunStatus.SUCCESS, result.error
                results.append(
                    {
                        "by_label": {
                            k: v - before_label.get(k, 0) for k, v in meter.by_label.items()
                        },
                        "by_label_op": {
                            k: v - before_op.get(k, 0) for k, v in meter.by_label_op.items()
                        },
                        "model_calls": llm.calls,
                        "wall_ms": (time.perf_counter() - t0) * 1000,
                    }
                )
                await agent.close()
            out["variants"][name] = results
    out["meter"] = meter.snapshot()
    await engine.dispose()
    return out


# --- followers ----------------------------------------------------------------------------


def _ts(value: Any) -> float:
    from datetime import UTC

    if value is None:
        return 0.0
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()


async def followers_round(
    ctx: Ctx,
    engine: Any,
    meter: Meter,
    url: str,
    keys: list[str],
    followers: int,
    notify: bool,
    idle_s: float,
) -> dict[str, Any]:
    import asyncpg

    from dendrux import Agent
    from dendrux.store import EventFeed, FollowEnd, FollowHeartbeat, PostgresEventNotifier, RunStore
    from dendrux.types import RunStatus, ToolResult

    store = make_store(engine, notify=notify)
    notifier = PostgresEventNotifier(dsn_of(url), channel=CHANNEL) if notify else None
    if notifier is not None:
        await notifier.start()
    reads = RunStore.from_engine(engine, notifier=notifier)
    feed = EventFeed(
        reads,
        max_followers=max(10, len(keys) * followers + 10),
        max_followers_per_tenant=None,
        notifier=notifier,
    )
    tools = make_tools(ctx.state)
    agent = Agent(
        **agent_kwargs(tools=[tools["client_read"]], provider=make_llm("client"), state_store=store)
    )
    book = await asyncpg.connect(
        dsn_of(url), server_settings={"application_name": "dendrux_load_book"}
    )
    result: dict[str, Any] = {}
    run_ids: dict[str, str] = {}
    pause: dict[str, Any] = {}

    # 1. create the runs and pause them
    meter.override = "create"
    mark = meter.n

    async def create(key: str) -> None:
        paused = await agent.run(f"go {key}")
        assert paused.status != RunStatus.SUCCESS
        run_ids[key] = paused.run_id
        pause[key] = paused

    t0 = time.perf_counter()
    await asyncio.gather(*(create(k) for k in keys))
    result["create"] = {"statements": meter.n - mark, "wall_s": time.perf_counter() - t0}
    heads = {
        r["agent_run_id"]: r["m"]
        for r in await book.fetch(
            "select agent_run_id, max(sequence_index) m from run_events group by 1"
        )
    }

    # 2. attach followers
    meter.override = "attach"
    mark = meter.n
    got: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    lags: list[float] = []
    lags_trickle: list[float] = []
    trickle_set: set[str] = set()
    progress_start = [float("inf")]
    ends: collections.Counter[str] = collections.Counter()
    caught_up = asyncio.Event()
    remaining = [len(keys) * followers]

    async def consume(key: str, j: int) -> None:
        seqs = got[(key, j)]
        done_attach = False
        target = heads.get(run_ids[key], 0)
        try:
            async with feed.follow(
                run_ids[key],
                tenant_id=None,
                after_sequence_index=None,
                until_terminal=True,
                max_duration_s=900.0,
            ) as follow:
                async for item in follow:
                    if isinstance(item, FollowEnd):
                        ends[str(item.reason)] += 1
                        break
                    if isinstance(item, FollowHeartbeat):
                        continue
                    seqs.append(item.sequence_index)
                    created = _ts(getattr(item, "created_at", None))
                    if created >= progress_start[0]:
                        bucket = lags_trickle if key in trickle_set else lags
                        bucket.append((time.time() - created) * 1000)
                    if not done_attach and item.sequence_index >= target:
                        done_attach = True
                        remaining[0] -= 1
                        if remaining[0] == 0:
                            caught_up.set()
        except Exception as exc:  # noqa: BLE001
            ctx.errors["follower_" + type(exc).__name__] += 1
            if not done_attach:
                remaining[0] -= 1
                if remaining[0] <= 0:
                    caught_up.set()

    t0 = time.perf_counter()
    tasks = [asyncio.create_task(consume(k, j)) for k in keys for j in range(followers)]
    try:
        await asyncio.wait_for(caught_up.wait(), timeout=300)
        attached = True
    except TimeoutError:
        attached = False
    result["attach"] = {
        "statements": meter.n - mark,
        "wall_s": time.perf_counter() - t0,
        "followers": len(tasks),
        "all_caught_up": attached,
    }

    # 3. idle with followers attached
    meter.override = "idle_warmup"  # feeds settle into their steady polling rhythm first
    await asyncio.sleep(min(3.0, max(1.0, idle_s / 2)))
    meter.override = "idle"
    mark = meter.n
    kinds_before = collections.Counter(meter.kinds)
    await asyncio.sleep(idle_s)
    result["idle"] = {
        "statements": meter.n - mark,
        "seconds": idle_s,
        "kinds": (meter.kinds - kinds_before).most_common(5),
    }

    # 4a. trickle: a few runs resume one at a time with every follower attached and the pool
    # otherwise idle. This isolates the wake latency from queueing behind the burst below.
    async def finish(key: str) -> None:
        paused = pause[key]
        call = paused.pending.tool_calls[0]
        await agent.submit_tool_results(
            paused.run_id,
            [ToolResult(name=call.name, call_id=call.id, payload='"ok"')],
            expected_pause_id=paused.pause_id,
        )

    trickle = keys[: min(10, len(keys) // 10)]
    burst = keys[len(trickle) :]
    trickle_set.update(trickle)
    progress_start[0] = time.time()
    meter.override = "trickle"
    mark = meter.n
    for key in trickle:
        await finish(key)
        await asyncio.sleep(0.3)
    result["trickle"] = {"statements": meter.n - mark, "runs": len(trickle)}

    # 4b. burst: resume every other run at once while followers watch
    meter.override = "progress"
    mark = meter.n
    t0 = time.perf_counter()
    await asyncio.gather(*(finish(k) for k in burst))
    submit_wall = time.perf_counter() - t0
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=300)
        drained = True
    except TimeoutError:
        drained = False
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    result["progress"] = {
        "statements": meter.n - mark,
        "runs": len(burst),
        "submit_wall_s": submit_wall,
        "drain_wall_s": time.perf_counter() - t0,
        "all_followers_ended": drained,
    }
    meter.override = None

    # 5. verify every follower saw exactly the stored events, in order
    db_events = {
        r["agent_run_id"]: list(r["s"])
        for r in await book.fetch(
            "select agent_run_id, array_agg(sequence_index order by sequence_index) s "
            "from run_events group by 1"
        )
    }
    bad = sum(1 for (key, _j), seqs in got.items() if seqs != db_events.get(run_ids[key], []))
    result["check"] = {
        "followers": len(got),
        "streams_not_matching_db": bad,
        "end_reasons": dict(ends),
        "follower_errors": {k: v for k, v in ctx.errors.items() if k.startswith("follower_")},
    }
    result["lag_ms"] = lat_summary(lags)
    result["lag_raw"] = lags
    result["lag_trickle_raw"] = lags_trickle
    await book.close()
    await agent.close()
    await feed.aclose()
    result["followers_open_after_close"] = getattr(feed, "_open", None)
    if notifier is not None:
        await notifier.stop()
    return result


async def w_followers(ctx: Ctx) -> dict[str, Any]:
    engine = make_engine(ctx.url, f"dendrux_load_w{ctx.index}")
    meter = Meter(engine)
    keys = ctx.shard([key_of(i) for i in range(ctx.p["runs"])])
    ctx.send("ready")
    ctx.pipe.recv()
    round_ = await followers_round(
        ctx, engine, meter, ctx.url, keys, ctx.p["followers"], ctx.p["notify"], ctx.p["idle_s"]
    )
    out = {
        "round": round_,
        "meter": meter.snapshot(),
        "errors": dict(ctx.errors),
        "runs": len(keys),
    }
    await engine.dispose()
    return out


# --- cancel ---------------------------------------------------------------------------------


def tree_links(nodes: int, fanout: int = 8) -> list[tuple[int, int | None]]:
    links: list[tuple[int, int | None]] = [(0, None)]
    frontier = [0]
    nxt = 1
    while nxt < nodes:
        new_frontier = []
        for parent in frontier:
            for _ in range(fanout):
                if nxt >= nodes:
                    break
                links.append((nxt, parent))
                new_frontier.append(nxt)
                nxt += 1
        frontier = new_frontier
    return links


async def w_cancel(ctx: Ctx) -> dict[str, Any]:
    """Cancel one large delegation tree, alone or while children are being created under it."""
    from sqlalchemy import func, insert, select

    from dendrux import Agent
    from dendrux.db.models import AgentRun
    from dendrux.errors import RunCancelledError
    from dendrux.types import generate_ulid

    engine = make_engine(ctx.url, "dendrux_load_w0")
    meter = Meter(engine)
    store = make_store(engine, notify=False)
    links = tree_links(ctx.p["nodes"])
    ids = [generate_ulid() for _ in links]
    LABEL.set("seed")
    async with store._session_factory() as session:
        rows = [
            {
                "id": ids[n],
                "agent_name": "cascade",
                "status": "paused",
                "tenant_id": None,
                "parent_run_id": None if p is None else ids[p],
            }
            for n, p in links
        ]
        for i in range(0, len(rows), 500):
            await session.execute(insert(AgentRun), rows[i : i + 500])
        await session.commit()
    deep = [ids[n] for n, _ in links[len(links) // 2 :]]
    agent = Agent(**agent_kwargs(state_store=store))
    stop = asyncio.Event()
    counts = {"admitted": 0, "refused": 0, "other": 0}
    ctx.send("ready")
    ctx.pipe.recv()

    async def creator(c: int) -> None:
        LABEL.set("creators")
        n = 0
        while not stop.is_set():
            parent = deep[(c * 7919 + n * 104729) % len(deep)]
            n += 1
            try:
                await store.create_run(
                    generate_ulid(), "late-child", parent_run_id=parent, delegation_level=1
                )
                counts["admitted"] += 1
            except RunCancelledError:
                counts["refused"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["other"] += 1
                ctx.errors[type(exc).__name__] += 1
            await asyncio.sleep(0.02)

    creators = []
    if ctx.p.get("contended"):
        creators = [asyncio.create_task(creator(c)) for c in range(ctx.p.get("creators", 8))]
        await asyncio.sleep(0.5)  # let admission traffic build before the cancel starts
    LABEL.set("cancel")
    t0 = time.perf_counter()
    result = await agent.cancel_run(ids[0], cascade=True)
    wall = time.perf_counter() - t0
    stop.set()
    await asyncio.gather(*creators)
    async with engine.connect() as conn:
        status_counts = dict(
            (
                await conn.execute(select(AgentRun.status, func.count()).group_by(AgentRun.status))
            ).all()
        )
        seeded_open = (
            await conn.execute(
                select(func.count())
                .select_from(AgentRun)
                .where(AgentRun.id.in_(ids), AgentRun.status != "cancelled")
            )
        ).scalar_one()
        unmarked = (
            await conn.execute(
                select(func.count())
                .select_from(AgentRun)
                .where(
                    AgentRun.cancel_requested.is_(False),
                    AgentRun.status.not_in(("success", "error", "cancelled", "max_iterations")),
                )
            )
        ).scalar_one()
    meta = getattr(result, "meta", None) or {}
    out = {
        "nodes": len(ids),
        "wall_s": wall,
        "cancel_statements": meter.by_label["cancel"],
        "seeded_not_cancelled": seeded_open,
        "nodes_without_cancel_request": unmarked,
        "status_counts": status_counts,
        "cascade_truncated": bool(meta.get("cascade_truncated")),
        "cascade_entries": len(meta.get("cascade", [])),
        "creators": counts,
        "creator_statements": meter.by_label["creators"],
        "meter": meter.snapshot(),
        "errors": dict(ctx.errors),
    }
    await engine.dispose()
    return out


# --- pool exhaustion ---------------------------------------------------------------------------


async def w_pool(ctx: Ctx) -> dict[str, Any]:
    from dendrux import Agent
    from dendrux.types import RunStatus

    out: dict[str, Any] = {"variants": {}}
    ctx.send("ready")
    ctx.pipe.recv()
    for name, timeout in (("queued", 120), ("short_timeout", 0.3)):
        engine = make_engine(
            ctx.url,
            "dendrux_load_w0",
            pool={
                "pool_size": 10,
                "max_overflow": 0,
                "pool_timeout": timeout,
                "pool_pre_ping": True,
            },
        )
        meter = Meter(engine)
        store = make_store(engine, notify=False)
        tools = make_tools(ctx.state)
        llm = make_llm("server", usage=True)
        agent = Agent(**agent_kwargs(tools=[tools["work"]], provider=llm, state_store=store))
        errors: collections.Counter[str] = collections.Counter()
        lat: list[float] = []

        async def one(
            i: int,
            errors: collections.Counter = errors,
            lat: list = lat,
            agent: Any = agent,
            prefix: str = name[0],
        ) -> bool:
            t0 = time.perf_counter()
            try:
                r = await agent.run(f"go {prefix}{i:05d}")
            except Exception as exc:  # noqa: BLE001
                errors[type(exc).__name__] += 1
                return False
            lat.append((time.perf_counter() - t0) * 1000)
            if r.status != RunStatus.SUCCESS:
                errors[f"status_{status_value(r)}"] += 1
                return False
            return True

        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(ctx.p["runs"])))
        out["variants"][name] = {
            "runs": ctx.p["runs"],
            "ok": sum(results),
            "wall_s": time.perf_counter() - t0,
            "errors": dict(errors),
            "peak_checked_out": meter.peak_active,
            "pool_size": 10,
            "pool_timeout_s": timeout,
            "lat": lat,
            "meter": meter.snapshot(),
        }
        await agent.close()
        await engine.dispose()
    return out


# --- history ----------------------------------------------------------------------------------


async def w_history(ctx: Ctx) -> dict[str, Any]:
    from dendrux import Agent
    from dendrux.types import RunStatus, ToolResult

    engine = make_engine(ctx.url, "dendrux_load_w0")
    meter = Meter(engine)
    store = make_store(engine, notify=False)
    tools = make_tools(ctx.state)
    calls = ctx.p["calls"]
    llm = make_llm("big", big_calls=calls)
    agent = Agent(
        **agent_kwargs(
            tools=[tools["client_read"], tools["big_work"]], provider=llm, state_store=store
        )
    )
    ctx.send("ready")
    ctx.pipe.recv()
    out: dict[str, Any] = {"role": ctx.role}
    if ctx.role == "create":
        t0 = time.perf_counter()
        meter.override = "create"
        paused = await agent.run("go history")
        assert paused.status != RunStatus.SUCCESS and paused.pending is not None, paused.status
        call = paused.pending.tool_calls[0]
        out.update(
            run_id=paused.run_id,
            pause_id=paused.pause_id,
            call_name=call.name,
            call_id=call.id,
            wall_s=time.perf_counter() - t0,
            big_results=calls,
            big_result_chars=BIG_CHARS,
        )
    else:
        gc.collect()
        rss0 = rss_kb()
        meter.override = "resume"
        peak = {"rss": rss0}

        async def watch() -> None:
            while True:
                peak["rss"] = max(peak["rss"], rss_kb())
                await asyncio.sleep(0.05)

        watcher = asyncio.create_task(watch())
        t0 = time.perf_counter()
        done = await agent.submit_tool_results(
            ctx.p["run_id"],
            [ToolResult(name=ctx.p["call_name"], call_id=ctx.p["call_id"], payload='"ok"')],
            expected_pause_id=ctx.p["pause_id"],
        )
        wall = time.perf_counter() - t0
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        out.update(
            status=status_value(done),
            error=str(getattr(done, "error", None))[:300],
            wall_s=wall,
            rss_before_kb=rss0,
            rss_peak_kb=max(peak["rss"], rss_kb()),
            provider_calls=llm.calls,
        )
    out["meter"] = meter.snapshot()
    await agent.close()
    await engine.dispose()
    return out


# --- memory -----------------------------------------------------------------------------------


async def _a_round(ctx: Ctx, engine: Any, keys: list[str]) -> int:
    from dendrux import Agent
    from dendrux.types import RunStatus, ToolResult

    store = make_store(engine, notify=False)
    tools = make_tools(ctx.state)
    ok = 0

    async def one(key: str) -> None:
        nonlocal ok
        agent = Agent(
            **agent_kwargs(
                tools=[tools["client_read"]], provider=make_llm("client"), state_store=store
            )
        )
        paused = await agent.run(f"go {key}")
        call = paused.pending.tool_calls[0]
        done = await agent.submit_tool_results(
            paused.run_id,
            [ToolResult(name=call.name, call_id=call.id, payload='"ok"')],
            expected_pause_id=paused.pause_id,
        )
        ok += done.status == RunStatus.SUCCESS
        await agent.close()

    await asyncio.gather(*(one(k) for k in keys))
    return ok


async def w_memory(ctx: Ctx) -> dict[str, Any]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base

    engine = make_engine(ctx.url, "dendrux_load_w0")
    admin = create_async_engine(
        ctx.url,
        pool_size=1,
        max_overflow=0,
        isolation_level="AUTOCOMMIT",
        connect_args={"server_settings": {"application_name": "dendrux_load_admin"}},
    )
    meter = Meter(engine)
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables if t.name != "dendrux_settings")
    cycles: list[dict[str, Any]] = []
    ctx.send("ready")
    ctx.pipe.recv()
    for c in range(ctx.p["cycles"]):
        t0 = time.perf_counter()
        async with admin.connect() as conn:
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
        keys = [key_of(i) for i in range(ctx.p["runs"])]
        a_ok = await _a_round(ctx, engine, keys)
        f_keys = [key_of(1000 + i) for i in range(10)]
        round_ = await followers_round(ctx, engine, meter, ctx.url, f_keys, 5, True, 0.5)
        gc.collect()
        gc.collect()
        pool = engine.pool
        cycles.append(
            {
                "cycle": c + 1,
                "wall_ms": (time.perf_counter() - t0) * 1000,
                "rss_kb": rss_kb(),
                "fds": fd_count(),
                "gc_objects": len(gc.get_objects()),
                "tasks": len(asyncio.all_tasks()),
                "checked_out": pool.checkedout(),
                "checked_in": pool.checkedin(),
                "overflow": pool.overflow(),
                "runs_ok": a_ok,
                "followers_mismatch": round_["check"]["streams_not_matching_db"],
                "followers_open_after_close": round_["followers_open_after_close"],
            }
        )
    out = {"cycles": cycles, "errors": dict(ctx.errors), "meter": meter.snapshot()}
    await admin.dispose()
    await engine.dispose()
    return out


WORKERS = {
    ("a", "main"): w_a,
    ("runs", "main"): w_runs,
    ("runs", "doomed"): w_runs,
    ("recover", "main"): w_recover,
    ("iteration", "main"): w_iteration,
    ("followers", "main"): w_followers,
    ("cancel", "main"): w_cancel,
    ("pool", "main"): w_pool,
    ("history", "create"): w_history,
    ("history", "resume"): w_history,
    ("memory", "main"): w_memory,
}


def worker_entry(pipe: Any, spec: dict[str, Any]) -> None:
    warnings.simplefilter("ignore")
    import logging

    logging.disable(logging.CRITICAL)
    ctx = Ctx(
        spec["scenario"], spec["role"], spec["index"], spec["workers"], spec["url"], spec["p"], pipe
    )
    try:
        result = asyncio.run(WORKERS[(ctx.scenario, ctx.role)](ctx))
        result["rss_kb_end"] = rss_kb()
        result["maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (
            1 if sys.platform != "darwin" else 1024
        )
        pipe.send(("result", result))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        raise


# ---------------------------------------------------------------------------
# Controller side
# ---------------------------------------------------------------------------

TERMINAL = ("success", "error", "cancelled", "max_iterations")
BIG_TABLES = (
    "agent_runs",
    "run_events",
    "react_traces",
    "tool_calls",
    "token_usage",
    "llm_interactions",
)
NEW_MARKERS = (
    "claim_epoch",
    "lease",
    "loop_state",
    "FOR SHARE",
    "cancel_requested = true",
    "parent_run_id = ANY",
    "pg_notify",
    "dendrux_settings",
    "trace_kind",
    "reclaim",
    "completion",
    "ANY (",
    "intent",
)


def log(message: str) -> None:
    print(f"[load {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


async def reset_db(url: str) -> None:
    import asyncpg
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    parsed = check_private_url(url)
    name = parsed.database
    maint = parsed.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    conn = await asyncpg.connect(maint)
    try:
        if not await conn.fetchval("select 1 from pg_database where datname = $1", name):
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()
    from dendrux.db.models import Base

    engine = create_async_engine(url, poolclass=NullPool)
    async with engine.begin() as c:
        from sqlalchemy import text

        await c.execute(text("DROP TABLE IF EXISTS load_billed"))
        await c.run_sync(Base.metadata.drop_all)
        await c.run_sync(Base.metadata.create_all)
        with contextlib.suppress(ImportError):
            from dendrux.db._settings import seed_settings

            await c.run_sync(lambda sc: seed_settings(sc, active=True))
        await c.execute(
            text(
                "CREATE TABLE load_billed "
                "(id bigserial primary key, run_key text not null, n_tool int not null)"
            )
        )
    await engine.dispose()
    # fresh planner statistics, as a just-migrated production database would have after autovacuum
    conn = await asyncpg.connect(dsn_of(url))
    try:
        await conn.execute("ANALYZE")
    finally:
        await conn.close()


class Sampler:
    """Samples ``pg_stat_activity``: connections by role, backends waiting on locks."""

    def __init__(self, url: str) -> None:
        self.dsn = dsn_of(url)
        self.task: asyncio.Task | None = None
        self.samples = 0
        self.peak_total = 0
        self.peak_by_app: collections.Counter[str] = collections.Counter()
        self.lock_wait_ms = 0.0
        self.lock_wait_ms_runrow = 0.0
        self.max_waiters = 0
        self.max_waiters_runrow = 0
        self.wait_events: collections.Counter[str] = collections.Counter()
        self.waiting_samples = 0

    async def start(self) -> None:
        import asyncpg

        conn = await asyncpg.connect(
            self.dsn, server_settings={"application_name": "dendrux_load_sampler"}
        )
        self.conn = conn
        self.task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        last = time.perf_counter()
        while True:
            rows = await self.conn.fetch(
                "select application_name app, wait_event_type wt, wait_event we, "
                "case when wait_event_type = 'Lock' then left(query, 300) end q "
                "from pg_stat_activity where datname = current_database() "
                "and backend_type = 'client backend'"
            )
            now = time.perf_counter()
            dt = (now - last) * 1000
            last = now
            self.samples += 1
            self.peak_total = max(self.peak_total, len(rows))
            apps = collections.Counter(r["app"] or "(unnamed: listener or other)" for r in rows)
            for app in apps:
                role = re.sub(r"\d+$", "N", app)
                self.peak_by_app[role] = max(
                    self.peak_by_app[role],
                    sum(c for a, c in apps.items() if re.sub(r"\d+$", "N", a) == role),
                )
            waiting = [r for r in rows if r["wt"] == "Lock"]
            runrow = [r for r in waiting if r["q"] and "agent_runs" in r["q"]]
            if waiting:
                self.waiting_samples += 1
            self.lock_wait_ms += len(waiting) * dt
            self.lock_wait_ms_runrow += len(runrow) * dt
            self.max_waiters = max(self.max_waiters, len(waiting))
            self.max_waiters_runrow = max(self.max_waiters_runrow, len(runrow))
            for r in rows:
                if r["wt"]:
                    self.wait_events[f"{r['wt']}:{r['we']}"] += 1
            await asyncio.sleep(SAMPLE_INTERVAL_S)

    async def stop(self) -> dict[str, Any]:
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.conn.close()
        return {
            "samples": self.samples,
            "interval_ms": SAMPLE_INTERVAL_S * 1000,
            "peak_connections": self.peak_total,
            "peak_connections_by_role": dict(sorted(self.peak_by_app.items())),
            "lock_wait_ms_total": round(self.lock_wait_ms, 1),
            "lock_wait_ms_run_row": round(self.lock_wait_ms_runrow, 1),
            "max_concurrent_lock_waiters": self.max_waiters,
            "max_concurrent_run_row_waiters": self.max_waiters_runrow,
            "samples_with_lock_waiter": self.waiting_samples,
            "top_wait_events": dict(self.wait_events.most_common(6)),
        }


async def db_counters(url: str) -> dict[str, Any]:
    import asyncpg

    conn = await asyncpg.connect(
        dsn_of(url), server_settings={"application_name": "dendrux_load_book"}
    )
    try:
        row = await conn.fetchrow(
            "select pg_current_wal_lsn()::text lsn, xact_commit + xact_rollback xacts, "
            "tup_inserted ins, tup_updated upd, tup_deleted del, deadlocks, "
            "(select coalesce(sum(pg_total_relation_size(c.oid)), 0)::bigint from pg_class c "
            " join pg_namespace n on n.oid = c.relnamespace "
            " where c.relkind = 'r' and n.nspname = 'public') size "
            "from pg_stat_database where datname = current_database()"
        )
        sizes = await conn.fetch(
            "select c.relname, pg_total_relation_size(c.oid)::bigint size from pg_class c "
            "join pg_namespace n on n.oid = c.relnamespace "
            "where c.relkind = 'r' and n.nspname = 'public'"
        )
        return {**dict(row), "sizes": {r["relname"]: r["size"] for r in sizes}}
    finally:
        await conn.close()


async def wal_delta(url: str, a: str, b: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(dsn_of(url))
    try:
        return int(
            await conn.fetchval(
                "select pg_wal_lsn_diff($1::text::pg_lsn, $2::text::pg_lsn)::bigint", b, a
            )
        )
    finally:
        await conn.close()


async def sql_one(url: str, query: str, *args: Any) -> Any:
    import asyncpg

    conn = await asyncpg.connect(
        dsn_of(url), server_settings={"application_name": "dendrux_load_book"}
    )
    try:
        return await conn.fetchrow(query, *args)
    finally:
        await conn.close()


async def sql_exec(url: str, query: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(
        dsn_of(url), server_settings={"application_name": "dendrux_load_book"}
    )
    try:
        await conn.execute(query)
    finally:
        await conn.close()


class Handle:
    def __init__(self, proc: Any, pipe: Any, spec: dict[str, Any]) -> None:
        self.proc, self.pipe, self.spec = proc, pipe, spec
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.ready = False
        self.killed = False

    @property
    def name(self) -> str:
        return f"{self.spec['scenario']}/{self.spec['role']}/{self.spec['index']}"


def spawn(url: str, scenario: str, role: str, n: int, p: dict[str, Any]) -> list[Handle]:
    ctx = multiprocessing.get_context("spawn")
    handles = []
    for i in range(n):
        spec = {"scenario": scenario, "role": role, "index": i, "workers": n, "url": url, "p": p}
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker_entry, args=(child, spec), daemon=True)
        proc.start()
        child.close()
        handles.append(Handle(proc, parent, spec))
    return handles


async def pump(
    handles: list[Handle], *, until: str, timeout: float, on_message: Any = None
) -> None:
    """Read worker messages until all reach ``until`` ('ready' or 'done')."""
    deadline = time.monotonic() + timeout
    while True:
        for h in handles:
            while h.pipe.poll():
                try:
                    kind, payload = h.pipe.recv()
                except EOFError:
                    break
                if kind == "ready":
                    h.ready = True
                elif kind == "result":
                    h.result = payload
                elif kind == "error":
                    h.error = payload
                elif on_message is not None:
                    on_message(h, kind, payload)
            if h.error:
                raise RuntimeError(f"worker {h.name} failed:\n{h.error}")
            if not h.proc.is_alive() and not h.killed and h.result is None and not h.pipe.poll():
                raise RuntimeError(f"worker {h.name} exited with code {h.proc.exitcode}")
        live = [h for h in handles if not h.killed]
        done = (
            all(h.ready for h in live)
            if until == "ready"
            else all(h.result is not None for h in live)
        )
        if done:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"workers did not reach {until} within {timeout}s")
        await asyncio.sleep(0.02)


def release(handles: list[Handle]) -> None:
    for h in handles:
        with contextlib.suppress(Exception):
            if h.proc.is_alive():
                h.proc.join(timeout=10)
            if h.proc.is_alive():
                h.proc.kill()
        with contextlib.suppress(Exception):
            h.pipe.close()


async def run_workers(
    url: str,
    scenario: str,
    role: str,
    n: int,
    p: dict[str, Any],
    *,
    timeout: float = 1800,
    on_message: Any = None,
    extra: list[Handle] | None = None,
) -> list[Handle]:
    handles = spawn(url, scenario, role, n, p)
    allh = handles + (extra or [])
    try:
        await pump(allh, until="ready", timeout=180)
        for h in allh:
            h.pipe.send("go")
        await pump(allh, until="done", timeout=timeout, on_message=on_message)
    finally:
        release(allh)
    return handles


class Measured:
    """Brackets one measured stretch: sampler, database counters, wall clock."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.sampler = Sampler(url)

    async def __aenter__(self) -> Measured:
        self.before = await db_counters(self.url)
        await self.sampler.start()
        self.t0 = time.perf_counter()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.wall_s = time.perf_counter() - self.t0
        self.samples = await self.sampler.stop()
        await asyncio.sleep(1.3)  # pg_stat_database is flushed about once a second
        self.after = await db_counters(self.url)
        self.wal_bytes = await wal_delta(self.url, self.before["lsn"], self.after["lsn"])

    def server(self) -> dict[str, Any]:
        return {
            "wal_bytes": self.wal_bytes,
            "table_growth_bytes": self.after["size"] - self.before["size"],
            "transactions": self.after["xacts"] - self.before["xacts"],
            "tuples_inserted": self.after["ins"] - self.before["ins"],
            "tuples_updated": self.after["upd"] - self.before["upd"],
            "deadlocks": self.after["deadlocks"] - self.before["deadlocks"],
            "growth_by_table": dict(
                sorted(
                    (
                        (t, v - self.before["sizes"].get(t, 0))
                        for t, v in self.after["sizes"].items()
                        if v - self.before["sizes"].get(t, 0) > 0
                    ),
                    key=lambda kv: -kv[1],
                )[:6]
            ),
            **self.samples,
        }


def lats(handles: list[Handle], key: str = "total") -> list[float]:
    out: list[float] = []
    for h in handles:
        out.extend((h.result or {}).get("lat", {}).get(key, []))
    return out


def per_op(merged: dict[str, Any]) -> dict[str, float]:
    return {
        label: round(merged["by_label"].get(label, 0) / count, 2)
        for label, count in merged["ops"].items()
        if count
    }


def unattributed(merged: dict[str, Any], ops: dict[str, float]) -> dict[str, int]:
    return {k: v for k, v in merged["by_label"].items() if k not in ops}


@dataclass
class Outcome:
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, float] = field(default_factory=dict)
    samples: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, value: Any = None, limit: Any = None) -> None:
        self.checks.append({"name": name, "ok": bool(ok), "value": value, "limit": limit})


async def integrity(url: str) -> dict[str, Any]:
    row = await sql_one(
        url,
        "select (select count(*) from agent_runs) total, "
        "(select count(*) from agent_runs where status = 'success') success, "
        "(select count(*) from agent_runs where status <> all($1::text[])) open, "
        "(select count(*) from (select agent_run_id, sequence_index from run_events "
        " group by 1, 2 having count(*) > 1) d) dup_events",
        list(TERMINAL),
    )
    return dict(row)


def base_metrics(
    m: Measured, runs: int, merged: dict[str, Any], handles: list[Handle]
) -> dict[str, Any]:
    server = m.server()
    ops = per_op(merged)
    total_lat = lats(handles)
    out = {
        "runs": runs,
        "wall_s": round(m.wall_s, 2),
        "runs_per_s": round(runs / m.wall_s, 2) if m.wall_s else None,
        "statements": merged["statements"],
        "statements_per_run": round(merged["statements"] / runs, 2) if runs else None,
        "statements_per_s": round(merged["statements"] / m.wall_s, 1) if m.wall_s else None,
        "statements_per_operation": ops,
        "unattributed_statements": unattributed(merged, ops),
        "latency_ms": lat_summary(total_lat),
        "app_bytes_sent": merged["bytes_sent"],
        "app_bytes_sent_per_run": round(merged["bytes_sent"] / runs) if runs else None,
        "wal_bytes_per_run": round(server["wal_bytes"] / runs) if runs else None,
        "pool_ceiling_per_worker": POOL_CEILING,
        "peak_pool_checkouts_sum": merged["peak_checked_out_sum"],
        "server": server,
        "top_statements": merged["top"].most_common(40),
    }
    return out


# --- scenario drivers -------------------------------------------------------------------------


async def ctl_a(
    url: str, cfg: dict[str, Any], workers: int, notify: bool, shared: bool = False
) -> Outcome:
    out = Outcome()
    n = cfg["a"]
    await reset_db(url)
    async with Measured(url) as m:
        handles = await run_workers(
            url, "a", "main", workers, {"runs": n, "notify": notify, "shared": shared}
        )
    results = [h.result for h in handles]
    merged = merge_meters([r["meter"] for r in results])
    out.samples = merged["samples"]
    ok = sum(r["ok"] for r in results)
    metrics = base_metrics(m, n, merged, handles)
    metrics["statements_per_llm_step"] = round(merged["statements"] / (n * 2), 2)
    metrics["latency_run_ms"] = lat_summary(lats(handles, "run"))
    metrics["latency_submit_ms"] = lat_summary(lats(handles, "submit"))
    metrics["errors"] = collections.Counter()
    for r in results:
        metrics["errors"].update(r["errors"])
    metrics["errors"] = dict(metrics["errors"])
    metrics["integrity"] = await integrity(url)
    metrics["notify"] = notify
    out.metrics = metrics
    out.check("every run completed once", ok == n and metrics["integrity"]["success"] == n, ok, n)
    out.check(
        "no duplicate event indexes",
        metrics["integrity"]["dup_events"] == 0,
        metrics["integrity"]["dup_events"],
        0,
    )
    out.check(
        "no run left open", metrics["integrity"]["open"] == 0, metrics["integrity"]["open"], 0
    )
    gate = "a_shared_agent" if shared else f"a_notify_{'on' if notify else 'off'}"
    out.gates[f"{gate}.statements_per_run"] = metrics["statements_per_run"]
    if shared:
        calls = sum(r["provider_calls"] for r in results)
        out.check("exactly two model calls per run", calls == 2 * n, calls, 2 * n)
    return out


async def ctl_iteration(url: str, cfg: dict[str, Any], workers: int) -> Outcome:
    out = Outcome()
    await reset_db(url)
    async with Measured(url) as m:
        handles = await run_workers(url, "iteration", "main", 1, {"runs": cfg["iteration_runs"]})
    r = handles[0].result
    merged = merge_meters([r["meter"]])
    out.samples = merged["samples"]
    variants: dict[str, Any] = {}
    for name, runs in r["variants"].items():
        phases: dict[str, list[int]] = collections.defaultdict(list)
        for run in runs:
            for label, count in run["by_label"].items():
                phases[label].append(count)
        last = runs[-1]
        per_op_last = collections.defaultdict(dict)
        for key, count in last["by_label_op"].items():
            label, op = key.split("|", 1)
            if count:
                per_op_last[label][op] = count
        variants[name] = {
            "startup": phases.get("startup", [0])[-1],
            "iteration_1": phases.get("iteration_1", [0])[-1],
            "iteration_2": phases.get("iteration_2", [0])[-1],
            "terminal": phases.get("terminal", [0])[-1],
            "max_non_terminal": max(
                max(phases.get("iteration_1", [0])), max(phases.get("iteration_2", [0]))
            ),
            "model_calls": last["model_calls"],
            "latency_ms": lat_summary([r["wall_ms"] for r in runs]),
            "by_operation_iteration_2": dict(per_op_last.get("iteration_2", {})),
            "by_operation_startup": dict(per_op_last.get("startup", {})),
            "by_operation_terminal": dict(per_op_last.get("terminal", {})),
        }
    out.metrics = {
        "variants": variants,
        "server": m.server(),
        "wall_s": round(m.wall_s, 2),
        "statements": merged["statements"],
        "app_bytes_sent": merged["bytes_sent"],
    }
    on_batch = variants["notify_on_batch"]["max_non_terminal"]
    on_stream = variants["notify_on_stream"]["max_non_terminal"]
    out.check(
        "one-tool iteration statements, notifier on (budget 25)", on_batch <= 25, on_batch, 25
    )
    out.check(
        "streaming iteration (50 chunks) statements, notifier on (budget 25)",
        on_stream <= 25,
        on_stream,
        25,
    )
    out.gates["iteration.notify_on_batch"] = on_batch
    out.gates["iteration.notify_on_stream"] = on_stream
    out.gates["iteration.notify_off_batch"] = variants["notify_off_batch"]["max_non_terminal"]
    return out


async def ctl_runs(url: str, cfg: dict[str, Any], workers: int) -> Outcome:
    out = Outcome()
    n = cfg["runs"]
    await reset_db(url)
    async with Measured(url) as m:
        handles = await run_workers(
            url, "runs", "main", workers, {"keys": list(range(n)), "notify": False}
        )
    results = [h.result for h in handles]
    merged = merge_meters([r["meter"] for r in results])
    out.samples = merged["samples"]
    metrics = base_metrics(m, n, merged, handles)
    calls = sum(r["provider_calls"] for r in results)
    metrics.update(
        ok=sum(r["ok"] for r in results), provider_calls=calls, integrity=await integrity(url)
    )
    errs: collections.Counter = collections.Counter()
    for r in results:
        errs.update(r["errors"])
    metrics["errors"] = dict(errs)
    out.metrics = metrics
    out.check(
        "every run completed",
        metrics["ok"] == n and metrics["integrity"]["success"] == n,
        metrics["ok"],
        n,
    )
    out.check("exactly two model calls per run", calls == 2 * n, calls, 2 * n)
    out.check(
        "no duplicate event indexes",
        metrics["integrity"]["dup_events"] == 0,
        metrics["integrity"]["dup_events"],
        0,
    )
    out.gates["runs.statements_per_run"] = metrics["statements_per_run"]
    return out


async def ctl_kill(url: str, cfg: dict[str, Any], workers: int, kill: str) -> Outcome:
    out = Outcome()
    n = cfg["kill_runs"]
    k = max(1, math.ceil(n * cfg["kill_fraction"]))
    await reset_db(url)
    doomed_keys = list(range(n - k, n))
    normal_keys = list(range(n - k))
    parked: dict[str, Any] = {"seen": False, "killed_at": None, "meter": None}
    doomed = spawn(
        url,
        "runs",
        "doomed",
        1,
        {"keys": doomed_keys, "kill": kill, "billing": True, "notify": False},
    )

    def on_message(h: Handle, kind: str, payload: Any) -> None:
        if kind == "parked" and not parked["seen"]:
            parked["seen"] = True
            parked["killed_at"] = time.perf_counter()
            parked["meter"] = payload["meter"]
            os.kill(h.proc.pid, signal.SIGKILL)
            h.killed = True

    async with Measured(url) as m:
        normal = spawn(
            url, "runs", "main", workers, {"keys": normal_keys, "billing": True, "notify": False}
        )
        allh = normal + doomed
        try:
            await pump(allh, until="ready", timeout=180)
            for h in allh:
                h.pipe.send("go")
            await pump(allh, until="done", timeout=1800, on_message=on_message)
        finally:
            release(allh)
        wall_load = time.perf_counter() - m.t0
        before_recovery = await sql_one(
            url,
            "select count(*) filter (where status = 'success') ok, "
            "count(*) filter (where status <> all($1::text[])) open from agent_runs",
            list(TERMINAL),
        )
        # lease expiry: the killed runs stop making progress; move their clocks back, not wait
        await sql_exec(
            url,
            "update agent_runs set last_progress_at = now() - interval '10 minutes' "
            "where status not in ('success','error','cancelled','max_iterations')",
        )
        t_rec = time.perf_counter()
        rec = await run_workers(url, "recover", "main", workers, {})
        recovery_wall = time.perf_counter() - t_rec
    normal_results = [h.result for h in normal]
    rec_results = [h.result for h in rec]
    merged = merge_meters([r["meter"] for r in normal_results] + [parked["meter"]])
    merged_rec = merge_meters([r["meter"] for r in rec_results])
    out.samples = {**merged["samples"], **merged_rec["samples"]}
    final = await integrity(url)
    billed = (await sql_one(url, "select count(*) c from load_billed"))["c"]
    extra = billed - 2 * n
    metrics = base_metrics(m, n, merged, normal)
    metrics.update(
        killed_processes=1,
        killed_worker_statements=parked["meter"]["statements"],
        runs_in_killed_process=k,
        kill_fraction_of_runs=round(k / n, 3),
        killed_at_boundary=kill,
        runs_open_after_load=before_recovery["open"],
        runs_success_before_recovery=before_recovery["ok"],
        load_wall_s=round(wall_load, 2),
        recovery_wall_s=round(recovery_wall, 2),
        swept=max((r["swept"] for r in rec_results), default=0),
        reclaimed=sum(r["reclaimed"] for r in rec_results),
        reclaim_ok=sum(r["ok"] for r in rec_results),
        recovery_statements=merged_rec["statements"],
        statements_including_recovery=merged["statements"] + merged_rec["statements"],
        recovery_statements_per_reclaim=round(merged_rec["statements"] / max(1, k), 2),
        recovery_by_label=dict(merged_rec["by_label"]),
        recovery_latency_ms=lat_summary(lats(rec, "reclaim")),
        billed_model_calls=billed,
        expected_model_calls=2 * n,
        extra_billed_calls=extra,
        integrity=final,
        errors={},
    )
    metrics["statements_per_run_including_recovery"] = round(
        (merged["statements"] + merged_rec["statements"]) / n, 2
    )
    out.metrics = metrics
    out.check(
        "the killed process left its runs open",
        before_recovery["open"] == k,
        before_recovery["open"],
        k,
    )
    out.check(
        "every run completed after recovery",
        final["success"] == n and final["open"] == 0,
        final["success"],
        n,
    )
    out.check(
        "reclaim finished every killed run", metrics["reclaim_ok"] == k, metrics["reclaim_ok"], k
    )
    out.check("no duplicate event indexes", final["dup_events"] == 0, final["dup_events"], 0)
    if kill == "checkpoint":
        out.check("zero extra billed model calls (checkpoint boundary)", extra == 0, extra, 0)
    else:
        out.notes.append(
            f"in-flight boundary: {extra} extra billed model call(s) for {k} killed run(s)"
        )
    out.gates[f"kill_{kill}.statements_per_run_including_recovery"] = metrics[
        "statements_per_run_including_recovery"
    ]
    return out


async def ctl_followers(url: str, cfg: dict[str, Any], workers: int, notify: bool) -> Outcome:
    out = Outcome()
    n = cfg["followers_runs"]
    f = cfg["followers_per_run"]
    await reset_db(url)
    async with Measured(url) as m:
        handles = await run_workers(
            url,
            "followers",
            "main",
            workers,
            {"runs": n, "followers": f, "notify": notify, "idle_s": cfg["idle_s"]},
            timeout=3600,
        )
    rounds = [h.result["round"] for h in handles]
    merged = merge_meters([h.result["meter"] for h in handles])
    out.samples = merged["samples"]
    total_followers = sum(r["attach"]["followers"] for r in rounds)
    attach_stmts = sum(r["attach"]["statements"] for r in rounds)
    idle_stmts = sum(r["idle"]["statements"] for r in rounds)
    idle_s = cfg["idle_s"]
    progress_stmts = sum(r["progress"]["statements"] for r in rounds)
    create_stmts = sum(r["create"]["statements"] for r in rounds)
    lag_raw: list[float] = []
    lag_trickle: list[float] = []
    for r in rounds:
        lag_raw.extend(r["lag_raw"])
        lag_trickle.extend(r["lag_trickle_raw"])
    burst_runs = sum(r["progress"]["runs"] for r in rounds)
    bad = sum(r["check"]["streams_not_matching_db"] for r in rounds)
    metrics = {
        "runs": n,
        "followers": total_followers,
        "followers_per_run": f,
        "notifier": notify,
        "create_statements": create_stmts,
        "attach_statements": attach_stmts,
        "attach_statements_per_follower": round(attach_stmts / total_followers, 2),
        "attach_wall_s": round(max(r["attach"]["wall_s"] for r in rounds), 2),
        "idle_seconds": idle_s,
        "idle_statements": idle_stmts,
        "idle_top_statements": sum(
            (collections.Counter(dict(r["idle"]["kinds"])) for r in rounds), collections.Counter()
        ).most_common(4),
        "idle_statements_per_s": round(idle_stmts / idle_s, 1),
        "idle_statements_per_s_per_watched_run": round(idle_stmts / idle_s / n, 3),
        "progress_statements": progress_stmts,
        "progress_statements_per_run": round(progress_stmts / max(1, burst_runs), 2),
        "trickle_runs": sum(r["trickle"]["runs"] for r in rounds),
        "trickle_delivery_lag_ms": lat_summary(lag_trickle),
        "progress_wall_s": round(max(r["progress"]["drain_wall_s"] for r in rounds), 2),
        "burst_delivery_lag_ms": lat_summary(lag_raw),
        "streams_not_matching_db": bad,
        "all_followers_ended": all(r["progress"]["all_followers_ended"] for r in rounds),
        "all_caught_up": all(r["attach"]["all_caught_up"] for r in rounds),
        "end_reasons": dict(
            sum(
                (collections.Counter(r["check"]["end_reasons"]) for r in rounds),
                collections.Counter(),
            )
        ),
        "followers_open_after_close": sum((r["followers_open_after_close"] or 0) for r in rounds),
        "statements_total": merged["statements"],
        "statements_by_phase": dict(merged["by_label"]),
        "app_bytes_sent": merged["bytes_sent"],
        "server": m.server(),
        "wall_s": round(m.wall_s, 2),
        "errors": {k: v for h in handles for k, v in h.result["errors"].items()},
        "top_statements": merged["top"].most_common(10),
    }
    out.metrics = metrics
    out.check("every follower stream matches the stored events", bad == 0, bad, 0)
    out.check(
        "every follower ended at the terminal event",
        metrics["end_reasons"] == {"terminal": total_followers},
        metrics["end_reasons"],
        total_followers,
    )
    out.check(
        "feeds closed with no follower left",
        metrics["followers_open_after_close"] == 0,
        metrics["followers_open_after_close"],
        0,
    )
    key = "followers_on" if notify else "followers_off"
    out.gates[f"{key}.attach_statements_per_follower"] = metrics["attach_statements_per_follower"]
    return out


async def ctl_cancel(url: str, cfg: dict[str, Any], workers: int) -> Outcome:
    out = Outcome()
    await reset_db(url)
    phases: dict[str, Any] = {}
    all_samples: dict[str, Any] = {}
    for name, contended in (("alone", False), ("with_concurrent_creators", True)):
        async with Measured(url) as m:
            handles = await run_workers(
                url,
                "cancel",
                "main",
                1,
                {"nodes": cfg["cancel_nodes"], "contended": contended, "creators": 8},
            )
        r = handles[0].result
        merged = merge_meters([r["meter"]])
        all_samples.update(merged["samples"])
        phases[name] = {
            "nodes": r["nodes"],
            "cancel_wall_s": round(r["wall_s"], 3),
            "cancel_statements": r["cancel_statements"],
            "cancel_statements_per_node": round(r["cancel_statements"] / r["nodes"], 3),
            "latency_ms": lat_summary([r["wall_s"] * 1000]),
            "app_bytes_sent": merged["bytes_sent"],
            "seeded_nodes_not_cancelled": r["seeded_not_cancelled"],
            "cascade_truncated": r["cascade_truncated"],
            "cascade_entries": r["cascade_entries"],
            "concurrent_creators": r["creators"],
            "creator_statements": r["creator_statements"],
            "server": m.server(),
            "errors": r["errors"],
            "status_counts": r["status_counts"],
            "nodes_without_cancel_request": r["nodes_without_cancel_request"],
        }
        out.check(
            f"{name}: every seeded node cancelled",
            r["seeded_not_cancelled"] == 0,
            r["seeded_not_cancelled"],
            0,
        )
        out.check(
            f"{name}: every nonterminal node including concurrent children marked for cancellation",
            r["nodes_without_cancel_request"] == 0,
            r["nodes_without_cancel_request"],
            0,
        )
        # The flag also turns on when the 999-entry outcome list is full, not only when the
        # traversal stopped (agent.py: cascade_count + 1 > len(outcomes)); every node being
        # cancelled is the real test, so the flag only fails a tree small enough to fit the list.
        out.check(
            f"{name}: cascade not truncated (flag counted only for trees under 1,000 nodes)",
            (not r["cascade_truncated"]) or r["nodes"] > 999,
            r["cascade_truncated"],
            False,
        )
        out.check(
            f"{name}: no unexpected creator errors",
            r["creators"]["other"] == 0,
            r["creators"]["other"],
            0,
        )
    if any(d["cascade_truncated"] and d["nodes"] > 999 for d in phases.values()):
        out.notes.append(
            "meta cascade_truncated was True although every node was cancelled: the flag also "
            "counts the 999-entry outcome list (agent.py cancel_run)"
        )
    out.samples = all_samples
    out.metrics = {
        "nodes": cfg["cancel_nodes"],
        "fanout": 8,
        "phases": phases,
        "top_statements": [],
    }
    out.gates["cancel.statements_per_node"] = phases["alone"]["cancel_statements_per_node"]
    return out


async def ctl_pool(url: str, cfg: dict[str, Any], workers: int) -> Outcome:
    out = Outcome()
    n = cfg["pool_runs"]
    await reset_db(url)
    async with Measured(url) as m:
        handles = await run_workers(url, "pool", "main", 1, {"runs": n})
    r = handles[0].result
    v = r["variants"]
    merged = merge_meters([x["meter"] for x in v.values()])
    out.samples = merged["samples"]
    final = await integrity(url)
    out.metrics = {
        "runs_per_variant": n,
        "pool_size": 10,
        "max_overflow": 0,
        "queued": {
            **{
                k: v["queued"][k]
                for k in ("ok", "wall_s", "errors", "peak_checked_out", "pool_timeout_s")
            },
            "latency_ms": lat_summary(v["queued"]["lat"]),
        },
        "short_timeout": {
            **{
                k: v["short_timeout"][k]
                for k in ("ok", "wall_s", "errors", "peak_checked_out", "pool_timeout_s")
            },
            "latency_ms": lat_summary(v["short_timeout"]["lat"]),
        },
        "integrity_after_both": final,
        "server": m.server(),
        "wall_s": round(m.wall_s, 2),
        "statements_per_run": round(v["queued"]["meter"]["statements"] / n, 2),
        "statements": merged["statements"],
        "app_bytes_sent": merged["bytes_sent"],
        "top_statements": merged["top"].most_common(6),
    }
    out.check(
        "pool exhaustion: all runs queued and completed",
        v["queued"]["ok"] == n and not v["queued"]["errors"],
        v["queued"]["ok"],
        n,
    )
    out.check(
        "pool never exceeded its size",
        v["queued"]["peak_checked_out"] <= 10,
        v["queued"]["peak_checked_out"],
        10,
    )
    short = v["short_timeout"]
    failed = n - short["ok"]
    out.notes.append(
        f"short pool_timeout (0.3 s): {short['ok']}/{n} runs ok, "
        f"{failed} failed with {short['errors']}"
    )
    return out


async def ctl_history(url: str, cfg: dict[str, Any], workers: int) -> Outcome:
    out = Outcome()
    calls = cfg["history_calls"]
    await reset_db(url)
    async with Measured(url) as m:
        create = (await run_workers(url, "history", "create", 1, {"calls": calls}))[0].result
        row = await sql_one(
            url,
            "select (select coalesce(sum(length(content)), 0) from react_traces) trace_chars, "
            "(select count(*) from react_traces) rows",
        )
        resume = (
            await run_workers(
                url,
                "history",
                "resume",
                1,
                {
                    "calls": calls,
                    "run_id": create["run_id"],
                    "pause_id": create["pause_id"],
                    "call_name": create["call_name"],
                    "call_id": create["call_id"],
                },
            )
        )[0].result
    merged = merge_meters([create["meter"], resume["meter"]])
    out.samples = merged["samples"]
    delta_mb = (resume["rss_peak_kb"] - resume["rss_before_kb"]) / 1024
    stored_mib = row["trace_chars"] / (1024 * 1024)
    out.metrics = {
        "big_results": calls,
        "chars_per_result": BIG_CHARS,
        "stored_trace_mib": round(stored_mib, 2),
        "trace_rows": row["rows"],
        "create_wall_s": round(create["wall_s"], 2),
        "create_statements": create["meter"]["statements"],
        "resume_status": resume["status"],
        "resume_error": resume["error"],
        "resume_wall_s": round(resume["wall_s"], 2),
        "resume_latency_ms": lat_summary([resume["wall_s"] * 1000]),
        "resume_statements": resume["meter"]["statements"],
        "resume_rss_before_mb": round(resume["rss_before_kb"] / 1024, 1),
        "resume_rss_peak_mb": round(resume["rss_peak_kb"] / 1024, 1),
        "resume_rss_growth_mb": round(delta_mb, 1),
        "resume_provider_calls": resume["provider_calls"],
        "resume_process_maxrss_mb": round(resume["maxrss_kb"] / 1024, 1),
        "server": m.server(),
        "wall_s": round(m.wall_s, 2),
        "errors": {},
        "statements": merged["statements"],
        "app_bytes_sent": merged["bytes_sent"],
        "top_statements": merged["top"].most_common(8),
    }
    out.check("history resume finished", resume["status"] == "success", resume["status"], "success")
    out.gates["history.resume_statements"] = resume["meter"]["statements"]
    return out


def slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den if den else 0.0


async def ctl_memory(url: str, cfg: dict[str, Any], workers: int) -> Outcome:
    out = Outcome()
    await reset_db(url)
    async with Measured(url) as m:
        h = (
            await run_workers(
                url,
                "memory",
                "main",
                1,
                {"cycles": cfg["memory_cycles"], "runs": cfg["memory_runs"]},
                timeout=7200,
            )
        )[0]
    cycles = h.result["cycles"]
    out.samples = {}
    warm = min(5, max(1, len(cycles) // 3))
    series = [c["rss_kb"] / 1024 for c in cycles]
    tail = series[warm:] or series
    s = slope(tail)
    out.metrics = {
        "cycles": len(cycles),
        "runs_per_cycle": cfg["memory_runs"],
        "cycle_latency_ms": lat_summary([c["wall_ms"] for c in cycles]),
        "statements": h.result["meter"]["statements"],
        "statements_per_cycle": round(h.result["meter"]["statements"] / len(cycles), 2),
        "app_bytes_sent": h.result["meter"]["bytes_sent"],
        "followers_per_cycle": 50,
        "rss_mb_series": [round(x, 1) for x in series],
        "warmup_cycles": warm,
        "rss_slope_mb_per_cycle": round(s, 3),
        "rss_first_mb": round(series[0], 1),
        "rss_last_mb": round(series[-1], 1),
        "max_checked_out": max(c["checked_out"] for c in cycles),
        "max_followers_open_after_close": max(
            (c["followers_open_after_close"] or 0) for c in cycles
        ),
        "fds_first_last": [cycles[0]["fds"], cycles[-1]["fds"]],
        "tasks_first_last": [cycles[0]["tasks"], cycles[-1]["tasks"]],
        "gc_objects_first_last": [cycles[0]["gc_objects"], cycles[-1]["gc_objects"]],
        "runs_ok_every_cycle": all(c["runs_ok"] == cfg["memory_runs"] for c in cycles),
        "followers_mismatch_total": sum(c["followers_mismatch"] for c in cycles),
        "server": m.server(),
        "wall_s": round(m.wall_s, 2),
        "errors": h.result["errors"],
        "top_statements": [],
    }
    if len(cycles) - warm >= 4:
        out.check("RSS growth after warm-up under 1 MB per cycle", s < 1.0, round(s, 3), 1.0)
    else:
        out.notes.append("too few cycles for the RSS slope check; informational only")
    out.check(
        "no connection checked out between cycles",
        out.metrics["max_checked_out"] == 0,
        out.metrics["max_checked_out"],
        0,
    )
    out.check(
        "no follower left after feed close",
        out.metrics["max_followers_open_after_close"] == 0,
        out.metrics["max_followers_open_after_close"],
        0,
    )
    out.check(
        "every cycle's runs and streams correct",
        out.metrics["runs_ok_every_cycle"] and out.metrics["followers_mismatch_total"] == 0,
    )
    return out


# ---------------------------------------------------------------------------
# EXPLAIN of the statements the scenarios actually ran
# ---------------------------------------------------------------------------


class _RollbackError(Exception):
    pass


def _is_table_statement(stmt: str) -> bool:
    flat = _norm(stmt).upper()
    if flat.startswith(
        ("COMMIT", "ROLLBACK", "BEGIN", "SELECT PG_", "CREATE", "DROP", "ALTER", "ANALYZE")
    ):
        return False
    return (
        re.search(
            r"\b(AGENT_RUNS|RUN_EVENTS|REACT_TRACES|TOOL_CALLS|TOKEN_USAGE|LLM_INTERACTIONS|RUN_INBOX|DENDRUX_SETTINGS)\b",
            flat,
        )
        is not None
    )


async def explain_samples(url: str, samples: dict[str, Any], cap: int = 30) -> list[dict[str, Any]]:
    import asyncpg

    conn = await asyncpg.connect(
        dsn_of(url), server_settings={"application_name": "dendrux_load_book"}
    )
    plans: list[dict[str, Any]] = []
    try:
        await conn.execute("ANALYZE")
        counts = {
            r["relname"]: int(r["n"])
            for r in await conn.fetch(
                "select relname, reltuples::bigint n from pg_class "
                "where relkind = 'r' and relnamespace = 'public'::regnamespace"
            )
        }
        chosen = [(k, v) for k, v in samples.items() if _is_table_statement(v[0])]
        chosen.sort(
            key=lambda kv: (not any(mk.lower() in kv[0].lower() for mk in NEW_MARKERS), kv[0])
        )
        for _kind, (stmt, params, label, many) in chosen[:cap]:
            args = params[0] if many and params else params
            record: dict[str, Any] = {
                "statement": _norm(stmt)[:600],
                "scenario_phase": label,
                "new_in_0_6": any(mk.lower() in stmt.lower() for mk in NEW_MARKERS),
            }
            try:
                plan = await _explain(conn, stmt, args, seqscan_off=False)
                record.update(plan)
                tables = re.findall(r"Seq Scan on (\w+)", plan["plan"])
                big = [t for t in tables if t in BIG_TABLES and counts.get(t, 0) >= 500]
                record["seq_scans_on_populated_tables"] = big
                if big:
                    forced = await _explain(conn, stmt, args, seqscan_off=True)
                    record["with_seqscan_off"] = {
                        "access": forced["access"],
                        "execution_ms": forced["execution_ms"],
                        "index_available": not re.search(
                            "Seq Scan on (" + "|".join(big) + ")", forced["plan"]
                        ),
                    }
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            plans.append(record)
    finally:
        await conn.close()
    return plans


async def _explain(conn: Any, stmt: str, args: Any, *, seqscan_off: bool) -> dict[str, Any]:
    plan_text = ""
    try:
        async with conn.transaction():
            if seqscan_off:
                await conn.execute("SET LOCAL enable_seqscan = off")
            # An INSERT would hit its own unique key when replayed, so it gets the plan only.
            options = (
                "COSTS OFF"
                if _norm(stmt).upper().startswith("INSERT")
                else "ANALYZE, BUFFERS, COSTS OFF, TIMING OFF"
            )
            rows = await conn.fetch(f"EXPLAIN ({options}) " + stmt, *args)
            plan_text = "\n".join(r[0] for r in rows)
            raise _RollbackError
    except _RollbackError:
        pass
    access = [
        line.strip()
        for line in plan_text.splitlines()
        if re.search(
            r"(Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan|Bitmap Index Scan)", line
        )
    ]
    match = re.search(r"Execution Time: ([0-9.]+) ms", plan_text)
    return {
        "access": access[:6],
        "execution_ms": float(match.group(1)) if match else None,
        "plan": plan_text[:2500],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    cells = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)
    ]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()  # noqa: E731
    return [line(headers), line(["-" * w for w in widths]), *(line(r) for r in cells)]


def machine_meta() -> dict[str, Any]:
    cpu = ""
    cores = os.cpu_count()
    mem_gb = None
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        ).stdout.strip()
        mem_gb = round(int(mem) / 2**30, 1) if mem else None
    else:
        with contextlib.suppress(OSError):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    mem_gb = round(int(line.split()[1]) / 2**20, 1)
                    break
    import sqlalchemy

    try:
        import asyncpg

        asyncpg_version = asyncpg.__version__
    except ImportError:
        asyncpg_version = None
    import dendrux

    commit = None
    with contextlib.suppress(Exception):
        commit = (
            subprocess.run(
                ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    return {
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu": cpu,
        "cores": cores,
        "memory_gb": mem_gb,
        "python": platform.python_version(),
        "sqlalchemy": sqlalchemy.__version__,
        "asyncpg": asyncpg_version,
        "dendrux_file": dendrux.__file__,
        "dendrux_version": getattr(dendrux, "__version__", None),
        "git_commit": commit,
        "load_average_1m_at_start": round(os.getloadavg()[0], 2),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


async def server_meta(url: str) -> dict[str, Any]:
    import asyncpg

    conn = await asyncpg.connect(
        dsn_of(url), server_settings={"application_name": "dendrux_load_book"}
    )
    try:
        out = {"version": await conn.fetchval("select version()")}
        for name in (
            "shared_buffers",
            "max_connections",
            "work_mem",
            "synchronous_commit",
            "fsync",
            "full_page_writes",
            "wal_level",
            "effective_cache_size",
            "max_wal_size",
        ):
            out[name] = await conn.fetchval(f"show {name}")
        out["pg_stat_statements"] = bool(
            await conn.fetchval("select 1 from pg_extension where extname = 'pg_stat_statements'")
        )
        return out
    finally:
        await conn.close()


def render(name: str, o: Outcome) -> list[str]:
    m = o.metrics
    lines = [f"== {name} =="]
    if "variants" in m and name == "iteration":
        rows = []
        for v, d in m["variants"].items():
            rows.append(
                [
                    v,
                    d["startup"],
                    d["iteration_1"],
                    d["iteration_2"],
                    d["terminal"],
                    d["max_non_terminal"],
                ]
            )
        lines += fmt_table(
            ["variant", "startup", "iter 1", "iter 2", "terminal", "max non-terminal"], rows
        )
        best = m["variants"]["notify_on_batch"]["by_operation_iteration_2"]
        lines.append("statements by store operation, iteration 2, notifier on:")
        lines += ["  " + f"{k}: {v}" for k, v in sorted(best.items())]
        for variant, d in m["variants"].items():
            lines.append(f"{variant} whole-run latency ms: {d['latency_ms']}")
    elif name.startswith("followers"):
        lines += fmt_table(
            ["metric", "value"],
            [
                ["followers / runs", f"{m['followers']} / {m['runs']}"],
                [
                    "attach statements (per follower)",
                    f"{m['attach_statements']} ({m['attach_statements_per_follower']})",
                ],
                [
                    "idle statements/s (per watched run)",
                    f"{m['idle_statements_per_s']} ({m['idle_statements_per_s_per_watched_run']})",
                ],
                [
                    "burst statements (per run)",
                    f"{m['progress_statements']} ({m['progress_statements_per_run']})",
                ],
                [
                    f"trickle delivery lag ms p50 / p95 ({m['trickle_runs']} runs)",
                    f"{m['trickle_delivery_lag_ms'].get('p50_ms')} / "
                    f"{m['trickle_delivery_lag_ms'].get('p95_ms')}",
                ],
                [
                    "burst delivery lag ms p50 / p95",
                    f"{m['burst_delivery_lag_ms'].get('p50_ms')} / "
                    f"{m['burst_delivery_lag_ms'].get('p95_ms')}",
                ],
                ["streams not matching the stored events", m["streams_not_matching_db"]],
                ["wall s", m["wall_s"]],
            ],
        )
    elif name == "cancel":
        rows = []
        for ph, d in m["phases"].items():
            srv = d["server"]
            rows.append(
                [
                    ph,
                    d["nodes"],
                    d["cancel_wall_s"],
                    f"{d['cancel_statements']} ({d['cancel_statements_per_node']})",
                    f"{d['concurrent_creators']['admitted']} / "
                    f"{d['concurrent_creators']['refused']}",
                    f"{srv['lock_wait_ms_run_row']} ms "
                    f"(max {srv['max_concurrent_run_row_waiters']})",
                    d["seeded_nodes_not_cancelled"],
                ]
            )
        lines += fmt_table(
            [
                "phase",
                "nodes",
                "cancel s",
                "statements (per node)",
                "creators admitted / refused",
                "run-row lock wait",
                "not cancelled",
            ],
            rows,
        )
        for ph, d in m["phases"].items():
            lines.append(
                f"{ph}: latency ms {d['latency_ms']} (one sample); "
                f"app bytes {d['app_bytes_sent']}; server {json.dumps(d['server'])}"
            )
    elif name == "pool":
        q, s = m["queued"], m["short_timeout"]
        lines += fmt_table(
            ["variant", "ok / runs", "peak checkouts", "p50 / p95 ms", "errors"],
            [
                [
                    "queued (pool 10, timeout 120 s)",
                    f"{q['ok']} / {m['runs_per_variant']}",
                    q["peak_checked_out"],
                    f"{q['latency_ms'].get('p50_ms')} / {q['latency_ms'].get('p95_ms')}",
                    q["errors"],
                ],
                [
                    "short timeout (0.3 s)",
                    f"{s['ok']} / {m['runs_per_variant']}",
                    s["peak_checked_out"],
                    f"{s['latency_ms'].get('p50_ms')} / {s['latency_ms'].get('p95_ms')}",
                    s["errors"],
                ],
            ],
        )
    elif name == "history":
        lines += fmt_table(
            ["metric", "value"],
            [
                [k, m[k]]
                for k in (
                    "big_results",
                    "stored_trace_mib",
                    "trace_rows",
                    "create_wall_s",
                    "create_statements",
                    "resume_status",
                    "resume_wall_s",
                    "resume_statements",
                    "resume_rss_before_mb",
                    "resume_rss_peak_mb",
                    "resume_rss_growth_mb",
                )
            ],
        )
    elif name == "memory":
        lines += fmt_table(
            ["metric", "value"],
            [
                [k, m[k]]
                for k in (
                    "cycles",
                    "rss_first_mb",
                    "rss_last_mb",
                    "rss_slope_mb_per_cycle",
                    "max_checked_out",
                    "max_followers_open_after_close",
                    "fds_first_last",
                    "tasks_first_last",
                    "gc_objects_first_last",
                )
            ],
        )
        lines.append("RSS MB per cycle: " + ", ".join(str(x) for x in m["rss_mb_series"]))
        lines.append(f"cycle latency ms: {m['cycle_latency_ms']}")
        lines.append(f"statements per cycle: {m['statements_per_cycle']}")
    else:
        rows = [
            ["runs", m.get("runs")],
            ["wall s", m.get("wall_s")],
            ["runs/s", m.get("runs_per_s")],
            ["statements (per run)", f"{m.get('statements')} ({m.get('statements_per_run')})"],
            ["statements/s", m.get("statements_per_s")],
            [
                "latency ms p50 / p95",
                f"{m['latency_ms'].get('p50_ms')} / {m['latency_ms'].get('p95_ms')}",
            ],
            ["app bytes sent per run", m.get("app_bytes_sent_per_run")],
            ["WAL bytes per run", m.get("wal_bytes_per_run")],
            ["peak connections (sampled, all roles)", m["server"]["peak_connections"]],
            ["run-row lock wait ms (sampled)", m["server"]["lock_wait_ms_run_row"]],
        ]
        for key in (
            "recovery_wall_s",
            "recovery_statements_per_reclaim",
            "billed_model_calls",
            "expected_model_calls",
            "extra_billed_calls",
            "statements_per_run_including_recovery",
        ):
            if key in m:
                rows.append([key, m[key]])
        lines += fmt_table(["metric", "value"], rows)
        if m.get("statements_per_operation"):
            lines.append("statements per operation: " + json.dumps(m["statements_per_operation"]))
    srv = m.get("server")
    if "app_bytes_sent" in m:
        lines.append(f"application write parameter bytes: {m['app_bytes_sent']}")
    if srv:
        lines.append(
            f"server: peak connections {srv['peak_connections']} "
            f"{srv['peak_connections_by_role']}; run-row lock wait "
            f"{srv['lock_wait_ms_run_row']} ms "
            f"(max {srv['max_concurrent_run_row_waiters']} waiters, {srv['interval_ms']:.0f} ms "
            f"sampling); WAL {srv['wal_bytes']} B; transactions {srv['transactions']}"
        )
    if srv and srv.get("growth_by_table"):
        grown = ", ".join(f"{t} {v / 2**20:.1f} MiB" for t, v in srv["growth_by_table"].items())
        lines.append(f"table growth (data + indexes + toast): {grown}")
    for n in o.notes:
        lines.append("note: " + n)
    for c in o.checks:
        verdict = "PASS" if c["ok"] else "FAIL"
        lines.append(f"[{verdict}] {c['name']} (value {c['value']}, limit {c['limit']})")
    for plan in m.get("explain", []):
        lines.append(f"EXPLAIN [{plan['scenario_phase']}]: {plan['statement']}")
        lines.append(plan.get("plan", plan.get("error", "no plan")))
    return lines


def load_baseline(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def compare_gates(
    current: dict[str, float], baseline: dict[str, Any], tolerance: float = 0.20
) -> tuple[list[list[Any]], bool]:
    rows: list[list[Any]] = []
    failed = False
    for key, base in sorted(baseline.get("gates", {}).items()):
        if key not in current:
            continue
        now = current[key]
        limit = base * (1 + tolerance) + (0.5 if base < 5 else 0)
        bad = now > limit
        failed |= bad
        rows.append(
            [
                key,
                base,
                now,
                f"{(now - base) / base * 100:+.1f}%" if base else "n/a",
                "FAIL" if bad else "ok",
            ]
        )
    return rows, failed


async def run_all(args: argparse.Namespace) -> int:
    url = str(check_private_url(args.url).render_as_string(hide_password=False))
    cfg = dict(PRESETS[args.scale])
    from dendrux.runtime.state import SQLAlchemyStateStore

    legacy = not hasattr(SQLAlchemyStateStore, "checkpoint_step")
    wanted = args.scenarios.split(",") if args.scenarios else list(ALL_SCENARIOS)
    if legacy:  # a 0.5 checkout can run the scenarios that need no 0.6 surface
        wanted = [w for w in wanted if w in LEGACY_SCENARIOS] or list(LEGACY_SCENARIOS)
    baseline = load_baseline(Path(args.baseline) if args.baseline else None)
    await reset_db(url)  # creates the private database when it is missing
    results: dict[str, Any] = {
        "meta": {
            **machine_meta(),
            "server": await server_meta(url),
            "scale": args.scale,
            "workers": args.workers,
            "pool_per_worker": POOL,
            "aggregate_pool_ceiling": POOL_CEILING * args.workers,
            "config": cfg,
            "legacy_0_5_checkout": legacy,
        },
        "scenarios": {},
        "checks": [],
        "gates": {},
    }
    log(f"dendrux at {results['meta']['dendrux_file']} (legacy 0.5 checkout: {legacy})")
    log(f"postgres: {results['meta']['server']['version'][:60]}")
    outcomes: dict[str, Outcome] = {}
    drivers = {
        "iteration": lambda: ctl_iteration(url, cfg, args.workers),
        "runs": lambda: ctl_runs(url, cfg, args.workers),
        "kill_checkpoint": lambda: ctl_kill(url, cfg, args.workers, "checkpoint"),
        "kill_inflight": lambda: ctl_kill(url, cfg, args.workers, "inflight"),
        "followers_on": lambda: ctl_followers(url, cfg, args.workers, True),
        "followers_off": lambda: ctl_followers(url, cfg, args.workers, False),
        "cancel": lambda: ctl_cancel(url, cfg, args.workers),
        "pool": lambda: ctl_pool(url, cfg, args.workers),
        "history": lambda: ctl_history(url, cfg, args.workers),
        "memory": lambda: ctl_memory(url, cfg, args.workers),
    }
    explain_for = {
        "a_notify_off",
        "iteration",
        "runs",
        "kill_checkpoint",
        "kill_inflight",
        "cancel",
        "followers_on",
        "followers_off",
        "pool",
        "history",
    }
    jobs: list[tuple[str, Any]] = []
    for name in wanted:
        if name == "a":
            jobs.append(("a_notify_off", lambda: ctl_a(url, cfg, args.workers, False)))
            jobs.append(("a_shared_agent", lambda: ctl_a(url, cfg, args.workers, False, True)))
            if not legacy:
                jobs.append(("a_notify_on", lambda: ctl_a(url, cfg, args.workers, True)))
        else:
            jobs.append((name, drivers[name]))
    failures = 0
    for name, make in jobs:
        log(f"scenario {name} ...")
        t0 = time.perf_counter()
        try:
            o = await make()
            if name in explain_for and o.samples and not args.no_explain:
                o.metrics["explain"] = await explain_samples(url, o.samples)
            outcomes[name] = o
            results["scenarios"][name] = {
                "metrics": o.metrics,
                "checks": o.checks,
                "notes": o.notes,
            }
            results["gates"].update(o.gates)
            log(f"scenario {name} done in {time.perf_counter() - t0:.1f}s")
        except BaseException as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log(f"scenario {name} FAILED: {exc!r}")
            results["scenarios"][name] = {"error": tb}
            results["checks"].append(
                {
                    "name": f"{name} ran to completion",
                    "ok": False,
                    "value": repr(exc)[:200],
                    "limit": None,
                }
            )
            failures += 1
            if isinstance(exc, KeyboardInterrupt):
                raise
    # cross-scenario budget checks
    ref = baseline.get("v050_reference", {}).get("scenario_a_statements_per_run", V050_REFERENCE_A)
    results["meta"]["v050_reference_scenario_a_statements_per_run"] = ref
    for variant in ("a_notify_off", "a_notify_on"):
        if variant in outcomes and not legacy:
            val = outcomes[variant].metrics["statements_per_run"]
            ok = val <= ref * 1.10
            results["checks"].append(
                {
                    "name": f"0.5 scenario A within 10% of the 0.5 baseline ({variant})",
                    "ok": ok,
                    "value": val,
                    "limit": round(ref * 1.10, 2),
                }
            )
            outcomes[variant].metrics["vs_v050_percent"] = round((val - ref) / ref * 100, 1)
    for name, o in outcomes.items():
        results["checks"].extend({**c, "name": f"{name}: {c['name']}"} for c in o.checks)
    conn_ceiling = POOL_CEILING * args.workers
    for name in ("a_notify_off", "runs"):
        if name in outcomes:
            srv = outcomes[name].metrics["server"]
            by_role = srv["peak_connections_by_role"]
            worker_conns = sum(v for k, v in by_role.items() if k.startswith("dendrux_load_w"))
            results["checks"].append(
                {
                    "name": f"{name}: worker connections within the aggregate pool ceiling",
                    "ok": worker_conns <= conn_ceiling,
                    "value": worker_conns,
                    "limit": conn_ceiling,
                }
            )
    rows, regressed = (
        compare_gates(results["gates"], baseline) if baseline.get("gates") else ([], False)
    )
    results["baseline_comparison"] = rows

    lines: list[str] = []
    for name, o in outcomes.items():
        lines += render(name, o) + [""]
    if rows:
        lines.append("== baseline comparison (20% tolerance, statement counts only) ==")
        lines += fmt_table(["gate", "baseline", "now", "change", ""], rows) + [""]
    lines.append("== checks ==")
    failed_checks = [c for c in results["checks"] if not c["ok"]]
    lines += [
        f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']} (value {c['value']}, limit {c['limit']})"
        for c in results["checks"]
    ]
    text = "\n".join(lines)
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        log(f"wrote {args.json}")
    if args.write_baseline:
        write_baseline(Path(args.write_baseline), results, baseline)
    return 1 if (failed_checks or regressed or failures) else 0


def write_baseline(path: Path, results: dict[str, Any], previous: dict[str, Any]) -> None:
    meta = results["meta"]
    data = {
        "_comment": (
            "Baseline for bench.yml. Gated metrics are statement counts (hardware independent); "
            "a regression of more than 20% fails the weekly run. Change this file only in a "
            "release commit, from a local `make load-quick` on a quiet machine."
        ),
        "scale": meta["scale"],
        "workers": meta["workers"],
        "recorded": {
            k: meta[k]
            for k in (
                "date",
                "machine",
                "cpu",
                "cores",
                "memory_gb",
                "python",
                "sqlalchemy",
                "asyncpg",
                "git_commit",
            )
        },
        "postgres": {
            k: meta["server"][k]
            for k in ("version", "shared_buffers", "max_connections", "synchronous_commit", "fsync")
        },
        "gates": results["gates"],
        "v050_reference": previous.get(
            "v050_reference",
            {
                "scenario_a_statements_per_run": V050_REFERENCE_A,
                "source": "owner-docs/build-0.5.0/load-test-results.md, commit dca4f05e",
            },
        ),
        "informational": {
            name: {
                k: s["metrics"].get(k)
                for k in ("runs_per_s", "statements_per_s", "latency_ms", "wal_bytes_per_run")
                if k in s.get("metrics", {})
            }
            for name, s in results["scenarios"].items()
            if "metrics" in s
        },
    }
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    log(f"wrote baseline {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("DENDRUX_LOAD_PG_URL"),
        help="private *_test database (env DENDRUX_LOAD_PG_URL)",
    )
    parser.add_argument("--scale", choices=sorted(PRESETS), default="full")
    parser.add_argument(
        "--workers", type=int, default=2, help="OS processes, each with its own pool (default 2)"
    )
    parser.add_argument("--scenarios", help=f"comma list from: {','.join(ALL_SCENARIOS)}")
    parser.add_argument("--json", help="write the full results here")
    parser.add_argument(
        "--baseline",
        help="compare statement-count gates with this file; exit 1 on >20%% regression",
    )
    parser.add_argument("--write-baseline", help="write a baseline file from this run")
    parser.add_argument("--no-explain", action="store_true")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("--url (or DENDRUX_LOAD_PG_URL) is required")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.scenarios:
        unknown = set(args.scenarios.split(",")) - set(ALL_SCENARIOS)
        if unknown:
            parser.error(f"unknown scenarios: {sorted(unknown)}")
    try:
        check_private_url(args.url)
    except ValueError as exc:
        print(f"load.py: {exc}", file=sys.stderr)
        return 2
    return asyncio.run(run_all(args))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
