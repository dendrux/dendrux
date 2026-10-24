"""Safe failure sinks, telemetry ownership, budget enforcement and migrations."""

import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rich.console import Console

from dendrux.errors import ProviderTimeoutError
from dendrux.llm._retry_telemetry import telemetry_stream
from dendrux.loops._budget import BudgetTracker
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.runtime.tasks import RunTaskManager
from dendrux.types import Budget, RunResult, RunStatus, ToolCall, UsageStats

PRIVATE = "vendor-private-request-echo"


def provider_error():
    error = ProviderTimeoutError(
        message=PRIVATE, provider="fixture", timeout_reason="call_deadline"
    )
    error.__cause__ = RuntimeError(PRIVATE)
    return error


@pytest.mark.parametrize("path", ["run", "llm", "orphan_llm", "orphan_tool", "result"])
async def test_provider_error_is_safe_in_every_span_path(path):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    notifier = OpenTelemetryNotifier(tracer=provider.get_tracer("test"))
    error = provider_error()
    await notifier.on_run_started("r")
    if path in {"llm", "orphan_llm"}:
        await notifier.on_llm_call_started("r", 1)
    if path == "orphan_tool":
        await notifier.on_tool_started("r", ToolCall(name="work", params={}), 1)
    if path == "llm":
        await notifier.on_llm_call_failed("r", 1, error)
    elif path == "result":
        from dendrux.types import ErrorInfo

        await notifier.on_run_finished(
            "r",
            RunResult(
                run_id="r",
                status=RunStatus.ERROR,
                error=error.safe_message,
                error_info=ErrorInfo.from_error(error),
            ),
        )
    else:
        await notifier.on_run_failed("r", error)
    spans = exporter.get_finished_spans()
    for span in spans:
        serialized = json.dumps(
            {
                "attributes": dict(span.attributes),
                "status": span.status.description,
                "events": [{"name": e.name, "attributes": dict(e.attributes)} for e in span.events],
            }
        )
        assert PRIVATE not in serialized, (
            "span paths must not serialize vendor text or exception chains"
        )
        if path != "llm" or span.name.startswith("chat"):
            assert span.status.description == error.safe_message
            assert span.attributes["dendrux.error.category"] == "timeout"
    provider.shutdown()


@pytest.mark.parametrize("path", ["run", "llm"])
async def test_provider_error_console_is_safe(monkeypatch, path):
    from dendrux.notifiers import console

    output = io.StringIO()
    monkeypatch.setattr(console, "_console", Console(file=output, width=200, color_system=None))
    error = provider_error()
    notifier = ConsoleNotifier()
    if path == "run":
        await notifier.on_run_failed("r", error)
    else:
        await notifier.on_llm_call_failed("r", 1, error)
    assert PRIVATE not in output.getvalue(), "console failures must use ErrorInfo.safe_message"
    assert error.safe_message in output.getvalue()


async def test_unstarted_telemetry_close_releases_eager_provider_once():
    class EagerStream:
        closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            self.closed += 1

    inner = EagerStream()
    wrapped = telemetry_stream(inner, run_id="eager", iteration=1, recorder=None, notifier=None)
    await asyncio.gather(wrapped.aclose(), wrapped.aclose())
    assert inner.closed == 1, "closing an unstarted telemetry wrapper must release its eager owner"


async def test_budget_tracker_unknown_numeric_cost_stops_as_unpriced():
    tracker = BudgetTracker(
        Budget(max_cost_usd=1.0, on_exceed="cancel"),
        run_id="budget",
        usage=UsageStats(cost_usd=0.0),
        recorder=None,
        notifier=None,
        warnings=[],
    )
    tracker.add_response(UsageStats(cost_usd=0.4, cost_unknown=True))
    action, breaches = await tracker.checkpoint(2)
    assert action == "cancel" and breaches, (
        "provider unknown cost must trigger the unpriced hard-budget rule"
    )
    assert tracker.pause_fields()["budget_unpriced_calls"] == 1
    assert breaches[0].used == 0.0


async def test_non_provider_task_failure_keeps_diagnostic_stack(caplog):
    async def fail():
        raise ValueError("ordinary internal bug")

    with pytest.raises(ValueError):
        await RunTaskManager().spawn("ordinary", fail())
    record = next(r for r in caplog.records if r.name == "dendrux.runtime.tasks")
    assert record.exc_info is not None, "ordinary bugs must retain their diagnostic traceback"


def test_migration_keeps_existing_dendrux_loggers_enabled(tmp_path):
    migrations = Path(__file__).resolve().parents[2] / "src/dendrux/db/migrations"
    # Run the real env.py/fileConfig branch in an isolated interpreter: logging
    # configuration is process-global and must not alter pytest's own handlers.
    script = f"""
import io, logging
from alembic import command
from alembic.config import Config
log = logging.getLogger('dendrux.runtime.runner')
log.disabled = False
cfg = Config('alembic.ini', output_buffer=io.StringIO())
cfg.set_main_option('script_location', {str(migrations)!r})
cfg.set_main_option('sqlalchemy.url', 'sqlite+aiosqlite:///unused.db')
command.upgrade(cfg, '2ce3bb57b0c5', sql=True)
assert not log.disabled, 'migration must not disable existing dendrux loggers'
log.error('migration logger remains active')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "migration logger remains active" in result.stderr


async def test_exhausted_telemetry_releases_eager_provider():
    class Eager:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    inner = Eager()
    wrapped = telemetry_stream(inner, run_id="eager", iteration=1, recorder=None, notifier=None)
    assert [item async for item in wrapped] == []
    assert inner.closed, "exhaustion must release an eager provider as well as explicit close"
