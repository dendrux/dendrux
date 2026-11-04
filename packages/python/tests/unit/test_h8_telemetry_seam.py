"""The retry-telemetry seam (H8b): the provider layer reports through two sinks.

``telemetry_context`` / ``telemetry_stream`` take a ``TelemetrySinks`` pair of
async callables ``(event_type, iteration, data)`` instead of the loop's
recorder and notifier; the loop builds the pair with
``loops._helpers.governance_sinks``. The provider layer never imports the loops.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from dendrux.llm import _retry_telemetry as telemetry
from dendrux.llm._resilience import RETRYABLE_HTTP_STATUSES
from dendrux.llm._retry_telemetry import TelemetrySinks, telemetry_context, telemetry_stream
from dendrux.loops._helpers import governance_sinks
from dendrux.loops.base import BaseNotifier, BaseRecorder

if TYPE_CHECKING:
    import pytest

LLM_DIR = Path(__file__).resolve().parents[2] / "src" / "dendrux" / "llm"


class _Recorder(BaseRecorder):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.calls.append((run_id, event_type, iteration, data, correlation_id))


class _Notifier(BaseNotifier):
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail = fail

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        if self.fail:
            raise RuntimeError("notifier down")
        self.calls.append((run_id, event_type, iteration, data, correlation_id))


def test_no_sinks_without_recorder_or_notifier() -> None:
    assert governance_sinks(None, None, "run-1") is None


async def test_record_sink_binds_run_id_and_passes_correlation_none() -> None:
    recorder = _Recorder()
    sinks = governance_sinks(recorder, None, "run-1")
    assert sinks is not None and sinks.notify is None and sinks.record is not None
    await sinks.record("provider.retry", 3, {"attempt": 1})
    assert recorder.calls == [("run-1", "provider.retry", 3, {"attempt": 1}, None)]


async def test_notify_sink_binds_run_id_and_swallows(caplog: pytest.LogCaptureFixture) -> None:
    notifier = _Notifier()
    sinks = governance_sinks(None, notifier, "run-2")
    assert sinks is not None and sinks.record is None and sinks.notify is not None
    await sinks.notify("provider.retry", 1, {"attempt": 2})
    assert notifier.calls == [("run-2", "provider.retry", 1, {"attempt": 2}, None)]

    failing = governance_sinks(None, _Notifier(fail=True), "run-2")
    assert failing is not None and failing.notify is not None
    with caplog.at_level(logging.WARNING, logger="dendrux"):
        await failing.notify("provider.retry", 1, {})
    assert "Notifier.on_governance_event failed" in caplog.text


async def test_hook_returns_before_counting_without_sinks() -> None:
    attempts = [0]
    token = telemetry._attempts.set(attempts)
    try:
        with telemetry_context(run_id="r", iteration=1, sinks=None):
            await telemetry._on_response(
                httpx.Response(503, request=httpx.Request("GET", "https://x.test/v1/a")),
                retryable_statuses=RETRYABLE_HTTP_STATUSES,
            )
        with telemetry_context(run_id="r", iteration=1, sinks=TelemetrySinks(None, None)):
            await telemetry._on_response(
                httpx.Response(503, request=httpx.Request("GET", "https://x.test/v1/a")),
                retryable_statuses=RETRYABLE_HTTP_STATUSES,
            )
    finally:
        telemetry._attempts.reset(token)
    assert attempts == [0]


async def test_one_failing_sink_does_not_stop_the_other(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[tuple[str, int, dict[str, Any]]] = []

    async def record(event_type: str, iteration: int, data: dict[str, Any]) -> None:
        raise RuntimeError("store down")

    async def notify(event_type: str, iteration: int, data: dict[str, Any]) -> None:
        seen.append((event_type, iteration, data))

    with (
        caplog.at_level(logging.WARNING, logger="dendrux"),
        telemetry_context(run_id="r", iteration=4, sinks=TelemetrySinks(record, notify)),
    ):
        await telemetry._on_response(
            httpx.Response(429, request=httpx.Request("GET", "https://x.test/v1/a")),
            retryable_statuses=RETRYABLE_HTTP_STATUSES,
        )
    assert [(e, i) for e, i, _ in seen] == [("provider.retry", 4)]
    assert "provider.retry recorder write failed" in caplog.text


async def test_stream_binds_sinks_on_every_pull() -> None:
    sinks = TelemetrySinks(None, None)
    observed: list[Any] = []

    async def inner() -> Any:
        observed.append(telemetry._sinks.get() is sinks and telemetry._run_id.get())
        yield 1
        observed.append(telemetry._run_id.get())

    wrapped = telemetry_stream(inner(), run_id="s1", iteration=1, sinks=sinks)
    assert [item async for item in wrapped] == [1]
    assert observed == ["s1", "s1"]
    assert telemetry._run_id.get() is None


def test_provider_layer_never_imports_the_loops() -> None:
    for path in sorted(LLM_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("dendrux.loops"), (path.name, node.lineno)


async def test_hook_uses_the_status_set_it_is_given() -> None:
    """H8c: the retryable statuses are handed to the hook factory (no import per response)."""
    from dendrux.llm._retry_telemetry import telemetry_event_hooks

    seen: list[int] = []

    async def record(event_type: str, iteration: int, data: dict[str, Any]) -> None:
        seen.append(data["status_code"])

    (hook,) = telemetry_event_hooks(frozenset({418}))["response"]
    request = httpx.Request("GET", "https://x.test/v1/a")
    with telemetry_context(run_id="r", iteration=1, sinks=TelemetrySinks(record, None)):
        await hook(httpx.Response(418, request=request))
        await hook(httpx.Response(503, request=request))
    assert seen == [418]


def test_retry_telemetry_has_no_function_local_imports() -> None:
    tree = ast.parse((LLM_DIR / "_retry_telemetry.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            inner = [n for n in ast.walk(node) if isinstance(n, ast.Import | ast.ImportFrom)]
            assert not inner, node.name
