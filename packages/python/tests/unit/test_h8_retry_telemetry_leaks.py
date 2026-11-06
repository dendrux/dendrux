"""Retry telemetry warnings omit database statements and their parameters.

Production notifier callbacks also retain their safe exception projection.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from sqlalchemy.exc import OperationalError

from dendrux.llm._resilience import RETRYABLE_HTTP_STATUSES
from dendrux.llm._retry_telemetry import (
    TelemetrySinks,
    call_attempt_tracking,
    telemetry_context,
    telemetry_event_hooks,
)
from dendrux.loops._helpers import governance_sinks
from dendrux.loops.base import BaseNotifier, BaseRecorder

SECRET = "SECRET-H8-sink-91c2"


def _db_error() -> OperationalError:
    return OperationalError(f"INSERT INTO run_events VALUES ('{SECRET}')", {}, Exception("x"))


class _FailingRecorder(BaseRecorder):
    async def on_governance_event(self, *args: Any, **kwargs: Any) -> None:
        raise _db_error()


class _FailingNotifier(BaseNotifier):
    async def on_governance_event(self, *args: Any, **kwargs: Any) -> None:
        raise _db_error()


async def _one_retryable_response(recorder: Any, notifier: Any) -> None:
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(5.0),
        event_hooks=telemetry_event_hooks(RETRYABLE_HTTP_STATUSES),
    )
    client._transport = httpx.MockTransport(lambda request: httpx.Response(503))  # type: ignore[attr-defined]
    try:
        with (
            telemetry_context(
                run_id="run-leak",
                iteration=1,
                sinks=governance_sinks(recorder, notifier, "run-leak"),
            ),
            call_attempt_tracking(),
        ):
            await client.get("https://api.example.com/v1/chat/completions")
    finally:
        await client.aclose()


def _formatted(caplog: pytest.LogCaptureFixture, message: str) -> str:
    handler = logging.Handler()
    handler.setFormatter(logging.Formatter())
    records = [r for r in caplog.records if r.getMessage() == message]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    return handler.format(records[0])


async def test_recorder_sink_failure_log_omits_the_db_statement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    await _one_retryable_response(_FailingRecorder(), None)
    text = _formatted(caplog, "provider.retry recorder write failed")
    assert SECRET not in text
    assert "INSERT INTO" not in text


async def test_notifier_sink_failure_is_logged_safely(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    await _one_retryable_response(None, _FailingNotifier())
    assert SECRET not in caplog.text
    assert "Notifier.on_governance_event failed" in caplog.text
    assert "provider.retry notifier broadcast failed" not in caplog.text


@pytest.mark.parametrize("database_failure", [False, True])
async def test_hook_failure_is_logged_and_swallowed(
    caplog: pytest.LogCaptureFixture,
    database_failure: bool,
) -> None:
    """The outer fail-open guard keeps its wording."""
    caplog.set_level(logging.WARNING, logger="dendrux")

    class Exploding:
        def get(self, name: str, default: Any = None) -> Any:
            if database_failure:
                raise _db_error()
            raise RuntimeError("header read failed")

    from dendrux.llm import _retry_telemetry as telemetry

    response = httpx.Response(503, request=httpx.Request("GET", "https://x.test/v1/a"))
    response.headers = Exploding()  # type: ignore[assignment]
    with telemetry_context(
        run_id="r", iteration=1, sinks=governance_sinks(_FailingRecorder(), None, "r")
    ):
        await telemetry._on_response(response, retryable_statuses=RETRYABLE_HTTP_STATUSES)
    assert SECRET not in _formatted(caplog, "retry telemetry hook raised")


async def test_direct_notifier_sink_failure_log_omits_the_db_statement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dendrux.llm import _retry_telemetry as telemetry

    caplog.set_level(logging.WARNING, logger="dendrux")

    async def fail(*args: Any) -> None:
        raise _db_error()

    response = httpx.Response(503, request=httpx.Request("GET", "https://x.test/v1/a"))
    with telemetry_context(run_id="r", iteration=1, sinks=TelemetrySinks(None, fail)):
        await telemetry._on_response(response, retryable_statuses=RETRYABLE_HTTP_STATUSES)
    text = _formatted(caplog, "provider.retry notifier broadcast failed")
    assert SECRET not in text
    assert "INSERT INTO" not in text
