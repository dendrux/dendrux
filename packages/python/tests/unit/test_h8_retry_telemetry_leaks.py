"""Sentinel leak scan of the retry-telemetry warnings (H8).

The hook logs a failed recorder write with ``exc_info=True``. For a database
failure that carries the statement (and so request data), ``exc_info=True``
puts the sentinel in the log text, where the loop helpers'
``safe_exception_info()`` would replace the error with a bare
``PersistenceError``. Pinned as today; listed as a named-fix candidate.
The notifier path goes through ``notify_governance``, which already logs with
``safe_exception_info()``: no sentinel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy.exc import OperationalError

from dendrux.errors import safe_exception_info
from dendrux.llm._retry_telemetry import (
    call_attempt_tracking,
    make_telemetry_http_client,
    telemetry_context,
)
from dendrux.loops.base import BaseNotifier, BaseRecorder

if TYPE_CHECKING:
    import pytest

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
    client = make_telemetry_http_client(httpx.Timeout(5.0))
    client._transport = httpx.MockTransport(lambda request: httpx.Response(503))  # type: ignore[attr-defined]
    try:
        with (
            telemetry_context(run_id="run-leak", iteration=1, recorder=recorder, notifier=notifier),
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


async def test_recorder_sink_failure_log_today_formats_the_db_statement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    await _one_retryable_response(_FailingRecorder(), None)
    text = _formatted(caplog, "provider.retry recorder write failed")
    # Today: exc_info=True formats the full DB error (named-fix candidate).
    assert SECRET in text
    # What safe_exception_info() would have logged instead: no sentinel.
    safe = safe_exception_info(_db_error())
    assert isinstance(safe, tuple)
    assert SECRET not in "".join(str(part) for part in safe[:2])


async def test_notifier_sink_failure_is_logged_safely(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux")
    await _one_retryable_response(None, _FailingNotifier())
    assert SECRET not in caplog.text
    assert "Notifier.on_governance_event failed" in caplog.text
    assert "provider.retry notifier broadcast failed" not in caplog.text


async def test_hook_failure_is_logged_and_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    """The outer fail-open guard keeps its wording."""
    caplog.set_level(logging.WARNING, logger="dendrux")

    class Exploding:
        def get(self, name: str, default: Any = None) -> Any:
            raise RuntimeError(f"header read {SECRET}")

    from dendrux.llm import _retry_telemetry as telemetry

    response = httpx.Response(503, request=httpx.Request("GET", "https://x.test/v1/a"))
    response.headers = Exploding()  # type: ignore[assignment]
    with telemetry_context(run_id="r", iteration=1, recorder=_FailingRecorder(), notifier=None):
        await telemetry._on_response(response)
    assert "retry telemetry hook raised" in caplog.text
