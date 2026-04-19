"""Tests for the provider retry telemetry hook.

Drives an httpx mock transport that returns retryable failures
(503/429) followed by success, and asserts the hook emits
``provider.retry`` events through the recorder/notifier set in the
contextvars.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from dendrux.llm._retry_telemetry import (
    begin_call_attempt_tracking,
    call_attempt_tracking,
    end_call_attempt_tracking,
    make_telemetry_http_client,
    telemetry_context,
)


class _CapturingRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, dict[str, Any], str | None]] = []

    async def on_message_appended(self, *a: Any, **k: Any) -> None: ...
    async def on_llm_call_completed(self, *a: Any, **k: Any) -> None: ...
    async def on_tool_completed(self, *a: Any, **k: Any) -> None: ...

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.events.append((event_type, iteration, data, correlation_id))


class _CapturingNotifier(_CapturingRecorder):
    """Same shape; notifier and recorder protocols overlap here."""


def _seq_transport(*statuses: int) -> httpx.MockTransport:
    """Mock transport that returns one response per call, in order."""
    queue = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            return httpx.Response(200, json={"ok": True})
        status = queue.pop(0)
        headers = {"retry-after": "1"} if status == 429 else {}
        return httpx.Response(status, headers=headers, json={"err": status})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_hook_emits_retry_event_on_retryable_failure() -> None:
    recorder = _CapturingRecorder()
    notifier = _CapturingNotifier()

    client = make_telemetry_http_client(httpx.Timeout(10.0))
    client._transport = _seq_transport(503)  # type: ignore[attr-defined]

    with (
        telemetry_context(
            run_id="run-1",
            iteration=2,
            recorder=recorder,
            notifier=notifier,
        ),
        call_attempt_tracking(),
    ):
        await client.get("https://api.example.com/v1/messages")

    await client.aclose()

    assert len(recorder.events) == 1
    event_type, iteration, data, correlation_id = recorder.events[0]
    assert event_type == "provider.retry"
    assert iteration == 2
    assert data["status_code"] == 503
    assert data["attempt"] == 1
    assert data["endpoint"] == "/v1/messages"
    assert data["retry_after"] is None
    assert correlation_id is None

    # Notifier got the same event.
    assert notifier.events == recorder.events


@pytest.mark.asyncio
async def test_hook_counts_attempts_within_one_logical_call() -> None:
    recorder = _CapturingRecorder()

    client = make_telemetry_http_client(httpx.Timeout(10.0))
    client._transport = _seq_transport(503, 429, 200)  # type: ignore[attr-defined]

    with (
        telemetry_context(run_id="run-2", iteration=1, recorder=recorder, notifier=None),
        call_attempt_tracking(),
    ):
        # Three sequential calls within one tracking scope —
        # equivalent to SDK retry loop (each retry = one HTTP call).
        await client.get("https://api.example.com/v1/messages")
        await client.get("https://api.example.com/v1/messages")
        await client.get("https://api.example.com/v1/messages")

    await client.aclose()

    # Only the two failures fire events; the 200 is silent.
    assert len(recorder.events) == 2
    assert recorder.events[0][2]["attempt"] == 1
    assert recorder.events[0][2]["status_code"] == 503
    assert recorder.events[1][2]["attempt"] == 2
    assert recorder.events[1][2]["status_code"] == 429
    assert recorder.events[1][2]["retry_after"] == "1"


@pytest.mark.asyncio
async def test_attempt_counter_resets_between_calls() -> None:
    recorder = _CapturingRecorder()

    client = make_telemetry_http_client(httpx.Timeout(10.0))
    client._transport = _seq_transport(503, 503, 503, 503)  # type: ignore[attr-defined]

    with telemetry_context(run_id="run-3", iteration=1, recorder=recorder, notifier=None):
        # First logical call (2 retryable responses)
        with call_attempt_tracking():
            await client.get("https://api.example.com/v1/messages")
            await client.get("https://api.example.com/v1/messages")
        # Second logical call (2 more retryable responses) — counter resets
        with call_attempt_tracking():
            await client.get("https://api.example.com/v1/messages")
            await client.get("https://api.example.com/v1/messages")

    await client.aclose()

    attempts = [ev[2]["attempt"] for ev in recorder.events]
    assert attempts == [1, 2, 1, 2]


@pytest.mark.asyncio
async def test_non_retryable_status_codes_do_not_emit() -> None:
    recorder = _CapturingRecorder()

    client = make_telemetry_http_client(httpx.Timeout(10.0))
    # 401 is auth; 404 is wrong model — neither is a transient retryable.
    client._transport = _seq_transport(401, 404, 422)  # type: ignore[attr-defined]

    with (
        telemetry_context(run_id="run-4", iteration=1, recorder=recorder, notifier=None),
        call_attempt_tracking(),
    ):
        await client.get("https://api.example.com/v1/messages")
        await client.get("https://api.example.com/v1/messages")
        await client.get("https://api.example.com/v1/messages")

    await client.aclose()

    assert recorder.events == []


@pytest.mark.asyncio
async def test_hook_silent_when_no_telemetry_context() -> None:
    """Hook fires but skips emission — no recorder, no notifier."""
    client = make_telemetry_http_client(httpx.Timeout(10.0))
    client._transport = _seq_transport(503)  # type: ignore[attr-defined]

    # No context manager around the call — contextvars are at default
    # (None). The hook must not raise.
    response = await client.get("https://api.example.com/v1/messages")

    await client.aclose()
    assert response.status_code == 503  # call still completed normally


@pytest.mark.asyncio
async def test_recorder_failure_is_swallowed_not_propagated() -> None:
    """An exception in the recorder must NOT break the LLM call.

    Observability is fail-open. A buggy recorder cannot take down a
    production agent.
    """

    class _BrokenRecorder:
        async def on_governance_event(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("recorder is broken")

        async def on_message_appended(self, *a: Any, **k: Any) -> None: ...
        async def on_llm_call_completed(self, *a: Any, **k: Any) -> None: ...
        async def on_tool_completed(self, *a: Any, **k: Any) -> None: ...

    client = make_telemetry_http_client(httpx.Timeout(10.0))
    client._transport = _seq_transport(503)  # type: ignore[attr-defined]

    with (
        telemetry_context(run_id="run-5", iteration=1, recorder=_BrokenRecorder(), notifier=None),
        call_attempt_tracking(),
    ):
        response = await client.get("https://api.example.com/v1/messages")

    await client.aclose()
    # Hook swallowed the recorder exception — call completed.
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_token_based_call_tracking_resets_correctly() -> None:
    recorder = _CapturingRecorder()

    client = make_telemetry_http_client(httpx.Timeout(10.0))
    client._transport = _seq_transport(503, 503)  # type: ignore[attr-defined]

    with telemetry_context(run_id="run-6", iteration=1, recorder=recorder, notifier=None):
        token = begin_call_attempt_tracking()
        try:
            await client.get("https://api.example.com/v1/messages")
            await client.get("https://api.example.com/v1/messages")
        finally:
            end_call_attempt_tracking(token)

    await client.aclose()
    attempts = [ev[2]["attempt"] for ev in recorder.events]
    assert attempts == [1, 2]
