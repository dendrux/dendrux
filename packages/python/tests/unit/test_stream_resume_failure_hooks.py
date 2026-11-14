"""Failure hooks preserve persistence order and never swallow cancellation."""

import asyncio
from types import SimpleNamespace

import pytest

from dendrux.loops import _helpers
from dendrux.runtime import _run_resume_loop, _run_start, runner
from dendrux.types import RunStatus


@pytest.mark.parametrize("failure_site", ["recorder", "notifier", "extra_notifier"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_stream_failure_hook_error_scope(monkeypatch, caplog, failure_site, cancel):
    calls = []
    store = object()
    recorder = object()
    notifier = object()
    extra_notifier = object()
    original_error = RuntimeError("resume failed")
    hook_error = asyncio.CancelledError() if cancel else RuntimeError("hook failed")
    ctx = (
        None
        if failure_site == "extra_notifier"
        else SimpleNamespace(recorder=recorder, notifier=notifier)
    )

    async def persist(actual_store, run_id, exc, *, safe_event):
        assert (actual_store, run_id, exc, safe_event) == (store, "run", original_error, True)
        calls.append("persist")

    async def record(actual_recorder, run_id, exc):
        assert (actual_recorder, run_id, exc) == (recorder, "run", original_error)
        calls.append("recorder")
        if failure_site == "recorder":
            raise hook_error

    async def notify(actual_notifier, run_id, exc):
        expected = extra_notifier if ctx is None else notifier
        assert (actual_notifier, run_id, exc) == (expected, "run", original_error)
        calls.append("extra_notifier" if ctx is None else "notifier")
        raise hook_error

    async def released_status(actual_store, run_id):
        assert (actual_store, run_id) == (store, "run")
        calls.append("released_status")
        return RunStatus.WAITING_APPROVAL

    monkeypatch.setattr(_run_start, "_persist_run_failure", persist)
    monkeypatch.setattr(_helpers, "record_run_failed", record)
    monkeypatch.setattr(_helpers, "notify_run_failed", notify)
    monkeypatch.setattr(_run_resume_loop, "_released_resume_status", released_status)
    shared = {"claimed": True, "released_prelude": store}
    pending = runner._finish_stream_resume_failure(
        shared, store, ctx, extra_notifier, "run", original_error
    )
    if cancel:
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        assert caught.value is hook_error
    else:
        assert await pending == RunStatus.WAITING_APPROVAL
        assert (
            "Lifecycle hooks failed during resume_stream error handler for run run" in caplog.text
        )
    expected_calls = ["persist"]
    if ctx is not None:
        expected_calls.append("recorder")
    if failure_site != "recorder":
        expected_calls.append(failure_site)
    if not cancel:
        expected_calls.append("released_status")
    assert calls == expected_calls
