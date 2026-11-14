"""Characterize legacy-store and observer fallbacks preserved by the PR3 move."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent
from dendrux._meta_keys import STOP_REASON_META_KEY
from dendrux.runtime import runner
from dendrux.runtime.replay import ReplayRequest
from dendrux.runtime.state import StoreCapabilities
from dendrux.skills._session import SkillSession
from dendrux.types import Message, Role, RunResult, RunStatus, generate_ulid


def test_replay_defers_history_seeding_without_mutating_caller_history():
    history = [Message(role=Role.USER, content="prior turn")]
    request = ReplayRequest((generate_ulid(),), "all", False)
    session = SkillSession.prepare(Agent(prompt="prompt"))
    assert runner._new_run_history(request, history, None, "new turn", session, None) is None
    assert history == [Message(role=Role.USER, content="prior turn")]


def test_checkpointed_store_without_atomic_completion_gets_no_atomic_evidence():
    store = SimpleNamespace(storage_format=2, capabilities=StoreCapabilities())
    result = RunResult(run_id=generate_ulid(), status=RunStatus.SUCCESS, answer="done")
    assert runner._completion_evidence(store, result, None, "run.completed", {}) == {}


@pytest.mark.parametrize("missing", [False, True])
async def test_stop_reason_metadata_is_best_effort_for_legacy_stores(missing, caplog):
    update = AsyncMock(side_effect=RuntimeError("metadata write failed"))
    store = SimpleNamespace() if missing else SimpleNamespace(update_run_meta=update)
    run_id = generate_ulid()
    await runner._record_stop_reason(store, run_id, "max_tokens")
    if missing:
        update.assert_not_awaited()
    else:
        update.assert_awaited_once_with(run_id, {STOP_REASON_META_KEY: "max_tokens"})
        assert "Failed to record stop_reason" in caplog.text


async def test_stream_failure_hook_failure_remains_best_effort(caplog):
    recorder = SimpleNamespace(on_run_failed=AsyncMock(side_effect=RuntimeError("sink failed")))
    notifier = SimpleNamespace(on_run_failed=AsyncMock())
    failure = RuntimeError("run failed")
    await runner._stream_run_failure(None, {}, recorder, notifier, generate_ulid(), failure)
    recorder.on_run_failed.assert_awaited_once()
    assert recorder.on_run_failed.await_args.args[-1] is failure
    notifier.on_run_failed.assert_not_awaited()
    assert "Lifecycle hooks failed during stream error handler" in caplog.text
