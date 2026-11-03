"""Recovery marks dispatch before side effects and finalizes later failures."""

from unittest.mock import AsyncMock

import pytest

from dendrux.runtime import _reclaim, runner
from dendrux.types import RunEventType
from tests.integration.test_cross_path_invariants import paused_recovery


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_resume"])
async def test_recovery_work_is_marked_before_dispatch_and_never_released(
    store, engine, monkeypatch, entry
):
    agent, paused, effects = await paused_recovery(store, engine)
    before = await store.get_run(paused.run_id)
    release = AsyncMock(wraps=store.release_run_claim)
    resolve = _reclaim.resolve
    marked = []
    failure_state = []
    error = RuntimeError("after_recovery_dispatch")

    async def dispatch_then_fail(*args, **kwargs):
        on_work = kwargs.get("on_work")
        if on_work is not None:

            def mark():
                marked.append(list(effects))
                on_work()

            kwargs["on_work"] = mark
        result = await resolve(*args, **kwargs)
        assert result is None and effects == [True]
        raise error

    failed = runner._stream_resume_failure

    async def capture_failure(shared, *args):
        failure_state.append(shared.get("work_started", False))
        return await failed(shared, *args)

    monkeypatch.setattr(_reclaim, "resolve", dispatch_then_fail)
    monkeypatch.setattr(runner, "_stream_resume_failure", capture_failure)
    monkeypatch.setattr(store, "release_run_claim", release)
    scope = {"tenant_id": "tenant", "expected_pause_id": paused.pause_id}
    if entry == "resume_stream":
        async with agent.resume_stream(paused.run_id, **scope) as stream:
            events = [event async for event in stream]
        assert events[-1].type == RunEventType.RUN_ERROR
    else:
        with pytest.raises(RuntimeError) as caught:
            await getattr(agent, entry)(paused.run_id, **scope)
        assert caught.value is error
    release.assert_not_awaited()
    row = await store.get_run(paused.run_id)
    assert row.status == "error"
    assert row.claim_epoch == before.claim_epoch + 1
    assert effects == [True]
    assert agent._provider.calls == 0
    assert marked == [[]]
    assert failure_state == ([True] if entry == "resume_stream" else [])
    rows = await store.get_tool_calls(paused.run_id)
    assert len(rows) == 1 and rows[0].outcome == "completed"
