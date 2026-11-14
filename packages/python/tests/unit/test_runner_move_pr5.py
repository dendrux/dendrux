"""Characterize claimed continuation setup and durable replay before the PR5 move."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux.errors import RunFencedError
from dendrux.runtime import runner
from dendrux.types import Message, Role, RunEventType, RunStatus
from tests._helpers.runtime_paths import path_case
from tests._helpers.scripted_provider import ScriptedLLM
from tests.unit.test_runner_move_contract import store as store


@pytest.mark.parametrize("path", ["resume", "resume_stream"])
@pytest.mark.parametrize("site", ["_resume_session", "_read_resume_state", "_claim_resume"])
async def test_setup_fault_fires_before_provider_and_releases_claim(store, monkeypatch, path, site):
    if site == "_resume_session" and path == "resume_stream":
        site = "continuation"
    provider = ScriptedLLM([])
    case = await path_case(store, None, path, provider)
    before = await store.get_run(case.run_id)
    failure = ValueError("pr5_setup_fault")
    fault = AsyncMock(side_effect=failure)
    monkeypatch.setattr(runner, site, fault)
    try:
        if path == "resume_stream":
            await case.invoke()
            assert case.events[-1].type == RunEventType.RUN_ERROR
            assert case.events[-1].error == "pr5_setup_fault"
        else:
            with pytest.raises(ValueError) as caught:
                await case.invoke()
            assert caught.value is failure
        fault.assert_awaited_once()
        after = await store.get_run(case.run_id)
        assert after.status == before.status == RunStatus.WAITING_CLIENT_TOOL.value
        assert after.claim_epoch == before.claim_epoch
        assert case.agent._task_manager._owned == {}
    finally:
        await case.agent.close()


@pytest.mark.parametrize("consumes", [False, True])
async def test_replay_consumes_projected_history_before_any_user_effect(monkeypatch, consumes):
    old = Message(role=Role.USER, content="old")
    injected = Message(role=Role.USER, content="injected")
    clean = Message(role=Role.USER, content="clean")
    order = []
    recorder = SimpleNamespace(_loop_state={"submission": {"kind": "input"}})

    async def project(target, message):
        assert target is recorder and message is injected
        order.append("project")
        return clean

    async def consume(messages):
        assert messages == [clean]
        order.append("consume")
        if consumes:
            recorder._loop_state = {}

    async def replay(ctx, run_id, callback, blocked):
        assert ctx.history == [old, clean]
        order.append("effect")

    recorder.consume_resume = consume
    ctx = SimpleNamespace(
        recorder=recorder, pause_state=SimpleNamespace(history=[old]), history=[old, injected]
    )
    monkeypatch.setattr(runner, "project_message", project)
    monkeypatch.setattr(runner, "_replay_user_input", replay)
    monkeypatch.setattr(runner, "order_observations", lambda *args: order.append("order"))
    await runner._replay_resume_injected_history(
        ctx,
        "run",
        tool_results=None,
        user_input="injected",
        on_consumed=lambda: order.append("work"),
    )
    assert order == ["project", "consume", *(["work"] if consumes else []), "effect", "order"]


async def test_failed_consumption_does_not_mark_work_or_replay(monkeypatch):
    failure = RuntimeError("pr5_consume_fault")
    recorder = SimpleNamespace(
        _loop_state={"submission": {}}, consume_resume=AsyncMock(side_effect=failure)
    )
    ctx = SimpleNamespace(recorder=recorder, pause_state=SimpleNamespace(history=[]), history=[])
    replay = AsyncMock()
    work = Mock()
    monkeypatch.setattr(runner, "_replay_user_input", replay)
    with pytest.raises(RuntimeError) as caught:
        await runner._replay_resume_injected_history(
            ctx, "run", tool_results=None, user_input="injected", on_consumed=work
        )
    assert caught.value is failure
    recorder.consume_resume.assert_awaited_once_with([])
    work.assert_not_called()
    replay.assert_not_awaited()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fenced", [False, True])
async def test_setup_failure_preserves_exception_and_release_boundary(monkeypatch, enabled, fenced):
    failure = RunFencedError("run") if fenced else RuntimeError("pr5_setup_cleanup")
    release = AsyncMock()
    monkeypatch.setattr(runner, "_release_claim", release)
    store = object()
    with pytest.raises(type(failure)) as caught:
        async with runner._release_setup_failure(
            store, "run", "waiting_client_tool", enabled=enabled
        ):
            raise failure
    assert caught.value is failure
    if enabled and not fenced:
        release.assert_awaited_once_with(store, "run", "waiting_client_tool")
    else:
        release.assert_not_awaited()


@pytest.mark.parametrize("path", ["resume", "resume_stream"])
@pytest.mark.parametrize("site", ["submission_validator", "require_pause_id"])
async def test_validation_fault_fires_before_claim_in_both_resume_modes(
    store, monkeypatch, path, site
):
    case = await path_case(store, None, path, ScriptedLLM([]))
    before = await store.get_run(case.run_id)
    failure = ValueError("pr5_validation_fault")
    fault = Mock(side_effect=failure)
    monkeypatch.setattr(runner, site, fault)
    try:
        if path == "resume_stream":
            await case.invoke()
            assert case.events[-1].type == RunEventType.RUN_ERROR
            assert case.events[-1].error == "pr5_validation_fault"
        else:
            with pytest.raises(ValueError) as caught:
                await case.invoke()
            assert caught.value is failure
        fault.assert_called_once()
        after = await store.get_run(case.run_id)
        assert after.status == before.status
        assert after.claim_epoch == before.claim_epoch
    finally:
        await case.agent.close()
