"""Post-preflight losses retain their typed errors on each SQL backend."""

import pytest
from sqlalchemy import delete, update

from dendrux.db.models import AgentRun
from dendrux.errors import (
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.runtime import submit
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.runtime_paths import path_case
from tests._helpers.scripted_provider import ScriptedLLM


@pytest.mark.parametrize(
    "status,error,tail",
    [
        (None, RunNotFoundError, "not found."),
        ("success", RunAlreadyTerminalError, "is already terminal (success); cannot resume."),
        ("error", RunAlreadyTerminalError, "is already terminal (error); cannot resume."),
        ("cancelled", RunAlreadyTerminalError, "is already terminal (cancelled); cannot resume."),
        (
            "running",
            RunAlreadyClaimedError,
            "was claimed by another submit. Poll for completion instead.",
        ),
        (
            "waiting_approval",
            PauseStatusMismatchError,
            "is waiting_approval, but this method requires waiting_client_tool.",
        ),
        (
            "waiting_client_tool",
            PauseStatusMismatchError,
            "is waiting_client_tool, but this method requires waiting_client_tool.",
        ),
        ("pending", RunNotPausedError, "is not paused (current status: pending)."),
    ],
)
async def test_post_cas_loss_mapping(store, engine, monkeypatch, status, error, tail):
    case = await path_case(store, engine, "tool_results", ScriptedLLM([LLMResponse(text="done")]))

    async def lose(run_id, **kwargs):
        async with engine.begin() as conn:
            if status is None:
                await conn.execute(delete(AgentRun).where(AgentRun.id == run_id))
            else:
                await conn.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(status=status)
                )
        return False

    monkeypatch.setattr(store, "submit_and_claim", lose)
    try:
        with pytest.raises(error) as caught:
            await case.invoke()
        assert type(caught.value) is error
        assert str(caught.value) == f"Run '{case.run_id}' {tail}"
        if error is PauseStatusMismatchError:
            assert caught.value.current_status == RunStatus(status)
            assert caught.value.expected_status == RunStatus.WAITING_CLIENT_TOOL
            assert caught.value.reason == "status"
    finally:
        await case.agent.close()


@pytest.mark.parametrize("reason", ["abandoned_waiting", "stale_running", "never_started"])
async def test_swept_submit_keeps_terminal_error(store, engine, reason):
    case = await path_case(store, engine, "tool_results", ScriptedLLM([]))
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == case.run_id)
            .values(status="error", failure_reason=reason)
        )
    try:
        with pytest.raises(RunAlreadyTerminalError) as caught:
            await case.invoke()
        assert (
            str(caught.value) == f"Run '{case.run_id}' is already terminal (error); cannot resume."
        )
    finally:
        await case.agent.close()


@pytest.mark.parametrize("missing", [False, True])
async def test_legacy_post_claim_failure_restores_actual_status(store, monkeypatch, missing):
    rid = "00000000000000000000000102"
    await store.create_run(rid, "legacy")
    await store.pause_run(rid, status="paused", pause_data={"pause_id": "observed"})
    original = store.claim_paused_run
    read = store.get_pause_state
    claimed = False

    async def legacy(run_id, *, expected_status):
        nonlocal claimed
        result = await original(run_id, expected_status=expected_status)
        claimed = True
        return result

    async def changed(run_id):
        if claimed:
            return None if missing else {"pause_id": "different"}
        return await read(run_id)

    monkeypatch.setattr(store, "claim_paused_run", legacy)
    monkeypatch.setattr(store, "get_pause_state", changed)
    submit._claim_validation_support.cache_clear()
    with pytest.raises(ValueError if missing else PauseStatusMismatchError) as caught:
        await submit.claim_or_raise(store, rid, expected_status=RunStatus.PAUSED)
    if missing:
        assert str(caught.value) == "Claimed run has no pause state"
    assert (await store.get_run(rid)).status == "paused"
