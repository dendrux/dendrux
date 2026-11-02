"""Postgres claim fencing and validation after a competing committed transaction."""

import pytest
from sqlalchemy import select, update

from dendrux import tool
from dendrux.db.models import AgentRun
from dendrux.errors import PauseStatusMismatchError
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import LLMResponse, RunStatus, generate_ulid
from tests._fixtures.storage import pin_storage_format
from tests._helpers.h0_races import Gate, connection_snapshot, running, snapshot, wait_for_lock
from tests._helpers.h0_races import peer as peer
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.runtime_paths import PATHS, SUBMIT_KINDS, path_case
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.parametrize("engine", ["postgres"], indirect=True)


class StaleProviderEffectError(AssertionError):
    pass


CONTINUATIONS = ["resume", "resume_stream", *SUBMIT_KINDS]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "cut",
    [
        pytest.param(
            "provider",
            marks=pytest.mark.xfail(
                strict=True,
                raises=StaleProviderEffectError,
                reason="a stale owner can start a provider call after its last control checkpoint",
            ),
        ),
        "provider_result",
        "tool",
    ],
)
async def test_new_claim_fences_old_owner_before_next_effect(
    store, engine, peer, monkeypatch, path, cut
):
    await pin_storage_format(engine, 2)
    effects = []

    @tool()
    async def effect() -> str:
        """Record an externally visible server effect."""
        effects.append(True)
        return "ok"

    provider = ScriptedLLM([tool_call("effect", pid="effect"), LLMResponse(text="done")])
    case = await path_case(store, engine, path, provider)
    # Public tool configuration is resolved on entry; no private provider replacement.
    case.agent.tools.append(effect)
    gate = Gate()
    gate.install(
        monkeypatch,
        PersistenceRecorder,
        {
            "provider": "on_llm_call_started",
            "provider_result": "on_llm_call_completed",
            "tool": "on_tool_started",
        }[cut],
        when=lambda recorder, run_id, *args, **kwargs: cut != "tool" or args[0].name == "effect",
    )
    other, other_engine = peer
    try:
        async with running(case.invoke(), gate) as task:
            await gate.entered.wait()
            run_id = gate.args[1]
            old = await other.get_run(run_id)
            item = await expire(other, other_engine, run_id)
            claim = await other.reclaim_run(
                run_id,
                tenant_id="tenant",
                agent_name=case.agent.name,
                expected_claim_epoch=item.claim_epoch,
                governance=governance_snapshot(case.agent),
                governance_policy="warn",
                skill_pins={},
                loop_type="react",
            )
            assert item.claim_epoch == old.claim_epoch + 1  # sweep fences the expired owner
            assert claim.claim_epoch == item.claim_epoch + 1
            before = await snapshot(other_engine)
            calls = provider.calls
            gate.release.set()
            result = await task
            assert effects == [], "the old owner started a server effect after reclaim"
            assert await snapshot(other_engine) == before, "old epoch wrote after the new claim"
            assert result.status == RunStatus.RUNNING
            if provider.calls != calls:
                raise StaleProviderEffectError("the old owner started the next provider call")
    finally:
        gate.release.set()
        await case.agent.close()


@pytest.mark.parametrize("path", CONTINUATIONS)
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_claim_revalidates_after_competing_generation_change(
    store, engine, peer, monkeypatch, path, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = ScriptedLLM([LLMResponse(text="done")])
    case = await path_case(store, engine, path, provider)
    before = await store.get_run(case.run_id)
    gate = Gate()
    method = (
        "claim_paused_run"
        if path == "submit_resume"
        or (storage_format == 1 and path == "approve")
        or (storage_format == 1 and path in {"resume", "resume_stream"})
        else "submit_and_claim"
    )
    gate.install(monkeypatch, store, method)
    _, other_engine = peer
    try:
        async with running(case.invoke(), gate) as task:
            await gate.entered.wait()  # all wrapper/preflight reads have already succeeded
            async with other_engine.connect() as conn:
                tx = await conn.begin()
                row = (
                    (
                        await conn.execute(
                            select(AgentRun.__table__)
                            .where(AgentRun.id == case.run_id)
                            .with_for_update()
                        )
                    )
                    .mappings()
                    .one()
                )
                pause = dict(row["pause_data"])
                new_id = generate_ulid()
                pause["pause_id"] = new_id
                await conn.execute(
                    update(AgentRun).where(AgentRun.id == case.run_id).values(pause_data=pause)
                )
                committed = await connection_snapshot(conn)
                gate.release.set()
                await wait_for_lock(conn, task)
                await tx.commit()
            if path == "resume_stream":
                result = await task
                assert result.status == RunStatus.ERROR
                assert result.meta["error_type"] == "PauseStatusMismatchError"
            else:
                with pytest.raises(PauseStatusMismatchError) as caught:
                    await task
                assert caught.value.reason == "pause_id"
                assert caught.value.current_pause_id == new_id
            row = await store.get_run(case.run_id)
            assert row.status == before.status and row.claim_epoch == before.claim_epoch
            assert await store.get_pause_state(case.run_id) == pause
            assert row.tenant_id == "tenant"
            assert provider.calls == 0
            assert await snapshot(other_engine) == committed
    finally:
        gate.release.set()
        await case.agent.close()
