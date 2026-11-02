"""Postgres claim fencing and validation after a competing committed transaction."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from sqlalchemy import select, update

from dendrux import tool
from dendrux.db.models import AgentRun
from dendrux.errors import (
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
)
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import LLMResponse, RunStatus, generate_ulid
from tests._fixtures.storage import pin_storage_format
from tests._helpers.h0_races import (
    Gate,
    connection_snapshot,
    independent_callers,
    running,
    snapshot,
    wait_for_lock,
)
from tests._helpers.h0_races import peer as peer
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.runtime_paths import PATHS, SUBMIT_KINDS, path_case
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.parametrize("engine", ["postgres"], indirect=True)


class StaleProviderEffectError(AssertionError):
    pass


CONTINUATIONS = ["resume", "resume_stream", *SUBMIT_KINDS]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("cut", ["provider", "provider_result", "tool"])
async def test_new_claim_fences_old_owner_before_next_effect(
    store, engine, peer, monkeypatch, caplog, path, cut
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
            assert any(
                getattr(record, "code", None) == "run_fenced"
                and getattr(record, "run_id", None) == run_id
                for record in caplog.records
            )
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


async def test_eight_claimants_have_one_winner(store, engine, monkeypatch):
    await pin_storage_format(engine, 2)
    seed = await path_case(store, engine, "submit_resume", ScriptedLLM([]))
    run_id = seed.run_id
    pause = await store.get_pause_state(run_id)
    original = await store.get_run(run_id)
    providers = [ScriptedLLM([LLMResponse(text="done")]) for _ in range(8)]
    try:
        async with independent_callers(engine, 8) as callers, AsyncExitStack() as stack:
            tasks = []
            gates = []
            for (caller, _), provider in zip(callers, providers, strict=True):
                agent = type(seed.agent)(
                    name=seed.agent.name,
                    prompt=seed.agent.prompt,
                    tools=seed.agent.tools,
                    provider=provider,
                    state_store=caller,
                    strict_tenancy=False,
                )
                stack.push_async_callback(agent.close)
                gate = Gate()
                gate.install(monkeypatch, caller, "claim_paused_run")
                gates.append(gate)
                tasks.append(
                    await stack.enter_async_context(
                        running(
                            agent.submit_resume(
                                run_id, tenant_id="tenant", expected_pause_id=pause["pause_id"]
                            ),
                            gate,
                        )
                    )
                )
            await asyncio.gather(*(gate.entered.wait() for gate in gates))
            before = await snapshot(engine)
            async with engine.connect() as conn:
                tx = await conn.begin()
                await conn.execute(
                    select(AgentRun.id).where(AgentRun.id == run_id).with_for_update()
                )
                for gate in gates:
                    gate.release.set()
                await wait_for_lock(conn, tasks[0], count=8)
                await tx.commit()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            winners = [i for i, result in enumerate(results) if not isinstance(result, Exception)]
            assert len(winners) == 1
            assert results[winners[0]].status == RunStatus.SUCCESS
            for i, result in enumerate(results):
                if i != winners[0]:
                    assert isinstance(result, (RunAlreadyClaimedError, RunAlreadyTerminalError))
                    assert providers[i].calls == 0
            assert providers[winners[0]].calls == 1
            final = await store.get_run(run_id)
            assert final.claim_epoch == original.claim_epoch + 1
            assert final.status == RunStatus.SUCCESS.value
            assert final.tenant_id == original.tenant_id
            assert final.meta == original.meta
            after = await snapshot(engine)
            # Every prior row survives; every added child belongs to the one run.
            for table, rows in after.items():
                assert {row._mapping["id"] for row in before[table] if "id" in row._mapping} <= {
                    row._mapping["id"] for row in rows if "id" in row._mapping
                }
                for row in rows:
                    for key in ("run_id", "agent_run_id"):
                        if key in row._mapping:
                            assert row._mapping[key] == run_id
            events = await store.get_run_events(run_id)
            assert sum(event.event_type == "run.completed" for event in events) == 1
            assert sum(event.event_type == "run.resumed" for event in events) == 1
    finally:
        await seed.agent.close()
