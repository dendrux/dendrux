"""Every supported format-1 continuation checks ownership at provider start."""

import pytest

from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import LLMResponse, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.h0_races import Gate, running, snapshot
from tests._helpers.runtime_paths import PATHS, path_case
from tests._helpers.scripted_provider import ScriptedLLM


@pytest.mark.parametrize("path", [p for p in PATHS if p != "recovery"])
async def test_legacy_stale_owner_never_dispatches(store, engine, monkeypatch, caplog, path):
    await pin_storage_format(engine, 1)
    provider = ScriptedLLM([LLMResponse(text="must not dispatch")])
    case = await path_case(store, engine, path, provider)
    gate = Gate()
    gate.install(monkeypatch, PersistenceRecorder, "on_llm_call_started")
    gated = PersistenceRecorder.on_llm_call_started

    async def with_recent_lease(self, *args, **kwargs):
        # A recent real renewal must not substitute for the dispatch ownership read.
        async with self.lease_scope():
            pass
        return await gated(self, *args, **kwargs)

    monkeypatch.setattr(PersistenceRecorder, "on_llm_call_started", with_recent_lease)
    try:
        async with running(case.invoke(), gate) as task:
            await gate.entered.wait()
            run_id = gate.args[1]
            current = await store.get_run_control(run_id)
            assert await store.try_pause_run(
                run_id,
                status="paused",
                pause_data={"pause_id": "next"},
                claim_epoch=current.claim_epoch,
            )
            winner = await store.claim_paused_run(run_id, expected_status="paused")
            assert winner.claim_epoch == current.claim_epoch + 1
            before = await snapshot(engine)
            gate.release.set()
            result = await task
            assert provider.calls == 0
            assert result.status == RunStatus.RUNNING
            assert await snapshot(engine) == before
            assert any(
                getattr(r, "code", None) == "run_fenced" and r.run_id == run_id
                for r in caplog.records
            )
    finally:
        gate.release.set()
        await case.agent.close()


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("streaming", [False, True])
async def test_each_provider_start_costs_one_ownership_read(
    store, engine, monkeypatch, storage_format, streaming
):
    from sqlalchemy import event

    from dendrux import tool
    from tests._helpers.runtime_paths import make_agent
    from tests._helpers.scripted_provider import tool_call

    await pin_storage_format(engine, storage_format)

    @tool()
    async def work() -> str:
        """Complete a step."""
        return "ok"

    hook = PersistenceRecorder.on_llm_call_started
    statements = []
    per_call = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    async def started(self, *args, **kwargs):
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await hook(self, *args, **kwargs)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        per_call.append(list(statements))
        statements.clear()

    monkeypatch.setattr(PersistenceRecorder, "on_llm_call_started", started)
    agent = make_agent(
        store, [tool_call("work", pid="first"), LLMResponse(text="done")], tools=[work]
    )
    if streaming:
        async with agent.stream("go") as stream:
            async for _ in stream:
                pass
        result = stream.result
    else:
        result = await agent.run("go")
    assert result.status == RunStatus.SUCCESS
    assert len(per_call) == 2
    # Format 1 adds exactly one indexed SELECT per provider call; format 2 is unchanged.
    assert [len(sql) for sql in per_call] == [1, 1]
    assert all("SELECT" in sql[0] and "agent_runs.id =" in sql[0] for sql in per_call)
