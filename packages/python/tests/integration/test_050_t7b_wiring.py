"""Run wiring contracts, exercised against both durable backends."""

import asyncio
import datetime as dt
import json

import pytest
from pydantic import BaseModel
from sqlalchemy import event, update

from dendrux import Agent, sweep, tool
from dendrux.db.models import AgentRun
from dendrux.errors import ProviderTimeoutError
from dendrux.guardrails import PII, SecretDetection
from dendrux.loops.single import SingleCall
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.runner import _build_cached_result
from dendrux.store import RunStore
from dendrux.types import Budget, LLMResponse, RunStatus, ToolCall, ToolResult, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import agent, consume, result_for


class FailingProvider(ScriptedLLM):
    async def complete(self, *args, **kwargs):
        if self.calls == 0 and self._responses:
            return await super().complete(*args, **kwargs)
        raise ProviderTimeoutError(
            message="vendor private request",
            provider="fixture",
            model="scripted",
            timeout_reason="call_deadline",
            retryable=False,
        )


@pytest.mark.parametrize("entry", ["run", "stream", "retry", "resume", "resume_stream", "submit"])
async def test_provider_error_safe_across_result_row_event_and_read(store, entry):
    llm = FailingProvider(
        [tool_call("client_read", pid="a")]
        if entry in {"resume", "resume_stream", "submit"}
        else []
    )
    a = agent(store, llm)
    pause = await a.run("go") if llm._responses else None
    if entry == "retry":
        original = await agent(store, [LLMResponse(text="prior")]).run("go")
        action = a.retry(original.run_id)
    elif entry == "resume":
        action = a.resume(pause.run_id, tool_results=[result_for(pause)])
    elif entry == "submit":
        action = a.submit_tool_results(pause.run_id, [result_for(pause)])
    else:
        action = a.run("go")
    result = None
    if entry in {"stream", "resume_stream"}:
        if hasattr(action, "close"):
            action.close()
        result, events = await consume(
            a.stream("go")
            if entry == "stream"
            else a.resume_stream(pause.run_id, tool_results=[result_for(pause)])
        )
        assert events[-1].error == "fixture timeout (HTTP n/a, code n/a)"
        assert result.error_info is not None, "RUN_ERROR must carry ErrorInfo"
    else:
        with pytest.raises(ProviderTimeoutError):
            await action
    rid = result.run_id if result else pause.run_id if pause else (await store.list_runs())[0].id
    row = await store.get_run(rid)
    assert row.error == "fixture timeout (HTTP n/a, code n/a)", "row must use safe_message"
    info = row.meta["dendrux.error_info"]
    assert info["timeout_reason"] == "call_deadline"
    error = next(e for e in await store.get_run_events(rid) if e.event_type == "run.error")
    assert error.data["error_info"] == info
    cached = await _build_cached_result(store, rid)
    assert cached.error_info.to_dict() == info
    detail = await RunStore(store).get_run(rid)
    assert detail.error == row.error
    assert "vendor private" not in json.dumps(error.data)


@pytest.mark.parametrize("loop", [None, SingleCall()])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
async def test_guardrail_block_has_durable_reason_and_structured_evidence(store, loop, direction):
    secret = "api_key=sk_test_abcdefghijklmnopqrstuvwxyz123456"
    a = Agent(
        name="blocked",
        prompt="p",
        provider=ScriptedLLM([LLMResponse(text=secret)]),
        state_store=store,
        loop=loop,
        guardrails=[SecretDetection(action="block")],
        strict_tenancy=False,
    )
    result = await a.run(secret if direction == "incoming" else "go")
    assert result.status == RunStatus.ERROR
    row = await store.get_run(result.run_id)
    assert row.error == result.error, "block-ended run must persist its safe reason"
    assert row.failure_reason == "guardrail_blocked"
    payload = next(
        e.data
        for e in await store.get_run_events(result.run_id)
        if e.event_type == "guardrail.blocked"
    )
    assert payload["code"] == "guardrail_blocked"
    assert payload["kinds"] and sum(payload["counts"].values()) >= 1
    assert payload["location"] == ("message 1 (role=user)" if direction == "incoming" else "output")
    assert secret not in json.dumps(payload)
    if direction == "incoming":
        assert "message 1" in result.error


async def test_typed_output_survives_persisted_wrapper(store):
    class Answer(BaseModel):
        word: str

    from dataclasses import replace

    llm = ScriptedLLM([LLMResponse(text='{"word":"ok"}')])
    llm.capabilities = replace(llm.capabilities, supports_structured_output=True)
    a = Agent(
        name="typed",
        prompt="p",
        provider=llm,
        state_store=store,
        loop=SingleCall(),
        strict_tenancy=False,
    )
    result = await a.run("go", output_type=Answer)
    assert result.output == Answer(word="ok"), "persisted wrapper must retain output"


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit"])
async def test_pause_only_retains_safe_provider_options_and_restores_them(store, entry):
    class Options(ScriptedLLM):
        async def complete(self, *args, **kwargs):
            self.options = getattr(self, "options", []) + [kwargs]
            return await super().complete(*args, **kwargs)

    llm = Options([tool_call("client_read", pid="a"), LLMResponse(text="done")])
    a = agent(store, llm)
    safe = dict(model="override", max_tokens=41, temperature=0.2, thinking=True, effort="high")
    pause = await a.run(
        "go", **safe, extra_headers={"x-private": "hidden"}, extra_body={"private": "hidden"}
    )
    raw = await store.get_pause_state(pause.run_id)
    assert raw.get("provider_kwargs") == safe, "pause must persist only allowlisted options"
    assert "hidden" not in json.dumps(raw)
    results = [result_for(pause)]
    if entry == "submit":
        await a.submit_tool_results(pause.run_id, results)
    elif entry == "resume":
        await a.resume(pause.run_id, tool_results=results)
    else:
        await consume(a.resume_stream(pause.run_id, tool_results=results))
    assert {k: llm.options[-1].get(k) for k in safe} == safe
    assert "extra_headers" not in llm.options[-1] and "extra_body" not in llm.options[-1]


@pytest.mark.parametrize("claim", ["plain", "submit"])
async def test_resume_claim_refreshes_progress_before_provider_starts(store, claim):
    pause = await agent(store, [tool_call("client_read", pid="a")]).run("go")
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == pause.run_id).values(last_progress_at=old)
        )
        await session.commit()
    if claim == "plain":
        assert await store.claim_paused_run(pause.run_id, expected_status="waiting_client_tool")
    else:
        assert await store.submit_and_claim(
            pause.run_id, expected_status="waiting_client_tool", submitted_data={}
        )
    assert not await store.sweep_stale_runs(dt.timedelta(hours=1)), "new claim must not look stale"


async def test_public_sweep_zero_threshold(store):
    await store.create_run("zero", agent_name="a", input_data={})
    result = await sweep(state_store=store, stale_running=dt.timedelta(0))
    assert [r.run_id for r in result.stale_running] == ["zero"], "zero is a valid sweep threshold"


async def test_sweep_projects_and_bounds_candidates(store, engine):
    for n in range(7):
        await store.create_run(f"batch-{n}", agent_name="a", input_data={"large": "x" * 10000})
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT") and "agent_runs" in statement:
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        results = await asyncio.gather(*[store.sweep_stale_runs(dt.timedelta(0)) for _ in range(3)])
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert sum(map(len, results)) == 7
    candidates = [s for s in statements if "last_progress_at" in s]
    assert candidates and all("LIMIT" in s for s in candidates), "candidate reads must be bounded"
    assert all("pause_data" not in s and "input_data" not in s for s in candidates)


async def test_budget_enforcement_survives_lost_forensics_and_retries_usage(store, monkeypatch):
    @tool()
    async def work() -> str:
        return "ok"

    usage_attempts = 0
    original = store.save_usage

    async def flaky(*args, **kwargs):
        nonlocal usage_attempts
        usage_attempts += 1
        if usage_attempts == 1:
            from sqlalchemy.exc import OperationalError

            raise OperationalError("fixture", {}, Exception("database is locked"))
        return await original(*args, **kwargs)

    async def lost(*args, **kwargs):
        raise RuntimeError("forensics unavailable")

    monkeypatch.setattr(store, "save_llm_interaction", lost)
    monkeypatch.setattr(store, "save_usage", flaky)
    responses = [tool_call("work", pid=str(i)) for i in range(3)]
    for r in responses:
        r.usage = UsageStats(input_tokens=300, total_tokens=300)
    llm = ScriptedLLM(responses + [LLMResponse(text="should not run")])
    a = Agent(
        name="budget",
        prompt="p",
        tools=[work],
        provider=llm,
        state_store=store,
        budget=Budget(max_tokens=500, count_cache_tokens=True, on_exceed="cancel"),
        strict_tenancy=False,
    )
    result = await a.run("go")
    assert result.status == RunStatus.CANCELLED and llm.calls == 2, (
        "local spend must enforce despite lost DB evidence"
    )
    assert usage_attempts == 3, "failed usage write must be retried"


async def test_retry_removes_unpaired_calls(store, monkeypatch):
    @tool()
    async def work() -> str:
        return "ok"

    original = store.save_tool_call

    async def lost(*args, **kwargs):
        raise RuntimeError("fixture disk failure")

    monkeypatch.setattr(store, "save_tool_call", lost)
    a = Agent(
        name="retry",
        prompt="p",
        tools=[work],
        provider=ScriptedLLM([tool_call("work", pid="a")]),
        state_store=store,
        strict_tenancy=False,
    )
    with pytest.raises(RuntimeError):
        await a.run("go")
    rid = (await store.list_runs())[0].id
    monkeypatch.setattr(store, "save_tool_call", original)
    llm = ScriptedLLM([LLMResponse(text="done")])
    a = Agent(
        name="retry",
        prompt="p",
        tools=[work],
        provider=llm,
        state_store=store,
        strict_tenancy=False,
    )
    await a.retry(rid)
    assert not any(m.tool_calls for m in llm.messages[-1]), (
        "retry must pair persisted calls with results"
    )


async def test_tool_completed_projects_safe_codes_and_zombie_count(store):
    await store.create_run("codes", agent_name="a", input_data={})
    recorder = PersistenceRecorder(store, "codes")
    tc = ToolCall(name="work", params={})
    tr = ToolResult(
        name="work",
        call_id=tc.id,
        payload='"timeout"',
        success=False,
        meta={"code": "timeout_outcome_unknown", "zombie_threads": 3, "private": "hidden"},
    )
    await recorder.on_tool_completed("codes", tc, tr, 1)
    data = (await store.get_run_events("codes"))[-1].data
    assert data.get("code") == "timeout_outcome_unknown" and data.get("zombie_threads") == 3
    assert "private" not in data


async def test_public_pii_mapping_hides_internal_origins_but_storage_keeps_them(store):
    a = agent(store, [tool_call("client_read", pid="a")], guardrails=[PII(restore="allowlisted")])
    result = await a.run("email alice@example.com")
    assert "__dendrux_origins__" not in result.meta["pii_mapping"], (
        "internal provenance is not a public mapping entry"
    )
    assert "__dendrux_origins__" in await store.get_pii_mapping(result.run_id)


async def test_in_process_cancel_already_returns_cancelled(store):
    llm = ScriptedLLM([LLMResponse(text="unused")], gate_at=0)
    a = agent(store, llm)
    task = asyncio.create_task(a.run("go"))
    await llm.entered.wait()
    rid = next(iter(a._task_manager._owned))
    await a.cancel_run(rid)
    result = await task
    assert result.status == RunStatus.CANCELLED


async def test_failed_usage_write_is_retried(store, monkeypatch):
    attempts = 0
    original = store.save_usage

    async def flaky(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            from sqlalchemy.exc import OperationalError

            raise OperationalError("fixture", {}, Exception("database is locked"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "save_usage", flaky)
    await agent(store, [LLMResponse(text="done")]).run("go")
    assert attempts == 2, "retry transient usage writes"


async def test_failed_interaction_write_is_retried(store, monkeypatch):
    attempts = 0
    original = store.save_llm_interaction

    async def flaky(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            from sqlalchemy.exc import OperationalError

            raise OperationalError("fixture", {}, Exception("database is locked"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "save_llm_interaction", flaky)
    await agent(store, [LLMResponse(text="done")]).run("go")
    assert attempts == 2, "retry transient interaction usage writes too"


@pytest.mark.parametrize("waiting", [False, True])
async def test_sweep_update_rechecks_staleness(store, monkeypatch, waiting):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.dml import Update

    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
    await store.create_run("refreshed", agent_name="a", input_data={})
    if waiting:
        await store.pause_run("refreshed", status="paused", pause_data={})
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "refreshed")
            .values(last_progress_at=old, updated_at=old, created_at=old)
        )
        await session.commit()
    original = AsyncSession.execute
    refreshed = False

    async def execute(session, statement, *args, **kwargs):
        nonlocal refreshed
        if isinstance(statement, Update) and "failure_reason" in str(statement) and not refreshed:
            refreshed = True
            await original(
                session,
                update(AgentRun)
                .where(AgentRun.id == "refreshed")
                .values(
                    last_progress_at=dt.datetime.now(dt.UTC), updated_at=dt.datetime.now(dt.UTC)
                ),
            )
        return await original(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", execute)
    swept = await (
        store.sweep_abandoned_runs(dt.timedelta(hours=1))
        if waiting
        else store.sweep_stale_runs(dt.timedelta(hours=1))
    )
    assert refreshed and not swept, "sweep CAS must re-check the timestamp it selected"


async def test_public_sweep_has_overridable_page_and_work_limits(store):
    for n in range(5):
        await store.create_run(f"limit-{n}", agent_name="a", input_data={})
    result = await sweep(state_store=store, stale_running=dt.timedelta(0), batch_size=1, max_runs=2)
    assert len(result.stale_running) == 2
    assert len((await sweep(state_store=store, stale_running=dt.timedelta(0))).stale_running) == 3


@pytest.mark.parametrize("entry", ["resume", "submit"])
async def test_resumed_in_process_cancel_returns_result(store, entry):
    llm = ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="unused")], gate_at=1)
    a = agent(store, llm)
    pause = await a.run("go")
    action = (
        a.resume(pause.run_id, tool_results=[result_for(pause)])
        if entry == "resume"
        else a.submit_tool_results(pause.run_id, [result_for(pause)])
    )
    task = asyncio.create_task(action)
    await asyncio.wait_for(llm.entered.wait(), 5)
    await a.cancel_run(pause.run_id)
    # A direct resume was not tracked in 7a: release its provider gate as well.
    llm.gate.set()
    result = await task
    assert result.status == RunStatus.CANCELLED and result.meta.get("interrupted")


async def test_failed_call_persists_timeout_evidence(store):
    recorder = PersistenceRecorder(store, "failed")
    await store.create_run("failed", agent_name="a", input_data={})
    exc = ProviderTimeoutError(provider="fixture", timeout_reason="stream_idle")
    await recorder.on_llm_call_failed("failed", 1, exc, duration_ms=12)
    events = await store.get_run_events("failed")
    assert len(events) == 1, "failed provider calls need durable timeout evidence"
    assert events[0].event_type == "llm.failed"
    assert events[0].data["timeout_reason"] == "stream_idle"
    assert events[0].data["error_info"]["safe_message"] == exc.safe_message


async def test_lost_finalize_cannot_return_proposed_output(store):
    from dendrux.runtime.runner import _persist_loop_outcome
    from dendrux.types import RunResult

    await store.create_run("lost", agent_name="a", input_data={})
    await store.finalize_run("lost", status="cancelled", expected_current_status="running")
    proposal = RunResult(
        run_id="lost", status=RunStatus.SUCCESS, answer="stale", output={"stale": True}
    )
    result = await _persist_loop_outcome(state_store=store, run_id="lost", result=proposal)
    assert result.status == RunStatus.CANCELLED
    assert result.output is None, "a lost CAS cannot return the loser's typed output"


@pytest.mark.parametrize("waiting", [False, True])
async def test_postgres_concurrent_sweeps_cannot_revoke_refreshed_run(
    store, engine, monkeypatch, waiting
):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    if engine.dialect.name != "postgresql":
        pytest.skip("concurrent independent transactions require Postgres")
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
    await store.create_run("live", agent_name="a", input_data={})
    if waiting:
        await store.pause_run("live", status="paused", pause_data={})
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "live")
            .values(created_at=old, updated_at=old, last_progress_at=old)
        )
        await session.commit()
    selected, refreshed = asyncio.Event(), asyncio.Event()
    count = 0
    original = AsyncSession.execute

    async def execute(session, statement, *args, **kwargs):
        nonlocal count
        result = await original(session, statement, *args, **kwargs)
        if (
            isinstance(statement, Select)
            and "started" in statement.selected_columns
            and "agent_runs" in str(statement)
        ):
            count += 1
            if count == 8:
                selected.set()
            await refreshed.wait()
        return result

    monkeypatch.setattr(AsyncSession, "execute", execute)
    method = store.sweep_abandoned_runs if waiting else store.sweep_stale_runs
    tasks = [asyncio.create_task(method(dt.timedelta(hours=1))) for _ in range(8)]
    try:
        await asyncio.wait_for(selected.wait(), 10)
        await store.touch_progress("live")
    finally:
        refreshed.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks), 10)
    assert not any(outcomes)
    assert (await store.get_run("live")).status == ("paused" if waiting else "running")


async def test_concurrent_error_finalize_keeps_winners_info_atomic(store):
    await store.create_run("atomic-info", agent_name="a", input_data={})

    async def finish(index):
        return await store.finalize_run(
            "atomic-info",
            status="error",
            error=f"safe-{index}",
            error_info={"category": str(index)},
            expected_current_status="running",
        )

    outcomes = await asyncio.gather(*(finish(i) for i in range(8)))
    assert sum(outcomes) == 1
    winner = outcomes.index(True)
    row = await store.get_run("atomic-info")
    assert row.error == f"safe-{winner}" and row.meta["dendrux.error_info"] == {
        "category": str(winner)
    }


async def test_oversized_mapping_resume_warning_has_run_id(store, monkeypatch, caplog):
    from dendrux.guardrails import _engine

    monkeypatch.setattr(_engine, "MAX_MAPPING_ENTRIES", 2)
    a = agent(
        store,
        [tool_call("client_read", pid="a"), LLMResponse(text="done")],
        guardrails=[PII(restore="allowlisted")],
    )
    pause = await a.run("go")
    mapping = {f"<<EMAIL_ADDRESS_{i}>>": f"person{i}@example.com" for i in range(3)}
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == pause.run_id).values(pii_mapping=mapping)
        )
        await session.commit()
    await a.resume(pause.run_id, tool_results=[result_for(pause)])
    lines = [r.message for r in caplog.records if "mapping exceeds cap" in r.message]
    assert len(lines) == 1 and f"run_id={pause.run_id}" in lines[0], (
        "resume warning needs its run id"
    )
