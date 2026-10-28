"""Revert-sensitive runner fencing, lifecycle and client-result compatibility."""

import asyncio
import datetime as dt
import json
import warnings
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, ApprovalDecision, tool
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import (
    agent,
    client_read,
    consume,
    epoch_of,
    result_for,
)


@pytest.mark.parametrize("entry", ["run", "stream", "resume", "resume_stream"])
async def test_runner_finalize_cannot_overwrite_running_newer_claim(store, entry):
    resumed = entry.startswith("resume")
    llm = ScriptedLLM(
        ([tool_call("client_read", pid="a")] if resumed else [])
        + [LLMResponse(text="stale answer")],
        gate_at=1 if resumed else 0,
    )
    a = agent(store, llm)
    if resumed:
        paused = await a.run("go")
        rid = paused.run_id
        action = (
            a.resume(rid, tool_results=[result_for(paused)])
            if entry == "resume"
            else consume(a.resume_stream(rid, tool_results=[result_for(paused)]))
        )
    else:
        action = a.run("go") if entry == "run" else consume(a.stream("go"))
    old = asyncio.create_task(action)
    try:
        await asyncio.wait_for(llm.entered.wait(), 5)
        if not resumed:
            rid = (await store.list_runs())[0].id
        epoch = await epoch_of(store, rid)
        assert await store.release_run_claim(rid, status="paused", claim_epoch=epoch)
        newer = (await store.claim_paused_run(rid, expected_status="paused")).claim_epoch
        assert newer == epoch + 1
    finally:
        llm.gate.set()
        await asyncio.wait_for(old, 5)
    record = await store.get_run(rid)
    assert record.status == "running" and record.answer is None
    assert await epoch_of(store, rid) == newer
    assert not any(
        e.event_type in {"run.completed", "run.error", "run.cancelled"}
        for e in await store.get_run_events(rid)
    )
    assert await store.finalize_run(
        rid,
        status="success",
        answer="new owner",
        expected_current_status="running",
        claim_epoch=newer,
    )


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
@pytest.mark.parametrize(
    ("payload", "value"), [("Revenue: $100M", "Revenue: $100M"), ('{"rows": 1}', {"rows": 1})]
)
async def test_plain_text_and_json_client_results_are_canonical_before_transform(
    store, entry, payload, value
):
    seen = []

    def transform(ctx):
        seen.append(ctx.result)
        return {"received": ctx.result}

    llm = ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="done")])
    a = agent(store, llm, tool_result_transform=transform)
    paused = await a.run("go")
    results = [result_for(paused, payload)]
    if entry == "submit":
        result = await a.submit_tool_results(paused.run_id, results)
    elif entry == "resume":
        result = await a.resume(
            paused.run_id, tool_results=results, expected_pause_id=paused.pause_id
        )
    else:
        result, _ = await consume(
            a.resume_stream(paused.run_id, tool_results=results, expected_pause_id=paused.pause_id)
        )
    assert result.status == RunStatus.SUCCESS and seen == [value]
    message = next(m for m in llm.messages[-1] if m.role.value == "tool")
    assert json.loads(message.content) == {"received": value}
    rows = await store.get_tool_calls(paused.run_id)
    assert rows[-1].result == {"received": value}


class Lifecycle(BaseNotifier):
    def __init__(self):
        self.started, self.finished, self.failed = [], [], []

    async def on_run_started(self, run_id, **kwargs):
        self.started.append(run_id)

    async def on_run_finished(self, run_id, result):
        self.finished.append((run_id, result.status))

    async def on_run_failed(self, run_id, error, **kwargs):
        self.failed.append(run_id)


@pytest.mark.parametrize("entry", ["run", "resume", "retry"])
async def test_fenced_batch_lifecycle_closes_both_sinks(store, monkeypatch, entry):
    @tool()
    async def never() -> str:
        raise AssertionError("fenced worker executed a tool")

    recorder_start, recorder_finish = AsyncMock(), AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_started", recorder_start)
    monkeypatch.setattr(PersistenceRecorder, "on_run_finished", recorder_finish)
    resumed = entry == "resume"
    provider = ScriptedLLM(
        ([tool_call("client_read", pid="client")] if resumed else [])
        + [tool_call("never", pid="n")],
        gate_at=1 if resumed else 0,
    )
    a = Agent(
        name="life",
        prompt="p",
        tools=[client_read, never],
        state_store=store,
        provider=provider,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    if entry == "resume":
        pause = await a.run("go")
    elif entry == "retry":
        source = Agent(
            name="life",
            prompt="p",
            tools=[client_read, never],
            state_store=store,
            provider=ScriptedLLM([LLMResponse(text="original")]),
            strict_tenancy=False,
        )
        prior = await source.run("go")
    recorder_start.reset_mock()
    recorder_finish.reset_mock()
    notifier = Lifecycle()
    action = (
        a.resume(
            pause.run_id,
            tool_results=[result_for(pause)],
            notifier=notifier,
            expected_pause_id=pause.pause_id,
        )
        if resumed
        else a.retry(prior.run_id, notifier=notifier)
        if entry == "retry"
        else a.run("go", notifier=notifier)
    )
    task = asyncio.create_task(action)
    try:
        await asyncio.wait_for(provider.entered.wait(), 5)
        assert await store.sweep_stale_runs(dt.timedelta(0))
    finally:
        provider.gate.set()
        result = await asyncio.wait_for(task, 5)
    assert result.status == RunStatus.ERROR
    assert notifier.started == [result.run_id]
    assert notifier.finished == [(result.run_id, RunStatus.ERROR)]
    assert not notifier.failed
    assert recorder_start.await_count == recorder_finish.await_count == 1


@tool()
async def server() -> str:
    return "ok"


@pytest.mark.parametrize("entry", ["batch", "decisions", "resume", "resume_stream"])
@pytest.mark.parametrize("legacy", [False, True])
async def test_approval_without_token_is_rejected_for_current_and_legacy_pauses(
    store, entry, legacy
):
    a = Agent(
        name="warning",
        prompt="p",
        tools=[server],
        require_approval=["server"],
        state_store=store,
        provider=ScriptedLLM([tool_call("server", pid="a"), LLMResponse(text="done")]),
        strict_tenancy=False,
        on_governance_change="warn",
    )
    paused = await a.run("go")
    if legacy:
        raw = await store.get_pause_state(paused.run_id)
        raw.pop("pause_id")
        await store.pause_run(paused.run_id, status="waiting_approval", pause_data=raw)
    from dendrux.types import InputValidationError

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        if entry == "resume_stream":
            result, _ = await consume(a.resume_stream(paused.run_id))
            assert result.status == RunStatus.ERROR
            assert "pause_id_required" in result.error
        else:
            with pytest.raises(InputValidationError):
                if entry == "batch":
                    await a.submit_approval(paused.run_id, approved=True)
                elif entry == "decisions":
                    call = paused.pending.tool_calls[0]
                    await a.submit_approval(
                        paused.run_id, decisions={call.id: ApprovalDecision.approve()}
                    )
                else:
                    await a.resume(paused.run_id)
    assert not [w for w in caught if "expected_pause_id" in str(w.message)]
    assert (await store.get_run(paused.run_id)).status == "waiting_approval"


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
async def test_resume_never_dispatches_newly_denied_client_tool(store, entry):
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("server", {}), ToolCall("client_read", {})]),
            LLMResponse(text="done"),
        ]
    )
    a = Agent(
        name="denied",
        prompt="p",
        tools=[server, client_read],
        require_approval=["server"],
        provider=llm,
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    paused = await a.run("go")
    b = Agent(
        name="denied",
        prompt="p",
        tools=[server, client_read],
        deny=["client_read"],
        provider=llm,
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    if entry == "submit":
        result = await b.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )
    elif entry == "resume":
        result = await b.resume(paused.run_id, expected_pause_id=paused.pause_id)
    else:
        result, _ = await consume(b.resume_stream(paused.run_id, expected_pause_id=paused.pause_id))
    assert result.status == RunStatus.SUCCESS
    assert await store.get_pause_state(paused.run_id) is None
    messages = [m for m in llm.messages[-1] if m.name == "client_read" and m.role.value == "tool"]
    assert len(messages) == 1 and messages[0].meta["is_error"]
    assert any(
        e.event_type == "policy.denied" and e.data["tool_name"] == "client_read"
        for e in await store.get_run_events(paused.run_id)
    )


async def test_example_09_plain_text_round_trip(store, monkeypatch):
    import importlib.util
    from pathlib import Path

    import dotenv
    import httpx

    import dendrux.llm.anthropic as anthropic

    path = Path(__file__).resolve().parents[2] / "examples/09_client_tools_streaming/server.py"
    spec = importlib.util.spec_from_file_location("client_tool_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = ScriptedLLM(
        [
            tool_call("read_excel_range", {"sheet": "S1", "range": "A1"}, pid="a"),
            LLMResponse(text="read the plain-text result"),
        ]
    )
    # The example's app factory normally loads local credentials and a home DB.
    # Replace those dependencies before invoking it; no .env file or home DB is read.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(anthropic, "AnthropicProvider", lambda **kwargs: provider)

    def configured_agent(**kwargs):
        kwargs.pop("database_url")
        return Agent(**kwargs, state_store=store, strict_tenancy=False, on_governance_change="warn")

    monkeypatch.setattr(module, "Agent", configured_agent)
    app = module.create_demo_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        started = await client.post("/chat", json={"input": "read A1"})
        events = [json.loads(line) for line in started.text.splitlines()]
        rid = events[0]["run_id"]
        pending = next(e for e in events if e["type"] == "run_paused")["run_result"][
            "pending_tool_calls"
        ][0]
        resumed = await client.post(
            f"/resume/{rid}",
            json={
                "tool_results": [
                    {"name": pending["name"], "call_id": pending["id"], "result": "Revenue: $100M"}
                ]
            },
        )
    events = [json.loads(line) for line in resumed.text.splitlines()]
    assert events[-1]["type"] == "run_completed"
    assert events[-1]["run_result"]["status"] == "success"
    message = next(m for m in provider.messages[-1] if m.role.value == "tool")
    assert message.content == "Revenue: $100M"


@pytest.mark.parametrize("entry", ["batch", "decisions", "resume", "resume_stream"])
async def test_fenced_approval_never_warns_at_repeated_call_site(store, entry):
    a = Agent(
        name="dedup",
        prompt="p",
        tools=[server],
        require_approval=["server"],
        state_store=store,
        provider=ScriptedLLM(
            [
                response
                for _ in range(3)
                for response in [tool_call("server", pid="a"), LLMResponse(text="done")]
            ]
        ),
        strict_tenancy=False,
        on_governance_change="warn",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", FutureWarning)
        for _ in range(3):
            paused = await a.run("go")
            if entry == "batch":
                await a.submit_approval(
                    paused.run_id, approved=True, expected_pause_id=paused.pause_id
                )
            elif entry == "decisions":
                call = paused.pending.tool_calls[0]
                await a.submit_approval(
                    paused.run_id,
                    decisions={call.id: ApprovalDecision.approve()},
                    expected_pause_id=paused.pause_id,
                )
            elif entry == "resume":
                await a.resume(paused.run_id, expected_pause_id=paused.pause_id)
            else:
                await consume(a.resume_stream(paused.run_id, expected_pause_id=paused.pause_id))
    token_warnings = [w for w in caught if "expected_pause_id" in str(w.message)]
    assert not token_warnings


@pytest.mark.parametrize(
    "entry", ["tool", "input", "approval", "paused", "resume", "resume_stream", "retry"]
)
async def test_omitted_scope_missing_run_is_not_found(store, entry):
    from dendrux.errors import RunNotFoundError

    a = agent(store, [])
    with pytest.raises(RunNotFoundError):
        if entry == "tool":
            await a.submit_tool_results("missing", [])
        elif entry == "input":
            await a.submit_input("missing", "text")
        elif entry == "approval":
            await a.submit_approval("missing", approved=True)
        elif entry == "paused":
            await a.submit_resume("missing")
        elif entry == "resume":
            await a.resume("missing")
        elif entry == "resume_stream":
            await consume(a.resume_stream("missing"))
        else:
            await a.retry("missing")


async def test_legacy_control_override_keeps_fencing_without_false_stop(engine):
    from dendrux.runtime.state import RunControl, SQLAlchemyStateStore

    class OlderControlStore(SQLAlchemyStateStore):
        async def get_run_control(self, run_id):
            return RunControl(cancel_requested=await self.is_cancel_requested(run_id))

    store = OlderControlStore(engine)
    a = agent(store, [LLMResponse(text="done")])
    assert (await a.run("go")).status == RunStatus.SUCCESS


async def test_raw_claimed_entry_can_keep_original_epoch(store):
    from dendrux.runtime.runner import resume_claimed
    from dendrux.runtime.submit import serialize_tool_results

    a = agent(store, [tool_call("client_read", pid="a"), LLMResponse(text="must not call")])
    paused = await a.run("go")
    old = (
        await store.submit_and_claim(
            paused.run_id,
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": serialize_tool_results([result_for(paused)])},
        )
    ).claim_epoch
    assert await store.release_run_claim(
        paused.run_id, status="waiting_client_tool", claim_epoch=old
    )
    newer = (
        await store.claim_paused_run(paused.run_id, expected_status="waiting_client_tool")
    ).claim_epoch
    result = await resume_claimed(
        paused.run_id,
        state_store=store,
        agent=a,
        provider=a.provider,
        pre_claim_status="waiting_client_tool",
        claim_epoch=old,
    )
    assert result.status == RunStatus.RUNNING
    assert await epoch_of(store, paused.run_id) == newer
    assert a.provider.calls == 1
    assert await store.finalize_run(paused.run_id, status="cancelled", claim_epoch=newer)


async def test_client_dispatch_rechecks_lowered_cap(store):
    def limited(cap):
        @tool(target="client", max_calls_per_run=cap)
        async def limited_read() -> str:
            raise AssertionError("client tool executed on server")

        return limited_read

    llm = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("server", {}),
                    ToolCall("limited_read", {}),
                    ToolCall("limited_read", {}),
                ]
            ),
            LLMResponse(text="done"),
        ]
    )
    a = Agent(
        name="cap",
        prompt="p",
        tools=[server, limited(3)],
        require_approval=["server"],
        provider=llm,
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    paused = await a.run("go")
    b = Agent(
        name="cap",
        prompt="p",
        tools=[server, limited(1)],
        provider=llm,
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    pending = await b.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )
    assert pending.status == RunStatus.WAITING_CLIENT_TOOL
    assert len(pending.pending.tool_calls) == 1
    assert (
        await b.submit_tool_results(paused.run_id, [result_for(pending, "rows")])
    ).status == RunStatus.SUCCESS


@pytest.mark.parametrize("streaming", [False, True])
async def test_governance_warning_points_to_public_call_site(store, streaming):
    from dendrux import Limits

    a = agent(store, [tool_call("client_read", pid="a")], limits=Limits(max_tool_result_bytes=100))
    paused = await a.run("go")
    b = Agent(
        name="integrity",
        prompt="p",
        tools=[client_read],
        state_store=store,
        provider=ScriptedLLM([LLMResponse(text="done")]),
        strict_tenancy=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        if streaming:
            await consume(
                b.resume_stream(
                    paused.run_id,
                    tool_results=[result_for(paused)],
                    expected_pause_id=paused.pause_id,
                )
            )
        else:
            await b.submit_tool_results(paused.run_id, [result_for(paused)])
    changed = [w for w in caught if "Changed governance" in str(w.message)]
    assert len(changed) == 1 and changed[0].filename == __file__
