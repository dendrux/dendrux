"""Submitted and resumed text retains its trusted blocked provenance."""

import json

import pytest
from sqlalchemy import select

from dendrux import Agent
from dendrux.db.models import Base
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.runtime._blocked_content import _blocked_write
from dendrux.tool import tool
from dendrux.types import ApprovalDecision, LLMResponse, RunStatus, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format

SECRET = "AKIAIOSFODNN7EXAMPLE"


@tool(target="client")
async def client_read() -> str:
    return "unused"


async def paused_agent(store, engine, fmt, *, guarded=False, approval=False, blob_store=None):
    await pin_storage_format(engine, fmt)
    effects = []

    @tool(parallel=False)
    async def write_value() -> str:
        effects.append("effect")
        return "okay"

    name = "write_value" if approval else "client_read"
    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name=name, params={})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        name="continuation",
        blob_store=blob_store,
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        tools=[write_value] if approval else [client_read],
        require_approval=[name] if approval else None,
        guardrails=[SecretDetection()] if guarded else None,
    )
    result = await agent.run("go", tenant_id="t")
    raw = await store.get_pause_state(result.run_id)
    return agent, provider, result.run_id, raw, effects


async def rows_are_clean(engine):
    async with engine.connect() as connection:
        for table in Base.metadata.sorted_tables:
            rows = (await connection.execute(select(table))).mappings().all()
            assert SECRET not in repr(rows), table.name


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("path", ["submit", "resume", "stream", "approve", "paused"])
async def test_061_marked_continuation_finalizes_error(store, engine, fmt, path):
    agent, provider, run_id, raw, effects = await paused_agent(
        store, engine, fmt, approval=path == "approve"
    )
    try:
        if path == "paused":
            await store.pause_run(run_id, status="paused", pause_data=raw)
        with _blocked_write(store, run_id):
            await store.save_run_event(run_id, event_type="guardrail.detected", data={"count": 1})
        results = [
            ToolResult(
                name="client_read", call_id=raw["pending_tool_calls"][0]["id"], payload='"okay"'
            )
        ]
        kwargs = {"tenant_id": "t", "expected_pause_id": raw["pause_id"]}
        if path == "submit":
            result = await agent.submit_tool_results(run_id, results, **kwargs)
        elif path == "resume":
            result = await agent.resume(run_id, tool_results=results, **kwargs)
        elif path == "stream":
            stream = agent.resume_stream(run_id, tool_results=results, **kwargs)
            async for _ in stream:
                pass
            result = stream.result
        elif path == "approve":
            result = await agent.submit_approval(run_id, approved=True, **kwargs)
        else:
            result = await agent.submit_resume(run_id, **kwargs)
        assert result.status == RunStatus.ERROR
        assert provider._call_count == 1 and effects == []
        record = await store.get_run(run_id)
        assert record.failure_reason == "guardrail_blocked"
        assert record.meta["dendrux.blocked"] is True
        events = await store.get_run_events(run_id, event_types=["run.error"])
        assert len(events) == 1
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("surface", ["client", "resume", "input", "rejection", "decision", "note"])
async def test_submitted_text_is_projected_before_first_write(
    store, engine, fmt, surface, monkeypatch
):
    from dendrux.runtime import runner

    agent, provider, run_id, raw, effects = await paused_agent(
        store, engine, fmt, guarded=True, approval=surface in ("rejection", "decision")
    )
    finalize = runner._finalize_running_event
    snapshots = []

    async def clean_before_terminal_commit(*args, **kwargs):
        # Between the claim COMMIT and the terminal COMMIT (pause data still present).
        await rows_are_clean(engine)
        snapshots.append(True)
        return await finalize(*args, **kwargs)

    monkeypatch.setattr(runner, "_finalize_running_event", clean_before_terminal_commit)
    try:
        kwargs = {"tenant_id": "t", "expected_pause_id": raw["pause_id"]}
        call_id = raw["pending_tool_calls"][0]["id"]
        results = [
            ToolResult(
                name="client_read",
                call_id=call_id,
                payload=json.dumps(SECRET),
                error=SECRET,
                meta={"copy": SECRET},
            )
        ]
        if surface == "client":
            result = await agent.submit_tool_results(run_id, results, **kwargs)
        elif surface == "resume":
            result = await agent.resume(run_id, tool_results=results, **kwargs)
        elif surface == "input":
            raw["pending_tool_calls"] = []
            await store.pause_run(run_id, status="waiting_human_input", pause_data=raw)
            result = await agent.submit_input(run_id, SECRET, **kwargs)
        elif surface == "rejection":
            result = await agent.submit_approval(
                run_id, approved=False, rejection_reason=SECRET, **kwargs
            )
        elif surface == "decision":
            result = await agent.submit_approval(
                run_id, decisions={call_id: ApprovalDecision.deny(SECRET)}, **kwargs
            )
        else:
            await agent.submit_note(run_id, SECRET, metadata={"copy": SECRET}, tenant_id="t")
            await rows_are_clean(engine)
            result = await agent.submit_tool_results(
                run_id,
                [ToolResult(name="client_read", call_id=call_id, payload='"okay"')],
                **kwargs,
            )
        await rows_are_clean(engine)
        assert result.status == RunStatus.ERROR and snapshots
        assert provider._call_count == 1 and effects == []
        assert (await store.get_run(run_id)).meta["dendrux.blocked"] is True
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_live_tool_blocks_before_next_effect_and_callback(store, engine, fmt):
    from dendrux.loops.base import BaseNotifier

    await pin_storage_format(engine, fmt)
    effects = []
    observed = []

    @tool(parallel=False)
    async def read_secret() -> str:
        return SECRET

    @tool(parallel=False)
    async def next_effect() -> str:
        effects.append("effect")
        return "done"

    class Capture(BaseNotifier):
        async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
            observed.append(tool_result)

    provider = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="read_secret", params={}),
                    ToolCall(name="next_effect", params={}),
                ]
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[read_secret, next_effect],
        guardrails=[SecretDetection()],
    )
    try:
        result = await agent.run("go", tenant_id="t", notifier=Capture())
        await rows_are_clean(engine)
        assert SECRET not in repr(observed)
        assert result.status == RunStatus.ERROR
        assert effects == [] and provider._call_count == 1
        assert (await store.get_run(result.run_id)).meta["dendrux.blocked"] is True
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_note_block_during_provider_stops_subsequent_tool(store, engine, fmt):
    import asyncio

    await pin_storage_format(engine, fmt)
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    @tool()
    async def write_after_note() -> str:
        effects.append("effect")
        return "done"

    class WaitingProvider(MockLLM):
        async def complete(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return await super().complete(*args, **kwargs)

    provider = WaitingProvider(
        [
            LLMResponse(tool_calls=[ToolCall(name="write_after_note", params={})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[write_after_note],
        guardrails=[SecretDetection()],
    )
    task = asyncio.create_task(agent.run("go", tenant_id="t"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        from dendrux.db.models import AgentRun

        async with engine.connect() as connection:
            run = (await connection.execute(select(AgentRun.id))).scalar_one()
        await agent.submit_note(run, SECRET, tenant_id="t")
        release.set()
        result = await task
        assert result.status == RunStatus.ERROR
        assert effects == [] and provider._call_count == 1
        await rows_are_clean(engine)
    finally:
        release.set()
        await task
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_note_terminal_winner_does_not_acquire_rejected_mark(store, engine, fmt, monkeypatch):
    from functools import wraps

    from dendrux.errors import RunAlreadyTerminalError

    agent, _, run_id, _, _ = await paused_agent(store, engine, fmt, guarded=True)
    original = store.add_note

    @wraps(original)
    async def win_before_note(*args, **kwargs):
        import asyncio
        from contextvars import Context

        await asyncio.create_task(store.finalize_run(run_id, status="cancelled"), context=Context())
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "add_note", win_before_note)
    try:
        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_note(run_id, SECRET, tenant_id="t")
        record = await store.get_run(run_id)
        assert record.status == "cancelled"
        assert not record.meta.get("dendrux.blocked")
        assert await store.get_notes(run_id) == []
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_rejected_submission_emits_existing_block_event(store, engine, fmt):
    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)
    try:
        await agent.submit_tool_results(
            run_id,
            [
                ToolResult(
                    name="client_read",
                    call_id=raw["pending_tool_calls"][0]["id"],
                    payload=json.dumps(SECRET),
                )
            ],
            tenant_id="t",
            expected_pause_id=raw["pause_id"],
        )
        assert await store.get_run_events(run_id, event_types=["guardrail.blocked"])
        assert provider._call_count == 1
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("wrong", ["tenant", "token", "call", "terminal"])
async def test_marked_submission_preserves_validation_precedence(store, engine, fmt, wrong):
    from dendrux.errors import (
        InvalidToolResultError,
        PauseStatusMismatchError,
        RunAlreadyTerminalError,
        RunNotFoundError,
    )

    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)
    try:
        with _blocked_write(store, run_id):
            await store.save_run_event(run_id, event_type="guardrail.detected", data={})
        if wrong == "terminal":
            await store.finalize_run(run_id, status="cancelled")
        old = await store.get_run(run_id)
        expected = {
            "tenant": RunNotFoundError,
            "token": PauseStatusMismatchError,
            "call": InvalidToolResultError,
            "terminal": RunAlreadyTerminalError,
        }[wrong]
        with pytest.raises(expected):
            await agent.submit_tool_results(
                run_id,
                [
                    ToolResult(
                        name="client_read",
                        call_id="wrong" if wrong == "call" else raw["pending_tool_calls"][0]["id"],
                        payload=json.dumps(SECRET),
                    )
                ],
                tenant_id="wrong" if wrong == "tenant" else "t",
                expected_pause_id="wrong" if wrong == "token" else raw["pause_id"],
            )
        assert (await store.get_run(run_id)).status == old.status
        assert provider._call_count == 1
        await rows_are_clean(engine)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_claim_commit_then_restart_uses_durable_mark_without_guardrails(
    store, engine, fmt, monkeypatch
):
    from functools import wraps

    from dendrux.runtime.runner import resume_claimed

    class Crash(BaseException):
        pass

    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)
    original = store.submit_and_claim

    @wraps(original)
    async def crash_after_commit(*args, **kwargs):
        result = await original(*args, **kwargs)
        if result:
            raise Crash()
        return result

    monkeypatch.setattr(store, "submit_and_claim", crash_after_commit)
    restarted = Agent(
        on_governance_change="warn",
        name="continuation",
        prompt="Answer briefly.",
        provider=MockLLM([LLMResponse(text="unreachable")]),
        state_store=store,
        tools=[client_read],
    )
    try:
        with pytest.raises(Crash):
            await agent.submit_tool_results(
                run_id,
                [
                    ToolResult(
                        name="client_read",
                        call_id=raw["pending_tool_calls"][0]["id"],
                        payload=json.dumps(SECRET),
                    )
                ],
                tenant_id="t",
                expected_pause_id=raw["pause_id"],
            )
        record = await store.get_run(run_id)
        assert record.status == "running" and record.meta["dendrux.blocked"] is True
        await rows_are_clean(engine)
        result = await resume_claimed(
            run_id,
            state_store=store,
            agent=restarted,
            provider=restarted._require_provider(),
            pre_claim_status="waiting_client_tool",
            claim_epoch=record.claim_epoch,
        )
        assert result.status == RunStatus.ERROR
        assert restarted._require_provider()._call_count == 0
        assert (await store.get_run(run_id)).failure_reason == "guardrail_blocked"
    finally:
        await restarted.close()
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("path", ["client", "resume", "live"])
async def test_unchanged_tool_payload_is_scanned_once(store, engine, fmt, path):
    await pin_storage_format(engine, fmt)
    seen = []
    payload = json.dumps("COUNT_THIS_PAYLOAD")

    class Count:
        action = "block"

        async def scan(self, text):
            if text == payload:
                seen.append(text)
            return []

    @tool()
    async def read_count() -> str:
        return "COUNT_THIS_PAYLOAD"

    name = "read_count" if path == "live" else "client_read"
    provider = MockLLM(
        [LLMResponse(tool_calls=[ToolCall(name=name, params={})]), LLMResponse(text="done")]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[read_count if path == "live" else client_read],
        guardrails=[Count()],
    )
    try:
        result = await agent.run("go", tenant_id="t")
        if path != "live":
            raw = await store.get_pause_state(result.run_id)
            results = [
                ToolResult(name=name, call_id=raw["pending_tool_calls"][0]["id"], payload=payload)
            ]
            if path == "client":
                result = await agent.submit_tool_results(
                    result.run_id, results, tenant_id="t", expected_pause_id=raw["pause_id"]
                )
            else:
                result = await agent.resume(
                    result.run_id,
                    tool_results=results,
                    tenant_id="t",
                    expected_pause_id=raw["pause_id"],
                )
        assert result.status == RunStatus.SUCCESS
        assert seen == [payload]
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("surface", ["client", "resume", "input", "rejection", "decision", "note"])
async def test_same_call_block_keeps_guardrail_error(store, engine, fmt, surface):
    # A block found in this call reports the guardrail and finding kinds (never the
    # matched text); only a mark left by an earlier call gets the generic message.
    agent, provider, run_id, raw, _ = await paused_agent(
        store, engine, fmt, guarded=True, approval=surface in ("rejection", "decision")
    )
    try:
        kwargs = {"tenant_id": "t", "expected_pause_id": raw["pause_id"]}
        call_id = raw["pending_tool_calls"][0]["id"]
        results = [ToolResult(name="client_read", call_id=call_id, payload=json.dumps(SECRET))]
        if surface == "client":
            result = await agent.submit_tool_results(run_id, results, **kwargs)
        elif surface == "resume":
            result = await agent.resume(run_id, tool_results=results, **kwargs)
        elif surface == "input":
            raw["pending_tool_calls"] = []
            await store.pause_run(run_id, status="waiting_human_input", pause_data=raw)
            result = await agent.submit_input(run_id, SECRET, **kwargs)
        elif surface == "rejection":
            result = await agent.submit_approval(
                run_id, approved=False, rejection_reason=SECRET, **kwargs
            )
        elif surface == "decision":
            result = await agent.submit_approval(
                run_id, decisions={call_id: ApprovalDecision.deny(SECRET)}, **kwargs
            )
        else:
            await agent.submit_note(run_id, SECRET, tenant_id="t")
            result = await agent.submit_tool_results(
                run_id,
                [ToolResult(name="client_read", call_id=call_id, payload='"okay"')],
                **kwargs,
            )
        assert result.status == RunStatus.ERROR and SECRET not in (result.error or "")
        assert (await store.get_run(run_id)).error == result.error
        if surface == "note":
            assert result.error == "Run stopped because it contains blocked content."
        else:
            assert result.error.startswith("Guardrail 'SecretDetection' blocked: 1 ")
        if surface in ("client", "resume"):
            assert result.error.endswith(f"at message 3 (role=tool, tool_call_id={call_id})")
            assert result.iteration_count == 2
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_adopted_run_with_legacy_block_event_stops(store, engine, fmt, monkeypatch):
    # 0.6.0 data: a crash after a blocked event committed but before finalize leaves an
    # unmarked `running` run. Adoption without guardrails must still see that event.
    from functools import wraps

    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.runtime.runner import resume_claimed

    class Crash(BaseException):
        pass

    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt)
    original = store.submit_and_claim

    @wraps(original)
    async def crash_after_commit(*args, **kwargs):
        result = await original(*args, **kwargs)
        if result:
            raise Crash()
        return result

    monkeypatch.setattr(store, "submit_and_claim", crash_after_commit)
    adopter_provider = MockLLM([LLMResponse(text="unreachable")])
    adopter = Agent(
        on_governance_change="warn",
        name="continuation",
        prompt="Answer briefly.",
        provider=adopter_provider,
        state_store=store,
        tools=[client_read],
    )
    try:
        with pytest.raises(Crash):
            await agent.submit_tool_results(
                run_id,
                [
                    ToolResult(
                        name="client_read",
                        call_id=raw["pending_tool_calls"][0]["id"],
                        payload=json.dumps("legacy"),
                    )
                ],
                tenant_id="t",
                expected_pause_id=raw["pause_id"],
            )
        await store.save_run_event(run_id, event_type="guardrail.blocked", data={})
        record = await store.get_run(run_id)
        meta = {k: v for k, v in record.meta.items() if k != "dendrux.blocked"}
        async with engine.begin() as connection:
            await connection.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(meta=meta)
            )
        record = await store.get_run(run_id)
        assert record.status == "running" and "dendrux.blocked" not in record.meta
        result = await resume_claimed(
            run_id,
            state_store=store,
            agent=adopter,
            provider=adopter._require_provider(),
            pre_claim_status="waiting_client_tool",
            claim_epoch=record.claim_epoch,
        )
        assert result.status == RunStatus.ERROR
        assert adopter_provider._call_count == 0
        assert (await store.get_run(run_id)).failure_reason == "guardrail_blocked"
    finally:
        await adopter.close()
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("path", ["submit", "resume"])
async def test_no_blocked_detail_outlives_the_call(store, engine, fmt, path):
    # Nothing from a blocked submission (raw message, findings) stays in the caller's
    # context once the call returns.
    from contextvars import copy_context

    agent, _, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)
    try:
        results = [
            ToolResult(
                name="client_read",
                call_id=raw["pending_tool_calls"][0]["id"],
                payload=json.dumps(SECRET),
            )
        ]
        kwargs = {"tenant_id": "t", "expected_pause_id": raw["pause_id"]}
        if path == "submit":
            result = await agent.submit_tool_results(run_id, results, **kwargs)
        else:
            result = await agent.resume(run_id, tool_results=results, **kwargs)
        assert result.status == RunStatus.ERROR
        leftovers = [
            var.name
            for var, value in copy_context().items()
            if var.name.startswith("dendrux")
            and value is not None
            and not isinstance(value, bool)
            and (SECRET in repr(value) or "Finding(" in repr(value))
        ]
        assert leftovers == []
    finally:
        await agent.close()
