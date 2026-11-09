"""Continuation failures keep validation, mapping and terminal ownership intact."""

import asyncio
import json

import pytest
from sqlalchemy import select, update

from dendrux import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, RunFencedError
from dendrux.guardrails import PII, SecretDetection
from dendrux.guardrails._engine import _entry_engine
from dendrux.llm.mock import MockLLM
from dendrux.runtime._blocked_content import _blocked_write
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format
from tests.integration.test_061_secrets_continuation import (
    SECRET,
    client_read,
    paused_agent,
    rows_are_clean,
)


@pytest.mark.parametrize("fmt", [1, 2])
async def test_later_block_keeps_prior_allowed_pii_mapping(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(tool_calls=[ToolCall(name="client_read", params={})])])
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[client_read],
        guardrails=[PII(action="redact"), SecretDetection()],
    )
    try:
        paused = await agent.run("owner person@example.com", tenant_id="t")
        mapping = await store.get_pii_mapping(paused.run_id)
        assert "person@example.com" in mapping.values()
        raw = await store.get_pause_state(paused.run_id)
        result = await agent.submit_tool_results(
            paused.run_id,
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
        assert result.status == RunStatus.ERROR
        assert result.meta["pii_mapping"] == mapping
        assert await store.get_pii_mapping(paused.run_id) == mapping
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("failure", ["scan", "cancel", "fenced"])
async def test_submission_scan_failure_and_interruptions(store, engine, fmt, failure):
    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)

    class Fail:
        action = "block"

        async def scan(self, text):
            if SECRET in text:
                if failure == "cancel":
                    raise asyncio.CancelledError()
                if failure == "fenced":
                    raise RunFencedError(run_id)
                raise ValueError(SECRET)
            return []

    agent._guardrails = [Fail()]
    agent._on_governance_change = "warn"
    try:
        kwargs = dict(tenant_id="t", expected_pause_id=raw["pause_id"])
        results = [
            ToolResult(
                name="client_read",
                call_id=raw["pending_tool_calls"][0]["id"],
                payload=json.dumps(SECRET),
            )
        ]
        if failure == "scan":
            result = await agent.submit_tool_results(run_id, results, **kwargs)
            assert result.status == RunStatus.ERROR and SECRET not in str(result.error)
        else:
            with pytest.raises(asyncio.CancelledError if failure == "cancel" else RunFencedError):
                await agent.submit_tool_results(run_id, results, **kwargs)
            row = await store.get_run(run_id)
            assert row.status == "waiting_client_tool" and not row.meta.get("dendrux.blocked")
        assert provider._call_count == 1 and _entry_engine(run_id) is None
        await rows_are_clean(engine)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_known_legacy_block_stops_continuation(store, engine, fmt):
    # Every earlier blocked event finalized its run as an error, so a continuable
    # run carries provenance only as the mark or the recorded block reason.
    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt)
    try:
        async with engine.begin() as connection:
            row = await store.get_run(run_id)
            meta = {k: v for k, v in row.meta.items() if k != "dendrux.blocked"}
            await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(meta=meta, failure_reason="guardrail_blocked")
            )
        result = await agent.submit_tool_results(
            run_id,
            [
                ToolResult(
                    name="client_read", call_id=raw["pending_tool_calls"][0]["id"], payload='"okay"'
                )
            ],
            tenant_id="t",
            expected_pause_id=raw["pause_id"],
        )
        assert result.status == RunStatus.ERROR and provider._call_count == 1
    finally:
        await agent.close()


@pytest.mark.parametrize("honors_mark", [False, True])
async def test_custom_submission_requires_safe_mark_contract(store, engine, honors_mark):
    from tests._fixtures.stores import LegacyInMemoryStore

    original, provider, run_id, raw, _ = await paused_agent(store, engine, 1, guarded=True)

    class ContractStore(LegacyInMemoryStore):
        async def update_run_meta(self, run, meta, **kwargs):
            if honors_mark and meta.get("dendrux.blocked") is True:
                with _blocked_write(self.wrapped, run):
                    return await self.wrapped.update_run_meta(run, meta, **kwargs)
            return await self.wrapped.update_run_meta(run, meta, **kwargs)

    agent = Agent(
        name="continuation",
        prompt="Answer briefly.",
        provider=provider,
        state_store=ContractStore(store),
        tools=[client_read],
        guardrails=[SecretDetection()],
    )
    try:
        results = [
            ToolResult(
                name="client_read",
                call_id=raw["pending_tool_calls"][0]["id"],
                payload=json.dumps(SECRET),
            )
        ]
        if honors_mark:
            result = await agent.submit_tool_results(
                run_id, results, tenant_id="t", expected_pause_id=raw["pause_id"]
            )
            assert result.status == RunStatus.ERROR
        else:
            with pytest.raises(PersistenceError) as caught:
                await agent.submit_tool_results(
                    run_id, results, tenant_id="t", expected_pause_id=raw["pause_id"]
                )
            assert caught.value.__context__ is None
        assert provider._call_count == 1
        await rows_are_clean(engine)
    finally:
        await agent.close()
        await original.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_blocked_note_between_serial_tools_stops_remaining_effect(store, engine, fmt):
    from dendrux.tool import tool

    await pin_storage_format(engine, fmt)
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    @tool(parallel=False)
    async def first_effect() -> str:
        effects.append("first")
        entered.set()
        await release.wait()
        return "okay"

    @tool(parallel=False)
    async def second_effect() -> str:
        effects.append("second")
        return "okay"

    provider = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="first_effect", params={}),
                    ToolCall(name="second_effect", params={}),
                ]
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[first_effect, second_effect],
        guardrails=[SecretDetection()],
    )
    task = asyncio.create_task(agent.run("go", tenant_id="t"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        from sqlalchemy import select

        async with engine.connect() as connection:
            run_id = (await connection.execute(select(AgentRun.id))).scalar_one()
        await agent.submit_note(run_id, SECRET, tenant_id="t")
        release.set()
        result = await task
        assert result.status == RunStatus.ERROR
        assert effects == ["first"] and provider._call_count == 1
    finally:
        release.set()
        await task
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_marked_continuation_retains_committed_usage(store, engine, fmt):
    from dendrux.types import UsageStats

    await pin_storage_format(engine, fmt)
    provider = MockLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="client_read", params={})],
                usage=UsageStats(input_tokens=7, output_tokens=3, total_tokens=10, cost_usd=0.5),
            )
        ]
    )
    agent = Agent(prompt="p", provider=provider, state_store=store, tools=[client_read])
    try:
        paused = await agent.run("go", tenant_id="t")
        raw = await store.get_pause_state(paused.run_id)
        with _blocked_write(store, paused.run_id):
            await store.save_run_event(paused.run_id, event_type="guardrail.detected", data={})
        result = await agent.submit_tool_results(
            paused.run_id,
            [
                ToolResult(
                    name="client_read", call_id=raw["pending_tool_calls"][0]["id"], payload='"okay"'
                )
            ],
            tenant_id="t",
            expected_pause_id=raw["pause_id"],
        )
        assert result.status == RunStatus.ERROR
        assert result.usage == paused.usage
        assert result.iteration_count == paused.iteration_count
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_client_error_text_is_governed_before_storage(store, engine, fmt):
    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)
    try:
        result = await agent.submit_tool_results(
            run_id,
            [
                ToolResult(
                    name="client_read",
                    call_id=raw["pending_tool_calls"][0]["id"],
                    payload='"failed"',
                    success=False,
                    error=SECRET,
                )
            ],
            tenant_id="t",
            expected_pause_id=raw["pause_id"],
        )
        assert result.status == RunStatus.ERROR and provider._call_count == 1
        await rows_are_clean(engine)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_marked_continuation_preserves_terminal_winner(store, engine, fmt, monkeypatch):
    from contextvars import Context

    from dendrux.runtime import runner

    agent, provider, run_id, raw, effects = await paused_agent(store, engine, fmt, guarded=True)
    original = runner._finalize_running_event

    async def win_before_finalize(claimed, target, *args, **kwargs):
        assert target == run_id
        await asyncio.create_task(
            store.finalize_run(run_id, status="cancelled", answer="winner"), context=Context()
        )
        return await original(claimed, target, *args, **kwargs)

    monkeypatch.setattr(runner, "_finalize_running_event", win_before_finalize)
    try:
        result = await agent.submit_tool_results(
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
        assert result.status == RunStatus.CANCELLED and result.answer == "winner"
        row = await store.get_run(run_id)
        assert row.status == "cancelled" and row.answer == "winner"
        assert row.failure_reason is None and row.meta["dendrux.blocked"] is True
        assert await store.get_run_events(run_id, event_types=["run.error"]) == []
        assert provider._call_count == 1 and effects == []
        await rows_are_clean(engine)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_submitted_text_parts_are_projected(store, engine, fmt):
    from pathlib import Path

    from dendrux.media import Image, InMemoryBlobStore

    png = (Path(__file__).resolve().parents[1] / "fixtures/images/zebra42.png").read_bytes()
    agent, provider, run_id, raw, _ = await paused_agent(
        store, engine, fmt, guarded=True, blob_store=InMemoryBlobStore()
    )
    try:
        raw["pending_tool_calls"] = []
        await store.pause_run(run_id, status="waiting_human_input", pause_data=raw)
        result = await agent.submit_input(
            run_id,
            ["prefix ", Image.from_bytes(png), SECRET],
            tenant_id="t",
            expected_pause_id=raw["pause_id"],
        )
        assert result.status == RunStatus.ERROR and provider._call_count == 1
        await rows_are_clean(engine)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_approved_live_tool_result_is_projected(store, engine, fmt):
    from dendrux.tool import tool

    await pin_storage_format(engine, fmt)

    @tool()
    async def read_approved() -> str:
        return SECRET

    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="read_approved", params={})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[read_approved],
        require_approval=["read_approved"],
        guardrails=[SecretDetection()],
    )
    try:
        paused = await agent.run("go", tenant_id="t")
        raw = await store.get_pause_state(paused.run_id)
        result = await agent.submit_approval(
            paused.run_id, approved=True, tenant_id="t", expected_pause_id=raw["pause_id"]
        )
        assert result.status == RunStatus.ERROR and provider._call_count == 1
        assert (await store.get_run(paused.run_id)).failure_reason == "guardrail_blocked"
        await rows_are_clean(engine)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_marked_stream_reports_terminal_winner(store, engine, fmt, monkeypatch):
    from contextvars import Context

    from dendrux.runtime import runner
    from dendrux.types import RunEventType

    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt)
    original = runner._finalize_running_event

    async def win_before_finalize(claimed, target, *args, **kwargs):
        await asyncio.create_task(store.finalize_run(run_id, status="cancelled"), context=Context())
        return await original(claimed, target, *args, **kwargs)

    monkeypatch.setattr(runner, "_finalize_running_event", win_before_finalize)
    try:
        with _blocked_write(store, run_id):
            await store.save_run_event(run_id, event_type="guardrail.detected", data={})
        stream = agent.resume_stream(
            run_id,
            tool_results=[
                ToolResult(
                    name="client_read", call_id=raw["pending_tool_calls"][0]["id"], payload='"okay"'
                )
            ],
            tenant_id="t",
            expected_pause_id=raw["pause_id"],
        )
        events = [event async for event in stream]
        assert stream.result.status == RunStatus.CANCELLED
        assert events[-1].type == RunEventType.RUN_CANCELLED
        assert provider._call_count == 1
        assert await store.get_run_events(run_id, event_types=["run.error"]) == []
    finally:
        await agent.close()


@pytest.mark.parametrize("raw_error", [False, True])
async def test_native_dispatch_lookup_failure_is_not_best_effort(
    store, engine, monkeypatch, raw_error
):
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.types import generate_ulid

    await pin_storage_format(engine, 1)
    run_id = generate_ulid()
    await store.create_run(run_id, agent_name="dispatch", input_data={})
    with _blocked_write(store, run_id):
        await store.save_run_event(run_id, event_type="guardrail.detected", data={})

    async def unavailable(*args):
        if raw_error:
            raise ValueError(SECRET)
        raise PersistenceError()

    monkeypatch.setattr(store, "_check_dispatch_block", unavailable)
    recorder = PersistenceRecorder(store, run_id)
    with pytest.raises(PersistenceError) as caught:
        await recorder.on_tool_started(run_id, ToolCall(name="effect", params={}), 1)
    assert (await store.get_run(run_id)).meta["dendrux.blocked"] is True
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("entry", ["resume", "submit"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_continuation_lookup_failure_uses_existing_claim_cleanup(
    store, engine, fmt, entry, cancel, monkeypatch
):
    agent, provider, run_id, raw, _ = await paused_agent(store, engine, fmt)

    from dendrux.runtime import _blocked_content

    def unavailable(*args, **kwargs):
        if cancel:
            raise asyncio.CancelledError()
        raise PersistenceError()

    monkeypatch.setattr(_blocked_content, "_continuation_blocked", unavailable)
    results = [
        ToolResult(name="client_read", call_id=raw["pending_tool_calls"][0]["id"], payload='"okay"')
    ]
    kwargs = dict(tenant_id="t", expected_pause_id=raw["pause_id"])
    try:
        with pytest.raises(asyncio.CancelledError if cancel else PersistenceError):
            if entry == "resume":
                await agent.resume(run_id, tool_results=results, **kwargs)
            else:
                await agent.submit_tool_results(run_id, results, **kwargs)
        record = await store.get_run(run_id)
        assert record.status == ("cancelled" if cancel else "waiting_client_tool")
        assert provider._call_count == 1
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_guardrail_free_note_delivery_adds_no_run_read(store, engine, fmt, monkeypatch):
    # The pending-note mark check runs only for guarded runs; a guardrail-free run
    # delivers notes with exactly the reads it used before.
    from dendrux.tool import tool

    await pin_storage_format(engine, fmt)
    reads = []
    original = store.get_run

    async def counted(run_id, *args, **kwargs):
        reads.append(run_id)
        return await original(run_id, *args, **kwargs)

    monkeypatch.setattr(store, "get_run", counted)

    async def run_once(with_note):
        holder = {}

        @tool()
        async def leave_note() -> str:
            if with_note:
                async with engine.connect() as connection:
                    run_id = (
                        await connection.execute(
                            select(AgentRun.id).where(AgentRun.status == "running")
                        )
                    ).scalar_one()
                await holder["agent"].submit_note(run_id, "clean note", tenant_id="t")
            return "ok"

        provider = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="leave_note", params={})]),
                LLMResponse(text="done"),
            ]
        )
        agent = holder["agent"] = Agent(
            prompt="p", provider=provider, state_store=store, tools=[leave_note]
        )
        try:
            before = len(reads)
            result = await agent.run("go", tenant_id="t")
            assert result.status == RunStatus.SUCCESS
            if with_note:
                notes = await store.get_notes(result.run_id)
                assert [note.text for note in notes] == ["clean note"]
                assert all(note.delivered_at is not None for note in notes)
            return len(reads) - before
        finally:
            await agent.close()

    assert await run_once(True) == await run_once(False)
