"""Format-2 intent and completion protocol through actual Agent entry paths."""

import asyncio
from functools import wraps

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse, ToolCall


async def test_a10_intent_precedes_execution_and_completion_is_atomic(store, monkeypatch):
    starts = []
    original = store.start_tool_call

    async def started(*args, **kwargs):
        starts.append(kwargs["tool_name"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "start_tool_call", started)

    @tool(idempotent=True)
    async def work() -> str:
        """Check durable intent before the effect."""
        run = get_delegation_context().run_id
        rows = await store.get_tool_calls(run)
        row = next((row for row in rows if row.tool_name == "work"), None)
        assert row is not None and row.outcome == "started" and row.idempotent
        return "worked"

    @tool()
    async def denied() -> str:
        """Never executes."""
        raise AssertionError("denied execution")

    model = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("missing"), ToolCall("denied"), ToolCall("work")]),
            LLMResponse(text="done"),
        ]
    )
    result = await Agent(
        prompt="p",
        provider=model,
        tools=[work, denied],
        deny=["denied"],
        state_store=store,
        strict_tenancy=False,
    ).run("go")
    assert starts == ["work"]
    rows = await store.get_tool_calls(result.run_id)
    assert next(r for r in rows if r.tool_name == "work").outcome == "completed"
    state = await store.get_loop_state(result.run_id)
    assert state["pending"] == []
    traces = await store.get_traces(result.run_id)
    assert len([t for t in traces if (t.meta or {}).get("tool_name") == "work"]) == 1


async def test_a20_completion_commits_while_sibling_is_running(store):
    slow_started = asyncio.Event()
    fast_done = asyncio.Event()
    checks = []

    @tool(parallel=True, idempotent=True)
    async def slow() -> str:
        """Wait for a sibling's committed completion."""
        slow_started.set()
        await asyncio.wait_for(fast_done.wait(), 5)
        run = get_delegation_context().run_id
        for _ in range(100):
            rows = await store.get_tool_calls(run)
            if any(r.tool_name == "fast" and r.outcome == "completed" for r in rows):
                state = await store.get_loop_state(run)
                checks.append([p["tool"] for p in state["pending"]])
                break
            await asyncio.sleep(0.005)
        return "slow"

    @tool(parallel=True, idempotent=True)
    async def fast() -> str:
        """Finish before the first call."""
        await slow_started.wait()
        fast_done.set()
        return "fast"

    model = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("slow"), ToolCall("fast")]), LLMResponse(text="done")]
    )
    result = await Agent(
        prompt="p", provider=model, tools=[slow, fast], state_store=store, strict_tenancy=False
    ).run("go")
    assert checks == [["slow"]]
    traces = await store.get_traces(result.run_id)
    assert [(t.meta or {}).get("tool_name") for t in traces if t.role == "tool"] == ["slow", "fast"]
    state = await store.get_loop_state(result.run_id)
    assert state["tool_counts"] == {"slow": 1, "fast": 1}


async def test_stream_lease_survives_consumer_yield_without_cancellation(store, monkeypatch):
    from datetime import timedelta

    from dendrux.types import RunEventType

    touches = []
    original = store.touch_progress

    async def touch(run_id, *, claim_epoch=None):
        touches.append(1)
        return await original(run_id, claim_epoch=claim_epoch)

    monkeypatch.setattr(store, "touch_progress", touch)
    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="hello")]),
        state_store=store,
        strict_tenancy=False,
    )
    a.lease_interval = timedelta(seconds=0.01)
    async with a.stream("go") as stream:
        async for event in stream:
            if event.type == RunEventType.TEXT_DELTA:
                count = len(touches)
                await asyncio.sleep(0.06)
                assert len(touches) > count
                assert asyncio.current_task().cancelling() == 0
                break


async def test_final_trace_and_event_share_finalization(store, monkeypatch):
    boundaries = []
    original = store.finalize_with_event

    async def finalize(run_id, *, claim_epoch=None, trace=None, event=None, **kwargs):
        before = await store.get_traces(run_id)
        boundaries.append((trace, event, before))
        return await original(run_id, claim_epoch=claim_epoch, trace=trace, event=event, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", finalize)
    await Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="final answer")]),
        state_store=store,
        strict_tenancy=False,
    ).run("go")
    assert len(boundaries) == 1
    trace, event, before = boundaries[0]
    assert trace.content == "final answer" and event.event_type == "run.completed"
    assert not any(t.content == "final answer" for t in before)


@pytest.mark.parametrize("approved", [False, True])
async def test_a19_approval_consumes_matching_generation_before_execution(
    store, monkeypatch, approved
):
    submissions, consumed, effects = [], [], []
    submit = store.submit_and_claim
    consume = store.consume_submission

    async def submitting(*args, **kwargs):
        submissions.append(kwargs.get("submission"))
        return await submit(*args, **kwargs)

    @wraps(consume)
    async def consuming(run_id, *, claim_epoch, **kwargs):
        result = await consume(run_id, claim_epoch=claim_epoch, **kwargs)
        consumed.append(await store.get_loop_state(run_id))
        return result

    monkeypatch.setattr(store, "submit_and_claim", submitting)
    monkeypatch.setattr(store, "consume_submission", consuming)

    @tool()
    async def change() -> str:
        """Require a decision before applying an effect."""
        run = get_delegation_context().run_id
        state = await store.get_loop_state(run)
        assert state["submission"] is None
        assert consumed[0]["pending"][0]["phase"] == "approved"
        assert state["pending"][0]["phase"] == "dispatchable"
        assert state["pending"][0]["decision_id"] is None
        effects.append("changed")
        return "ok"

    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("change")]), LLMResponse(text="done")]),
        tools=[change],
        require_approval=["change"],
        state_store=store,
        strict_tenancy=False,
    )
    paused = await a.run("go")
    await a.submit_approval(paused.run_id, approved=approved, expected_pause_id=paused.pause_id)
    assert submissions and submissions[0]["kind"] == "approval"
    assert submissions[0]["pause_id"] == paused.pause_id
    assert consumed and consumed[0]["submission"] is None
    assert effects == (["changed"] if approved else [])


async def test_format2_submission_requires_descriptor_and_rolls_back(store):
    from dendrux.errors import PersistenceError
    from dendrux.types import generate_ulid

    run = generate_ulid()
    await store.create_run(run, "descriptor")
    await store.pause_run(run, status="waiting_human_input", pause_data={"schema_version": 2})
    with pytest.raises(PersistenceError) as error:
        await store.submit_and_claim(
            run,
            expected_status="waiting_human_input",
            submitted_data={"submitted_user_input": "answer"},
        )
    assert error.value.code == "submission_missing"
    assert (await store.get_run(run)).status == "waiting_human_input"
    assert (await store.get_pause_state(run)) == {"schema_version": 2}


async def test_client_stays_pending_until_validated_submission(store):
    from dendrux.runtime.persistence import resolve_pending
    from dendrux.types import ToolResult

    @tool(target="client")
    async def read() -> str:
        """The client supplies this result."""
        raise AssertionError("client tool executed on server")

    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("read")]), LLMResponse(text="done")]),
        tools=[read],
        state_store=store,
        strict_tenancy=False,
    )
    paused = await a.run("go")
    state = await store.get_loop_state(paused.run_id)
    for _ in range(2):
        assert [a.kind for a in resolve_pending(state, [])] == ["wait_client"]
        assert await store.get_loop_state(paused.run_id) == state
    with pytest.raises(ValueError):
        await a.submit_tool_results(paused.run_id, [], expected_pause_id=paused.pause_id)
    assert await store.get_loop_state(paused.run_id) == state
    call = state["pending"][0]["call_id"]
    await a.submit_tool_results(
        paused.run_id,
        [ToolResult(name="read", call_id=call, payload='"ok"')],
        expected_pause_id=paused.pause_id,
    )
    assert (await store.get_loop_state(paused.run_id))["submission"] is None
    observations = [t for t in await store.get_traces(paused.run_id) if t.role == "tool"]
    assert len(observations) == 1
    assert not [
        e for e in await store.get_run_events(paused.run_id) if e.event_type == "tool.started"
    ]


async def test_terminal_phase_commits_with_final_trace(store):
    result = await Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=store,
        strict_tenancy=False,
    ).run("go")
    state = await store.get_loop_state(result.run_id)
    assert state["phase"] == "finalizing"
    assert state["iteration"] == 1


async def test_patch_cannot_grow_checkpoint_past_bound(store):
    from dendrux.errors import PersistenceError
    from dendrux.runtime.state import LoopStatePatch
    from dendrux.types import generate_ulid

    run = generate_ulid()
    await store.create_run(run, "patch")
    await store.checkpoint_step(
        run, claim_epoch=0, loop_state={"prefix": "a" * 20000}, pii_mapping=None, trace=None
    )
    async with store._session_factory() as session:
        with pytest.raises(PersistenceError) as error:
            await store._patch_fence(
                session, run, 0, LoopStatePatch(set_fields={"suffix": "b" * 20000})
            )
        assert error.value.code == "checkpoint_too_large"
        await session.rollback()
    assert await store.get_loop_state(run) == {"prefix": "a" * 20000}
    async with store._session_factory() as session:
        await store._patch_fence(session, run, 0, LoopStatePatch(set_fields={"suffix": "b" * 4000}))
        await session.commit()
    assert len((await store.get_loop_state(run))["suffix"]) == 4000


async def test_unconsumed_approval_cannot_begin_an_execution(store):
    from dendrux.errors import PersistenceError
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.state import ClaimResult, claimed_store

    call = ToolCall("change")
    await store.create_run("00000000000000000000000110", "a")
    state = {
        "v": 1,
        "phase": "awaiting_approval",
        "pending": [
            {
                "call_id": call.id,
                "tool": "change",
                "target": "server",
                "phase": "gated",
                "order_index": 1,
                "decision_id": "generation:" + call.id,
            }
        ],
    }
    await store.checkpoint_step(
        "00000000000000000000000110", claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    owned = claimed_store(store, "00000000000000000000000110", ClaimResult(0, 2, state, None))
    recorder = PersistenceRecorder(owned, "00000000000000000000000110")
    with pytest.raises(PersistenceError) as error:
        await recorder.on_tool_started("00000000000000000000000110", call, 1)
    assert error.value.code == "submission_missing"
    assert await store.get_tool_calls("00000000000000000000000110") == []


@pytest.mark.parametrize("streaming", [False, True])
async def test_single_call_renews_only_during_provider_steps(store, monkeypatch, streaming):
    from datetime import timedelta

    from dendrux.loops import SingleCall
    from dendrux.types import RunEventType, RunStatus

    touches = []
    renewed = asyncio.Event()
    contexts = []
    original = store.touch_progress

    async def touch(run_id, *, claim_epoch=None):
        result = await original(run_id, claim_epoch=claim_epoch)
        touches.append(1)
        renewed.set()
        return result

    async def wait_for_renewals(count):
        target = len(touches) + count
        async with asyncio.timeout(60):
            while len(touches) < target:
                await renewed.wait()
                renewed.clear()

    class SlowModel(MockLLM):
        async def complete(self, *args, **kwargs):
            contexts.append(get_delegation_context())
            await wait_for_renewals(2)
            return await super().complete(*args, **kwargs)

        async def complete_stream(self, *args, **kwargs):
            from dendrux.types import StreamEvent, StreamEventType

            response = await self.complete(*args, **kwargs)
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=response.text)
            yield StreamEvent(type=StreamEventType.DONE, raw=response)

    monkeypatch.setattr(store, "touch_progress", touch)
    a = Agent(
        prompt="p",
        provider=SlowModel([LLMResponse(text="done")]),
        loop=SingleCall(),
        state_store=store,
        strict_tenancy=False,
    )
    a.lease_interval = timedelta(seconds=5)
    if streaming:
        async with a.stream("go") as events:
            async for event in events:
                if event.type == RunEventType.TEXT_DELTA:
                    # The whole provider stream owns the ticker while yielded.
                    await wait_for_renewals(1)
            assert event.type == RunEventType.RUN_COMPLETED
    else:
        assert (await a.run("go")).status == RunStatus.SUCCESS
    assert contexts and all(ctx._lease_ticker[0] is None for ctx in contexts)
    assert get_delegation_context() is None


async def test_dynamic_gate_is_durable_before_approval_notification(store):
    from dendrux import ToolDecision
    from dendrux.loops.base import BaseNotifier

    seen = []

    class Observer(BaseNotifier):
        async def on_governance_event(
            self, run_id, event_type, iteration, data, *, correlation_id=None
        ):
            if event_type == "approval.requested":
                state = await store.get_loop_state(run_id)
                seen.append((state["phase"], state["pending"][0]["phase"]))

    @tool()
    async def change() -> str:
        return "ok"

    async def gate(call):
        return ToolDecision.require_approval("review")

    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("change")])]),
        tools=[change],
        before_tool=gate,
        state_store=store,
        strict_tenancy=False,
    )
    await a.run("go", notifier=Observer())
    assert seen == [("awaiting_approval", "gated")]
