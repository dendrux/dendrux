"""Runtime consumers rebuild the bounded, effective conversation."""

import json

import pytest

from dendrux.errors import PersistenceError, RunNotReclaimableError
from dendrux.runtime import replay
from dendrux.types import Limits, PauseState, generate_ulid


async def test_rebuild_uses_rewrite_and_excludes_state(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "rebuild")
    await store.save_trace(run_id, "user", "old", order_index=0)
    await store.save_trace(
        run_id,
        "rewrite",
        json.dumps([{"role": "user", "content": "summary", "meta": {}}]),
        order_index=1,
        trace_kind="rewrite",
    )
    await store.save_trace(
        run_id, "state", '{"kind":"skills_session","v":1}', order_index=2, trace_kind="state"
    )
    await store.save_trace(run_id, "assistant", "answer", order_index=3)
    messages = await replay.rebuild_history(store, run_id)
    assert [m.content for m in messages.messages] == ["summary", "answer"]


@pytest.mark.parametrize("limits", [Limits(max_rebuild_rows=2), Limits(max_rebuild_bytes=8)])
async def test_a18_rebuild_refuses_overflow(store, limits, caplog):
    run_id = generate_ulid()
    await store.create_run(run_id, "rebuild")
    for i in range(3):
        await store.save_trace(run_id, "user", "a realistic message", order_index=i)
    with pytest.raises(RunNotReclaimableError) as caught:
        await replay.rebuild_history(store, run_id, limits=limits)
    assert caught.value.reason == "history_too_large"
    assert any(getattr(r, "run_id", None) == run_id for r in caplog.records)


async def test_a18_realistic_rebuild_crosses_pages(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "rebuild")
    for i in range(501):
        await store.save_trace(run_id, "user", str(i), order_index=i)
    messages = await replay.rebuild_history(store, run_id)
    assert [m.content for m in messages.messages] == [str(i) for i in range(501)]


def test_pause_versions():
    pause = PauseState.from_dict(
        {
            "agent_name": "test",
            "schema_version": 2,
            "image_view_ref": "image",
            "loop_state_ref": True,
        }
    )
    encoded = pause.to_dict()
    assert encoded["schema_version"] == 2
    assert encoded["image_view_ref"] == "image"
    assert encoded["loop_state_ref"] is True
    assert not {"history", "steps", "image_view"} & encoded.keys()
    with pytest.raises(PersistenceError) as caught:
        PauseState.from_dict({"agent_name": "test", "schema_version": 3})
    assert caught.value.code == "pause_format_unsupported"


@pytest.mark.parametrize("streaming", [False, True])
async def test_runtime_seeds_and_full_checkpoint(store, engine, streaming):
    from sqlalchemy import update

    from dendrux.agent import Agent
    from dendrux.db.models import DendruxSetting
    from dendrux.llm.mock import MockLLM
    from dendrux.tool import tool
    from dendrux.types import LLMResponse, Message, Role, ToolCall

    async with engine.begin() as conn:
        await conn.execute(
            update(DendruxSetting).where(DendruxSetting.key == "storage_format").values(value="2")
        )
    observed = []

    @tool()
    async def inspect_checkpoint() -> str:
        """Observe the committed assistant boundary."""
        from dendrux.runtime.context import get_delegation_context

        ctx = get_delegation_context()
        observed.append(await store.get_loop_state(ctx.run_id))
        return "ok"

    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="inspect_checkpoint", params={})]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        name="checkpoint",
        prompt="system",
        provider=provider,
        state_store=store,
        tools=[inspect_checkpoint],
        strict_tenancy=False,
    )
    history = [
        Message(role=Role.USER, content="earlier"),
        Message(role=Role.ASSISTANT, content="prior answer"),
    ]
    if streaming:
        stream = agent.stream("now", history=history)
        async for _ in stream:
            pass
        result = stream.result
    else:
        result = await agent.run("now", history=history)
    traces = await store.get_traces_after(result.run_id, after_order_index=None)
    assert [t.content for t in traces if t.trace_kind == "seed"] == [
        "system",
        "earlier",
        "prior answer",
        "now",
    ]
    assert observed and observed[0] is not None
    assert observed[0]["steps_count"] == 1
    assert observed[0]["iteration"] == 1
    assert observed[0]["phase"] == "dispatch"
    assert len(observed[0]["pending"]) == 1
    assert observed[0]["pending"][0]["tool"] == "inspect_checkpoint"
    assert observed[0]["pending"][0]["order_index"] > 0
    assert set(observed[0]) == {
        "v",
        "loop",
        "phase",
        "iteration",
        "trace_order_offset",
        "steps_count",
        "usage",
        "budget",
        "tool_counts",
        "pii",
        "provider_kwargs",
        "skills",
        "pending",
        "submission",
        "image_view_ref",
        "clarification_ref",
    }


async def test_checkpoint_accepts_64_realistic_pending_entries(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "bound")
    pending = [
        {
            "call_id": generate_ulid(),
            "tool": "lookup_customer",
            "phase": "approved",
            "decision_id": generate_ulid() + ":" + generate_ulid(),
            "order_index": i + 20,
            "target": "server",
        }
        for i in range(64)
    ]
    checkpoint = {
        "v": 1,
        "loop": "react",
        "pending": pending,
        "provider_kwargs": {"model": "model", "max_tokens": 4096},
    }
    assert 8192 < len(json.dumps(checkpoint).encode()) < 32768
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state=checkpoint, pii_mapping=None, trace=None
    )
    assert await store.get_loop_state(run_id) == checkpoint


async def test_rebuild_steps_survive_compaction_retry_and_replay(store):
    from dendrux import Agent, tool
    from dendrux.llm.mock import MockLLM
    from dendrux.strategies.native import NativeToolCalling
    from dendrux.types import LLMResponse, Message, Role, ToolCall, _step_to_dict

    @tool()
    async def echo() -> str:
        """Produce an observation."""
        return "observed"

    class Compact(NativeToolCalling):
        def build_messages(self, *, system_prompt, history, tool_defs):
            if len(history) > 2:
                history[:] = [Message(Role.USER, "summary"), *history[-2:]]
            return super().build_messages(
                system_prompt=system_prompt, history=history, tool_defs=tool_defs
            )

    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("echo")]), LLMResponse(text="done")])
    agent = Agent(
        prompt="system", provider=provider, tools=[echo], state_store=store, strict_tenancy=False
    )
    result = await agent.run("superseded input", strategy=Compact())
    await store.save_trace(
        result.run_id,
        "state",
        '{"kind":"skills_session","v":1}',
        order_index=1000,
        trace_kind="state",
    )
    rebuilt = await replay.rebuild_history(store, result.run_id)
    assert [_step_to_dict(s) for s in rebuilt.steps] == [_step_to_dict(s) for s in result.steps]
    assert [m.content for m in rebuilt.messages][:2] == ["system", "summary"]
    for path in ("retry", "replay"):
        follow = MockLLM([LLMResponse(text="next")])
        a = Agent(
            prompt="system", provider=follow, tools=[echo], state_store=store, strict_tenancy=False
        )
        if path == "retry":
            output = await a.retry(result.run_id)
        else:
            output = await a.run("follow up", runs=[result.run_id], replay_last="all")
        restored = await replay.rebuild_history(store, output.run_id)
        contents = [m.content for m in restored.messages]
        assert "summary" in contents
        assert "superseded input" not in contents
        assert '"observed"' in contents


async def test_large_clarification_and_serializer_state_rows(store):
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.state import ClaimResult, claimed_store
    from dendrux.skills import Skill
    from dendrux.skills._session import SkillSession
    from dendrux.strategies.native import NativeToolCalling
    from dendrux.types import AgentStep, Clarification, ImageViewState, LLMResponse

    question = "Choose the next action. " * 100

    class Ask(NativeToolCalling):
        def parse_response(self, response):
            return AgentStep(reasoning=None, action=Clarification(question=question))

    a = Agent(
        prompt="system",
        provider=MockLLM([LLMResponse(text=question)]),
        state_store=store,
        strict_tenancy=False,
        skills=[Skill(name="demo", description="description", body="body")],
    )
    paused = await a.run("start", strategy=Ask(), activate_skills=["demo"])
    checkpoint = await store.get_loop_state(paused.run_id)
    assert checkpoint["phase"] == "awaiting_input"
    row = await store.get_trace(paused.run_id, checkpoint["clarification_ref"])
    assert json.loads(row.content) == {
        "kind": "clarification",
        "v": 1,
        "question": question,
        "pause_message": None,
    }
    row = await store.get_trace(paused.run_id, checkpoint["skills"]["session_ref"])
    session = SkillSession.prepare(a)
    session.restore_snapshot(json.loads(row.content))
    assert session.snapshot() == {"v": 1, "activated": ["demo"]}
    raw = await store.get_pause_state(paused.run_id)
    assert raw["schema_version"] == 2 and "history" not in raw

    run = generate_ulid()
    await store.create_run(run, "image-state")
    recorder = PersistenceRecorder(
        claimed_store(store, run, ClaimResult(0, 2, None, None)),
        run,
        model="mock",
        provider_name="mock",
    )
    image_view = ImageViewState(
        epoch=3,
        sent_through=100,
        placeholders={i: "purged" for i in range(100)},
        described={("-", "a" * 64): "description " * 400},
    )
    ref = await recorder._state_ref("image_view", image_view.to_dict())
    row = await store.get_trace(run, ref)
    encoded = json.loads(row.content)
    assert encoded["v"] == 1 and encoded["kind"] == "image_view"
    assert ImageViewState.from_dict(encoded).to_dict() == image_view.to_dict()


async def test_restore_checkpoint_counters_after_history_compaction(store):
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.state import ClaimResult, claimed_store

    run = generate_ulid()
    await store.create_run(run, "counts")
    saved = {"tool_counts": {"lookup": 19}, "pii": {"counters": {"EMAIL_ADDRESS": 8}}}
    recorder = PersistenceRecorder(claimed_store(store, run, ClaimResult(0, 2, saved, None)), run)
    counts, counters = {"lookup": 1}, {"EMAIL_ADDRESS": 1}
    recorder.bind_loop(lambda: {"tool_counts": counts, "pii_counters": counters}, None)
    assert counts == {"lookup": 19}
    assert counters == {"EMAIL_ADDRESS": 8}
