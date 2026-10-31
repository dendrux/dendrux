"""Checkpoint CPU work and state-row identity remain bounded across recorder restarts."""

from dataclasses import replace
from functools import wraps

import pytest

from dendrux import Agent
from dendrux.errors import PersistenceError
from dendrux.runtime import persistence
from dendrux.runtime.state import ClaimResult, claimed_store
from dendrux.skills import Skill
from dendrux.skills._session import SkillSession
from dendrux.types import ImageViewState, Message, Role, generate_ulid


async def test_unchanged_history_serializes_each_entry_once_per_boundary(store, monkeypatch):
    run_id = generate_ulid()
    messages = [Message(role=Role.USER, content=str(i) + "x" * 131072) for i in range(16)]
    seeds = [
        replace(persistence.message_trace(m, i), trace_kind="seed") for i, m in enumerate(messages)
    ]
    await store.create_run(run_id, "cpu", seed_traces=seeds)
    recorder = persistence.PersistenceRecorder(
        claimed_store(store, run_id, ClaimResult(0, 2, None, None)),
        run_id,
        initial_order_index=len(seeds),
        seed_traces=seeds,
    )
    identity = persistence._view_identity
    serializations = []

    def measured(entry):
        serializations.append(len(entry["content"]))
        return identity(entry)

    monkeypatch.setattr(persistence, "_view_identity", measured)
    for iteration in range(1, 4):
        serializations.clear()
        await recorder.checkpoint_view(messages, iteration)
        assert len(serializations) == len(messages)
        assert sum(serializations) > 2 * 1024 * 1024
    assert all(
        isinstance(position, int) and len(digest) == 64
        for position, digest in recorder._known_history
    )
    assert not [row for row in await store.get_traces(run_id) if row.trace_kind == "rewrite"]


async def test_state_refs_survive_three_recorder_restarts_without_duplicate_rows(
    store, monkeypatch
):
    run_id = generate_ulid()
    await store.create_run(run_id, "states")
    session = SkillSession.prepare(
        Agent(prompt="p", skills=[Skill(name="demo", description="d", body="b")]), explicit=["demo"]
    )
    payloads = {
        "skills_session": session.snapshot(),
        "image_view": ImageViewState(
            epoch=3,
            sent_through=10,
            placeholders={2: "purged"},
            described={("-", "a" * 64): "description"},
        ).to_dict(),
    }
    original = store.get_trace
    reads = []

    @wraps(original)
    async def read(*args, **kwargs):
        reads.append(args[1])
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "get_trace", read)
    first = None
    for _restart in range(4):
        state = await store.get_loop_state(run_id) or {}
        recorder = persistence.PersistenceRecorder(
            claimed_store(store, run_id, ClaimResult(0, 2, state, None)),
            run_id,
            initial_order_index=state.get("trace_order_offset", 0),
        )
        reads.clear()
        refs = {
            kind: await recorder._state_ref(kind, payload) for kind, payload in payloads.items()
        }
        for _ in range(3):
            for kind, payload in payloads.items():
                assert (
                    await recorder._state_ref(kind, dict(reversed(list(payload.items()))))
                    == refs[kind]
                )
        await recorder._checkpoint()
        if first is None:
            first = refs
            assert reads == []
        else:
            assert refs == first
            assert sorted(reads) == sorted(first.values())
    rows = [r for r in await store.get_traces(run_id) if r.trace_kind == "state"]
    assert len(rows) == 2
    # A real content change still allocates one new state row and reference.
    changed = await recorder._state_ref("image_view", ImageViewState(epoch=4).to_dict())
    assert changed != first["image_view"]
    await recorder._checkpoint()
    assert len([r for r in await store.get_traces(run_id) if r.trace_kind == "state"]) == 3


@pytest.mark.parametrize(
    "bad", ["missing", "wrong_kind", "wrong_version", "invalid_json", "not_state"]
)
async def test_invalid_durable_state_reference_fails_closed(store, engine, bad):
    run_id = generate_ulid()
    await store.create_run(run_id, "bad-state")
    ref = generate_ulid()
    if bad != "missing":
        content = {
            "wrong_kind": '{"kind":"skills_session","v":1}',
            "wrong_version": '{"kind":"image_view","v":2}',
            "invalid_json": '{"kind":"image_view","v":1}',
            "not_state": '{"kind":"image_view","v":1}',
        }[bad]
        await store.save_trace(
            run_id,
            "state",
            content,
            order_index=0,
            trace_id=ref,
            trace_kind="turn" if bad == "not_state" else "state",
        )
    if bad == "invalid_json":
        from sqlalchemy import update

        from dendrux.db.models import ReactTrace

        # Simulate corrupted persisted evidence; normal writers reject this JSON.
        async with engine.begin() as conn:
            await conn.execute(
                update(ReactTrace).where(ReactTrace.id == ref).values(content="not json")
            )
    state = {"image_view_ref": ref}
    recorder = persistence.PersistenceRecorder(
        claimed_store(store, run_id, ClaimResult(0, 2, state, None)), run_id, initial_order_index=1
    )
    with pytest.raises(PersistenceError) as caught:
        await recorder._state_ref("image_view", ImageViewState().to_dict())
    assert caught.value.code == "pause_format_unsupported"


async def test_real_resumes_reuse_the_activated_skill_state_row(store):
    from dendrux import tool
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse, ToolCall, ToolResult

    @tool(target="client")
    async def lookup() -> str:
        """Fetch a client value."""
        return ""

    model = MockLLM(
        [
            *[LLMResponse(tool_calls=[ToolCall("lookup")]) for _ in range(3)],
            LLMResponse(text="done"),
        ]
    )

    def agent():
        return Agent(
            name="dedup",
            prompt="p",
            provider=model,
            tools=[lookup],
            state_store=store,
            strict_tenancy=False,
            skills=[Skill(name="demo", description="d", body="b")],
        )

    paused = await agent().run("go", activate_skills=["demo"])
    reference = (await store.get_loop_state(paused.run_id))["skills"]["session_ref"]
    for _ in range(3):
        assert paused.status.value == "waiting_client_tool"
        paused = await agent().resume(
            paused.run_id,
            expected_pause_id=paused.pause_id,
            tool_results=[
                ToolResult(name="lookup", call_id=paused.pending.tool_calls[0].id, payload='"ok"')
            ],
        )
        state = await store.get_loop_state(paused.run_id)
        assert state["skills"]["session_ref"] == reference
        rows = [
            row
            for row in await store.get_traces(paused.run_id)
            if (row.meta or {}).get("state_kind") == "skills_session"
        ]
        assert len(rows) == 1
    assert paused.status.value == "success"
