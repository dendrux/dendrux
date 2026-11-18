"""0.8 fan-out adoption (T4c): claim first, then N completions in call order (spec §7.3).

A parent is parked at a v4 wait through the real ``bind_managed_children``; its children are
driven terminal with ordinary store writes; adoption then runs through ``adopt_settled``
(claim + completions) or, after a cut, through the internal completion-resume entry
``finish_fanout_adoption`` under a fresh recovery claim. Tests count side effects (transform
invocations, observations, events) with ``tests/_helpers/fanout_effects.py``, not only final
state. Behaviour D1–D5 runs on SQLite and on Postgres (``DENDRUX_TEST_PG_URL``, a private
``*_test`` database).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux._managed_wait import ManagedFanout, ManagedWait
from dendrux._meta_keys import BLOCKED_META_KEY
from dendrux.db.models import AgentRun, ReactTrace, RunEvent, ToolCallRecord
from dendrux.errors import ManagedStateError
from dendrux.runtime import _reclaim
from dendrux.runtime._managed_adoption import (
    adopt_fanout,
    adopt_settled,
    finish_fanout_adoption,
)
from dendrux.runtime._managed_budget import UNKNOWN, UNPRICED_ACCEPTED
from dendrux.runtime._managed_fanout_store import BindManagedChildrenRequest, ManagedChildSpec
from dendrux.runtime._managed_tree import LeafNode, WaitNode
from dendrux.runtime.persistence import message_trace
from dendrux.runtime.state import (
    MANAGED_HEADER_KEY,
    ClaimResult,
    EventWrite,
    LoopStatePatch,
    ManagedHeader,
    claimed_store,
)
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import (
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunStatus,
    ToolCall,
    UsageStats,
    _usage_to_dict,
    generate_ulid,
)
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM

TENANT = "t1"
EMAIL_A = "ada.sentinel@example.com"
EMAIL_B = "bob.sentinel@example.org"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


class CrashError(Exception):
    """Stands in for a process dying at a chosen boundary."""


# ---------------------------------------------------------------------------
# Building a parked v4 fan-out through the real store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Child:
    """How one bound child ends: its terminal status (``running``: left unfinished), its answer
    and saved checkpoint extras."""

    answer: str | None = "ok"
    status: str = RunStatus.SUCCESS.value
    state: dict[str, Any] | None = None


@dataclass(frozen=True)
class Fanout:
    run_id: str
    pause_id: str
    calls: tuple[ToolCall, ...]
    child_ids: tuple[str, ...]


def reviewer() -> Agent:
    return Agent(
        name="reviewer",
        description="Reviews one draft and reports.",
        prompt="You review.",
        provider=ScriptedLLM([]),
    )


def lead(llm: Any = None, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="You lead.",
        provider=llm or ScriptedLLM([]),
        subagents=[reviewer()],
        on_governance_change="raise",
        **kwargs,
    )


def _parent_state(run_id: str, calls: list[ToolCall]) -> dict[str, Any]:
    """The checkpoint a ReAct parent writes before it binds its managed batch (0.7 shape)."""
    return {
        "v": 1,
        "loop": "react",
        MANAGED_HEADER_KEY: ManagedHeader(root_run_id=run_id, path=()).to_dict(),
        "managed_limits": {"max_managed_runs": 128},
        UNPRICED_ACCEPTED: 0,
        "iteration": 1,
        "steps_count": 1,
        "usage": _usage_to_dict(UsageStats()),
        "tool_counts": {},
        "provider_kwargs": {},
        "pending": [
            {
                "call_id": c.id,
                "tool": c.name,
                "phase": "dispatchable",
                "decision_id": None,
                "order_index": 3 + i,
                "target": "agent",
            }
            for i, c in enumerate(calls)
        ],
        "submission": None,
        "phase": "dispatch",
        "skills": {"session_ref": None, "pins": {}},
        "trace_order_offset": 3 + len(calls),
    }


def _spec(call: ToolCall, child_id: str, child: Child) -> ManagedChildSpec:
    task = str(call.params["task"])
    return ManagedChildSpec(
        tool_call_id=call.id,
        row_id=generate_ulid(),
        provider_tool_call_id=call.provider_tool_call_id,
        tool_name=call.name,
        params=dict(call.params),
        child_run_id=child_id,
        child_agent_name="reviewer",
        child_path=("reviewer",),
        child_input_data={"input": task},
        child_loop_state={"v": 1, "loop": "react", **(child.state or {})},
        intent_event=EventWrite(
            generate_ulid(),
            "tool.started",
            1,
            {"tool_name": call.name, "call_id": call.id, "idempotent": False},
            call.id,
            None,
        ),
        child_seed_traces=(message_trace(Message(Role.USER, task), 0, kind="seed"),),
    )


async def parked(store: Any, root: Agent, children: list[Child]) -> Fanout:
    """A managed root parked at a v4 wait over ``children``, each driven to its terminal state."""
    run_id = generate_ulid()
    calls = [
        ToolCall(
            name="reviewer",
            params={"task": f"brief {i}"},
            id=generate_ulid(),
            provider_tool_call_id=f"r{i}",
        )
        for i in range(len(children))
    ]
    seeds = [
        message_trace(Message(Role.SYSTEM, root.prompt), 0, kind="seed"),
        message_trace(Message(Role.USER, "go"), 1, kind="seed"),
    ]
    await store.create_run(
        run_id,
        root.name,
        tenant_id=TENANT,
        input_data={"input": "go"},
        seed_traces=seeds,
        meta={"dendrux.governance": governance_snapshot(root)},
    )
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state=_parent_state(run_id, calls),
        pii_mapping=None,
        trace=message_trace(Message(Role.ASSISTANT, "", tool_calls=calls), 2),
    )
    child_ids = [generate_ulid() for _ in children]
    wait = PauseState(
        agent_name=root.name,
        pending_tool_calls=[replace(c, params={}) for c in calls],
        pending_targets={c.id: "agent" for c in calls},
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=2,
        usage=UsageStats(),
        pause_reason="managed_wait",
        pause_id=generate_ulid(),
        schema_version=4,
        loop_state_ref=True,
        managed_fanout=ManagedFanout(
            tuple(ManagedWait(c.id, child) for c, child in zip(calls, child_ids, strict=True))
        ),
    )
    request = BindManagedChildrenRequest(
        tenant_id=TENANT,
        root_run_id=run_id,
        parent_run_id=run_id,
        parent_claim_epoch=0,
        iteration_index=1,
        parent_wait=wait,
        parent_patch=LoopStatePatch(set_fields={"iteration": 1}),
        children=tuple(
            _spec(c, cid, ch) for c, cid, ch in zip(calls, child_ids, children, strict=True)
        ),
        max_parallel=32,
    )
    assert (await store.bind_managed_children(request)).outcome == "created"
    for child_id, child in zip(child_ids, children, strict=True):
        if child.status == RunStatus.RUNNING.value:
            continue  # left running for the test to finish
        assert await store.finalize_run(
            child_id,
            status=child.status,
            answer=child.answer,
            claim_epoch=0,
            expected_current_status=RunStatus.RUNNING.value,
        )
    return Fanout(run_id, str(wait.pause_id), tuple(calls), tuple(child_ids))


def node_of(fanout: Fanout, root: Agent) -> WaitNode:
    child = root.subagents[0]
    return WaitNode(
        run_id=fanout.run_id,
        root_run_id=fanout.run_id,
        tenant_id=TENANT,
        agent=root,
        agent_path=(),
        pause_id=fanout.pause_id,
        version=4,
        iteration=1,
        children=tuple(
            LeafNode(
                run_id=child_id,
                agent=child,
                agent_path=("reviewer",),
                status=RunStatus.SUCCESS.value,
                call=call,
            )
            for call, child_id in zip(fanout.calls, fanout.child_ids, strict=True)
        ),
    )


# ---------------------------------------------------------------------------
# Cuts and recovery
# ---------------------------------------------------------------------------


def crash_on_completion(monkeypatch: Any, store: Any, at: int) -> None:
    """The ``at``-th completion (1-based) dies before it commits; earlier ones commit."""
    real = store.complete_tool_call
    seen = {"n": 0}

    async def complete(*args: Any, **kwargs: Any) -> Any:
        seen["n"] += 1
        if seen["n"] == at:
            raise CrashError
        return await real(*args, **kwargs)

    monkeypatch.setattr(store, "complete_tool_call", complete)


async def adopt_with_cut(
    store: Any, engine: Any, monkeypatch: Any, root: Agent, fanout: Fanout, cut: int | None
) -> ClaimResult:
    """Adopt live; with a cut, die at completion ``cut`` and finish in a "fresh process"."""
    if cut is None:
        claim = await adopt_settled(store, node_of(fanout, root))
        assert isinstance(claim, ClaimResult)
        return claim
    with monkeypatch.context() as patch:
        crash_on_completion(patch, store, cut)
        with pytest.raises(CrashError):
            await adopt_settled(store, node_of(fanout, root))
    return await recover(store, engine, fanout.run_id, root)


async def recover(store: Any, engine: Any, run_id: str, agent: Agent) -> ClaimResult:
    """A new recovery claim (new epoch) and the internal completion-resume entry."""
    item = await expire(store, engine, run_id)
    claim = await store.reclaim_run(
        run_id,
        tenant_id=TENANT,
        agent_name=agent.name,
        expected_claim_epoch=item.claim_epoch,
        governance=governance_snapshot(agent),
        governance_policy="raise",
        skill_pins={},
        loop_type="react",
    )
    assert isinstance(claim, ClaimResult)
    return await finish_fanout_adoption(agent, claimed_store(store, run_id, claim), run_id, claim)


@pytest.fixture(autouse=True)
def v4_continuation(monkeypatch: Any) -> None:
    """Contract gap G2 (T6 owns ``_reclaim.py``): continue from a stale v4 wait as from v3.

    After adoption the parent's raw pause still holds the v4 wait; the 0.7
    ``_continuation_base`` reads only a v3 one strictly and refuses v4 as
    ``pause_format_unsupported``. This stand-in is the same rule for v4: decode
    strictly, keep only the continuation fields. Remove it when T6 lands the fix.
    """
    original = _reclaim._continuation_base

    def base(raw: dict[str, Any], run_id: str) -> PauseState:
        if raw.get("schema_version") != 4:
            return original(raw, run_id)
        wait = PauseState.from_managed_dict(raw, run_id=run_id)
        return replace(
            wait,
            managed_fanout=None,
            schema_version=2,
            pause_reason=None,
            pending_tool_calls=[],
            pending_targets={},
        )

    monkeypatch.setattr(_reclaim, "_continuation_base", base)


# ---------------------------------------------------------------------------
# Reading durable effects
# ---------------------------------------------------------------------------


async def _rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def observations(engine: Any, run_id: str) -> list[Any]:
    traces = await _rows(engine, ReactTrace, ReactTrace.agent_run_id == run_id)
    return sorted((t for t in traces if t.role == "tool"), key=lambda t: t.order_index)


async def intents(engine: Any, run_id: str, fanout: Fanout) -> list[Any]:
    rows = {
        r.tool_call_id: r
        for r in await _rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == run_id)
    }
    return [rows[c.id] for c in fanout.calls]


async def completed_events(engine: Any, run_id: str) -> list[Any]:
    events = await _rows(engine, RunEvent, RunEvent.agent_run_id == run_id)
    return [e for e in events if e.event_type == "tool.completed"]


async def parent_row(engine: Any, run_id: str) -> Any:
    return (await _rows(engine, AgentRun, AgentRun.id == run_id))[0]


def transform_counts(effects: FanoutEffects, fanout: Fanout) -> list[int]:
    counts = effects.counts()
    return [counts.transform_calls(c.id) for c in fanout.calls]


async def assert_adopted_once(engine: Any, fanout: Fanout, claim: ClaimResult) -> None:
    """One resolved row, one observation and one completed event per call; the wait cleared."""
    rows = await intents(engine, fanout.run_id, fanout)
    assert [r.outcome for r in rows] != ["started"] * len(rows)
    assert all(r.outcome in ("completed", "failed") and r.content_digest for r in rows)
    assert [r.meta["managed"]["child_run_id"] for r in rows] == list(fanout.child_ids)
    seen = await observations(engine, fanout.run_id)
    assert [t.order_index for t in seen] == [3 + i for i in range(len(fanout.calls))]
    assert len(await completed_events(engine, fanout.run_id)) == len(fanout.calls)
    state = claim.loop_state or {}
    assert state.get("managed_wait") is None and state.get("submission") is None
    assert state.get("managed_admitted") is None and state.get("pending") == []
    assert state.get("phase") == "model_next"
    row = await parent_row(engine, fanout.run_id)
    assert row.status == RunStatus.RUNNING.value and row.claim_epoch == claim.claim_epoch


CUTS = {"none": None, "before_row_1": 1, "after_row_1": 2}


# ---------------------------------------------------------------------------
# D1, PII placeholders are numbered across calls and survive recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", list(CUTS))
async def test_d1_each_child_email_gets_its_own_placeholder(
    store: Any, engine: Any, monkeypatch: Any, cut: str
) -> None:
    from dendrux.guardrails import PII

    received: list[str] = []

    @tool()
    async def notify(to: str) -> str:
        """Send a note."""
        received.append(to)
        return "sent"

    def make(llm: Any = None) -> Agent:
        return lead(llm, guardrails=[PII(action="redact")], tools=[notify])

    fanout = await parked(store, make(), [Child(f"contact {EMAIL_A}"), Child(f"contact {EMAIL_B}")])
    claim = await adopt_with_cut(store, engine, monkeypatch, make(), fanout, CUTS[cut])

    await assert_adopted_once(engine, fanout, claim)
    first, second = (t.content for t in await observations(engine, fanout.run_id))
    assert "<<EMAIL_ADDRESS_1>>" in first and "<<EMAIL_ADDRESS_2>>" in second
    assert EMAIL_A not in first + second and EMAIL_B not in first + second
    assert await store.get_pii_mapping(fanout.run_id) == {
        "<<EMAIL_ADDRESS_1>>": EMAIL_A,
        "<<EMAIL_ADDRESS_2>>": EMAIL_B,
    }

    llm = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="notify",
                        params={"to": "<<EMAIL_ADDRESS_2>>"},
                        provider_tool_call_id="n2",
                    ),
                    ToolCall(
                        name="notify",
                        params={"to": "<<EMAIL_ADDRESS_1>>"},
                        provider_tool_call_id="n1",
                    ),
                ]
            ),
            LLMResponse(text="done"),
        ]
    )
    result = await _reclaim.continue_claimed(make(llm), store, fanout.run_id, claim, notifier=None)
    assert result.status == RunStatus.SUCCESS
    assert received == [EMAIL_B, EMAIL_A]
    sent = json.dumps([m.content for m in llm.messages[0]])
    assert "<<EMAIL_ADDRESS_1>>" in sent and EMAIL_A not in sent and EMAIL_B not in sent


# ---------------------------------------------------------------------------
# D2, blocked provenance is per call and survives recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["blocked_first", "clean_first"])
@pytest.mark.parametrize("cut", list(CUTS))
async def test_d2_blocked_provenance_is_per_call(
    store: Any, engine: Any, monkeypatch: Any, cut: str, order: str
) -> None:
    from dendrux.guardrails import SecretDetection

    def make(llm: Any = None) -> Agent:
        return lead(llm, guardrails=[SecretDetection(action="block")])

    blocked, clean = Child(f"key {AWS_KEY}"), Child("clean report")
    children = [blocked, clean] if order == "blocked_first" else [clean, blocked]
    fanout = await parked(store, make(), children)
    position = 0 if order == "blocked_first" else 1
    if CUTS[cut] == 2:
        with monkeypatch.context() as patch:
            crash_on_completion(patch, store, 2)
            with pytest.raises(CrashError):
                await adopt_settled(store, node_of(fanout, make()))
        marked = (await parent_row(engine, fanout.run_id)).meta.get(BLOCKED_META_KEY) is True
        assert marked is (position == 0), "the mark commits with the blocked call's own row"
        claim = await recover(store, engine, fanout.run_id, make())
    else:
        claim = await adopt_with_cut(store, engine, monkeypatch, make(), fanout, CUTS[cut])

    await assert_adopted_once(engine, fanout, claim)
    seen = [t.content for t in await observations(engine, fanout.run_id)]
    assert AWS_KEY not in json.dumps(seen)
    assert "clean report" in seen[1 - position] and "clean report" not in seen[position]
    row = await parent_row(engine, fanout.run_id)
    assert (row.meta or {}).get(BLOCKED_META_KEY) is True

    llm = ScriptedLLM([LLMResponse(text="never")])
    result = await _reclaim.continue_claimed(make(llm), store, fanout.run_id, claim, notifier=None)
    assert result.status == RunStatus.ERROR and llm.calls == 0
    assert (await parent_row(engine, fanout.run_id)).failure_reason == "guardrail_blocked"


# ---------------------------------------------------------------------------
# D3, accepted-unpriced counts only rise; a usage fault is sticky
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", list(CUTS))
async def test_d3_accepted_unpriced_count_never_drops(
    store: Any, engine: Any, monkeypatch: Any, cut: str
) -> None:
    fault = {"code": "managed_usage_unknown", "origin_run_id": generate_ulid()}
    children = [
        Child(state={UNPRICED_ACCEPTED: 10}),
        Child(state={UNPRICED_ACCEPTED: 3, UNKNOWN: fault}),
        Child(state={UNPRICED_ACCEPTED: 1}),
    ]
    fanout = await parked(store, lead(), children)
    claim = await adopt_with_cut(store, engine, monkeypatch, lead(), fanout, CUTS[cut])

    await assert_adopted_once(engine, fanout, claim)
    state = await store.get_loop_state(fanout.run_id)
    assert state[UNPRICED_ACCEPTED] == 10 == claim.loop_state[UNPRICED_ACCEPTED]
    assert state[UNKNOWN]["code"] == "managed_usage_unknown"
    assert state[UNKNOWN]["origin_run_id"] == fault["origin_run_id"]


async def test_d3_count_is_persisted_with_each_completion(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Every cut after row 1 holds A's 10; the fault arrives with B's own row and stays."""
    children = [Child(state={UNPRICED_ACCEPTED: 10}), Child(state={UNPRICED_ACCEPTED: 3}), Child()]
    fanout = await parked(store, lead(), children)
    persisted: list[tuple[int, Any]] = []
    real = store.complete_tool_call

    async def complete(*args: Any, **kwargs: Any) -> Any:
        await real(*args, **kwargs)
        state = await store.get_loop_state(fanout.run_id)
        persisted.append((state[UNPRICED_ACCEPTED], state.get("managed_wait") is None))

    monkeypatch.setattr(store, "complete_tool_call", complete)
    await adopt_settled(store, node_of(fanout, lead()))
    assert persisted == [(10, False), (10, False), (10, True)]


# ---------------------------------------------------------------------------
# D4, two contending adopters: one claim, N transforms in all
# ---------------------------------------------------------------------------


async def test_d4_contending_adopters_run_transforms_once(store: Any, engine: Any) -> None:
    import asyncio

    effects = FanoutEffects()
    gate = asyncio.Event()
    arrived: list[str] = []

    def make() -> Agent:
        return lead(tool_result_transform=effects.transform())

    fanout = await parked(store, make(), [Child("a"), Child("b"), Child("c")])
    real = store.submit_and_claim

    async def contend(*args: Any, **kwargs: Any) -> Any:
        arrived.append("x")
        if len(arrived) == 2:
            gate.set()
        await gate.wait()  # both adopters reach the claim before either takes it
        return await real(*args, **kwargs)

    store.submit_and_claim = contend
    try:
        outcomes = await asyncio.gather(
            adopt_settled(store, node_of(fanout, make())),
            adopt_settled(store, node_of(fanout, make())),
        )
    finally:
        del store.submit_and_claim
    winners = [o for o in outcomes if o is not None]
    assert len(winners) == 1 and len(arrived) == 2
    assert transform_counts(effects, fanout) == [1, 1, 1]
    await assert_adopted_once(engine, fanout, winners[0])
    assert (await parent_row(engine, fanout.run_id)).claim_epoch == 2


async def test_d4_loser_after_the_adoption_writes_nothing(store: Any, engine: Any) -> None:
    effects = FanoutEffects()
    root = lead(tool_result_transform=effects.transform())
    fanout = await parked(store, root, [Child("a"), Child("b")])
    claim = await adopt_settled(store, node_of(fanout, root))
    assert claim is not None
    before = [(t.id, t.content) for t in await observations(engine, fanout.run_id)]
    assert await adopt_settled(store, node_of(fanout, root)) is None
    assert transform_counts(effects, fanout) == [1, 1]
    assert [(t.id, t.content) for t in await observations(engine, fanout.run_id)] == before
    assert (await parent_row(engine, fanout.run_id)).claim_epoch == 2


# ---------------------------------------------------------------------------
# D5, a crash after a transform and before its row: only that call is transformed again
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [1, 2, 3])
async def test_d5_crash_after_a_transform_reruns_only_that_call(
    store: Any, engine: Any, monkeypatch: Any, cut: int
) -> None:
    effects = FanoutEffects()

    def make(llm: Any = None) -> Agent:
        return lead(llm, tool_result_transform=effects.transform(lambda ctx: {"t": ctx.result}))

    fanout = await parked(store, make(), [Child("a"), Child("b"), Child("c")])
    with monkeypatch.context() as patch:
        crash_on_completion(patch, store, cut)
        with pytest.raises(CrashError):
            await adopt_settled(store, node_of(fanout, make()))
    rows = await intents(engine, fanout.run_id, fanout)
    assert [r.outcome for r in rows] == ["completed"] * (cut - 1) + ["started"] * (4 - cut)
    committed = [(t.id, t.content) for t in await observations(engine, fanout.run_id)]
    assert (await parent_row(engine, fanout.run_id)).status == RunStatus.RUNNING.value

    claim = await recover(store, engine, fanout.run_id, make())

    assert transform_counts(effects, fanout) == [2 if i == cut - 1 else 1 for i in range(3)]
    await assert_adopted_once(engine, fanout, claim)
    seen = await observations(engine, fanout.run_id)
    assert [(t.id, t.content) for t in seen[: cut - 1]] == committed
    assert [json.loads(t.content) for t in seen] == [{"t": "a"}, {"t": "b"}, {"t": "c"}]
    llm = ScriptedLLM([LLMResponse(text="root done")])
    result = await _reclaim.continue_claimed(make(llm), store, fanout.run_id, claim, notifier=None)
    assert result.status == RunStatus.SUCCESS and llm.calls == 1
    tools = [m for m in llm.messages[0] if m.role == Role.TOOL]
    assert [m.call_id for m in tools] == [c.id for c in fanout.calls]


# ---------------------------------------------------------------------------
# Refusals: identity, order and state checks write nothing
# ---------------------------------------------------------------------------


async def test_claim_refuses_bindings_the_wait_does_not_hold(store: Any, engine: Any) -> None:
    effects = FanoutEffects()
    root = lead(tool_result_transform=effects.transform())
    fanout = await parked(store, root, [Child("a"), Child("b")])
    other = await parked(store, root, [Child("x"), Child("y")])
    node = node_of(fanout, root)
    swapped = replace(node.children[1], run_id=other.child_ids[1])
    with pytest.raises(ManagedStateError) as refused:
        await adopt_settled(store, replace(node, children=(node.children[0], swapped)))
    assert refused.value.reason == "binding_conflict"
    row = await parent_row(engine, fanout.run_id)
    assert (row.status, row.claim_epoch) == (RunStatus.PAUSED.value, 1)
    assert "submission" not in row.pause_data and effects.counts().transform_calls() == 0


async def test_a_child_whose_header_names_another_call_is_refused(store: Any, engine: Any) -> None:
    from sqlalchemy import update

    effects = FanoutEffects()
    root = lead(tool_result_transform=effects.transform())
    fanout = await parked(store, root, [Child("a"), Child("b")])
    state = await store.get_loop_state(fanout.child_ids[1])
    state[MANAGED_HEADER_KEY]["invocation"]["tool_call_id"] = fanout.calls[0].id
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == fanout.child_ids[1]).values(loop_state=state)
        )
    with pytest.raises(ManagedStateError) as refused:
        await adopt_settled(store, node_of(fanout, root))
    assert refused.value.reason == "child_mismatch"
    assert [r.outcome for r in await intents(engine, fanout.run_id, fanout)] == [
        "completed",
        "started",
    ]
    assert transform_counts(effects, fanout) == [1, 0]


async def test_adoption_needs_the_v4_marker_in_the_claim(store: Any, engine: Any) -> None:
    root = lead()
    fanout = await parked(store, root, [Child("a"), Child("b")])
    claim = await store.claim_paused_run(fanout.run_id, expected_status=RunStatus.PAUSED.value)
    assert isinstance(claim, ClaimResult)
    with pytest.raises(ManagedStateError) as refused:
        await adopt_fanout(store, node_of(fanout, root), claim)
    assert refused.value.reason == "binding_conflict"
    assert [r.outcome for r in await intents(engine, fanout.run_id, fanout)] == ["started"] * 2


# ---------------------------------------------------------------------------
# The claim's stored JSON on the real store (both forms), and nothing but ids
# ---------------------------------------------------------------------------


async def test_real_store_keeps_the_claim_json_exactly(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    secret = "SENTINEL-report-7c1e"
    fanout = await parked(store, lead(), [Child(secret), Child(secret), Child(secret)])
    ids = [c.id for c in fanout.calls]
    with monkeypatch.context() as patch:
        crash_on_completion(patch, store, 1)  # the claim committed; no completion did
        with pytest.raises(CrashError):
            await adopt_settled(store, node_of(fanout, lead()))
    row = await parent_row(engine, fanout.run_id)
    descriptor = row.loop_state["submission"]
    assert descriptor == {
        "submission_id": descriptor["submission_id"],
        "kind": "tool_results",
        "pause_id": fanout.pause_id,
        "payload_ref": {"pause_data_key": "submission"},
        "call_ids": ids,
        "reserved_order_index": 2,
        "managed_fanout_adoption": {"v": 1},
    }
    assert row.pause_data["submission"] == {
        "submission_id": descriptor["submission_id"],
        "kind": "tool_results",
        "pause_id": fanout.pause_id,
        "call_ids": ids,
        "payload": {"managed_fanout_adoption": {"v": 1, "calls": ids}},
    }
    # Call order lives in lists (JSONB may reorder object keys, never list items).
    assert [b["tool_call_id"] for b in row.pause_data["managed_fanout"]["bindings"]] == ids
    assert secret not in json.dumps([row.pause_data, row.loop_state])
    assert (row.status, row.claim_epoch) == (RunStatus.RUNNING.value, 2)
    assert [r.outcome for r in await intents(engine, fanout.run_id, fanout)] == ["started"] * 3


# ---------------------------------------------------------------------------
# Cost: a fixed claim plus a constant per call (no N+1), counted per dialect
# ---------------------------------------------------------------------------


async def test_adoption_statements_are_a_constant_per_call(store: Any, engine: Any) -> None:
    from sqlalchemy import event

    from dendrux.guardrails import PII

    root = lead(guardrails=[PII(action="redact")])
    totals: dict[int, int] = {}
    for n in (2, 3, 4, 8):
        fanout = await parked(store, root, [Child(f"report {i}") for i in range(n)])
        seen = {"n": 0}

        def count(*_: Any, seen: dict[str, int] = seen) -> None:
            seen["n"] += 1

        event.listen(engine.sync_engine, "before_cursor_execute", count)
        try:
            assert await adopt_settled(store, node_of(fanout, root)) is not None
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", count)
        totals[n] = seen["n"]
    per_call = totals[3] - totals[2]
    assert [totals[n] - totals[2] for n in (3, 4, 8)] == [per_call, 2 * per_call, 6 * per_call]
    # Measured: SQLite 9 + 9N, Postgres 9 + 7N (cancel check, three admission reads and the
    # claim transaction; one intent read; per call two child reads, one mapping read and
    # one completion transaction).
    assert per_call <= 9 and totals[2] - 2 * per_call <= 9, totals
