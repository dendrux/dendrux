"""Fan-out (0.8, T5): the tree walk, routing a root continuation and pending attachment.

A parent parked on a v4 wait has several children. Every leaf's external pause shows on the
root in stable order (``RunResult.pendings``, ``RunStore.get_pendings``); a root-id
continuation with a leaf's token reaches exactly that leaf, runs it in the caller's task and
then unwinds leaf-first: the leaf's parent settles first, then its parent, and so on.

The 0.8 driver (owner tasks, fan-in) is not built yet, so fan-out states are made directly
through the store: the parent is checkpointed with its pending managed calls, the batch is
bound with ``bind_managed_children``, and each child then runs from its bound checkpoint with
its own claim. The adoption step is a fake ``settle_parent`` that satisfies the T0b Protocol
(:class:`FakeSettle`); it claims and completes the wait through ordinary store writes.
Runs on SQLite and Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test`` database).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event

from dendrux import Agent, tool
from dendrux._lifecycle import TERMINAL_STATUS_VALUES, StoredEventType
from dendrux._managed_wait import ManagedFanout, ManagedWait
from dendrux.errors import PauseStatusMismatchError, RunNotFoundError, RunNotPausedError
from dendrux.runtime import _managed_budget, _managed_route, _reclaim
from dendrux.runtime._managed_fanout_store import BindManagedChildrenRequest, ManagedChildSpec
from dendrux.runtime._reclaim import continue_claimed
from dendrux.runtime._trace_shared import message_trace
from dendrux.runtime.persistence import seed_trace_kwargs
from dendrux.runtime.state import (
    MANAGED_HEADER_KEY,
    BindManagedChildRequest,
    ClaimResult,
    EventWrite,
    LoopStatePatch,
    ManagedHeader,
)
from dendrux.store import RunStore
from dendrux.types import (
    ApprovalDecision,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
    UsageStats,
    generate_ulid,
)
from tests._helpers.scripted_provider import ScriptedLLM

if TYPE_CHECKING:
    from dendrux.runtime._managed_tree import SettleParent, WaitNode
    from dendrux.runtime.state import StateStore

TENANT = "t1"
OTHER_TENANT = "t2"
BRIEF = "SENTINEL-fanout-brief-7c1d"
CHILD_TEXT = "SENTINEL-fanout-child-2e9a"
ROOT_TEXT = "root summary"


# ---------------------------------------------------------------------------
# Agents and tools
# ---------------------------------------------------------------------------

published: list[str] = []


@tool()
async def publish(text: str) -> str:
    """Publish a draft."""
    published.append(text)
    return f"published {text}"


@tool(target="client")
async def read_sheet(sheet: str) -> str:
    """Read a sheet in the user's browser."""
    return ""


@pytest.fixture(autouse=True)
def _reset_published() -> None:
    published.clear()


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


def approver(name: str, *responses: LLMResponse, **kwargs: Any) -> Agent:
    """A child that asks approval to publish, then answers."""
    script = [call("publish", {"text": name}, f"{name}-p"), *responses]
    script = script if responses else [*script, LLMResponse(text=f"{CHILD_TEXT} {name}")]
    return Agent(
        name=name,
        description=f"{name} reviews one draft.",
        prompt="You review.",
        provider=ScriptedLLM(script),
        tools=[publish],
        require_approval=["publish"],
        **kwargs,
    )


def client_child(name: str) -> Agent:
    """A child that waits for a client tool, then answers."""
    return Agent(
        name=name,
        description=f"{name} reads one sheet.",
        prompt="You read.",
        provider=ScriptedLLM(
            [call("read_sheet", {"sheet": "S1"}, f"{name}-c"), LLMResponse(text=f"{name} read")]
        ),
        tools=[read_sheet],
    )


def finisher(name: str) -> Agent:
    """A child that answers at once."""
    return Agent(
        name=name,
        description=f"{name} answers.",
        prompt="You answer.",
        provider=ScriptedLLM([LLMResponse(text=f"{CHILD_TEXT} {name}")]),
    )


def root_agent(store: Any, *children: Agent, responses: list[LLMResponse] | None = None) -> Agent:
    return Agent(
        name="lead",
        prompt="You lead.",
        provider=ScriptedLLM(responses or [LLMResponse(text=ROOT_TEXT)]),
        state_store=store,
        subagents=list(children),
    )


# ---------------------------------------------------------------------------
# The fan-out harness (no driver: store writes plus each child's own claim)
# ---------------------------------------------------------------------------


@dataclass
class Fan:
    """One bound fan-out: the parent, its wait generation and its children in call order."""

    parent_id: str
    wait_id: str
    calls: list[str]
    children: list[str]
    results: dict[str, RunResult] = field(default_factory=dict)
    claims: dict[str, ClaimResult] = field(default_factory=dict)
    """The unused execution tokens of children left unrun (an owner task's to use)."""

    def child(self, index: int) -> str:
        return self.children[index]


async def start_root(store: Any, root: Agent, *, tenant: str | None = TENANT) -> str:
    """A running format-2 root at epoch 0, as the runtime creates one."""
    root_id = generate_ulid()
    seeds = seed_trace_kwargs(store, [Message(role=Role.USER, content="go")])
    await store.create_run(
        root_id,
        root.name,
        input_data={"input": "go"},
        tenant_id=tenant,
        model="scripted",
        seed_traces=seeds["seed_traces"],
    )
    await store.save_run_event(root_id, event_type=StoredEventType.RUN_STARTED.value)
    return root_id


async def fan_out(
    store: Any,
    root: Agent,
    parent_id: str,
    children: list[Agent],
    *,
    tenant: str | None = TENANT,
    root_id: str | None = None,
    path: tuple[str, ...] = (),
    epoch: int = 0,
    run: tuple[int, ...] | None = None,
) -> Fan:
    """Park ``parent_id`` on a v4 wait over ``children`` (call order), then run the children.

    ``run`` lists the indexes to run from their bound checkpoint now (all by default); a
    child left unrun stays RUNNING with its claim unused, as an owner task would hold it.
    """
    root_id = root_id or parent_id
    calls = [
        ToolCall(name=c.name, params={"task": f"{BRIEF} {i}"}, id=generate_ulid())
        for i, c in enumerate(children)
    ]
    await _checkpoint_parent(store, parent_id, root_id, calls, epoch=epoch)
    specs = [
        _child_spec(store, call_, agent, path) for call_, agent in zip(calls, children, strict=True)
    ]
    wait = PauseState(
        agent_name=root.name if not path else path[-1],
        pending_tool_calls=[ToolCall(name=c.name, params={}, id=c.id) for c in calls],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=2,
        usage=UsageStats(),
        pending_targets={c.id: "agent" for c in calls},
        pause_reason="managed_wait",
        pause_id=generate_ulid(),
        schema_version=4,
        loop_state_ref=True,
        managed_fanout=ManagedFanout(
            tuple(ManagedWait(s.tool_call_id, s.child_run_id) for s in specs)
        ),
    )
    bound = await store.bind_managed_children(
        BindManagedChildrenRequest(
            tenant_id=tenant,
            root_run_id=root_id,
            parent_run_id=parent_id,
            parent_claim_epoch=epoch,
            iteration_index=1,
            parent_wait=wait,
            parent_patch=LoopStatePatch(set_fields={"iteration": 1}),
            children=tuple(specs),
            max_parallel=max(len(specs), 2),
        )
    )
    assert bound.outcome == "created"
    fan = Fan(parent_id, str(wait.pause_id), [c.id for c in calls], [s.child_run_id for s in specs])
    ran = run if run is not None else range(len(children))
    for index, bound_child in enumerate(bound.children):
        claim = bound_child.child_claim
        assert claim is not None
        if index not in ran:
            fan.claims[fan.child(index)] = claim
            continue
        execution = _managed_budget.inherit(children[index], root)
        fan.results[fan.child(index)] = await continue_claimed(
            execution, store, fan.child(index), claim, notifier=None
        )
    return fan


async def _checkpoint_parent(
    store: Any, parent_id: str, root_id: str, calls: list[ToolCall], *, epoch: int
) -> None:
    """The parent's handoff checkpoint: its model turn and N admitted managed calls."""
    state = await store.get_loop_state(parent_id) or {
        "v": 1,
        "loop": "react",
        MANAGED_HEADER_KEY: ManagedHeader(root_id, ()).to_dict(),
        "managed_limits": {"max_managed_runs": 128},
        "managed_unpriced_accepted": 0,
    }
    state.update(
        {
            "phase": "dispatch",
            "iteration": 1,
            "pending": [
                {
                    "call_id": c.id,
                    "tool": c.name,
                    "phase": "dispatchable",
                    "decision_id": None,
                    "order_index": 2 + i,
                    "target": "agent",
                }
                for i, c in enumerate(calls)
            ],
            "managed_admitted": [c.id for c in calls],
            "trace_order_offset": 2 + len(calls),
        }
    )
    turn = message_trace(Message(role=Role.ASSISTANT, content="", tool_calls=calls), 1)
    turn.meta["iteration"] = 1
    await store.checkpoint_step(
        parent_id, claim_epoch=epoch, loop_state=state, pii_mapping=None, trace=turn
    )


def _child_spec(
    store: Any, call_: ToolCall, agent: Agent, path: tuple[str, ...]
) -> ManagedChildSpec:
    brief = str(call_.params["task"])
    seeds = seed_trace_kwargs(store, [Message(role=Role.USER, content=brief)])["seed_traces"]
    return ManagedChildSpec(
        tool_call_id=call_.id,
        row_id=generate_ulid(),
        provider_tool_call_id=None,
        tool_name=call_.name,
        params=dict(call_.params),
        child_run_id=generate_ulid(),
        child_agent_name=agent.name,
        child_path=(*path, agent.name),
        child_input_data={"input": brief},
        child_loop_state=_managed_budget.child_state({"v": 1, "loop": "react"}, 0),
        intent_event=EventWrite(
            generate_ulid(),
            StoredEventType.TOOL_STARTED.value,
            1,
            {"tool_name": call_.name, "call_id": call_.id, "target": "agent"},
            call_.id,
            None,
        ),
        child_seed_traces=tuple(seeds),
        child_model="scripted",
    )


async def bind_one(
    store: Any, root: Agent, parent_id: str, child: Agent, *, root_id: str | None = None
) -> tuple[str, ClaimResult]:
    """Park ``parent_id`` on a 0.7 (v3) wait over one child; return the child and its claim."""
    root_id = root_id or parent_id
    call_ = ToolCall(name=child.name, params={"task": BRIEF}, id=generate_ulid())
    await _checkpoint_parent(store, parent_id, root_id, [call_], epoch=0)
    spec = _child_spec(store, call_, child, ())
    wait = PauseState(
        agent_name=root.name,
        pending_tool_calls=[ToolCall(name=call_.name, params={}, id=call_.id)],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=2,
        usage=UsageStats(),
        pending_targets={call_.id: "agent"},
        pause_reason="managed_wait",
        pause_id=generate_ulid(),
        schema_version=3,
        loop_state_ref=True,
        managed_wait=ManagedWait(call_.id, spec.child_run_id),
    )
    bound = await store.bind_managed_child(
        BindManagedChildRequest(
            tenant_id=TENANT,
            root_run_id=root_id,
            parent_run_id=parent_id,
            parent_claim_epoch=0,
            tool_call_id=call_.id,
            row_id=spec.row_id,
            provider_tool_call_id=None,
            tool_name=call_.name,
            params=dict(call_.params),
            iteration_index=1,
            child_run_id=spec.child_run_id,
            child_agent_name=child.name,
            child_path=spec.child_path,
            child_input_data=spec.child_input_data,
            child_loop_state=spec.child_loop_state,
            parent_wait=wait,
            parent_patch=LoopStatePatch(set_fields={"iteration": 1}),
            intent_event=spec.intent_event,
            child_seed_traces=spec.child_seed_traces,
            child_model="scripted",
        )
    )
    assert bound.child_claim is not None
    return spec.child_run_id, bound.child_claim


class FakeSettle:
    """A ``settle_parent`` (T0b Protocol) built on ordinary store writes, for routing tests.

    It reads the wait's children in a fresh read; when every one is terminal it claims the
    parked parent with the v4 adoption descriptor (one winner, store CAS) and completes each
    intent with the child's answer in call order. ``calls`` records every settle attempt as
    ``(parent run id, all children terminal)``; ``adoptions`` counts claims won.
    """

    def __init__(self, *, adopt: bool = True, gate: Any = None) -> None:
        self.adopt = adopt
        self.gate = gate  # awaited before the read: a race schedule's hook
        self.calls: list[tuple[str, bool]] = []
        self.adoptions = 0

    async def __call__(self, store: StateStore, node: WaitNode) -> ClaimResult | None:
        if self.gate is not None:
            await self.gate()
        ids = [child.run_id for child in node.children]
        rows = await store.get_managed_children(node.run_id, ids, tenant_id=node.tenant_id)  # type: ignore[attr-defined]
        settled = len(rows) == len(ids) and all(r.status in TERMINAL_STATUS_VALUES for r in rows)
        self.calls.append((node.run_id, settled))
        if not settled or not self.adopt:
            return None
        return await self._adopt(store, node)

    async def _adopt(self, store: Any, node: WaitNode) -> ClaimResult | None:
        calls = [child.call.id for child in node.children if child.call is not None]
        claim = await store.submit_and_claim(
            node.run_id,
            expected_status=RunStatus.PAUSED.value,
            submitted_data={"managed_fanout_adoption": {"v": 1, "calls": calls}},
            submission={
                "submission_id": generate_ulid(),
                "kind": "tool_results",
                "pause_id": node.pause_id,
                "payload_ref": {"pause_data_key": "submission"},
                "call_ids": calls,
                "reserved_order_index": 2,
                "managed_fanout_adoption": {"v": 1},
            },
        )
        if not isinstance(claim, ClaimResult):
            return None
        self.adoptions += 1
        state = dict(claim.loop_state or {})
        for index, child in enumerate(node.children):
            assert child.call is not None
            record = await store.get_run(child.run_id)
            last = index == len(node.children) - 1
            cleared = {"managed_wait": None, "submission": None, "managed_admitted": None}
            patch = LoopStatePatch(
                remove_pending=(child.call.id,),
                set_phase="model_next" if last else None,
                set_fields=cleared if last else {},
            )
            text = record.answer or record.status
            observation = message_trace(
                Message(role=Role.TOOL, content=text, call_id=child.call.id, name=child.call.name),
                2 + index,
            )
            observation.meta["iteration"] = 1
            await store.complete_tool_call(
                node.run_id,
                claim_epoch=claim.claim_epoch,
                tool_call_id=child.call.id,
                outcome="completed" if record.status == "success" else "failed",
                result_payload=text,
                success=record.status == "success",
                duration_ms=1,
                error_message=None,
                replayed=False,
                observation=observation,
                event=EventWrite(
                    generate_ulid(),
                    StoredEventType.TOOL_COMPLETED.value,
                    1,
                    {"call_id": child.call.id, "target": "agent"},
                    child.call.id,
                    None,
                ),
                loop_state_patch=patch,
                meta=None,
                pii_mapping=None,
            )
            state = {**state, **patch.set_fields}
            state["pending"] = [
                p for p in state.get("pending", []) if p["call_id"] != child.call.id
            ]
            if patch.set_phase:
                state["phase"] = patch.set_phase
        return ClaimResult(claim.claim_epoch, claim.storage_format, state, None)


@pytest.fixture
def settle(monkeypatch: pytest.MonkeyPatch) -> FakeSettle:
    """Route every v4 settle through a fake adoption (see :func:`install_fake_settle`)."""
    return install_fake_settle(monkeypatch)


def install_fake_settle(
    monkeypatch: pytest.MonkeyPatch, fake: FakeSettle | None = None
) -> FakeSettle:
    """Route every v4 settle through a fake adoption (the real one is the fan-in task).

    A parent continued after a v4 adoption still holds its old v4 envelope in its pause
    data. The 0.7 continuation reads only a stale v3 wait there; the fan-in task extends
    ``_reclaim._continuation_base`` to v4 (contract gap), mirrored here for the test.
    """
    fake = fake or FakeSettle()
    protocol: SettleParent = fake
    monkeypatch.setattr(_managed_route, "settle_fanout", protocol)
    base = _reclaim._continuation_base

    def continuation_base(raw: dict[str, Any], run_id: str) -> PauseState:
        if raw.get("schema_version") != 4:
            return base(raw, run_id)
        wait = PauseState.from_managed_dict(raw, run_id=run_id)
        return replace(
            wait,
            managed_fanout=None,
            schema_version=2,
            pause_reason=None,
            pending_tool_calls=[],
            pending_targets={},
        )

    monkeypatch.setattr(_reclaim, "_continuation_base", continuation_base)
    return fake


def assert_coherent(result: RunResult) -> None:
    """One attachment policy (§8.4): ``pending`` is ``pendings[0]``, or all are cleared."""
    if result.pendings:
        assert result.pending == result.pendings[0]
        assert result.pause_id == result.pending.pause_id
        assert result.status.value == result.pending.status
    else:
        assert result.pending is None and result.pause_id is None
        assert result.status is not None


async def snapshot(store: Any, *run_ids: str) -> list[tuple[Any, ...]]:
    """The durable control state of runs: status, epoch, pause and checkpoint."""
    out = []
    for run_id in run_ids:
        record = await store.get_run(run_id)
        out.append(
            (
                record.status,
                record.claim_epoch,
                await store.get_pause_state(run_id),
                await store.get_loop_state(run_id),
            )
        )
    return out


# ---------------------------------------------------------------------------
# A1, A7: the root view of a fan-out and its read cost
# ---------------------------------------------------------------------------


async def test_one_paused_child_among_finished_siblings_is_the_root_view(store: Any) -> None:
    """A1: 1 and 3 finish, 2 needs approval: the root waits for approval with one pending."""
    one, two, three = finisher("one"), approver("two"), finisher("three")
    root = root_agent(store, one, two, three)
    root_id = await start_root(store, root)

    fan = await fan_out(store, root, root_id, [one, two, three])

    leaf = fan.results[fan.child(1)]
    assert leaf.status == RunStatus.WAITING_APPROVAL
    for read in (
        await root.get_result(root_id, tenant_id=TENANT),
        await root.reclaim(root_id, tenant_id=TENANT),
    ):
        assert read.status == RunStatus.WAITING_APPROVAL
        assert len(read.pendings) == 1
        pending = read.pendings[0]
        assert (pending.run_id, pending.origin_run_id) == (root_id, fan.child(1))
        assert (pending.agent_name, pending.agent_path) == ("two", ["lead", "two"])
        assert pending.pause_id == leaf.pause_id
        assert pending.tool_calls[0].name == "publish"
        assert_coherent(read)
    pendings = await RunStore(store).get_pendings(root_id, tenant_id=TENANT)
    assert pendings == list(read.pendings)
    assert await RunStore(store).get_pending(root_id, tenant_id=TENANT) == pendings[0]
    assert root.provider.calls == 0  # no extra parent call  # type: ignore[union-attr]


async def test_every_paused_leaf_is_listed_depth_first_in_call_order(store: Any) -> None:
    """§8.3: pendings list every paused leaf, nested ones in place, siblings in call order."""
    inner_one, inner_two = approver("inner_one"), client_child("inner_two")
    middle = Agent(
        name="middle",
        description="Delegates.",
        prompt="m",
        provider=ScriptedLLM([]),
        subagents=[inner_one, inner_two],
    )
    first, last = approver("first"), approver("last")
    root = root_agent(store, first, middle, last)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [first, middle, last], run=(0, 2))
    inner = await fan_out(
        store, root, fan.child(1), [inner_one, inner_two], root_id=root_id, path=("middle",)
    )

    pendings = await RunStore(store).get_pendings(root_id, tenant_id=TENANT)

    assert [p.origin_run_id for p in pendings] == [
        fan.child(0),
        inner.child(0),
        inner.child(1),
        fan.child(2),
    ]
    assert [p.agent_path for p in pendings] == [
        ["lead", "first"],
        ["lead", "middle", "inner_one"],
        ["lead", "middle", "inner_two"],
        ["lead", "last"],
    ]
    assert [p.kind for p in pendings] == ["approval", "approval", "client_tool", "approval"]
    assert {p.run_id for p in pendings} == {root_id}
    # Read from the middle run, the same branch is addressed to that run.
    below = await RunStore(store).get_pendings(fan.child(1), tenant_id=TENANT)
    assert [p.origin_run_id for p in below] == [inner.child(0), inner.child(1)]
    assert [p.agent_path for p in below] == [["middle", "inner_one"], ["middle", "inner_two"]]
    result = await root.get_result(root_id, tenant_id=TENANT)
    assert list(result.pendings) == pendings
    assert_coherent(result)


async def test_all_children_running_or_finished_reads_running(store: Any) -> None:
    """A wait whose children run or await adoption reads RUNNING with nothing pending."""
    one, two = finisher("one"), finisher("two")
    root = root_agent(store, one, two)
    root_id = await start_root(store, root)
    await fan_out(store, root, root_id, [one, two], run=(0,))

    result = await root.get_result(root_id, tenant_id=TENANT)

    assert result.status == RunStatus.RUNNING
    assert result.pendings == () and result.pending is None and result.pause_id is None
    assert await RunStore(store).get_pendings(root_id, tenant_id=TENANT) == []


def _count_statements(engine: Any) -> tuple[list[str], Any]:
    statements: list[str] = []

    def capture(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement.strip().lower())

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    return statements, capture


async def test_get_pendings_is_read_only_and_within_its_counted_reads(
    store: Any, engine: Any
) -> None:
    """A7: no claim, no write; a fixed statement count, never per-child run reads or history."""
    a, b, c = approver("a"), approver("b"), finisher("c")
    root = root_agent(store, a, b, c)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b, c])
    before = await snapshot(store, root_id, *fan.children)

    statements, capture = _count_statements(engine)
    try:
        pendings = await RunStore(store).get_pendings(root_id, tenant_id=TENANT)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)

    assert [p.origin_run_id for p in pendings] == fan.children[:2]
    assert await snapshot(store, root_id, *fan.children) == before
    reads = [s for s in statements if s.startswith(("select", "with"))]
    assert reads == statements, "get_pendings wrote"
    assert not any("react_traces" in s for s in reads), "history read"
    # Root: pause summary + hop projection (2); the v4 wait: envelope + child projection (2);
    # each approval leaf: its summary, active pause sequence and approval params (3).
    assert len(reads) == 2 + 2 + 3 * len(pendings)


# ---------------------------------------------------------------------------
# A8 (read part): tenant scope before disclosure
# ---------------------------------------------------------------------------


async def test_foreign_tenant_reads_are_not_found(store: Any) -> None:
    a, b = approver("a"), approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])

    for scope in (OTHER_TENANT, None):
        with pytest.raises(RunNotFoundError) as caught:
            await RunStore(store).get_pendings(root_id, tenant_id=scope)
        assert not any(child in str(caught.value) for child in fan.children)
        with pytest.raises(RunNotFoundError):
            await root.get_result(root_id, tenant_id=scope)


# ---------------------------------------------------------------------------
# A13: one attachment policy on every result path (§8.4)
# ---------------------------------------------------------------------------


def solo(store: Any, *responses: LLMResponse, **kwargs: Any) -> Agent:
    """A plain agent (no subagents) whose own pauses go through every result path."""
    return Agent(
        name="solo",
        prompt="You work.",
        provider=ScriptedLLM(list(responses)),
        state_store=store,
        tools=[publish, read_sheet],
        require_approval=["publish"],
        **kwargs,
    )


def gated(text: str = "x") -> LLMResponse:
    return call("publish", {"text": text}, f"p-{text}")


def own_pause(result: RunResult) -> None:
    """A run's own pause: pendings is exactly that one pause."""
    assert result.pendings == (result.pending,)
    assert result.pending is not None and result.pending.origin_run_id is None
    assert_coherent(result)


async def test_own_pause_paths_attach_one_pending(store: Any) -> None:
    """Own-root pause, get_result, payload-free resume, approval and the cached result."""
    agent = solo(store, gated("a"), gated("b"), LLMResponse(text="done"))

    first = await agent.run("go", tenant_id=TENANT, idempotency_key="k-own")
    own_pause(first)
    own_pause(await agent.get_result(first.run_id, tenant_id=TENANT))

    again = await agent.resume(first.run_id, tenant_id=TENANT, expected_pause_id=first.pause_id)
    own_pause(again)
    assert again.pause_id != first.pause_id
    done = await agent.submit_approval(
        first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=again.pause_id
    )
    assert done.status == RunStatus.SUCCESS
    assert_coherent(done)
    assert done.pendings == ()
    assert_coherent(await agent.get_result(first.run_id, tenant_id=TENANT))
    cached = await agent.run("go", tenant_id=TENANT, idempotency_key="k-own")
    assert (cached.run_id, cached.status, cached.pendings) == (first.run_id, done.status, ())
    assert_coherent(cached)


async def test_ephemeral_pause_attaches_one_pending() -> None:
    agent = Agent(
        name="solo",
        prompt="You work.",
        provider=ScriptedLLM([gated()]),
        tools=[publish],
        require_approval=["publish"],
    )
    own_pause(await agent.run("go"))


async def test_submit_paths_attach_or_clear(store: Any) -> None:
    """submit_tool_results, submit_approval (reject) and submit_resume."""
    client = solo(store, call("read_sheet", {"sheet": "S"}, "c1"), gated(), LLMResponse(text="ok"))
    paused = await client.run("go", tenant_id=TENANT)
    own_pause(paused)
    call_id = paused.pending.tool_calls[0].id  # type: ignore[union-attr]
    approval = await client.submit_tool_results(
        paused.run_id,
        [ToolResult(name="read_sheet", call_id=call_id, payload='"v"')],
        tenant_id=TENANT,
        expected_pause_id=paused.pause_id,
    )
    own_pause(approval)
    done = await client.submit_approval(
        paused.run_id, approved=False, tenant_id=TENANT, expected_pause_id=approval.pause_id
    )
    assert done.status == RunStatus.SUCCESS and done.pendings == ()
    assert_coherent(done)

    pausing = solo(store, gated(), LLMResponse(text="ok"))
    run = await pausing.run("go", tenant_id=TENANT)
    own_pause(run)


async def test_stream_terminal_events_carry_coherent_results(store: Any) -> None:
    """stream's RUN_PAUSED and resume_stream's RUN_COMPLETED results follow the policy."""
    agent = solo(store, gated(), LLMResponse(text="ok"))
    paused = None
    async for item in agent.stream("go", tenant_id=TENANT):
        if item.run_result is not None:
            paused = item.run_result
    assert paused is not None
    own_pause(paused)
    final = None
    async for item in agent.resume_stream(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    ):
        if item.run_result is not None:
            final = item.run_result
    assert final is not None and final.status == RunStatus.SUCCESS
    assert_coherent(final)
    assert final.pendings == ()


async def test_fanout_root_views_attach_every_pause(store: Any) -> None:
    """get_result, reclaim and the cached root view of a fan-out list both paused leaves."""
    a, b = approver("a"), client_child("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])

    for result in (
        await root.get_result(root_id, tenant_id=TENANT),
        await root.reclaim(root_id, tenant_id=TENANT),
        await _managed_route.root_view(store, root_id),
    ):
        assert [p.origin_run_id for p in result.pendings] == fan.children
        assert result.status == RunStatus.WAITING_APPROVAL
        assert_coherent(result)


# ---------------------------------------------------------------------------
# Routing a root continuation (F1): A2, A3, A4, A8, A14 routing part
# ---------------------------------------------------------------------------


def pause_of(fan: Fan, index: int) -> str:
    pause_id = fan.results[fan.child(index)].pause_id
    assert pause_id is not None
    return pause_id


def client_call_id(fan: Fan, index: int) -> str:
    pending = fan.results[fan.child(index)].pending
    assert pending is not None
    return pending.tool_calls[0].id


async def test_answers_in_any_order_adopt_once_and_finish_the_root(
    store: Any, settle: FakeSettle
) -> None:
    """A2: approval in one child, a client tool in another; answer the second first."""
    a, b = approver("a"), client_child("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])

    first = await root.submit_tool_results(
        root_id,
        [ToolResult(name="read_sheet", call_id=client_call_id(fan, 1), payload='"v"')],
        tenant_id=TENANT,
        expected_pause_id=pause_of(fan, 1),
    )

    assert first.run_id == root_id and first.status == RunStatus.WAITING_APPROVAL
    assert [p.origin_run_id for p in first.pendings] == [fan.child(0)]
    assert first.pause_id == pause_of(fan, 0)
    assert_coherent(first)
    assert (settle.calls, settle.adoptions) == ([(root_id, False)], 0)

    final = await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
    )

    assert (final.run_id, final.status, final.answer) == (root_id, RunStatus.SUCCESS, ROOT_TEXT)
    assert_coherent(final)
    assert (settle.calls[-1], settle.adoptions) == ((root_id, True), 1)
    assert published == ["a"]
    assert root.provider.calls == 1  # type: ignore[union-attr]


async def test_a_stale_token_at_lookup_names_the_first_pending(store: Any) -> None:
    """A3, stale at lookup: no leaf has the token; nothing changes."""
    a, b = approver("a"), approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    before = await snapshot(store, root_id, *fan.children)

    with pytest.raises(PauseStatusMismatchError) as caught:
        await root.submit_approval(
            root_id, approved=True, tenant_id=TENANT, expected_pause_id=generate_ulid()
        )

    assert caught.value.reason == "pause_id"
    assert caught.value.current_pause_id == pause_of(fan, 0)
    assert await snapshot(store, root_id, *fan.children) == before
    assert published == []


async def test_a_token_that_goes_stale_before_the_claim_names_the_leaf_token(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3, stale inside the leaf's claim: the 0.7 claim error with the leaf's new token."""
    a = approver("a", call("publish", {"text": "again"}, "a-p2"), LLMResponse(text="a done"))
    b = approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    walk = _managed_route._walk
    raced: list[RunResult] = []
    racing: list[bool] = []

    async def walk_then_race(*args: Any) -> Any:
        walked = await walk(*args)
        if not racing:  # another caller answers the same leaf after this lookup
            racing.append(True)
            raced.append(
                await root.submit_approval(
                    root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
                )
            )
        return walked

    monkeypatch.setattr(_managed_route, "_walk", walk_then_race)
    with pytest.raises(PauseStatusMismatchError) as caught:
        await root.submit_approval(
            root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
        )

    leaf_now = raced[0].pendings[0]
    assert leaf_now.origin_run_id == fan.child(0) and leaf_now.pause_id != pause_of(fan, 0)
    assert caught.value.reason == "pause_id"
    assert caught.value.current_pause_id == leaf_now.pause_id
    assert published == ["a"]  # the racing answer applied once; the stale one never ran


async def test_a_denial_in_one_child_affects_only_that_call(store: Any) -> None:
    """A4: the denied call never runs; the sibling keeps its own pause and token."""
    a, b = approver("a"), approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])

    result = await root.submit_approval(
        root_id, approved=False, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
    )

    assert published == []
    child = await store.get_run(fan.child(0))
    assert child.status == RunStatus.SUCCESS.value
    assert [(p.origin_run_id, p.pause_id) for p in result.pendings] == [
        (fan.child(1), pause_of(fan, 1))
    ]
    assert_coherent(result)
    sibling = await store.get_run(fan.child(1))
    assert (sibling.status, sibling.claim_epoch) == (RunStatus.WAITING_APPROVAL.value, 0)


async def test_per_call_decisions_route_to_their_leaf_only(store: Any) -> None:
    """A14 (routing): decisions name the leaf's own gated calls; the sibling is untouched."""
    a = Agent(
        name="a",
        description="Publishes two drafts.",
        prompt="You publish.",
        provider=ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="publish", params={"text": "x"}, provider_tool_call_id="x"),
                        ToolCall(name="publish", params={"text": "y"}, provider_tool_call_id="y"),
                    ]
                ),
                LLMResponse(text="a done"),
            ]
        ),
        tools=[publish],
        require_approval=["publish"],
    )
    b = approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    gated_calls = [c.id for c in fan.results[fan.child(0)].pending.tool_calls]  # type: ignore[union-attr]

    result = await root.submit_approval(
        root_id,
        decisions={
            gated_calls[0]: ApprovalDecision.approve(),
            gated_calls[1]: ApprovalDecision.deny("no"),
        },
        tenant_id=TENANT,
        expected_pause_id=pause_of(fan, 0),
    )

    assert published == ["x"]
    assert [p.origin_run_id for p in result.pendings] == [fan.child(1)]
    assert (await store.get_run(fan.child(0))).status == RunStatus.SUCCESS.value


async def test_a_fanout_continuation_needs_a_token_and_a_paused_leaf(store: Any) -> None:
    a, b = approver("a"), finisher("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    before = await snapshot(store, root_id, *fan.children)

    with pytest.raises(ValueError, match="pause_id"):
        await root.submit_approval(root_id, approved=True, tenant_id=TENANT)
    assert await snapshot(store, root_id, *fan.children) == before

    running = await start_root(store, root)
    await fan_out(store, root, running, [finisher("a"), finisher("b")], run=())
    with pytest.raises(RunNotPausedError):
        await root.submit_resume(running, tenant_id=TENANT)


SUBMITS = {
    "submit_approval": lambda agent, run_id, tenant, pid: agent.submit_approval(
        run_id, approved=True, tenant_id=tenant, expected_pause_id=pid
    ),
    "submit_tool_results": lambda agent, run_id, tenant, pid: agent.submit_tool_results(
        run_id,
        [ToolResult(name="read_sheet", call_id="c", payload='"v"')],
        tenant_id=tenant,
        expected_pause_id=pid,
    ),
    "submit_input": lambda agent, run_id, tenant, pid: agent.submit_input(
        run_id, "yes", tenant_id=tenant, expected_pause_id=pid
    ),
    "submit_resume": lambda agent, run_id, tenant, pid: agent.submit_resume(
        run_id, tenant_id=tenant, expected_pause_id=pid
    ),
    "resume": lambda agent, run_id, tenant, pid: agent.resume(
        run_id, tenant_id=tenant, expected_pause_id=pid
    ),
}


@pytest.mark.parametrize("operation", sorted(SUBMITS))
async def test_foreign_tenant_submits_are_not_found_before_disclosure(
    store: Any, operation: str
) -> None:
    """A8: a foreign scope cannot submit to, or learn anything about, a fan-out leaf."""
    a, b = approver("a"), approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    before = await snapshot(store, root_id, *fan.children)

    with pytest.raises(RunNotFoundError) as caught:
        await SUBMITS[operation](root, root_id, OTHER_TENANT, pause_of(fan, 0))

    assert await snapshot(store, root_id, *fan.children) == before
    shown = str(caught.value)
    assert not any(secret in shown for secret in (*fan.children, pause_of(fan, 0), BRIEF))
    assert published == []


# ---------------------------------------------------------------------------
# Leaf-first unwind (§8.2 step 4), proven with the fake settle_parent
# ---------------------------------------------------------------------------


async def test_unwind_settles_the_leafs_parent_first_then_its_parent(
    store: Any, settle: FakeSettle
) -> None:
    """R waits on A and B; A waits on A1 and A2. Answering A2 settles A first, continues
    A to its end, then settles R (B still paused: R keeps waiting). Answering B then
    settles R and finishes the root."""
    a1, a2 = finisher("a1"), approver("a2")
    middle = Agent(
        name="middle",
        description="Delegates twice.",
        prompt="m",
        provider=ScriptedLLM([LLMResponse(text="middle done")]),
        subagents=[a1, a2],
    )
    b = approver("b")
    root = root_agent(store, middle, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [middle, b], run=(1,))
    inner = await fan_out(store, root, fan.child(0), [a1, a2], root_id=root_id, path=("middle",))

    first = await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(inner, 1)
    )

    assert settle.calls == [(fan.child(0), True), (root_id, False)]
    assert (await store.get_run(fan.child(0))).status == RunStatus.SUCCESS.value
    assert [p.origin_run_id for p in first.pendings] == [fan.child(1)]
    assert_coherent(first)

    final = await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 1)
    )

    assert settle.calls[-1] == (root_id, True) and settle.adoptions == 2
    assert (final.status, final.answer) == (RunStatus.SUCCESS, ROOT_TEXT)
    assert_coherent(final)
    assert published == ["a2", "b"]


async def test_unwind_stops_at_a_parent_with_an_unsettled_child(
    store: Any, settle: FakeSettle
) -> None:
    """A1 still paused: answering A2 settles A (not adoptable) and never reaches R."""
    a1, a2 = approver("a1"), approver("a2")
    middle = Agent(
        name="middle",
        description="Delegates.",
        prompt="m",
        provider=ScriptedLLM([]),
        subagents=[a1, a2],
    )
    b = finisher("b")
    root = root_agent(store, middle, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [middle, b], run=(1,))
    inner = await fan_out(store, root, fan.child(0), [a1, a2], root_id=root_id, path=("middle",))

    result = await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(inner, 1)
    )

    assert settle.calls == [(fan.child(0), False)]
    assert [p.origin_run_id for p in result.pendings] == [inner.child(0)]
    assert result.status == RunStatus.WAITING_APPROVAL
    assert_coherent(result)


async def test_a_chain_above_a_fanout_shows_every_pause_and_unwinds_through_0_7_adoption(
    store: Any, settle: FakeSettle
) -> None:
    """R waits (v3) on M; M fans out to A and B. Reads re-address both pauses to R; the
    last answer settles M (v4), continues M, then adopts M into R the 0.7 way."""
    a, b = approver("a"), approver("b")
    middle = Agent(
        name="middle",
        description="Delegates twice.",
        prompt="m",
        provider=ScriptedLLM([LLMResponse(text="middle done")]),
        subagents=[a, b],
    )
    root = root_agent(store, middle)
    root_id = await start_root(store, root)
    middle_id, _ = await bind_one(store, root, root_id, middle)
    fan = await fan_out(store, root, middle_id, [a, b], root_id=root_id, path=("middle",))

    for view in (
        await root.get_result(root_id, tenant_id=TENANT),
        await root.reclaim(root_id, tenant_id=TENANT),
    ):
        assert [p.origin_run_id for p in view.pendings] == fan.children
        assert [p.agent_path for p in view.pendings] == [
            ["lead", "middle", "a"],
            ["lead", "middle", "b"],
        ]
        assert {p.run_id for p in view.pendings} == {root_id}
        assert_coherent(view)

    first = await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 1)
    )
    assert [p.origin_run_id for p in first.pendings] == [fan.child(0)]
    final = await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
    )

    assert settle.calls == [(middle_id, False), (middle_id, True)]
    assert (final.run_id, final.status, final.answer) == (root_id, RunStatus.SUCCESS, ROOT_TEXT)
    assert (await store.get_run(middle_id)).status == RunStatus.SUCCESS.value
    assert_coherent(final)
