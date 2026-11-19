"""Fan-out (0.8, T5): the tree walk, routing a root continuation and pending attachment.

A parent parked on a v4 wait has several children. Every leaf's external pause shows on the
root in stable order (``RunResult.pendings``, ``RunStore.get_pendings``); a root-id
continuation with a leaf's token reaches exactly that leaf, runs it in the caller's task and
then unwinds leaf-first: the leaf's parent settles first, then its parent, and so on.

The 0.8 driver (owner tasks, fan-in) is not built yet, so fan-out states are made directly
through the store: the parent is checkpointed with its pending managed calls, the batch is
bound with ``bind_managed_children``, and each child then runs from its bound checkpoint with
its own claim. Settlement is the real fan-in (``settle_parent``), observed by
:class:`SettleSpy`.
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
from dendrux.runtime import _managed_budget, _managed_fanout, _managed_route
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

PROGRESS_S = 60  # green-path waits wait for progress; generous for loaded runners

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


class SettleSpy:
    """Wraps the real ``settle_parent`` (T0b Protocol, T4c adoption) to observe the unwind.

    ``calls`` records every settle attempt as ``(parent run id, all children terminal)``
    from a read just before the real settle; ``adoptions`` counts claims it won. ``gate``
    (optional) is awaited first: a race schedule's hook.
    """

    def __init__(self, real: SettleParent, *, gate: Any = None) -> None:
        self.real = real
        self.gate = gate
        self.calls: list[tuple[str, bool]] = []
        self.adoptions = 0

    async def __call__(
        self, store: StateStore, node: WaitNode, **kwargs: Any
    ) -> ClaimResult | None:
        if self.gate is not None:
            await self.gate()
        ids = [child.run_id for child in node.children]
        rows = await store.get_managed_children(node.run_id, ids, tenant_id=node.tenant_id)  # type: ignore[attr-defined]
        settled = len(rows) == len(ids) and all(r.status in TERMINAL_STATUS_VALUES for r in rows)
        self.calls.append((node.run_id, settled))
        claim = await self.real(store, node, **kwargs)
        self.adoptions += claim is not None
        return claim


@pytest.fixture
def settle(monkeypatch: pytest.MonkeyPatch) -> SettleSpy:
    """Observe the real fan-in settlement (see :func:`spy_settle`)."""
    return spy_settle(monkeypatch)


def spy_settle(monkeypatch: pytest.MonkeyPatch, *, gate: Any = None) -> SettleSpy:
    """Observe the real fan-in settlement without replacing continuation decoding."""
    spy = SettleSpy(_managed_fanout.settle_parent, gate=gate)
    monkeypatch.setattr(_managed_fanout, "settle_parent", spy)
    return spy


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
    store: Any, settle: SettleSpy
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
    store: Any, settle: SettleSpy
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
    store: Any, settle: SettleSpy
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
    store: Any, settle: SettleSpy
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


@pytest.mark.parametrize("nested", [False, True])
async def test_route_walk_reads_are_counted(store: Any, engine: Any, nested: bool) -> None:
    """§14 row 3: root 2 (run, saved cap) + the 0.7 route checkpoint 1, then 3 per v4 wait
    (envelope, one child projection, one intent read); a fan-out child's pause token comes
    with its row, so paused fan-out leaves cost nothing more. Read-only."""
    a, b, c = approver("a"), approver("b"), finisher("c")
    inner = Agent(
        name="inner",
        description="Delegates.",
        prompt="i",
        provider=ScriptedLLM([]),
        subagents=[a, b],
    )
    root = root_agent(store, inner, c) if nested else root_agent(store, a, b, c)
    root_id = await start_root(store, root)
    if nested:
        fan = await fan_out(store, root, root_id, [inner, c], run=(1,))
        await fan_out(store, root, fan.child(0), [a, b], root_id=root_id, path=("inner",))
    else:
        fan = await fan_out(store, root, root_id, [a, b, c])
    before = await snapshot(store, root_id, *fan.children)

    statements, capture = _count_statements(engine)
    try:
        tree = await _managed_route.route_tree(root, store, root_id, tenant_id=TENANT)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)

    waits = 2 if nested else 1
    assert len(tree.paused_leaves()) == 2  # type: ignore[union-attr]
    assert all(s.startswith("select") for s in statements)
    assert not any("react_traces" in s for s in statements)
    assert len(statements) == 2 + 1 + 3 * waits
    assert await snapshot(store, root_id, *fan.children) == before


async def test_input_and_requested_pause_paths_attach_or_clear(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A13: submit_input and submit_resume (a requested pause) follow the same policy."""
    from tests.integration.test_managed_routing import patch_questions

    patch_questions(monkeypatch)
    asking = solo(store, LLMResponse(text="Which draft?"), LLMResponse(text="ok"))
    question = await asking.run("go", tenant_id=TENANT)
    own_pause(question)
    answered = await asking.submit_input(
        question.run_id, "the first", tenant_id=TENANT, expected_pause_id=question.pause_id
    )
    assert answered.status == RunStatus.SUCCESS and answered.pendings == ()
    assert_coherent(answered)

    async def pause_me(index: int, run_id: str | None) -> None:
        if index == 0 and run_id is not None:
            await store.request_pause(run_id)

    from tests.integration.test_managed_routing import ContextLLM

    pausing = Agent(
        name="solo",
        prompt="You work.",
        provider=ContextLLM(
            [call("publish", {"text": "p"}, "c"), LLMResponse(text="ok")], on_call=pause_me
        ),
        state_store=store,
        tools=[publish],
    )
    paused = await pausing.run("go", tenant_id=TENANT)
    assert paused.status == RunStatus.PAUSED
    own_pause(paused)
    resumed = await pausing.submit_resume(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    )
    assert (resumed.status, resumed.pendings) == (RunStatus.SUCCESS, ())
    assert_coherent(resumed)


async def test_a_refresh_that_sees_the_run_finish_clears_every_pause_field(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8.4: the run moves client tool -> approval -> success while its result is attached.

    The pending read sees the approval (the run moved), the refresh sees success: the
    result is SUCCESS with no pending, pendings or token.
    """
    from dendrux.runtime import _run_pending
    from dendrux.runtime._pending import _PendingReader

    agent = solo(store, call("read_sheet", {"sheet": "S"}, "c1"), gated(), LLMResponse(text="ok"))
    paused = await agent.run("go", tenant_id=TENANT)
    record = await store.get_run(paused.run_id)
    facing = _PendingReader.facing
    raced: list[bool] = []

    async def advancing(self: Any, run_id: str, *, tenant_id: Any) -> Any:
        if raced:
            return await facing(self, run_id, tenant_id=tenant_id)
        raced.append(True)
        call_id = paused.pending.tool_calls[0].id  # type: ignore[union-attr]
        await agent.submit_tool_results(
            run_id,
            [ToolResult(name="read_sheet", call_id=call_id, payload='"v"')],
            tenant_id=TENANT,
            expected_pause_id=paused.pause_id,
        )
        view = await facing(self, run_id, tenant_id=tenant_id)  # the approval
        await agent.submit_approval(
            run_id, approved=True, tenant_id=TENANT, expected_pause_id=view.pending.pause_id
        )
        return view

    monkeypatch.setattr(_PendingReader, "facing", advancing)
    result = RunResult(run_id=paused.run_id, status=RunStatus.WAITING_CLIENT_TOOL)

    attached = await _run_pending._attach_persisted_pending(store, result, record)

    assert attached.status == RunStatus.SUCCESS
    assert (attached.pending, attached.pendings, attached.pause_id) == (None, (), None)


async def test_a_routed_fanout_leaf_runs_in_the_callers_task(store: Any) -> None:
    """§8.2 step 3: the claimed leaf runs in the submitting task, with its ContextVars."""
    import asyncio
    from contextvars import ContextVar

    request_id: ContextVar[str] = ContextVar("t5_request_id", default="MISSING")
    seen: list[tuple[str, Any, str]] = []

    class Watching(ScriptedLLM):
        async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
            seen.append(("provider", asyncio.current_task(), request_id.get()))
            return await super().complete(messages, tools, **kwargs)

    @tool()
    async def stamp(text: str) -> str:
        """Stamp a draft."""
        seen.append(("tool", asyncio.current_task(), request_id.get()))
        return text

    a = Agent(
        name="a",
        description="Stamps one draft.",
        prompt="You stamp.",
        provider=Watching([call("stamp", {"text": "x"}, "s1"), LLMResponse(text="a done")]),
        tools=[stamp],
        require_approval=["stamp"],
    )
    b = approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    seen.clear()

    request_id.set("req-7")
    await root.submit_approval(
        root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
    )

    caller = asyncio.current_task()
    assert seen and [kind for kind, _, _ in seen] == ["tool", "provider"]
    assert all(task is caller for _, task, _ in seen)
    assert all(value == "req-7" for _, _, value in seen)


def _asks_twice(name: str) -> Agent:
    """A child that asks the human twice, then answers."""
    return Agent(
        name=name,
        description=f"{name} asks.",
        prompt="You ask.",
        provider=ScriptedLLM(
            [
                LLMResponse(text="Which one?"),
                LLMResponse(text="Which again?"),
                LLMResponse(text="ok"),
            ]
        ),
    )


def _reads_twice(name: str) -> Agent:
    """A child that waits for two client tools in turn, then answers."""
    return Agent(
        name=name,
        description=f"{name} reads.",
        prompt="You read.",
        provider=ScriptedLLM(
            [
                call("read_sheet", {"sheet": "S1"}, f"{name}-1"),
                call("read_sheet", {"sheet": "S2"}, f"{name}-2"),
                LLMResponse(text="ok"),
            ]
        ),
        tools=[read_sheet],
    )


@pytest.mark.parametrize("kind", ["client_tool", "input"])
async def test_a_token_that_goes_stale_before_the_claim_on_every_submit_kind(
    store: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A3 for client results and input: another caller answers the same leaf after this
    lookup and it pauses again the same way; the stale submit is a pause_id mismatch
    naming the leaf's new token, never a payload error."""
    from tests.integration.test_managed_routing import patch_questions

    patch_questions(monkeypatch)
    a = _reads_twice("a") if kind == "client_tool" else _asks_twice("a")
    b = approver("b")
    root = root_agent(store, a, b)
    root_id = await start_root(store, root)
    fan = await fan_out(store, root, root_id, [a, b])
    first = fan.results[fan.child(0)]
    assert first.pending is not None and first.pending.kind == kind

    def answer(token: str) -> Any:
        if kind == "client_tool":
            call_id = first.pending.tool_calls[0].id  # type: ignore[union-attr]
            return root.submit_tool_results(
                root_id,
                [ToolResult(name="read_sheet", call_id=call_id, payload='"v"')],
                tenant_id=TENANT,
                expected_pause_id=token,
            )
        return root.submit_input(root_id, "the first", tenant_id=TENANT, expected_pause_id=token)

    walk = _managed_route._walk
    raced: list[RunResult] = []

    async def walk_then_race(*args: Any) -> Any:
        walked = await walk(*args)
        if not raced:
            raced.append(RunResult(run_id=root_id, status=RunStatus.RUNNING))
            raced[0] = await answer(pause_of(fan, 0))
        return walked

    monkeypatch.setattr(_managed_route, "_walk", walk_then_race)
    with pytest.raises(PauseStatusMismatchError) as caught:
        await answer(pause_of(fan, 0))

    again = next(p for p in raced[0].pendings if p.origin_run_id == fan.child(0))
    assert again.kind == kind and again.pause_id != pause_of(fan, 0)
    assert caught.value.reason == "pause_id"
    assert caught.value.current_pause_id == again.pause_id


# ---------------------------------------------------------------------------
# Walk bounds: runs are counted before a batch is read, never after
# ---------------------------------------------------------------------------


class _BranchingStore:
    """A store-free tree of nested 32-way waits (each wait's first child waits again).

    Counts every run id it projects; enough for both walks (pending and routing).
    """

    def __init__(self, depth: int, width: int = 32) -> None:
        from types import SimpleNamespace

        from dendrux.runtime._managed_fanout_store import ManagedChildRow

        self.fetched = 0
        self.pauses: dict[str, dict[str, Any]] = {}
        self.rows: dict[str, ManagedChildRow] = {}
        self.intents: dict[str, list[Any]] = {}
        self.root = generate_ulid()
        parent, path = self.root, ()
        self.root_binding = None
        for level in range(depth):
            name = f"c{level}"
            calls = [ToolCall(name=name, params={}, id=generate_ulid()) for _ in range(width)]
            kids = [generate_ulid() for _ in range(width)]
            wait = PauseState(
                agent_name="lead",
                pending_tool_calls=calls,
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
                    tuple(ManagedWait(c.id, k) for c, k in zip(calls, kids, strict=True))
                ),
            )
            self.pauses[parent] = wait.to_dict()
            binding = wait.managed_fanout.to_checkpoint(pause_id=str(wait.pause_id))  # type: ignore[union-attr]
            if parent == self.root:
                self.root_binding = binding
            else:
                self.rows[parent] = replace(
                    self.rows[parent],
                    status="paused",
                    binding=binding,
                    pause_reason="managed_wait",
                    pause_id=wait.pause_id,
                )
            self.intents[parent] = [
                SimpleNamespace(
                    tool_call_id=c.id,
                    target="agent",
                    tool_name=c.name,
                    params={"task": "t"},
                    provider_tool_call_id=None,
                )
                for c in calls
            ]
            child_path = (*path, name)
            for c, kid in zip(calls, kids, strict=True):
                self.rows[kid] = ManagedChildRow(
                    kid,
                    parent,
                    TENANT,
                    "running",
                    ManagedHeader(self.root, child_path, parent, c.id).to_dict(),
                    None,
                    "react",
                    None,
                    None,
                )
            parent, path = kids[0], child_path

    async def get_pause_state(self, run_id: str) -> Any:
        return self.pauses.get(run_id)

    async def get_managed_children(self, parent: str, ids: Any, *, tenant_id: Any) -> list[Any]:
        self.fetched += len(ids)
        return [self.rows[i] for i in ids]

    async def get_tool_calls(self, run_id: str, **kwargs: Any) -> list[Any]:
        return self.intents[run_id] if kwargs.get("offset", 0) == 0 else []

    async def get_loop_state(self, run_id: str) -> Any:
        return {"managed_limits": {"max_managed_runs": 128}, "managed_wait": self.root_binding}


def _capable(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from dendrux.runtime import _managed_pending

    monkeypatch.setattr(
        _managed_pending, "fanout_support", lambda store: SimpleNamespace(capable=True)
    )


async def test_the_pending_walk_counts_runs_before_reading_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine nested 32-way waits hold 289 runs: the walk refuses before reading past 257."""
    from dendrux.errors import ManagedStateError
    from dendrux.runtime._managed_pending import MAX_MANAGED_WALK, ManagedHop, read_workflow

    _capable(monkeypatch)
    store = _BranchingStore(depth=9)
    start = ManagedHop(
        store.root,
        "paused",
        TENANT,
        None,
        "lead",
        ManagedHeader(store.root, ()).to_dict(),
        store.root_binding,
    )

    async def read_hop(run_id: str, tenant_id: Any) -> Any:
        return start

    with pytest.raises(ManagedStateError) as caught:
        await read_workflow(store, store.root, TENANT, read_hop=read_hop)

    assert caught.value.reason == "binding_conflict"
    assert 1 + store.fetched <= MAX_MANAGED_WALK


async def test_the_route_walk_counts_runs_before_reading_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The routing walk's bound is the root's saved cap (128): refused before 4 x 32 reads."""
    from types import SimpleNamespace

    from dendrux.errors import ManagedStateError

    _capable(monkeypatch)
    store = _BranchingStore(depth=9)
    agents = [finisher(f"c{level}") for level in range(9)]
    for parent, child in zip(agents, agents[1:], strict=False):
        parent._subagents = [child]  # type: ignore[attr-defined]
    root = Agent(name="lead", prompt="l", provider=ScriptedLLM([]), subagents=[agents[0]])
    record = SimpleNamespace(
        id=store.root, tenant_id=TENANT, status="paused", storage_format=2, parent_run_id=None
    )

    with pytest.raises(ManagedStateError) as caught:
        await _managed_route._walk(root, store, record)  # type: ignore[arg-type]

    assert caught.value.reason == "binding_conflict"
    assert 1 + store.fetched <= 128


# ---------------------------------------------------------------------------
# Root cancellation reaches a routed leaf running in the caller's task
# ---------------------------------------------------------------------------


def _held_after_approval(name: str) -> tuple[Agent, ScriptedLLM]:
    """A child that asks approval, then blocks in its next model call until released."""
    llm = ScriptedLLM(
        [call("publish", {"text": name}, f"{name}-p"), LLMResponse(text="done")], gate_at=1
    )
    agent = Agent(
        name=name,
        description=f"{name} reviews.",
        prompt="You review.",
        provider=llm,
        tools=[publish],
        require_approval=["publish"],
    )
    return agent, llm


@pytest.mark.parametrize("shape", ["fanout", "nested_fanout", "chain"])
async def test_cancelling_the_root_interrupts_a_routed_leaf(store: Any, shape: str) -> None:
    """The routed leaf's blocked provider call is interrupted; the submit returns and the
    leaf ends CANCELLED, never SUCCESS. ``chain`` checks the 0.7 single-child route."""
    import asyncio

    leaf, llm = _held_after_approval("leaf")
    other = approver("other")
    if shape == "fanout":
        root = root_agent(store, leaf, other)
        root_id = await start_root(store, root)
        fan = await fan_out(store, root, root_id, [leaf, other])
        leaf_id, token = fan.child(0), pause_of(fan, 0)
    elif shape == "nested_fanout":
        middle = Agent(
            name="middle",
            description="Delegates.",
            prompt="m",
            provider=ScriptedLLM([]),
            subagents=[leaf, other],
        )
        root = root_agent(store, middle, approver("side"))
        root_id = await start_root(store, root)
        top = await fan_out(store, root, root_id, [middle, root.subagents[1]], run=(1,))
        inner = await fan_out(
            store, root, top.child(0), [leaf, other], root_id=root_id, path=("middle",)
        )
        leaf_id, token = inner.child(0), pause_of(inner, 0)
    else:
        root = root_agent(store, leaf)
        root_id = await start_root(store, root)
        leaf_id, claim = await bind_one(store, root, root_id, leaf)
        paused = await continue_claimed(
            _managed_budget.inherit(leaf, root), store, leaf_id, claim, notifier=None
        )
        token = str(paused.pause_id)

    submit = asyncio.create_task(
        root.submit_approval(root_id, approved=True, tenant_id=TENANT, expected_pause_id=token)
    )
    await asyncio.wait_for(llm.entered.wait(), PROGRESS_S)
    await root.cancel_run(root_id, tenant_id=TENANT)
    try:
        # hang proof: a cancelled root must stop the routed leaf promptly
        await asyncio.wait_for(asyncio.shield(submit), 5)
    except TimeoutError:
        llm.gate.set()
        await asyncio.gather(submit, return_exceptions=True)
        pytest.fail("the routed leaf kept running after the root was cancelled")
    except Exception:  # noqa: BLE001 - a cancelled continuation may raise; the state decides
        pass
    finally:
        llm.gate.set()

    assert llm.cancelled
    assert (await store.get_run(leaf_id)).status == RunStatus.CANCELLED.value
