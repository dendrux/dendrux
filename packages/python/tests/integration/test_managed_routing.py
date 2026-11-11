"""Managed subagents (0.7, T3): a subagent's external pause flows up to the root.

A child (at any depth) that pauses for approval, a client tool, human input or a
requested pause surfaces on the root: ``run``, ``get_result`` and
``RunStore.get_pending(root)`` report the leaf's waiting status and pause token,
with ``agent_name``, ``agent_path`` and ``origin_run_id``. Root-id continuations
then route to that leaf, validate with the leaf's own definition, claim it with
the existing CAS, and unwind finished outcomes back to the root. Runs on SQLite
and Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test`` database).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ToolCallRecord
from dendrux.errors import (
    InvalidToolResultError,
    ManagedChildControlError,
    PauseStatusMismatchError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.runtime.context import get_delegation_context
from dendrux.store import RunStore
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    AgentStep,
    ApprovalDecision,
    Clarification,
    InputValidationError,
    Limits,
    LLMResponse,
    RunStatus,
    ToolCall,
    ToolResult,
)
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-route-brief-4d2a"
CHILD_TEXT = "SENTINEL-child-final-9c1e"
TENANT = "t1"


class ContextLLM(ScriptedLLM):
    """Scripted provider recording the run each call is served under (real ContextVar).

    ``on_call`` (optional) runs inside ``complete()`` with the serving run id,
    before the scripted response is returned.
    """

    def __init__(self, responses: list[LLMResponse], on_call: Any = None) -> None:
        super().__init__(responses)
        self.served: list[str | None] = []
        self.on_call = on_call

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        run_id = ctx.run_id if ctx else None
        self.served.append(run_id)
        if self.on_call is not None:
            await self.on_call(self.calls, run_id)
        return await super().complete(messages, tools, **kwargs)


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


published: list[str] = []


@tool()
async def publish(text: str) -> str:
    """Publish a draft."""
    published.append(text)
    return f"published {text}"


@tool()
async def count_words(text: str) -> int:
    """Count words."""
    return len(text.split())


@tool(target="client")
async def read_sheet(sheet: str) -> str:
    """Read a sheet in the user's browser."""
    return ""


@pytest.fixture
def ask_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Managed runs use the native strategy: text ending in '?' asks the human."""
    patch_questions(monkeypatch)


def patch_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the native strategy treat text ending in '?' as a clarification question."""
    native = NativeToolCalling.parse_response

    def parse_response(self: NativeToolCalling, response: LLMResponse) -> AgentStep:
        if (response.text or "").endswith("?"):
            return AgentStep(
                reasoning=None,
                action=Clarification(question=response.text or "", options=["yes", "no"]),
                raw_response=response.text,
                meta={},
            )
        return native(self, response)

    monkeypatch.setattr(NativeToolCalling, "parse_response", parse_response)


def reviewer(llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="reviewer",
        description="Reviews one draft and reports.",
        prompt="You review.",
        provider=llm,
        **kwargs,
    )


def auditor(llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="auditor",
        description="Audits one claim.",
        prompt="You audit.",
        provider=llm,
        **kwargs,
    )


def lead(store: Any, llm: Any, *children: Agent, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="You lead.",
        provider=llm,
        state_store=store,
        subagents=list(children),
        **kwargs,
    )


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def run_row(engine: Any, run_id: str) -> Any:
    return (await rows(engine, AgentRun, AgentRun.id == run_id))[0]


async def child_of(engine: Any, parent_id: str) -> Any:
    found = await rows(engine, AgentRun, AgentRun.parent_run_id == parent_id)
    assert len(found) == 1
    return found[0]


async def observations(engine: Any, run_id: str) -> list[Any]:
    from dendrux.db.models import ReactTrace

    found = await rows(engine, ReactTrace, ReactTrace.agent_run_id == run_id)
    return [t for t in found if t.role == "tool"]


def assert_root_facing(
    result: Any, root_id: str, leaf: Any, *, status: RunStatus, kind: str, path: list[str]
) -> None:
    """The root reports the leaf's waiting status, token and origin."""
    assert result.run_id == root_id
    assert result.status == status
    pending = result.pending
    assert pending is not None
    assert pending.run_id == root_id
    assert pending.status == status.value
    assert pending.kind == kind
    assert pending.pause_id == leaf.pause_data["pause_id"] == result.pause_id
    assert pending.origin_run_id == leaf.id
    assert pending.agent_name == path[-1]
    assert pending.agent_path == path
    shown = pending.to_dict()
    assert shown["origin_run_id"] == leaf.id and shown["agent_path"] == path


# ---------------------------------------------------------------------------
# 1. The four leaf pause kinds surface on the root
# ---------------------------------------------------------------------------


async def _approval_case(store: Any) -> tuple[Agent, RunStatus, str]:
    child = reviewer(
        ContextLLM([call("publish", {"text": "d"}, "c1")]),
        tools=[publish],
        require_approval=["publish"],
    )
    return (
        lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child),
        (RunStatus.WAITING_APPROVAL),
        "approval",
    )


async def _client_case(store: Any) -> tuple[Agent, RunStatus, str]:
    child = reviewer(ContextLLM([call("read_sheet", {"sheet": "S1"}, "c1")]), tools=[read_sheet])
    return (
        lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child),
        (RunStatus.WAITING_CLIENT_TOOL),
        "client_tool",
    )


async def _input_case(store: Any) -> tuple[Agent, RunStatus, str]:
    child = reviewer(ContextLLM([LLMResponse(text="Which draft?")]), tools=[count_words])
    return (
        lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child),
        (RunStatus.WAITING_HUMAN_INPUT),
        "input",
    )


async def _paused_case(store: Any) -> tuple[Agent, RunStatus, str]:
    async def pause_me(index: int, run_id: str | None) -> None:
        if index == 0:  # the root routes the request to the running leaf
            ctx = get_delegation_context()
            assert ctx is not None
            await store.request_pause(run_id)

    child = reviewer(
        ContextLLM([call("count_words", {"text": "a b"}, "c1")], on_call=pause_me),
        tools=[count_words],
    )
    return (
        lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child),
        (RunStatus.PAUSED),
        "paused",
    )


CASES = {
    "approval": _approval_case,
    "client_tool": _client_case,
    "input": _input_case,
    "paused": _paused_case,
}


@pytest.mark.parametrize("case", sorted(CASES))
async def test_leaf_pause_kinds_surface_on_the_root(
    store: Any, engine: Any, case: str, ask_questions: None
) -> None:
    root, status, kind = await CASES[case](store)

    result = await root.run("go", tenant_id=TENANT)

    root_id = result.run_id
    child = await child_of(engine, root_id)
    assert child.status == status.value
    root_row = await run_row(engine, root_id)
    assert root_row.status == "paused" and root_row.pause_data["schema_version"] == 3
    assert_root_facing(result, root_id, child, status=status, kind=kind, path=["lead", "reviewer"])
    assert "managed_origin" not in result.meta

    # The same view after a restart: a fresh read through the store and the Agent.
    pending = await RunStore(store).get_pending(root_id, tenant_id=TENANT)
    assert pending == result.pending
    loaded = await root.get_result(root_id, tenant_id=TENANT)
    assert_root_facing(loaded, root_id, child, status=status, kind=kind, path=["lead", "reviewer"])
    # A child read stays the child's own view.
    own = await RunStore(store).get_pending(child.id, tenant_id=TENANT)
    assert own is not None and own.run_id == child.id and own.origin_run_id is None


async def test_grandchild_approval_surfaces_with_its_path(store: Any, engine: Any) -> None:
    grandchild = auditor(
        ContextLLM([call("publish", {"text": "g"}, "g1")]),
        tools=[publish],
        require_approval=["publish"],
    )
    child = reviewer(ContextLLM([call("auditor", {"task": "audit"}, "c1")]), subagents=[grandchild])
    root = lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child)

    result = await root.run("go", tenant_id=TENANT)

    root_id = result.run_id
    middle = await child_of(engine, root_id)
    leaf = await child_of(engine, middle.id)
    assert middle.status == "paused" and leaf.status == "waiting_approval"
    path = ["lead", "reviewer", "auditor"]
    assert_root_facing(
        result, root_id, leaf, status=RunStatus.WAITING_APPROVAL, kind="approval", path=path
    )
    pending = await RunStore(store).get_pending(root_id, tenant_id=TENANT)
    assert pending == result.pending
    assert pending.tool_calls[0].name == "publish"
    assert pending.tool_calls[0].params == {"text": "g"}
    # Reading the middle run shows the branch below it, addressed to that run.
    below = await RunStore(store).get_pending(middle.id, tenant_id=TENANT)
    assert below.run_id == middle.id and below.agent_path == ["reviewer", "auditor"]


async def test_running_subagent_reads_as_running_root_without_pending(
    store: Any, engine: Any
) -> None:
    seen: list[Any] = []
    holder: dict[str, Any] = {}

    async def look(index: int, run_id: str | None) -> None:
        root_id = holder["root"]
        seen.append(await RunStore(store).get_pending(root_id, tenant_id=TENANT))
        seen.append((await holder["agent"].get_result(root_id, tenant_id=TENANT)).status)

    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="done")])

    async def remember_root(index: int, run_id: str | None) -> None:
        holder.setdefault("root", run_id)

    root_llm.on_call = remember_root
    child = reviewer(ContextLLM([LLMResponse(text=CHILD_TEXT)], on_call=look), tools=[])
    root = lead(store, root_llm, child)
    holder["agent"] = root

    result = await root.run("go", tenant_id=TENANT)

    assert result.status == RunStatus.SUCCESS
    assert seen == [None, RunStatus.RUNNING]


# ---------------------------------------------------------------------------
# 2. Root-id continuations route to the leaf and unwind to the root
# ---------------------------------------------------------------------------


def done(text: str) -> LLMResponse:
    return LLMResponse(text=text)


def build(store: Any, case: str, **child_kwargs: Any) -> Agent:
    """lead -> reviewer, where the reviewer pauses with ``case`` and then finishes."""
    first = {
        "approval": call("publish", {"text": "d"}, "c1"),
        "client_tool": call("read_sheet", {"sheet": "S1"}, "c1"),
        "input": LLMResponse(text="Which draft?"),
        "paused": call("count_words", {"text": "a b"}, "c1"),
    }[case]
    on_call = None
    if case == "paused":

        async def on_call(index: int, run_id: str | None) -> None:
            if index == 0:
                await store.request_pause(run_id)

    tools = {
        "approval": [publish],
        "client_tool": [read_sheet],
        "input": [count_words],
        "paused": [count_words],
    }[case]
    if case == "approval":
        child_kwargs.setdefault("require_approval", ["publish"])
    child = reviewer(
        ContextLLM([first, done(CHILD_TEXT)], on_call=on_call), tools=tools, **child_kwargs
    )
    return lead(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), done("root done")]),
        child,
    )


async def submit_family(root: Agent, case: str, paused: Any, **kwargs: Any) -> Any:
    root_id, token = paused.run_id, paused.pause_id
    kwargs.setdefault("tenant_id", TENANT)
    kwargs.setdefault("expected_pause_id", token)
    if case == "approval":
        return await root.submit_approval(root_id, approved=True, **kwargs)
    if case == "client_tool":
        call_id = paused.pending.tool_calls[0].id
        result = ToolResult(name="read_sheet", call_id=call_id, payload=json.dumps("cells"))
        return await root.submit_tool_results(root_id, [result], **kwargs)
    if case == "input":
        return await root.submit_input(root_id, "the second draft", **kwargs)
    return await root.submit_resume(root_id, **kwargs)


async def assert_unwound(engine: Any, root_id: str, result: Any) -> None:
    """Leaf finished, its report adopted once, root succeeded with its own answer."""
    assert result.run_id == root_id
    assert result.status == RunStatus.SUCCESS
    assert result.answer == "root done" and CHILD_TEXT not in result.answer
    assert result.pending is None
    child = await child_of(engine, root_id)
    assert child.status == "success"
    adopted = await observations(engine, root_id)
    assert len(adopted) == 1 and json.loads(adopted[0].content) == CHILD_TEXT
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "completed" and intent.target == "agent"


@pytest.mark.parametrize("case", sorted(CASES))
async def test_each_family_routes_through_the_root_and_unwinds(
    store: Any, engine: Any, case: str, ask_questions: None
) -> None:
    published.clear()
    root = build(store, case)
    paused = await root.run("go", tenant_id=TENANT)
    assert paused.pending.origin_run_id is not None

    result = await submit_family(root, case, paused)

    await assert_unwound(engine, paused.run_id, result)
    if case == "approval":
        assert published == ["d"]


@pytest.mark.parametrize("case", sorted(CASES))
async def test_managed_continuation_requires_the_pause_token(
    store: Any, engine: Any, case: str, ask_questions: None
) -> None:
    """Even without strict tenancy: a missing token is refused before any claim."""
    root = build(store, case)
    root._strict_tenancy = False
    root.subagents[0]._strict_tenancy = False  # the root's rule alone requires the token
    paused = await root.run("go", tenant_id=TENANT)
    child = await child_of(engine, paused.run_id)

    with pytest.raises(InputValidationError) as caught:
        await submit_family(root, case, paused, expected_pause_id=None)
    assert caught.value.code == "pause_id_required"
    with pytest.raises(PauseStatusMismatchError):
        await submit_family(root, case, paused, expected_pause_id="01ZZZZZZZZZZZZZZZZZZZZZZZZ")

    after = await run_row(engine, child.id)
    assert (after.status, after.claim_epoch) == (child.status, child.claim_epoch)
    assert (await run_row(engine, paused.run_id)).status == "paused"


async def test_resume_routes_each_mode_to_the_leaf(store: Any, engine: Any) -> None:
    published.clear()
    root = build(store, "approval")
    paused = await root.run("go", tenant_id=TENANT)
    result = await root.resume(paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id)
    await assert_unwound(engine, paused.run_id, result)
    assert published == ["d"]


async def test_resume_with_tool_results_routes_to_the_leaf(store: Any, engine: Any) -> None:
    root = build(store, "client_tool")
    paused = await root.run("go", tenant_id=TENANT)
    call_id = paused.pending.tool_calls[0].id
    result = await root.resume(
        paused.run_id,
        tool_results=[ToolResult(name="read_sheet", call_id=call_id, payload='"cells"')],
        tenant_id=TENANT,
        expected_pause_id=paused.pause_id,
    )
    await assert_unwound(engine, paused.run_id, result)


async def test_resume_with_input_routes_to_the_leaf(
    store: Any, engine: Any, ask_questions: None
) -> None:
    root = build(store, "input")
    paused = await root.run("go", tenant_id=TENANT)
    result = await root.resume(
        paused.run_id, user_input="yes", tenant_id=TENANT, expected_pause_id=paused.pause_id
    )
    await assert_unwound(engine, paused.run_id, result)


# ---------------------------------------------------------------------------
# 3. Each family validates with the leaf's own definition
# ---------------------------------------------------------------------------


async def test_tool_results_use_the_leafs_transform_and_limits(store: Any, engine: Any) -> None:
    seen: list[Any] = []

    def leaf_transform(ctx: Any) -> Any:
        seen.append(ctx.tool_name if hasattr(ctx, "tool_name") else ctx)
        return {"leaf": "transformed"}

    root = build(store, "client_tool", tool_result_transform=leaf_transform)
    paused = await root.run("go", tenant_id=TENANT)
    child = await child_of(engine, paused.run_id)
    call_id = paused.pending.tool_calls[0].id

    # Unknown call id: refused by the leaf's pending calls, nothing claimed.
    with pytest.raises(InvalidToolResultError):
        await root.submit_tool_results(
            paused.run_id,
            [ToolResult(name="read_sheet", call_id="other", payload='"x"')],
            tenant_id=TENANT,
            expected_pause_id=paused.pause_id,
        )
    assert (await run_row(engine, child.id)).status == "waiting_client_tool"

    result = await root.submit_tool_results(
        paused.run_id,
        [ToolResult(name="read_sheet", call_id=call_id, payload='"cells"')],
        tenant_id=TENANT,
        expected_pause_id=paused.pause_id,
    )
    await assert_unwound(engine, paused.run_id, result)
    assert len(seen) == 1
    leaf_obs = await observations(engine, child.id)
    assert [json.loads(o.content) for o in leaf_obs] == [{"leaf": "transformed"}]


async def test_approval_decisions_use_the_leafs_tool_catalog(store: Any, engine: Any) -> None:
    published.clear()
    root = build(store, "approval")
    paused = await root.run("go", tenant_id=TENANT)
    leaf_call = paused.pending.tool_calls[0].id

    with pytest.raises(ValueError):
        await root.submit_approval(
            paused.run_id,
            decisions={"not-a-pending-call": ApprovalDecision.approve()},
            tenant_id=TENANT,
            expected_pause_id=paused.pause_id,
        )
    result = await root.submit_approval(
        paused.run_id,
        decisions={leaf_call: ApprovalDecision.deny("no")},
        tenant_id=TENANT,
        expected_pause_id=paused.pause_id,
    )
    await assert_unwound(engine, paused.run_id, result)
    assert published == []  # denied with the leaf's own gate snapshot


async def test_input_meets_the_leafs_input_limit(
    store: Any, engine: Any, ask_questions: None
) -> None:
    root = build(store, "input", limits=Limits(max_input_bytes=40))  # the brief fits
    paused = await root.run("go", tenant_id=TENANT)
    assert paused.status == RunStatus.WAITING_HUMAN_INPUT
    child = await child_of(engine, paused.run_id)
    with pytest.raises(ValueError):
        await root.submit_input(
            paused.run_id, "x" * 64, tenant_id=TENANT, expected_pause_id=paused.pause_id
        )
    assert (await run_row(engine, child.id)).status == "waiting_human_input"
    result = await root.submit_input(
        paused.run_id, "short", tenant_id=TENANT, expected_pause_id=paused.pause_id
    )
    await assert_unwound(engine, paused.run_id, result)


async def test_requested_pause_through_the_root_reaches_the_running_leaf(
    store: Any, engine: Any
) -> None:
    holder: dict[str, Any] = {}

    async def pause_root(index: int, run_id: str | None) -> None:
        if index == 0:
            outcome = await holder["root"].request_pause(
                holder["root_id"], tenant_id=TENANT, cascade=False
            )
            holder["outcome"] = outcome

    async def remember(index: int, run_id: str | None) -> None:
        holder.setdefault("root_id", run_id)

    child = reviewer(
        ContextLLM(
            [call("count_words", {"text": "a b"}, "c1"), done(CHILD_TEXT)], on_call=pause_root
        ),
        tools=[count_words],
    )
    root = lead(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), done("root done")], on_call=remember),
        child,
    )
    holder["root"] = root
    paused = await root.run("go", tenant_id=TENANT)

    assert paused.status == RunStatus.PAUSED and paused.pending.kind == "paused"
    child_row = await child_of(engine, paused.run_id)
    assert paused.pending.origin_run_id == child_row.id
    affected = dict(holder["outcome"].affected)
    assert affected[child_row.id] == "requested"
    result = await root.submit_resume(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    )
    await assert_unwound(engine, paused.run_id, result)


# ---------------------------------------------------------------------------
# 4. Stale tokens, running leaves, child ids and tenant scope
# ---------------------------------------------------------------------------


async def test_stale_token_after_the_same_leaf_pauses_again(store: Any, engine: Any) -> None:
    published.clear()
    child = reviewer(
        ContextLLM(
            [
                call("publish", {"text": "one"}, "c1"),
                call("publish", {"text": "two"}, "c2"),
                done(CHILD_TEXT),
            ]
        ),
        tools=[publish],
        require_approval=["publish"],
    )
    root = lead(
        store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), done("root done")]), child
    )
    first = await root.run("go", tenant_id=TENANT)
    second = await root.submit_approval(
        first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=first.pause_id
    )
    assert second.status == RunStatus.WAITING_APPROVAL and second.pause_id != first.pause_id
    assert second.pending.origin_run_id == first.pending.origin_run_id

    with pytest.raises(PauseStatusMismatchError):
        await root.submit_approval(
            first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=first.pause_id
        )
    assert published == ["one"]
    result = await root.submit_approval(
        first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=second.pause_id
    )
    await assert_unwound(engine, first.run_id, result)
    assert published == ["one", "two"]


async def test_stale_token_never_approves_a_later_sibling(store: Any, engine: Any) -> None:
    published.clear()
    writer = Agent(
        name="writer",
        description="Writes.",
        prompt="You write.",
        provider=ContextLLM([call("publish", {"text": "w"}, "w1"), done("written")]),
        tools=[publish],
        require_approval=["publish"],
    )
    child = reviewer(
        ContextLLM([call("publish", {"text": "r"}, "c1"), done(CHILD_TEXT)]),
        tools=[publish],
        require_approval=["publish"],
    )
    root = lead(
        store,
        ContextLLM(
            [
                call("reviewer", {"task": BRIEF}, "r1"),
                call("writer", {"task": "write"}, "r2"),
                done("root done"),
            ]
        ),
        child,
        writer,
    )
    first = await root.run("go", tenant_id=TENANT)
    second = await root.submit_approval(
        first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=first.pause_id
    )
    assert second.status == RunStatus.WAITING_APPROVAL
    assert second.pending.agent_path == ["lead", "writer"]
    sibling = second.pending.origin_run_id
    before = await run_row(engine, sibling)

    with pytest.raises(PauseStatusMismatchError):
        await root.submit_approval(
            first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=first.pause_id
        )
    after = await run_row(engine, sibling)
    assert (after.status, after.claim_epoch) == (before.status, before.claim_epoch)
    assert published == ["r"]
    result = await root.submit_approval(
        first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=second.pause_id
    )
    assert result.status == RunStatus.SUCCESS and published == ["r", "w"]


async def test_submit_while_the_leaf_runs_reports_a_running_root(store: Any, engine: Any) -> None:
    holder: dict[str, Any] = {}

    async def try_submit(index: int, run_id: str | None) -> None:
        with pytest.raises(RunNotPausedError) as caught:
            await holder["root"].submit_approval(
                holder["root_id"],
                approved=True,
                tenant_id=TENANT,
                expected_pause_id="01ZZZZZZZZZZZZZZZZZZZZZZZZ",
            )
        holder["error"] = caught.value

    async def remember(index: int, run_id: str | None) -> None:
        holder.setdefault("root_id", run_id)

    child = reviewer(ContextLLM([done(CHILD_TEXT)], on_call=try_submit), tools=[])
    root = lead(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), done("root done")], on_call=remember),
        child,
    )
    holder["root"] = root
    result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS
    error = holder["error"]
    assert error.run_id == result.run_id and error.current_status == RunStatus.RUNNING


async def test_child_ids_are_refused_through_the_root_agent(store: Any, engine: Any) -> None:
    root = build(store, "approval")
    paused = await root.run("go", tenant_id=TENANT)
    child = await child_of(engine, paused.run_id)
    for attempt in (
        lambda: root.submit_approval(
            child.id, approved=True, tenant_id=TENANT, expected_pause_id=paused.pause_id
        ),
        lambda: root.resume(child.id, tenant_id=TENANT, expected_pause_id=paused.pause_id),
        lambda: root.request_pause(child.id, tenant_id=TENANT),
    ):
        with pytest.raises(ManagedChildControlError):
            await attempt()
    assert (await run_row(engine, child.id)).status == "waiting_approval"


@pytest.mark.parametrize("scope", ["foreign", "none"])
async def test_foreign_and_none_scopes_are_not_found(store: Any, engine: Any, scope: str) -> None:
    root = build(store, "approval")
    paused = await root.run("go", tenant_id=TENANT)
    child = await child_of(engine, paused.run_id)
    tenant = "other" if scope == "foreign" else None
    with pytest.raises(RunNotFoundError):
        await root.submit_approval(
            paused.run_id, approved=True, tenant_id=tenant, expected_pause_id=paused.pause_id
        )
    with pytest.raises(RunNotFoundError):
        await RunStore(store).get_pending(paused.run_id, tenant_id=tenant)
    with pytest.raises(RunNotFoundError):
        await root.get_result(paused.run_id, tenant_id=tenant)
    assert (await run_row(engine, child.id)).claim_epoch == child.claim_epoch


async def test_omitted_scope_under_strict_tenancy_is_refused(store: Any, engine: Any) -> None:
    from dendrux.errors import TenantScopeRequiredError

    root = build(store, "approval")
    paused = await root.run("go", tenant_id=TENANT)
    with pytest.raises(TenantScopeRequiredError):
        await root.submit_approval(paused.run_id, approved=True, expected_pause_id=paused.pause_id)


async def test_unscoped_trusted_caller_claims_with_the_roots_actual_tenant(
    store: Any, engine: Any
) -> None:
    published.clear()
    root = build(store, "approval")
    root._strict_tenancy = False
    paused = await root.run("go", tenant_id=TENANT)
    result = await root.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )
    await assert_unwound(engine, paused.run_id, result)
    child = await child_of(engine, paused.run_id)
    assert child.tenant_id == TENANT and published == ["d"]


# ---------------------------------------------------------------------------
# 5. The spec §3 journey, notes, and the root-vs-child control race
# ---------------------------------------------------------------------------


def journey(store: Any, *, restarted: bool = False) -> tuple[Agent, list[ContextLLM]]:
    """lead -> reviewer -> auditor (approval). ``restarted``: a fresh process whose
    providers continue after each run's first turn."""
    skip = 1 if restarted else 0
    llms = [
        ContextLLM([call("publish", {"text": "A1"}, "a1"), done("auditor report")][skip:]),
        ContextLLM([call("auditor", {"task": "audit"}, "t2"), done(CHILD_TEXT)][skip:]),
        ContextLLM([call("reviewer", {"task": BRIEF}, "t1"), done("root done")][skip:]),
    ]
    grand = auditor(llms[0], tools=[publish], require_approval=["publish"])
    child = reviewer(llms[1], subagents=[grand])
    return lead(store, llms[2], child), llms


async def event_types(engine: Any, run_id: str) -> list[str]:
    from dendrux.db.models import RunEvent

    found = await rows(engine, RunEvent, RunEvent.agent_run_id == run_id)
    return [e.event_type for e in sorted(found, key=lambda e: e.sequence_index)]


async def test_depth_two_journey_root_to_grandchild_approval_and_back(
    store: Any, engine: Any
) -> None:
    """Spec §3 sample journey: R -> C -> G approval, restart, root-id submit, unwind."""
    from dendrux.runtime.stores._factory import store_for_engine

    published.clear()
    first, _ = journey(store)
    paused = await first.run("go", tenant_id=TENANT)
    root_id = paused.run_id
    middle = await child_of(engine, root_id)
    leaf = await child_of(engine, middle.id)
    assert paused.status == RunStatus.WAITING_APPROVAL
    assert paused.pause_id == leaf.pause_data["pause_id"]
    for parent in (await run_row(engine, root_id), middle):  # private waits, never the token
        assert parent.pause_data["pause_id"] != paused.pause_id

    # -- process exits; the caller reloads pending through a fresh store and Agent --
    fresh_store = store_for_engine(engine)
    pending = await RunStore(fresh_store).get_pending(root_id, tenant_id=TENANT)
    assert pending.pause_id == paused.pause_id and pending.origin_run_id == leaf.id
    restarted, llms = journey(fresh_store, restarted=True)
    result = await restarted.submit_approval(
        root_id, approved=True, expected_pause_id=pending.pause_id, tenant_id=TENANT
    )

    assert (result.run_id, result.status, result.answer) == (
        root_id,
        RunStatus.SUCCESS,
        "root done",
    )
    assert published == ["A1"]
    assert [llm.calls for llm in llms] == [1, 1, 1]  # each run's final turn, once
    for run_id in (leaf.id, middle.id, root_id):
        assert (await run_row(engine, run_id)).status == "success"
        assert (await event_types(engine, run_id)).count("tool.completed") == 1
    for run_id, report in ((middle.id, "auditor report"), (root_id, CHILD_TEXT)):
        adopted = await observations(engine, run_id)
        assert [json.loads(o.content) for o in adopted] == [report]
    for parent_id in (root_id, middle.id):
        intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == parent_id))[0]
        assert intent.outcome == "completed"
    assert (await RunStore(fresh_store).get_pending(root_id, tenant_id=TENANT)) is None


async def test_root_note_is_delivered_when_the_root_runs_again(store: Any, engine: Any) -> None:
    """§5a: addressed-run notes; a root note is never forwarded to the active child."""
    note = "SENTINEL-root-note-55e1"
    root = build(store, "approval")
    paused = await root.run("go", tenant_id=TENANT)
    await root.submit_note(paused.run_id, note, tenant_id=TENANT)

    result = await root.submit_approval(
        paused.run_id, approved=True, tenant_id=TENANT, expected_pause_id=paused.pause_id
    )

    assert result.status == RunStatus.SUCCESS
    child_llm = root.subagents[0].provider
    root_llm = root.provider
    assert all(note not in str(m.content) for batch in child_llm.messages for m in batch)
    assert note not in "".join(str(m.content) for m in root_llm.messages[0])
    assert any(note in str(m.content) for m in root_llm.messages[-1])
    notes = await RunStore(store).get_notes(paused.run_id, tenant_id=TENANT)
    assert [n.delivered_at is not None for n in notes] == [True]


def _postgres(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("independent-session contenders run on Postgres")


async def test_pg_root_versus_child_control_race(store: Any, engine: Any) -> None:
    """Concurrent root-id approvals and direct child controls on independent engines:
    exactly one leaf claim, one effect, one adoption per parent; child controls change
    nothing."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.errors import RunAlreadyClaimedError
    from dendrux.runtime.stores._factory import store_for_engine

    _postgres(engine)
    for _ in range(3):
        published.clear()
        first, _ = journey(store)
        paused = await first.run("go", tenant_id=TENANT)
        root_id = paused.run_id
        middle = await child_of(engine, root_id)
        leaf = await child_of(engine, middle.id)
        second = create_async_engine(engine.url, pool_size=8)
        try:
            other = store_for_engine(second)
            roots = [journey(store if i % 2 else other, restarted=True) for i in range(6)]
            direct = [
                auditor(
                    ContextLLM([]),
                    tools=[publish],
                    require_approval=["publish"],
                    state_store=store if i % 2 else other,
                )
                for i in range(4)
            ]
            attempts = [
                agent.submit_approval(
                    root_id, approved=True, tenant_id=TENANT, expected_pause_id=paused.pause_id
                )
                for agent, _ in roots
            ] + [
                d.submit_approval(
                    leaf.id, approved=True, tenant_id=TENANT, expected_pause_id=paused.pause_id
                )
                for d in direct
            ]
            outcomes = await asyncio.gather(*attempts, return_exceptions=True)
        finally:
            await second.dispose()

        root_outcomes, child_outcomes = outcomes[:6], outcomes[6:]
        assert all(isinstance(o, ManagedChildControlError) for o in child_outcomes)
        winners = [o for o in root_outcomes if not isinstance(o, BaseException)]
        assert len(winners) == 1 and winners[0].status == RunStatus.SUCCESS
        for loser in root_outcomes:
            if isinstance(loser, BaseException):
                assert isinstance(
                    loser, (RunAlreadyClaimedError, PauseStatusMismatchError, RunNotPausedError)
                ), repr(loser)
        assert published == ["A1"]
        assert sum(llm.calls for _, llms in roots for llm in llms) == 3
        for run_id in (leaf.id, middle.id, root_id):
            assert (await run_row(engine, run_id)).status == "success"
            assert (await event_types(engine, run_id)).count("tool.completed") == 1
        for run_id in (middle.id, root_id):
            assert len(await observations(engine, run_id)) == 1


# ---------------------------------------------------------------------------
# Fix round 1, R1: a tokenless continuation never crosses RUNNING -> pause
# ---------------------------------------------------------------------------


def _own_pause_root(store: Any, entry: str, holder: dict[str, Any]) -> tuple[Agent, ContextLLM]:
    """A managed root that pauses itself: for input (a question) or a requested pause."""
    if entry == "submit_resume":
        first = call("count_words", {"text": "a b"}, "w1")
    else:
        first = LLMResponse(text="Which draft?")
    llm = ContextLLM([first, done("root done")])
    llm.gate_at = 0

    async def remember(index: int, run_id: str | None) -> None:
        holder.setdefault("root_id", run_id)

    llm.on_call = remember
    root = lead(store, llm, reviewer(ContextLLM([]), tools=[]), tools=[count_words])
    root._strict_tenancy = False
    return root, llm


async def _tokenless(agent: Agent, entry: str, root_id: str) -> Any:
    if entry == "submit_input":
        return await agent.submit_input(root_id, "late answer")
    if entry == "submit_resume":
        return await agent.submit_resume(root_id)
    if entry == "resume":
        return await agent.resume(root_id, user_input="late answer")
    stream = agent.resume_stream(root_id, user_input="late answer")
    async with stream:
        return [event async for event in stream]


@pytest.mark.parametrize("entry", ["submit_input", "submit_resume", "resume", "resume_stream"])
async def test_pg_tokenless_continuation_cannot_cross_root_running_to_its_own_pause(
    store: Any, engine: Any, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing reads the root RUNNING; the root then persists its own pause before the
    caller's ordinary preflight. The tokenless caller is refused; nothing is claimed."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime import _managed
    from dendrux.runtime.stores._factory import store_for_engine

    _postgres(engine)
    patch_questions(monkeypatch)
    holder: dict[str, Any] = {}
    root, llm = _own_pause_root(store, entry, holder)
    running = asyncio.create_task(root.run("go", tenant_id=TENANT))
    entered = asyncio.create_task(llm.entered.wait())
    await asyncio.wait({running, entered}, return_when=asyncio.FIRST_COMPLETED)
    if running.done():
        running.result()  # surface a setup failure instead of waiting forever
    root_id = holder["root_id"]
    if entry == "submit_resume":
        await store.request_pause(root_id)
    original = _managed.route

    async def barrier(*args: Any, **kwargs: Any) -> Any:
        found = await original(*args, **kwargs)
        assert found.leaf.status == "running"  # the routing snapshot
        llm.gate.set()
        holder["paused"] = await running  # the root persists its own pause now
        return found

    monkeypatch.setattr(_managed, "route", barrier)
    second = create_async_engine(engine.url, pool_size=2)
    try:
        other, other_llm = _own_pause_root(store_for_engine(second), entry, {})
        other_llm.gate_at = None  # if wrongly admitted, the continuation runs to completion
        with pytest.raises((InputValidationError, RunNotPausedError)):
            await _tokenless(other, entry, root_id)
    finally:
        await second.dispose()
    paused = holder["paused"]
    assert paused.status in (RunStatus.WAITING_HUMAN_INPUT, RunStatus.PAUSED)
    row = await run_row(engine, root_id)
    assert row.status == paused.status.value
    assert row.claim_epoch == (await run_row(engine, root_id)).claim_epoch
    assert not any(k.startswith("submitted") for k in (row.pause_data or {}))
    assert llm.calls == 1  # the root never continued


@pytest.mark.parametrize("entry", ["submit_input", "resume", "resume_stream"])
async def test_an_agent_without_registrations_still_needs_the_token_for_a_managed_run(
    store: Any, engine: Any, entry: str, ask_questions: None
) -> None:
    """Claim admission reads the persisted managed identity, not the caller's Agent."""
    root = build(store, "input")
    paused = await root.run("go", tenant_id=TENANT)
    root_row = await run_row(engine, paused.run_id)
    # A root that paused for its own question: rebuilt without subagents, non-strict.
    own = lead(store, ContextLLM([LLMResponse(text="Which one?")]), reviewer(ContextLLM([])))
    own._strict_tenancy = False
    mine = await own.run("go", tenant_id=TENANT)
    assert mine.status == RunStatus.WAITING_HUMAN_INPUT and mine.pending.origin_run_id is None
    plain = Agent(
        name="lead",
        prompt="You lead.",
        provider=ContextLLM([done("x")]),
        state_store=store,
        strict_tenancy=False,
    )
    for run_id in (mine.run_id,):
        before = await run_row(engine, run_id)
        if entry == "resume_stream":  # the ordinary stream reports setup errors as RUN_ERROR
            events = await _tokenless(plain, entry, run_id)
            assert events[-1].type.value == "run_error"
            assert "pause_id_required" in (events[-1].error or "")
        else:
            with pytest.raises(InputValidationError) as caught:
                await _tokenless(plain, entry, run_id)
            assert caught.value.code == "pause_id_required"
        after = await run_row(engine, run_id)
        assert (after.status, after.claim_epoch) == (before.status, before.claim_epoch)
    assert (await run_row(engine, paused.run_id)).status == root_row.status


# ---------------------------------------------------------------------------
# Fix round 1, R4: a result read never fabricates RUNNING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["legacy_requested", "managed_root"])
async def test_cancel_between_record_and_pending_reads_never_reads_as_running(
    store: Any, engine: Any, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dendrux.runtime._pending import _PendingReader

    if kind == "legacy_requested":

        async def pause_me(index: int, run_id: str | None) -> None:
            if index == 0:
                await store.request_pause(run_id)

        agent = Agent(
            name="solo",
            prompt="p",
            provider=ContextLLM([call("count_words", {"text": "a"}, "w1")], on_call=pause_me),
            tools=[count_words],
            state_store=store,
        )
        paused = await agent.run("go", tenant_id=TENANT)
        assert paused.status == RunStatus.PAUSED and paused.pending.kind == "paused"
    else:
        agent = build(store, "approval")
        paused = await agent.run("go", tenant_id=TENANT)
        assert paused.status == RunStatus.WAITING_APPROVAL
    run_id = paused.run_id
    original = _PendingReader._summary
    fired: list[bool] = []

    async def cancel_first(self: Any, target: str, tenant_id: Any) -> Any:
        if not fired:  # between the result-record read and the pending read
            fired.append(True)
            await agent.cancel_run(run_id, tenant_id=TENANT, cascade=True)
        return await original(self, target, tenant_id)

    monkeypatch.setattr(_PendingReader, "_summary", cancel_first)
    loaded = await agent.get_result(run_id, tenant_id=TENANT)

    assert fired
    assert loaded.status != RunStatus.RUNNING
    assert loaded.status in (RunStatus.PAUSED, RunStatus.WAITING_APPROVAL, RunStatus.CANCELLED)
    assert loaded.pending is None
    assert (await run_row(engine, run_id)).status == "cancelled"
