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

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.runtime.context import get_delegation_context
from dendrux.store import RunStore
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    AgentStep,
    Clarification,
    LLMResponse,
    RunStatus,
    ToolCall,
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
