"""Managed subagents (0.7, T4): the abandoned-waiting sweep and managed waits.

A parent's v3 managed wait is not an unanswered human request: the sweep skips it,
including when its child has just finished (adoption work, not abandonment). The
exemption is projected into the existing bounded sweep pages (no history read, no
JSON index) and rechecked at the final row CAS. Every other waiting row keeps
today's predicates exactly (golden test), including a legacy delegation parent
whose child still runs. A corrupt managed wait is reported, never recreated. A
subagent's own external pause is ordinary: the sweep may abandon it and the root
later adopts that failed outcome. Runs on SQLite and Postgres.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, sweep, tool
from dendrux._lifecycle import PAUSE_STATUS_VALUES
from dendrux.db.models import AgentRun, ReactTrace, RunEvent
from dendrux.errors import ManagedStateError
from dendrux.runtime import _managed
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-sweep-brief-2f7b"
REPORT = "SENTINEL-sweep-report-6d1a"
TENANT = "t1"
OLD = timedelta(hours=3)
THRESHOLD = timedelta(hours=1)


class ProcessDeathError(BaseException):
    """Ends the caller like a killed process: no runner cleanup finalizes the run."""


class ContextLLM(ScriptedLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.served: list[str | None] = []

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        self.served.append(ctx.run_id if ctx else None)
        return await super().complete(messages, tools, **kwargs)


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


published: list[str] = []


@tool()
async def publish(text: str) -> str:
    """Publish a draft."""
    published.append(text)
    return "published"


@tool(target="client")
async def read_sheet(sheet: str) -> str:
    """Read a sheet in the user's browser."""
    return ""


def reviewer(llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="reviewer", description="Reviews one draft.", prompt="r", provider=llm, **kwargs
    )


def lead(store: Any, llm: Any, child: Agent) -> Agent:
    return Agent(name="lead", prompt="l", provider=llm, state_store=store, subagents=[child])


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def run_row(engine: Any, run_id: str) -> Any:
    return (await rows(engine, AgentRun, AgentRun.id == run_id))[0]


async def age(engine: Any, *run_ids: str, by: timedelta = OLD) -> None:
    """Make waiting rows look abandoned: both sweep timestamps older than the threshold."""
    old = datetime.now(UTC) - by
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_(run_ids))
            .values(status_changed_at=old, updated_at=old)
        )


async def _approval_waiting(store: Any, engine: Any) -> tuple[str, str]:
    """Root waits (v3) on a child that waits for approval."""
    child = reviewer(
        ContextLLM([call("publish", {"text": "d"}, "c1")]),
        tools=[publish],
        require_approval=["publish"],
    )
    result = await lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child).run(
        "go", tenant_id=TENANT
    )
    assert result.status == RunStatus.WAITING_APPROVAL
    child_row = (await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id))[0]
    return result.run_id, child_row.id


async def _terminal_child_waiting(store: Any, engine: Any, monkeypatch: Any) -> tuple[str, str]:
    """Root waits (v3) on a child that finished; the process died before adoption."""

    async def die(*args: Any, **kwargs: Any) -> Any:
        raise ProcessDeathError

    root = lead(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
        reviewer(ContextLLM([LLMResponse(text=REPORT)])),
    )
    with monkeypatch.context() as patch:
        patch.setattr(_managed, "adopt", die)
        with pytest.raises(ProcessDeathError):
            await root.run("go", tenant_id=TENANT)
    root_id = (await rows(engine, AgentRun, AgentRun.agent_name == "lead"))[0].id
    child = (await rows(engine, AgentRun, AgentRun.parent_run_id == root_id))[0]
    assert child.status == "success"
    return root_id, child.id


async def _abandoned(store: Any) -> list[str]:
    result = await sweep(state_store=store, abandoned_waiting=THRESHOLD)
    return [r.run_id for r in result.abandoned_waiting]


# ---------------------------------------------------------------------------
# Managed waits are skipped
# ---------------------------------------------------------------------------


async def test_sweep_skips_a_managed_wait_and_abandons_the_leaf_pause_ordinarily(
    store: Any, engine: Any
) -> None:
    """The leaf's approval is an ordinary pause; its parent's managed wait is not."""
    root_id, child_id = await _approval_waiting(store, engine)
    before = await run_row(engine, root_id)
    await age(engine, root_id, child_id)

    assert await _abandoned(store) == [child_id]

    root = await run_row(engine, root_id)
    assert (root.status, root.claim_epoch) == ("paused", before.claim_epoch)
    assert root.pause_data == before.pause_data and root.pause_data["schema_version"] == 3
    child = await run_row(engine, child_id)
    assert (child.status, child.failure_reason) == ("error", "abandoned_waiting")

    # The root later adopts the abandoned leaf as a failed outcome, without rerunning it.
    child_llm = ContextLLM([])
    root_llm = ContextLLM([LLMResponse(text="handled")])
    child = reviewer(child_llm, tools=[publish], require_approval=["publish"])
    fresh = lead(store, root_llm, child)
    result = await fresh.reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and result.answer == "handled"
    assert child_llm.calls == 0 and published == []
    seen = [json.loads(m.content) for m in root_llm.messages[0] if m.role.value == "tool"]
    assert [o["code"] for o in seen] == ["managed_child_failed"]


async def test_sweep_skips_a_wait_whose_child_just_finished(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Terminal-child cut: the wait holds adoption work, not an unanswered request."""
    root_id, child_id = await _terminal_child_waiting(store, engine, monkeypatch)
    before = await run_row(engine, root_id)
    await age(engine, root_id, child_id)

    assert await _abandoned(store) == []
    root = await run_row(engine, root_id)
    assert (root.status, root.claim_epoch, root.pause_data) == (
        "paused",
        before.claim_epoch,
        before.pause_data,
    )

    child_llm = ContextLLM([])
    root_llm = ContextLLM([LLMResponse(text="root done")])
    result = await lead(store, root_llm, reviewer(child_llm)).reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 0
    seen = [json.loads(m.content) for m in root_llm.messages[0] if m.role.value == "tool"]
    assert seen == [REPORT]


async def test_sweep_rechecks_the_managed_exemption_at_the_final_cas(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """A row read as an ordinary pause that is a managed wait by its CAS is not swept."""
    root_id, child_id = await _terminal_child_waiting(store, engine, monkeypatch)
    wait = (await run_row(engine, root_id)).pause_data
    ordinary = {**wait, "schema_version": 2, "pause_reason": None}
    ordinary.pop("managed_wait")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == root_id).values(pause_data=ordinary)
        )
    await age(engine, root_id, child_id)
    original = type(store)._update_one_rows
    restored: list[bool] = []

    async def restore_then_update(self: Any, session: Any, **kwargs: Any) -> Any:
        if not restored:  # the page saw the ordinary pause; the row is a managed wait now
            restored.append(True)
            async with engine.begin() as conn:
                await conn.execute(
                    update(AgentRun)
                    .where(AgentRun.id == root_id)
                    .values(pause_data=wait, updated_at=datetime.now(UTC) - OLD)
                )
        return await original(self, session, **kwargs)

    monkeypatch.setattr(type(store), "_update_one_rows", restore_then_update)
    assert await _abandoned(store) == []
    assert restored
    root = await run_row(engine, root_id)
    assert root.status == "paused" and root.pause_data == wait


@pytest.mark.parametrize("corruption", ["no_binding", "reason_only", "version_only"])
async def test_a_corrupt_managed_wait_is_reported_never_swept_or_recreated(
    store: Any, engine: Any, monkeypatch: Any, caplog: Any, corruption: str
) -> None:
    root_id, child_id = await _terminal_child_waiting(store, engine, monkeypatch)
    wait = dict((await run_row(engine, root_id)).pause_data)
    if corruption == "no_binding":
        wait.pop("managed_wait")
    elif corruption == "reason_only":
        wait["schema_version"] = 2
    else:
        wait["pause_reason"] = "requested"
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == root_id).values(pause_data=wait))
    await age(engine, root_id, child_id)

    with caplog.at_level(logging.WARNING):
        assert await _abandoned(store) == []
    codes = [
        (getattr(r, "code", None), getattr(r, "run_id", None), getattr(r, "managed_reason", None))
        for r in caplog.records
    ]
    assert ("managed_state_invalid", root_id, "wait_invalid") in codes
    root = await run_row(engine, root_id)
    assert root.status == "paused" and root.pause_data == wait
    with pytest.raises(ManagedStateError):
        await lead(store, ContextLLM([]), reviewer(ContextLLM([]))).reclaim(
            root_id, tenant_id=TENANT
        )
    assert len(await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)) == 1


async def test_skipped_managed_waits_cost_no_per_row_statement_and_no_history_read(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """The exemption is projected into the page: no per-row read, no trace or pause load."""
    root_id, child_id = await _terminal_child_waiting(store, engine, monkeypatch)
    await age(engine, root_id, child_id)
    statements: list[str] = []

    def capture(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await _abandoned(store) == []
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    lowered = [s.lower() for s in statements]
    assert not any(ReactTrace.__tablename__ in s for s in lowered)
    assert not any(s.lstrip().startswith("update") for s in lowered)
    selects = [s for s in lowered if s.lstrip().startswith("select")]
    # Only candidate pages: one per waiting status plus the page after the skipped row.
    assert all("as sweep_age" in s for s in selects)
    assert len(selects) == len(PAUSE_STATUS_VALUES) + 1
    # pause_data is read only through JSON path projections, never loaded whole.
    for statement in selects:
        bare = re.sub(r"(json_type|json_extract|cast)\(agent_runs\.pause_data", "", statement)
        assert "agent_runs.pause_data" not in bare


# ---------------------------------------------------------------------------
# Golden: every non-managed waiting row keeps today's sweep
# ---------------------------------------------------------------------------


async def test_non_managed_waiting_rows_keep_todays_sweep(store: Any, engine: Any) -> None:
    """Legacy pauses of every kind, and a legacy delegation parent with a running child,
    are swept exactly as before; fresh rows are not."""
    from dendrux.types import generate_ulid

    async def paused_run(status: str, pause: dict[str, Any] | None, **fields: Any) -> str:
        run_id = generate_ulid()
        await store.create_run(run_id, "legacy", tenant_id=TENANT, **fields)
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(status=status, pause_data=pause)
            )
        return run_id

    v2 = {"schema_version": 2, "agent_name": "legacy", "pending_tool_calls": []}
    ids = {
        "client": await paused_run("waiting_client_tool", {**v2, "pause_reason": None}),
        "human": await paused_run("waiting_human_input", {**v2}),
        "approval": await paused_run("waiting_approval", {**v2, "approval_gates": {}}),
        "requested": await paused_run("paused", {**v2, "pause_reason": "requested"}),
        "v1": await paused_run("paused", {"agent_name": "legacy", "pending_tool_calls": []}),
        "no_pause_data": await paused_run("waiting_client_tool", None),
        "odd_reason": await paused_run("paused", {**v2, "pause_reason": "managed"}),
    }
    parent = await paused_run("waiting_client_tool", {**v2})
    child = generate_ulid()
    await store.create_run(child, "legacy-child", tenant_id=TENANT, parent_run_id=parent)
    ids["delegating_parent"] = parent
    fresh = await paused_run("waiting_client_tool", {**v2})
    epochs = {name: (await run_row(engine, run_id)).claim_epoch for name, run_id in ids.items()}
    await age(engine, *ids.values())

    result = await sweep(state_store=store, abandoned_waiting=THRESHOLD)

    swept = {r.run_id: r for r in result.abandoned_waiting}
    assert set(swept) == set(ids.values())
    for name, run_id in ids.items():
        row = await run_row(engine, run_id)
        assert (row.status, row.failure_reason, row.pause_data) == (
            "error",
            "abandoned_waiting",
            None,
        ), name
        assert row.error == "Run swept as abandoned_waiting", name
        assert row.claim_epoch == epochs[name] + 1, name
        assert (swept[run_id].status, swept[run_id].failure_reason) == (
            "error",
            "abandoned_waiting",
        )
        events = await rows(engine, RunEvent, RunEvent.agent_run_id == run_id)
        kinds = [e.event_type for e in events]
        assert kinds.count("run.abandoned") == 1, name
    assert (await run_row(engine, fresh)).status == "waiting_client_tool"
    assert (await run_row(engine, child)).status == "running"


# ---------------------------------------------------------------------------
# Stale running children keep ordinary stale/reclaim behaviour
# ---------------------------------------------------------------------------


async def test_a_stale_running_child_finalised_by_sweep_is_adopted_as_failed(
    store: Any, engine: Any
) -> None:
    class DyingLLM(ContextLLM):
        async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
            raise ProcessDeathError

    root = lead(
        store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), reviewer(DyingLLM([]))
    )
    with pytest.raises(ProcessDeathError):
        await root.run("go", tenant_id=TENANT)
    root_id = (await rows(engine, AgentRun, AgentRun.agent_name == "lead"))[0].id
    child = (await rows(engine, AgentRun, AgentRun.parent_run_id == root_id))[0]
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == child.id)
            .values(last_progress_at=datetime.now(UTC) - OLD)
        )
    await age(engine, root_id)
    swept = await sweep(
        state_store=store, stale_running=THRESHOLD, abandoned_waiting=THRESHOLD, on_stale="error"
    )
    assert [r.run_id for r in swept.stale_running] == [child.id]
    assert swept.abandoned_waiting == []

    child_llm = ContextLLM([])
    root_llm = ContextLLM([LLMResponse(text="handled")])
    result = await lead(store, root_llm, reviewer(child_llm)).reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 0
    seen = [json.loads(m.content) for m in root_llm.messages[0] if m.role.value == "tool"]
    assert [o["code"] for o in seen] == ["managed_child_failed"]


async def test_a_stale_running_child_marked_by_sweep_is_reclaimed_through_its_root(
    store: Any, engine: Any
) -> None:
    class DyingLLM(ContextLLM):
        async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
            raise ProcessDeathError

    root = lead(
        store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), reviewer(DyingLLM([]))
    )
    with pytest.raises(ProcessDeathError):
        await root.run("go", tenant_id=TENANT)
    root_id = (await rows(engine, AgentRun, AgentRun.agent_name == "lead"))[0].id
    child = (await rows(engine, AgentRun, AgentRun.parent_run_id == root_id))[0]
    item = await expire(store, engine, child.id)
    assert item.run_id == child.id

    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([LLMResponse(text="root done")])
    result = await lead(store, root_llm, reviewer(child_llm)).reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS
    assert child_llm.calls == 1 and set(child_llm.served) == {child.id}
    assert len(await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)) == 1
