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


@pytest.mark.parametrize(
    "corruption", ["no_binding", "reason_only", "version_only", "empty_child", "null_child"]
)
async def test_a_corrupt_managed_wait_is_reported_never_swept_or_recreated(
    store: Any, engine: Any, monkeypatch: Any, caplog: Any, corruption: str
) -> None:
    root_id, child_id = await _terminal_child_waiting(store, engine, monkeypatch)
    wait = dict((await run_row(engine, root_id)).pause_data)
    if corruption == "no_binding":
        wait.pop("managed_wait")
    elif corruption in {"empty_child", "null_child"}:
        wait["managed_wait"] = {
            **wait["managed_wait"],
            "child_run_id": "" if corruption == "empty_child" else None,
        }
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
    # One bounded page per status; a short page needs no empty follow-up read.
    assert all("as sweep_age" in s for s in selects)
    assert len(selects) == len(PAUSE_STATUS_VALUES)
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


# ---------------------------------------------------------------------------
# Fix round 1, R1: exempt managed waits never starve ordinary abandoned pauses
# ---------------------------------------------------------------------------


async def _another_terminal_wait(store: Any, engine: Any, monkeypatch: Any) -> str:
    before = {r.id for r in await rows(engine, AgentRun, AgentRun.agent_name == "lead")}
    await _terminal_child_waiting_any(store, monkeypatch)
    after = {r.id for r in await rows(engine, AgentRun, AgentRun.agent_name == "lead")}
    [root_id] = after - before
    return root_id


async def _terminal_child_waiting_any(store: Any, monkeypatch: Any) -> None:
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


async def test_bounded_sweeps_reach_an_ordinary_pause_behind_more_than_a_cap_of_waits(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Three old managed waits ahead of an ordinary requested pause, cap two rows per
    sweep: repeated bounded sweeps still abandon the ordinary pause."""
    from dendrux.types import generate_ulid

    waits = [await _another_terminal_wait(store, engine, monkeypatch) for _ in range(3)]
    ordinary = generate_ulid()
    await store.create_run(ordinary, "legacy", tenant_id=TENANT)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ordinary)
            .values(status="paused", pause_data={"schema_version": 2, "pause_reason": "requested"})
        )
    await age(engine, *waits, by=timedelta(hours=4))
    await age(engine, ordinary, by=timedelta(hours=3))

    swept: list[str] = []
    cursor = None
    for _ in range(3):
        result = await sweep(
            state_store=store, abandoned_waiting=THRESHOLD, max_runs=2, batch_size=1, cursor=cursor
        )
        cursor = result.cursor
        swept.extend(r.run_id for r in result.abandoned_waiting)
    assert swept == [ordinary]
    for run_id in waits:
        row = await run_row(engine, run_id)
        assert row.status == "paused" and row.pause_data["schema_version"] == 3


async def _seed_wait_prefix(engine: Any, count: int, *, corrupt: bool = False) -> list[str]:
    old = datetime.now(UTC) - OLD
    ids = [f"{i:026d}" for i in range(count)]
    wait = {"schema_version": 3, "pause_reason": "managed_wait"}
    if not corrupt:
        wait["managed_wait"] = {"child_run_id": "child"}
    async with engine.begin() as conn:
        await conn.execute(
            AgentRun.__table__.insert(),
            [
                dict(
                    id=run_id,
                    agent_name="wait",
                    status="paused",
                    claim_epoch=1,
                    storage_format=2,
                    updated_at=old,
                    status_changed_at=old,
                    pause_data=wait,
                )
                for run_id in ids
            ],
        )
    return ids


@pytest.mark.parametrize("population", [100, 1000, 10000])
async def test_all_exempt_scan_bounds_examined_work_and_continues(
    store: Any, engine: Any, monkeypatch: Any, population: int
) -> None:
    if engine.dialect.name != "sqlite":
        pytest.skip("SQLite VM instruction measurement")
    from dendrux.runtime.stores.sql.base import _SweepScan

    ids = await _seed_wait_prefix(engine, population)
    visited: list[str] = []
    original = _SweepScan.skipped

    def observe(self: Any, row: Any) -> bool:
        visited.append(row.id)
        return original(self, row)

    monkeypatch.setattr(_SweepScan, "skipped", observe)
    instructions = [0]

    def tick() -> int:
        instructions[0] += 1
        return 0

    cap = 37
    cursor = None
    costs = []
    async with engine.connect() as conn:
        raw = await conn.get_raw_connection()
        driver = raw.driver_connection
    for _ in range((population + cap - 1) // cap + 1):
        start = len(visited)
        instructions[0] = 0
        await driver.set_progress_handler(tick, 1)
        try:
            result = await sweep(
                state_store=store,
                abandoned_waiting=THRESHOLD,
                max_runs=cap,
                batch_size=cap,
                cursor=cursor,
            )
        finally:
            await driver.set_progress_handler(None, 1)
        costs.append(instructions[0])
        assert instructions[0] < 10000, costs
        assert len(visited) - start <= cap
        assert not result.abandoned_waiting
        cursor = result.cursor
        if cursor is None:
            break
    assert visited == ids
    assert cursor is None
    print(
        f"population={population} cap={cap} calls={len(costs)} "
        f"max_vm={max(costs)} examined={len(visited)}"
    )


@pytest.mark.parametrize("prefix_status", PAUSE_STATUS_VALUES)
async def test_corrupt_prefix_continues_without_repeating_reports(
    store: Any, engine: Any, caplog: Any, prefix_status: str
) -> None:
    from dendrux.types import generate_ulid

    ids = await _seed_wait_prefix(engine, 7, corrupt=True)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id.in_(ids)).values(status=prefix_status)
        )
    ordinary = generate_ulid()
    await store.create_run(ordinary, "ordinary")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ordinary)
            .values(status="paused", pause_data={"schema_version": 2})
        )
    await age(engine, ordinary, by=timedelta(hours=2))
    before = [(await run_row(engine, rid)).pause_data for rid in ids]
    cursor = None
    swept = []
    reported = []
    with caplog.at_level(logging.WARNING):
        for _ in range(5):  # ceil(7 / 2) + 1
            caplog.clear()
            result = await sweep(
                state_store=store,
                abandoned_waiting=THRESHOLD,
                max_runs=2,
                batch_size=1,
                cursor=cursor,
            )
            reports = [
                r.run_id
                for r in caplog.records
                if getattr(r, "code", None) == "managed_state_invalid"
            ]
            reported.extend(reports)
            assert len(reports) <= 2
            swept.extend(r.run_id for r in result.abandoned_waiting)
            cursor = result.cursor
            if cursor is None:
                break
    assert swept == [ordinary]
    assert reported == ids
    for rid, wait in zip(ids, before, strict=True):
        row = await run_row(engine, rid)
        assert (row.status, row.claim_epoch, row.pause_data) == (prefix_status, 1, wait)


async def test_old_running_cursor_can_begin_waiting_continuation(store: Any, engine: Any) -> None:
    ids = await _seed_wait_prefix(engine, 3)
    cursor = "0" * 26  # pre-composite cursors were raw running IDs
    for _ in range(4):
        result = await sweep(
            state_store=store,
            stale_running=THRESHOLD,
            abandoned_waiting=THRESHOLD,
            on_stale="error",
            max_runs=1,
            cursor=cursor,
        )
        assert not result.stale_running and not result.abandoned_waiting
        cursor = result.cursor
        if cursor is None:
            break
    assert _ == 3 and cursor is None
    for rid in ids:
        assert (await run_row(engine, rid)).status == "paused"


@pytest.mark.parametrize(("running_count", "waiting_count"), [(1, 5), (5, 1)])
async def test_combined_cursor_does_not_restart_a_completed_category(
    store: Any, engine: Any, monkeypatch: Any, running_count: int, waiting_count: int
) -> None:
    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.runtime.stores.sql.base import _SweepScan
    from dendrux.types import generate_ulid

    ids = await _seed_wait_prefix(engine, waiting_count)
    for _ in range(running_count):
        await store.create_run(generate_ulid(), "fresh-running")
    visited = []
    original = _SweepScan.skipped

    def observe(self: Any, row: Any) -> bool:
        visited.append(row.id)
        return original(self, row)

    monkeypatch.setattr(_SweepScan, "skipped", observe)
    running_calls = []
    original_running = type(store)._scan_expired_runs

    async def running(self: Any, **kwargs: Any) -> Any:
        running_calls.append(kwargs["cursor"])
        return await original_running(self, **kwargs)

    monkeypatch.setattr(type(store), "_scan_expired_runs", running)
    cursor = None
    for _ in range(7):
        result = await sweep(
            state_store=store_for_engine(engine),
            stale_running=THRESHOLD,
            abandoned_waiting=THRESHOLD,
            on_stale="error",
            max_runs=1,
            cursor=cursor,
        )
        assert not result.stale_running and not result.abandoned_waiting
        cursor = result.cursor
        if cursor is None:
            break
    assert cursor is None
    assert visited == ids
    assert len(running_calls) == running_count + 1


async def test_waiting_cursor_contenders_finalise_an_ordinary_pause_once(
    store: Any, engine: Any
) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent pools require PostgreSQL")
    import asyncio

    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.types import generate_ulid

    await _seed_wait_prefix(engine, 3)
    ordinary = generate_ulid()
    await store.create_run(ordinary, "ordinary")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ordinary)
            .values(status="paused", pause_data={"schema_version": 2})
        )
    await age(engine, ordinary, by=timedelta(hours=2))
    before_epoch = (await run_row(engine, ordinary)).claim_epoch
    first = await sweep(state_store=store, abandoned_waiting=THRESHOLD, max_runs=2)
    assert first.cursor is not None and not first.abandoned_waiting
    pools = [create_async_engine(engine.url) for _ in range(4)]
    try:
        results = await asyncio.gather(
            *(
                sweep(
                    state_store=store_for_engine(pool),
                    abandoned_waiting=THRESHOLD,
                    max_runs=2,
                    cursor=first.cursor,
                )
                for pool in pools
            )
        )
    finally:
        for pool in pools:
            await pool.dispose()
    assert [r.run_id for result in results for r in result.abandoned_waiting] == [ordinary]
    events = await rows(
        engine, RunEvent, RunEvent.agent_run_id == ordinary, RunEvent.event_type == "run.abandoned"
    )
    assert len(events) == 1
    assert (await run_row(engine, ordinary)).claim_epoch == before_epoch + 1


async def test_expanded_waiting_range_preserves_ties_and_time_order_on_sqlite(
    engine: Any,
) -> None:
    if engine.dialect.name != "sqlite":
        pytest.skip("Execute the MySQL-family predicate on SQLite without MySQL")
    from dendrux.runtime.stores.sql.base import _SweepScan
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore

    ids = await _seed_wait_prefix(engine, 7)
    old = (await run_row(engine, ids[0])).status_changed_at
    async with engine.begin() as conn:
        # A smaller ID at a later time must follow the same-time ID tail.
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ids[0])
            .values(status_changed_at=old + timedelta(seconds=1))
        )
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ids[-1])
            .values(status_changed_at=old - timedelta(seconds=1))
        )
        await conn.execute(
            update(AgentRun).where(AgentRun.id == ids[-2]).values(status="waiting_approval")
        )
    scan = _SweepScan(object.__new__(MySQLStateStore), waiting=True, older_than=THRESHOLD)
    # Execute the actual page WHERE, using SQLite projections instead of MySQL JSON functions.
    seen = []
    after = (old, ids[2])
    async with engine.connect() as conn:
        for _ in range(4):
            page = scan.page("paused", after, 1)
            statement = (
                select(AgentRun.id, AgentRun.status_changed_at)
                .where(page.whereclause)
                .order_by(AgentRun.status_changed_at, AgentRun.id)
                .limit(1)
            )
            row = (await conn.execute(statement)).first()
            if row is None:
                break
            seen.append(row.id)
            after = (row.status_changed_at, row.id)
    assert seen == [ids[3], ids[4], ids[0]]


@pytest.fixture(params=["mysql", "mariadb"])
def sweep_mysql_url(request: Any) -> Any:
    from tests.db.conftest import mysql_database

    yield from mysql_database(request.param)


def _explain_run_tables(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [table for value in node for table in _explain_run_tables(value)]
    if not isinstance(node, dict):
        return []
    if node.get("table_name") == "agent_runs":
        return [node]
    return [table for value in node.values() for table in _explain_run_tables(value)]


async def test_mysql_waiting_pages_use_bounded_index_ranges(
    sweep_mysql_url: str, record_property: Any
) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base
    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.runtime.stores.sql.base import _SweepScan

    engine = create_async_engine(
        sweep_mysql_url,
        isolation_level="READ COMMITTED",
        connect_args={"init_command": "SET time_zone='+00:00'", "charset": "utf8mb4"},
    )
    population, cap = 10000, 50
    now = datetime.now(UTC)
    old = now - OLD
    wait = {
        "schema_version": 3,
        "pause_reason": "managed_wait",
        "managed_wait": {"child_run_id": "child"},
    }
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # 10k eligible waits, 10k fresh waits, 20k terminal rows. All eligible
            # waits tie on time: deep continuation must also range on the ID.
            for start in range(0, 4 * population, 500):
                await conn.execute(
                    AgentRun.__table__.insert(),
                    [
                        dict(
                            id=f"{i:026d}",
                            agent_name="wait",
                            status="paused" if i < 2 * population else "success",
                            claim_epoch=1,
                            storage_format=2,
                            status_changed_at=old if i < population else now,
                            updated_at=old if i < population else now,
                            pause_data=wait,
                        )
                        for i in range(start, start + 500)
                    ],
                )
        scan = _SweepScan(store_for_engine(engine), waiting=True, older_than=THRESHOLD)
        positions = {"first": None, "deep": (old, f"{population - cap - 1:026d}")}
        async with engine.connect() as conn:
            (await conn.execute(text("ANALYZE TABLE agent_runs"))).all()
            for name, after in positions.items():
                statement = scan.page("paused", after, cap)
                sql = str(
                    statement.compile(
                        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
                    )
                )
                raw = (await conn.execute(text("EXPLAIN FORMAT=JSON " + sql))).scalar_one()
                plan = json.loads(raw) if isinstance(raw, str) else raw
                record_property(f"waiting_range_{name}", json.dumps(plan))
                [table] = _explain_run_tables(plan)
                assert table["key"] == "ix_agent_runs_sweep_waiting", plan
                assert table["access_type"] == "range", plan
                estimate = table.get("rows_examined_per_scan", table.get("rows"))
                assert estimate is not None, plan
                # EXPLAIN estimates the matching range, not LIMIT rows. Allow
                # statistics noise; a deep page must exclude the 9,950-row prefix.
                ceiling = 2 * population if after is None else 4 * cap
                assert 0 < int(estimate) <= ceiling, plan
                required = {"status", "status_changed_at"}
                if after is not None:
                    required.add("id")
                assert required <= set(table["used_key_parts"]), plan
                page = (await conn.execute(statement)).all()
                first = 0 if after is None else population - cap
                assert [row.id for row in page] == [f"{i:026d}" for i in range(first, first + cap)]
    finally:
        await engine.dispose()
