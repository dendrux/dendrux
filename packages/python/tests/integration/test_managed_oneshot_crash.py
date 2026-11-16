"""Managed subagents (0.7, B9): a SingleCall child interrupted mid-call ends failed.

Phase one runs in a child process that dies with ``os._exit`` at the cut (no cleanup,
no finally). Phase two is a fresh process view: new Agents, providers and a new engine
on the same database. It follows the documented loop: ``sweep()`` finalises the stale
child (it is never marked reclaimable), the developer walks ``parent_run_id`` to the
root, and a root control adopts the failure. The child's model is never called again.

Runs on file SQLite and Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test``
database); the contender race runs on Postgres only.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, sweep
from dendrux.db.models import AgentRun, ReactTrace, RunEvent, TokenUsage, ToolCallRecord
from dendrux.errors import RunNotPausedError, RunNotReclaimableError
from dendrux.loops.single import SingleCall
from dendrux.runtime import _managed
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult, UsageStats
from tests._helpers.gates import wait_for_gate
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-oneshot-crash-brief-2f81"
REPORT = "SENTINEL-oneshot-crash-report-7d3a"
TENANT = "t1"
CRASHED = 17
"""Exit code of a phase-one process that reached its crash cut."""


def _usage() -> UsageStats:
    return UsageStats(input_tokens=10, output_tokens=1, total_tokens=11)


def _said(text: str) -> LLMResponse:
    return LLMResponse(text=text, usage=_usage())


def _asked(pid: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(name="scorer", params={"task": BRIEF}, provider_tool_call_id=pid)],
        usage=_usage(),
    )


class ContextLLM(ScriptedLLM):
    """Scripted provider recording the run each call is served under (real ContextVar)."""

    def __init__(self, responses: list[LLMResponse], **kwargs: Any) -> None:
        super().__init__(responses, **kwargs)
        self.served: list[str | None] = []

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        self.served.append(ctx.run_id if ctx else None)
        return await super().complete(messages, tools, **kwargs)


class DyingLLM(ContextLLM):
    """The process dies while the provider call is in flight (after ``llm.started``)."""

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        os._exit(CRASHED)


def scorer(llm: Any) -> Agent:
    return Agent(
        name="scorer",
        description="Scores one draft in a single model call.",
        prompt="You score.",
        provider=llm,
        loop=SingleCall(),
    )


def lead(store: Any, llm: Any, *children: Agent) -> Agent:
    return Agent(
        name="lead", prompt="You lead.", provider=llm, state_store=store, subagents=list(children)
    )


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def count(engine: Any, model: Any, *where: Any) -> int:
    async with AsyncSession(engine) as session:
        return int(await session.scalar(select(func.count()).select_from(model).where(*where)))


async def run_row(engine: Any, run_id: str) -> Any:
    return (await rows(engine, AgentRun, AgentRun.id == run_id))[0]


# ---------------------------------------------------------------------------
# Phase one: run until the cut, where the process dies
# ---------------------------------------------------------------------------


def _install_cut(cut: str) -> None:
    if cut == "bind_committed":
        original = _managed.bind_child

        async def bound_then_die(*args: Any, **kwargs: Any) -> Any:
            await original(*args, **kwargs)
            os._exit(CRASHED)

        _managed.bind_child = bound_then_die
    elif cut == "after_usage":
        recorded = PersistenceRecorder.on_llm_call_completed

        async def recorded_then_die(self: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
            await recorded(self, run_id, *args, **kwargs)
            if self._loop_state.get("loop") == "single_call":
                os._exit(CRASHED)  # the child's usage is durable; its finalise is not

        PersistenceRecorder.on_llm_call_completed = recorded_then_die  # type: ignore[method-assign]


async def _phase_one(cut: str, url: str) -> None:
    engine = create_async_engine(url)
    store = store_for_engine(engine)
    child_llm = ContextLLM([_said(REPORT)]) if cut in {"after_usage"} else DyingLLM([])
    root = lead(store, ContextLLM([_asked("r1")]), scorer(child_llm))
    _install_cut(cut)
    if cut == "in_call_stream":
        async with root.stream("go", tenant_id=TENANT) as stream:
            async for _ in stream:
                pass
    else:
        await root.run("go", tenant_id=TENANT)
    raise SystemExit(3)  # the cut was never reached


def _crash_in_new_process(cut: str, engine: Any) -> None:
    here = Path(__file__).resolve()
    package = here.parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(package / "src"), str(package)]),
        "B9_CRASH_URL": engine.url.render_as_string(hide_password=False),
    }
    done = subprocess.run(
        [sys.executable, "-m", "tests.integration.test_managed_oneshot_crash", cut],
        cwd=package,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == CRASHED, done.stderr.decode()[-2000:]


# ---------------------------------------------------------------------------
# Phase two helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _fresh_process(engine: Any) -> Any:
    """Phase two's store: a new engine and pool on the same database."""
    other = create_async_engine(engine.url.render_as_string(hide_password=False))
    try:
        yield store_for_engine(other)
    finally:
        await other.dispose()


async def _crashed(cut: str, engine: Any) -> tuple[str, str]:
    _crash_in_new_process(cut, engine)
    [root] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == root.id)
    assert (root.status, child.status) == ("paused", "running")
    assert child.loop_state["loop"] == "single_call"
    return root.id, child.id


async def _snapshot(engine: Any, *run_ids: str) -> list[tuple[Any, ...]]:
    found = []
    for run_id in run_ids:
        row = await run_row(engine, run_id)
        events = await count(engine, RunEvent, RunEvent.agent_run_id == run_id)
        found.append(
            (row.status, row.claim_epoch, row.reclaim_attempts, row.lease_expired_at, events)
        )
    return found


async def _age(engine: Any, run_id: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )


async def _sweep(store: Any, on_stale: Any) -> Any:
    kwargs = {"state_store": store, "stale_running": timedelta(minutes=1), "on_stale": on_stale}
    if on_stale is None:
        with pytest.warns(FutureWarning):
            return await sweep(**kwargs)
    return await sweep(**kwargs)


async def _root_of(store: Any, run_id: str) -> str:
    """The documented loop: walk ``parent_run_id`` from a swept child to its root."""
    record = await store.get_run(run_id)
    while record.parent_run_id is not None:
        record = await store.get_run(record.parent_run_id)
    return str(record.id)


async def _refusals_before_the_sweep(root: Agent, root_id: str) -> None:
    """While the leaf is RUNNING nothing can continue it: every control refuses."""
    with pytest.raises(RunNotReclaimableError) as refused:
        await root.reclaim(root_id, tenant_id=TENANT)
    assert refused.value.reason == "loop_unsupported"
    attempts = [
        lambda: root.resume(root_id, tenant_id=TENANT),
        lambda: root.resume(root_id, tenant_id=TENANT, expected_pause_id="stale"),
        lambda: root.submit_input(root_id, "late", tenant_id=TENANT, expected_pause_id="stale"),
        lambda: root.submit_approval(
            root_id, approved=True, tenant_id=TENANT, expected_pause_id="stale"
        ),
        lambda: root.submit_tool_results(
            root_id,
            [ToolResult(name="x", call_id="c", payload='"ok"')],
            tenant_id=TENANT,
            expected_pause_id="stale",
        ),
    ]
    for attempt in attempts:
        with pytest.raises(RunNotPausedError):
            await attempt()
    with pytest.raises(RunNotPausedError):
        async with root.resume_stream(root_id, tenant_id=TENANT) as stream:
            async for _ in stream:
                pass


async def _continue(root: Agent, root_id: str, entry: str) -> Any:
    if entry == "reclaim":
        return await root.reclaim(root_id, tenant_id=TENANT)
    if entry == "resume":
        return await root.resume(root_id, tenant_id=TENANT)
    async with root.resume_stream(root_id, tenant_id=TENANT) as stream:
        async for _ in stream:
            pass
    return stream.result


async def _assert_failed_and_adopted_once(
    engine: Any, root_id: str, child_id: str, *, reason: str, child_usage: int
) -> None:
    """One child row, failed; one ``managed_child_failed`` observation; usage once."""
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    assert child.id == child_id
    assert (child.status, child.failure_reason) == ("error", reason)
    [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert intent.outcome == "failed" and intent.meta["managed"]["child_run_id"] == child_id
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == root_id)
    [observation] = [t for t in traces if t.role == "tool"]
    body = json.loads(observation.content)
    assert (body["code"], body["child_run_id"]) == ("managed_child_failed", child_id)
    events = await rows(engine, RunEvent, RunEvent.agent_run_id == root_id)
    [completed] = [e for e in events if e.event_type == "tool.completed"]
    assert completed.data.get("managed_outcome") == "failed"
    assert [e.event_type for e in await rows(engine, RunEvent, RunEvent.agent_run_id == child_id)
            ].count("run.interrupted") == 1  # fmt: skip
    assert await count(engine, TokenUsage, TokenUsage.agent_run_id == root_id) == 2
    assert await count(engine, TokenUsage, TokenUsage.agent_run_id == child_id) == child_usage
    async with RunStore.from_engine(engine) as reader:
        tree = await reader.get_subtree(root_id, tenant_id=TENANT, max_nodes=16)
    assert tree.usage.input_tokens == 10 * (2 + child_usage)


crash_engines = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


# ---------------------------------------------------------------------------
# Interrupted mid-call: failed, never rerun, adopted through every root entry
# ---------------------------------------------------------------------------


@crash_engines
@pytest.mark.parametrize("cut", ["in_call", "in_call_stream"])
@pytest.mark.parametrize("entry", ["reclaim", "resume", "resume_stream"])
async def test_a_child_interrupted_mid_call_fails_and_is_adopted_once(
    engine: Any, cut: str, entry: str
) -> None:
    root_id, child_id = await _crashed(cut, engine)
    assert await count(engine, TokenUsage, TokenUsage.agent_run_id == child_id) == 0
    async with _fresh_process(engine) as store:
        child_llm = ContextLLM([_said(REPORT)])
        root_llm = ContextLLM([_said("root done")])
        root = lead(store, root_llm, scorer(child_llm))
        before = await _snapshot(engine, root_id, child_id)
        await _refusals_before_the_sweep(root, root_id)
        assert await _snapshot(engine, root_id, child_id) == before

        await _age(engine, child_id)
        swept = await _sweep(store, "reclaim")
        assert [(s.run_id, s.status, s.failure_reason) for s in swept.stale_running] == [
            (child_id, "error", "stale_running")
        ]
        assert swept.reclaimable == []
        assert await _root_of(store, swept.stale_running[0].run_id) == root_id

        result = await _continue(root, root_id, entry)
        assert result.status == RunStatus.SUCCESS and result.answer == "root done"
        assert child_llm.calls == 0 and root_llm.calls == 1
        await _assert_failed_and_adopted_once(
            engine, root_id, child_id, reason="stale_running", child_usage=0
        )

        again = await _sweep(store, "reclaim")
        assert again.stale_running == [] and again.reclaimable == []
        repeat = await root.reclaim(root_id, tenant_id=TENANT)
        assert repeat.status == RunStatus.SUCCESS and root_llm.calls == 1
        assert child_llm.calls == 0


@crash_engines
@pytest.mark.parametrize(
    ("cut", "on_stale", "reason", "child_usage"),
    [
        ("bind_committed", "reclaim", "never_started", 0),
        ("in_call", "error", "stale_running", 0),
        ("in_call", None, "stale_running", 0),
        ("after_usage", "reclaim", "stale_running", 1),
        ("after_usage", "error", "stale_running", 1),
    ],
)
async def test_every_sweep_mode_fails_an_interrupted_child_and_counts_usage_once(
    engine: Any, cut: str, on_stale: Any, reason: str, child_usage: int
) -> None:
    root_id, child_id = await _crashed(cut, engine)
    usage_rows = await count(engine, TokenUsage, TokenUsage.agent_run_id == child_id)
    assert usage_rows == child_usage
    async with _fresh_process(engine) as store:
        child_llm = ContextLLM([_said(REPORT)])
        root_llm = ContextLLM([_said("root done")])
        root = lead(store, root_llm, scorer(child_llm))
        await _age(engine, child_id)
        swept = await _sweep(store, on_stale)
        assert [(s.run_id, s.failure_reason) for s in swept.stale_running] == [(child_id, reason)]
        assert swept.reclaimable == []
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        assert child_llm.calls == 0 and root_llm.calls == 1
        await _assert_failed_and_adopted_once(
            engine, root_id, child_id, reason=reason, child_usage=child_usage
        )


async def test_pg_four_concurrent_root_reclaims_adopt_the_failed_child_once(
    engine: Any,
) -> None:
    """After the sweep, 4 contending root reclaims on two pools: exactly one adoption,
    zero child calls, one root continuation."""
    if engine.dialect.name != "postgresql":
        pytest.skip("independent-session race: Postgres only")
    root_id, child_id = await _crashed("in_call", engine)
    await _age(engine, child_id)
    await _sweep(store_for_engine(engine), "reclaim")
    child_llms = [ContextLLM([_said(REPORT)]) for _ in range(4)]
    root_llms = [ContextLLM([_said("root done")]) for _ in range(4)]
    async with _fresh_process(engine) as other:
        stores = [store_for_engine(engine), other]
        outcomes = await asyncio.gather(
            *(
                lead(stores[i % 2], root_llms[i], scorer(child_llms[i])).reclaim(
                    root_id, tenant_id=TENANT
                )
                for i in range(4)
            ),
            return_exceptions=True,
        )
    statuses = [getattr(o, "status", None) for o in outcomes]
    assert statuses.count(RunStatus.SUCCESS) == 1, outcomes  # exactly one adopts
    for outcome in outcomes:  # losers see the owned root as running, or are refused
        assert isinstance(
            outcome, (RunNotReclaimableError, RunNotPausedError)
        ) or outcome.status in {RunStatus.SUCCESS, RunStatus.RUNNING}, outcome
    assert sum(llm.calls for llm in child_llms) == 0
    assert sum(llm.calls for llm in root_llms) == 1
    await _assert_failed_and_adopted_once(
        engine, root_id, child_id, reason="stale_running", child_usage=0
    )


# ---------------------------------------------------------------------------
# A live process cut off from the database (P2-4): the documented limit
# ---------------------------------------------------------------------------


@crash_engines
async def test_a_live_child_swept_mid_call_is_fenced_and_never_recorded(engine: Any) -> None:
    """The sweep cannot tell a dead process from a live one cut off from the database.
    The child ends failed; the root adopts that once; the live call still completes and
    is billed, but nothing it writes after the fence is recorded (usage undercounts by
    that one call)."""
    child_llm = ContextLLM([_said(REPORT)], gate_at=0)
    owner_root = lead(store_for_engine(engine), ContextLLM([_asked("r1")]), scorer(child_llm))
    owner = asyncio.create_task(owner_root.run("go", tenant_id=TENANT))
    await wait_for_gate(child_llm.entered, task=owner)
    [root_row] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    [child_row] = await rows(engine, AgentRun, AgentRun.parent_run_id == root_row.id)
    try:
        async with _fresh_process(engine) as other:
            await _age(engine, child_row.id)
            swept = await _sweep(other, "reclaim")
            assert [s.run_id for s in swept.stale_running] == [child_row.id]
            root_llm = ContextLLM([_said("root done")])
            recovered = await lead(other, root_llm, scorer(ContextLLM([]))).reclaim(
                root_row.id, tenant_id=TENANT
            )
            assert recovered.status == RunStatus.SUCCESS and root_llm.calls == 1
    finally:
        child_llm.gate.set()
    late = await asyncio.wait_for(owner, 30)  # the owner reads the finished root view
    assert (late.status, late.answer) == (RunStatus.SUCCESS, "root done")
    assert child_llm.calls == 1, "the cut-off call completed (and was billed)"
    child = await run_row(engine, child_row.id)
    assert (child.status, child.failure_reason, child.output_data) == (
        "error",
        "stale_running",
        None,
    )
    assert await count(engine, TokenUsage, TokenUsage.agent_run_id == child_row.id) == 0
    await _assert_failed_and_adopted_once(
        engine, root_row.id, child_row.id, reason="stale_running", child_usage=0
    )


@crash_engines
async def test_a_child_fenced_before_its_call_makes_no_provider_call(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lease renewal before the call sees the sweep's new epoch: no provider call."""
    started = PersistenceRecorder.on_llm_call_started
    child_llm = ContextLLM([_said(REPORT)])

    async def swept_then_started(self: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
        if self._loop_state.get("loop") == "single_call":
            await _age(engine, run_id)
            async with _fresh_process(engine) as other:
                await _sweep(other, "reclaim")
        return await started(self, run_id, *args, **kwargs)

    monkeypatch.setattr(PersistenceRecorder, "on_llm_call_started", swept_then_started)
    root_llm = ContextLLM([_asked("r1"), _said("root done")])
    result = await lead(store_for_engine(engine), root_llm, scorer(child_llm)).run(
        "go", tenant_id=TENANT
    )
    assert child_llm.calls == 0
    assert result.status == RunStatus.SUCCESS and root_llm.calls == 2
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    await _assert_failed_and_adopted_once(
        engine, result.run_id, child.id, reason="stale_running", child_usage=0
    )


if __name__ == "__main__":
    asyncio.run(_phase_one(sys.argv[1], os.environ["B9_CRASH_URL"]))
