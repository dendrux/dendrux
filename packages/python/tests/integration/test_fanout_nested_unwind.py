"""Real-driver nested fan-in, routed unwind and reconnect after process death."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, sweep, tool
from dendrux.db.models import AgentRun
from dendrux.errors import PauseStatusMismatchError, RunFencedError, RunNotPausedError
from dendrux.loops.single import SingleCall
from dendrux.runtime import _managed_fanout
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult, generate_ulid
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_adoption import (
    TENANT as ADOPTION_TENANT,
)
from tests.integration.test_fanout_adoption import (
    Child,
    lead,
    observations,
    parked,
)

TENANT = "nested-recovery"
pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


@tool(target="client")
async def read() -> str:
    """Read a value in the caller's client."""
    raise AssertionError("client tool executed on server")


def calls(*names):
    return LLMResponse(tool_calls=[ToolCall(name, {"task": "go"}) for name in names])


def workflow(store, effects, *, both=False, again=False, root_width=2, nested_width=2):
    def agent(name, script, *, children=(), tools=()):
        return Agent(
            name=name,
            description="Reviews",
            prompt="Review",
            state_store=store if name == "root" else None,
            provider=effects.provider(name, ScriptedLLM(script)),
            subagents=list(children),
            tools=list(tools),
            tool_result_transform=effects.transform() if children else None,
        )

    client = LLMResponse(tool_calls=[ToolCall("read", {})])
    end = LLMResponse(text="done")
    a1 = agent("a1", [client, end] if both else [end, end], tools=[read])
    a2 = agent("a2", [client, end, end], tools=[read])
    names = ("a1", "a2") if nested_width == 2 else ("a2",)
    script = [calls(*names), *([calls("a1", "a2")] if again else []), end]
    a = agent("a", script, children=[a1, a2])
    b = agent("b", [end])
    root = agent(
        "root", [calls(*(("a", "b") if root_width == 2 else ("a",))), end], children=[a, b]
    )
    return root


async def records(engine):
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(AgentRun))).scalars())


async def answer(root, run_id, pending):
    call = pending.tool_calls[0]
    return await root.submit_tool_results(
        run_id,
        [ToolResult(call_id=call.id, name=call.name, payload="value")],
        tenant_id=TENANT,
        expected_pause_id=pending.pause_id,
    )


@pytest.mark.parametrize("again", [False, True])
async def test_nested_submit_unwinds_and_may_delegate_again(store, engine, again):
    effects = FanoutEffects()
    root = workflow(store, effects, again=again)
    paused = await root.run("go", tenant_id=TENANT)
    assert [p.agent_path for p in paused.pendings] == [["root", "a", "a2"]]
    before = {
        row.id: await store.get_run(row.id)
        for row in await records(engine)
        if row.status == "success"
    }
    assert {row.agent_name for row in before.values()} == {"a1", "b"}
    result = await answer(root, paused.run_id, paused.pending)
    assert result.status == RunStatus.SUCCESS and result.run_id == paused.run_id
    assert result.pending is None and result.pendings == () and result.pause_id is None
    after = await records(engine)
    assert all(row.status == "success" for row in after)
    assert len(after) == (7 if again else 5)
    assert all([(await store.get_run(key)) == value for key, value in before.items()])
    counts = effects.counts()
    assert dict(counts.provider) == {
        "root": 2,
        "a": 3 if again else 2,
        "a1": 2 if again else 1,
        "a2": 3 if again else 2,
        "b": 1,
    }
    assert counts.transform_calls() == (6 if again else 4)
    for name, width in (("root", 2), ("a", 4 if again else 2)):
        row = next(row for row in after if row.agent_name == name)
        intents = await store.get_tool_calls(row.id)
        assert len(intents) == width and all(intent.outcome == "completed" for intent in intents)
    await root.close()


async def test_nested_concurrent_submits_adopt_each_wait_once(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent claims require Postgres")
    effects = FanoutEffects()
    root = workflow(store, effects, both=True)
    paused = await root.run("go", tenant_id=TENANT)
    a = next(row for row in await records(engine) if row.agent_name == "a")
    ready, arrivals = asyncio.Event(), 0
    original = _managed_fanout.settle_parent

    async def settle(store, node, **kwargs):
        nonlocal arrivals
        if node.run_id == a.id:
            arrivals += 1
            if arrivals == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 10)
        return await original(store, node, **kwargs)

    monkeypatch.setattr(_managed_fanout, "settle_parent", settle)
    results = await asyncio.gather(*(answer(root, paused.run_id, p) for p in paused.pendings))
    assert arrivals == 2 and any(result.status == RunStatus.SUCCESS for result in results)
    assert all(row.status == "success" for row in await records(engine))
    counts = effects.counts()
    assert dict(counts.provider) == {"root": 2, "a": 2, "a1": 2, "a2": 2, "b": 1}
    assert counts.transform_calls() == 4
    for run_id in (a.id, paused.run_id):
        rows = await store.get_tool_calls(run_id)
        assert len(rows) == 2 and all(row.outcome == "completed" for row in rows)
        events = await store.get_run_events(run_id)
        assert sum(event.event_type == "run.completed" for event in events) == 1
    await root.close()


@pytest.mark.parametrize(("root_width", "nested_width"), [(1, 2), (2, 1), (1, 1)])
async def test_reconnect_carries_fanout_policy_through_single_child_waits(
    store, root_width, nested_width
):
    effects = FanoutEffects()
    root = workflow(store, effects, root_width=root_width, nested_width=nested_width)
    paused = await root.run("go", tenant_id=TENANT)
    before = effects.counts()
    result = await root.reclaim(paused.run_id, tenant_id=TENANT)
    assert result.status == RunStatus.WAITING_CLIENT_TOOL
    assert result.pendings == paused.pendings
    assert effects.counts() == before
    assert (await answer(root, paused.run_id, result.pending)).status == RunStatus.SUCCESS
    assert effects.counts().transform_calls() == root_width + nested_width
    await root.close()


CRASH_CUTS = ["bind-react", "bind-mixed", "0", "1", "2", "all", "finisher", "nested", "chain"]


@pytest.mark.parametrize("cut", CRASH_CUTS)
async def test_public_restart_skips_terminal_children_and_fences_retired_writers(
    store, engine, tmp_path, cut
):
    effects_path = tmp_path / "effects.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": "src:.",
        "T6_CRASH_URL": engine.url.render_as_string(hide_password=False),
        "T6_EFFECTS": str(effects_path),
    }
    crashed = subprocess.run(
        [sys.executable, "-m", __name__, "crash", cut],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert crashed.returncode == 17, crashed.stderr[-3000:]
    runs = await records(engine)
    root = next(row for row in runs if row.parent_run_id is None)
    bound_epoch = root.claim_epoch
    await fenced_checkpoint(store, root.id, 0)
    saved = {row.id: await store.get_run(row.id) for row in runs if row.status == "success"}
    old_epochs = {row.id: row.claim_epoch for row in runs if row.status == "running"}
    before = FanoutEffects(effects_path).counts()
    assert_crash_cut(runs, before, cut)
    recovered = subprocess.run(
        [sys.executable, "-m", __name__, "recover", cut],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert recovered.returncode == 0, recovered.stderr[-3000:]
    assert (await store.get_run(root.id)).status == "success"
    assert all([(await store.get_run(key)) == value for key, value in saved.items()])
    after = FanoutEffects(effects_path).counts()
    for row in runs:
        if row.id in saved:
            assert before.for_child(row.agent_name) == after.for_child(row.agent_name)
    for run_id, epoch in old_epochs.items():
        assert (await store.get_run(run_id)).claim_epoch > epoch
        await fenced_checkpoint(store, run_id, epoch)
    await fenced_checkpoint(store, root.id, bound_epoch)
    assert_crash_effects(after, cut)


def assert_crash_cut(runs, counts, cut):
    assert next(row for row in runs if row.parent_run_id is None).status == "paused"
    terminal = {row.agent_name for row in runs if row.status == "success"}
    assert counts.provider_calls("root") == 1
    if cut in ("nested", "chain"):
        assert terminal == ({"a", "a1", "a2"} if cut == "nested" else {"a1", "a2"})
        assert counts.transform_calls() == (2 if cut == "nested" else 0)
        expected = {"root": 1, "a": 2 if cut == "nested" else 1, "a1": 2, "a2": 2}
        if cut == "nested":
            expected["b"] = 1
        assert dict(counts.provider) == expected
    else:
        done = int(cut) if cut in ("0", "1", "2") else 0 if cut.startswith("bind") else 3
        assert terminal == {f"c{i}" for i in range(done)}
        assert counts.transform_calls() == 0
        assert [counts.provider_calls(f"c{i}") for i in range(3)] == [
            0 if cut.startswith("bind") else 2 if i < done else 1 for i in range(3)
        ]
        assert dict(counts.tools) == {(f"c{i}", "touch"): 1 for i in range(done)}


async def fenced_checkpoint(store, run_id, epoch):
    before = await store.get_run(run_id)
    with pytest.raises(RunFencedError):
        await store.checkpoint_step(
            run_id, claim_epoch=epoch, loop_state={"v": 1}, pii_mapping=None, trace=None
        )
    assert await store.get_run(run_id) == before


def assert_crash_effects(counts, cut):
    assert counts.provider_calls("root") == 2
    if cut in ("nested", "chain"):
        expected = {"root": 2, "a": 2, "a1": 2, "a2": 2}
        if cut == "nested":
            expected["b"] = 3
        assert dict(counts.provider) == expected
        assert counts.transform_calls() == (4 if cut == "nested" else 3)
        assert dict(counts.tools) == {
            (name, "touch"): 1 for name in (("a1", "a2", "b") if cut == "nested" else ("a1", "a2"))
        }
    else:
        expected = [0, 2, 0] if cut == "bind-mixed" else [2] * 3
        if cut in ("0", "1", "2"):
            expected = [2 if i < int(cut) else 3 for i in range(3)]
        assert [counts.provider_calls(f"c{i}") for i in range(3)] == expected
        assert counts.transform_calls() == (1 if cut == "bind-mixed" else 3)
        assert dict(counts.tools) == {
            (f"c{i}", "touch"): 1 for i in range(3) if cut != "bind-mixed" or i == 1
        }


class CrashSchedule:
    def __init__(self, cut):
        self.cut = cut
        self.entered, self.finished = set(), set()

    def check(self):
        if self.cut in ("0", "1", "2"):
            count = int(self.cut)
            if len(self.entered) == 3 - count and len(self.finished) == count:
                os._exit(17)
        elif (
            self.cut == "nested"
            and self.entered == {"b"}
            and "a" in self.finished
            or self.cut == "finisher"
            and len(self.finished) == 3
        ):
            os._exit(17)

    async def before_provider(self, name):
        blocked = (
            self.cut == "nested"
            and name == "b"
            or self.cut in ("0", "1", "2")
            and name.startswith("c")
            and int(name[1:]) >= int(self.cut)
        )
        if blocked:
            self.entered.add(name)
            self.check()
            await asyncio.Event().wait()


class CrashProvider(ScriptedLLM):
    def __init__(self, name, responses, schedule):
        super().__init__(responses)
        self.name, self.schedule = name, schedule

    async def complete(self, messages, tools=None, **kwargs):
        await self.schedule.before_provider(self.name)
        return await super().complete(messages, tools, **kwargs)


def crash_workflow(store, effects, mode, cut, schedule):
    def child(name, *, children=(), script=None, single=False):
        @tool()
        async def touch() -> str:
            """Record an idempotent test effect."""
            effects.record_tool(name, "touch")
            return "ok"

        script = script or [LLMResponse(tool_calls=[ToolCall("touch", {})]), LLMResponse(text="ok")]
        provider = CrashProvider(name, script, schedule) if mode == "crash" else ScriptedLLM(script)
        return Agent(
            name=name,
            description="Reviews",
            prompt="Review",
            state_store=store if name == "root" else None,
            tools=[] if children or single else [touch],
            subagents=list(children),
            loop=SingleCall() if single else None,
            provider=effects.provider(name, provider),
            tool_result_transform=effects.transform() if children else None,
        )

    if cut in ("nested", "chain"):
        a = child(
            "a",
            children=[child("a1"), child("a2")],
            script=[calls("a1", "a2"), LLMResponse(text="a done")]
            if mode == "crash"
            else [LLMResponse(text="a done")],
        )
        children = [a] if cut == "chain" else [a, child("b")]
    else:
        children = [child(f"c{i}", single=cut == "bind-mixed" and i != 1) for i in range(3)]
    script = [LLMResponse(text="done")]
    if mode == "crash":
        script.insert(0, calls(*(agent.name for agent in children)))
    return child("root", children=children, script=script)


def install_crash_cut(store, schedule, cut):
    if cut.startswith("bind"):
        original = store.bind_managed_children

        async def bind(*args, **kwargs):
            await original(*args, **kwargs)
            os._exit(17)

        store.bind_managed_children = bind
    elif cut in ("all", "chain"):

        async def settle(*args, **kwargs):
            os._exit(17)

        _managed_fanout.settle_parent = settle
    else:
        original = store.finalize_with_event

        async def finalise(run_id, **kwargs):
            result = await original(run_id, **kwargs)
            row = await store.get_run(run_id)
            if result is not None and row.parent_run_id is not None:
                schedule.finished.add(row.agent_name)
                schedule.check()
            return result

        store.finalize_with_event = finalise


def install_adoption_fence_probe(store):
    original, claims = store.submit_and_claim, []

    async def claim(run_id, *args, **kwargs):
        before = await store.get_run(run_id)
        result = await original(run_id, *args, **kwargs)
        if result:
            during = await store.get_run(run_id)
            assert during.status == "running" and during.claim_epoch == before.claim_epoch + 1
            await asyncio.create_task(fenced_checkpoint(store, run_id, before.claim_epoch))
            claims.append(run_id)
        return result

    store.submit_and_claim = claim
    return claims


async def recover_public(root, store, engine, cut):
    old = datetime.now(UTC) - timedelta(hours=2)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.status.in_(["running", "paused"]))
            .values(last_progress_at=old, status_changed_at=old, updated_at=old)
        )
    run_id = next(row.id for row in await records(engine) if row.parent_run_id is None)
    result = await sweep(
        state_store=store,
        stale_running=timedelta(minutes=1),
        abandoned_waiting=timedelta(minutes=1),
        adoptable_after=timedelta(minutes=1),
    )
    assert result.abandoned_waiting == []
    assert result.adoptable == ([run_id] if cut in ("all", "finisher", "chain") else [])
    if cut == "bind-mixed":
        assert len(result.stale_running) == 2
        assert {item.failure_reason for item in result.stale_running} == {"never_started"}
    claims = install_adoption_fence_probe(store)
    assert (await root.reclaim(run_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    assert len(claims) == (2 if cut == "chain" else 1) and run_id in claims
    runs = await records(engine)
    assert len(runs) == (5 if cut == "nested" else 4)
    for row in runs:
        assert row.status == (
            "error" if cut == "bind-mixed" and row.agent_name in ("c0", "c2") else "success"
        )
        rows = await store.get_tool_calls(row.id)
        width = (
            (1 if cut == "chain" else 2 if cut == "nested" else 3)
            if row.id == run_id
            else 2
            if row.agent_name == "a"
            else 0
            if cut == "bind-mixed" and row.agent_name in ("c0", "c2")
            else 1
        )
        assert len(rows) == width
        assert len(await observations(engine, row.id)) == width
        assert all(item.outcome in ("completed", "failed") for item in rows)


async def crash_phase(mode, cut):
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    effects = FanoutEffects(Path(os.environ["T6_EFFECTS"]))
    schedule = CrashSchedule(cut)
    root = crash_workflow(store, effects, mode, cut, schedule)
    if mode == "crash":
        install_crash_cut(store, schedule, cut)
        await root.run("go", tenant_id=TENANT)
        raise RuntimeError("crash cut was not reached")
    await recover_public(root, store, engine, cut)
    await root.close()
    await engine.dispose()


async def test_reconnect_adopts_a_single_child_ancestor_after_nested_fanout(
    store, engine, monkeypatch
):
    class ProcessCut(BaseException):
        pass

    effects = FanoutEffects()
    root = crash_workflow(store, effects, "crash", "chain", CrashSchedule("noop"))

    async def cut(*args, **kwargs):
        raise ProcessCut

    with monkeypatch.context() as patch:
        patch.setattr(_managed_fanout, "settle_parent", cut)
        with pytest.raises(ProcessCut):
            await root.run("go", tenant_id=TENANT)
    runs = await records(engine)
    run_id = next(row.id for row in runs if row.parent_run_id is None)
    assert (await root.reclaim(run_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    assert all(row.status == "success" for row in await records(engine))
    assert_crash_effects(effects.counts(), "chain")
    await root.close()


@pytest.mark.parametrize("entry", ["resume", "reclaim"])
@pytest.mark.parametrize("live", [False, True])
async def test_tokenless_resume_matches_public_reclaim(store, engine, entry, live):
    effects = FanoutEffects()
    root = lead(
        state_store=store,
        llm=effects.provider("root", ScriptedLLM([LLMResponse(text="done")])),
        tool_result_transform=effects.transform(),
    )
    fanout = await parked(store, root, [Child(status="running") if live else Child(), Child()])
    parent = await store.get_run(fanout.run_id)
    before = [await store.get_run(child) for child in fanout.child_ids]
    try:
        result = await getattr(root, entry)(fanout.run_id, tenant_id=ADOPTION_TENANT)
        assert result.run_id == fanout.run_id
        assert result.status == (RunStatus.RUNNING if live else RunStatus.SUCCESS)
        assert result.pending is None and result.pendings == () and result.pause_id is None
        assert [await store.get_run(child) for child in fanout.child_ids] == before
        after = await store.get_run(fanout.run_id)
        assert after.claim_epoch == parent.claim_epoch + int(not live)
        assert after.reclaim_attempts == parent.reclaim_attempts
        assert effects.counts().provider_calls("root") == int(not live)
        assert effects.counts().transform_calls() == (0 if live else 2)
        assert root.subagents[0].provider.calls == 0
        rows = await store.get_tool_calls(fanout.run_id)
        assert len(rows) == 2
        assert all(row.outcome == ("started" if live else "completed") for row in rows)
        assert len(await observations(engine, fanout.run_id)) == (0 if live else 2)
    finally:
        await root.close()


@pytest.mark.parametrize(
    "entry", ["approval", "client", "input", "submit_resume", "resume_client", "resume_input"]
)
async def test_tokenless_submissions_do_not_reconnect_an_all_terminal_wait(store, entry):
    root = lead(state_store=store)
    fanout = await parked(store, root, [Child(), Child()])
    ids = [fanout.run_id, *fanout.child_ids]
    before = [await store.get_run(run_id) for run_id in ids]
    common = dict(tenant_id=ADOPTION_TENANT)
    entries = {
        "approval": lambda: root.submit_approval(fanout.run_id, approved=True, **common),
        "client": lambda: root.submit_tool_results(fanout.run_id, [], **common),
        "input": lambda: root.submit_input(fanout.run_id, "", **common),
        "submit_resume": lambda: root.submit_resume(fanout.run_id, **common),
        "resume_client": lambda: root.resume(fanout.run_id, tool_results=[], **common),
        "resume_input": lambda: root.resume(fanout.run_id, user_input="", **common),
    }
    try:
        with pytest.raises(RunNotPausedError):
            await entries[entry]()
        assert [await store.get_run(run_id) for run_id in ids] == before
        assert root.provider.calls == 0 and root.subagents[0].provider.calls == 0
        assert all(row.outcome == "started" for row in await store.get_tool_calls(fanout.run_id))
    finally:
        await root.close()


async def test_tokenless_resume_keeps_mixed_paused_and_terminal_leaves_unchanged(store, engine):
    effects = FanoutEffects()
    root = workflow(store, effects)
    paused = await root.run("go", tenant_id=TENANT)
    rows = await records(engine)
    assert any(row.status == "success" for row in rows)
    assert [p.agent_path for p in paused.pendings] == [["root", "a", "a2"]]
    before = [(await store.get_run(row.id), await store.get_tool_calls(row.id)) for row in rows]
    counts = effects.counts()
    try:
        with pytest.raises(ValueError, match="pause_id"):
            await root.resume(paused.run_id, tenant_id=TENANT)
        assert [
            (await store.get_run(row.id), await store.get_tool_calls(row.id)) for row in rows
        ] == before
        assert effects.counts() == counts
    finally:
        await root.close()


async def test_tokenless_resume_preserves_a_generationless_pause_refusal(store, engine):
    effects = FanoutEffects()
    root = workflow(store, effects)
    paused = await root.run("go", tenant_id=TENANT)
    leaf_id = paused.pending.origin_run_id
    raw = await store.get_pause_state(leaf_id)
    raw.pop("pause_id")
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun).where(AgentRun.id == leaf_id).values(pause_data=raw)
        )
    rows = await records(engine)
    before = [await store.get_run(row.id) for row in rows]
    counts = effects.counts()
    try:
        with pytest.raises(RunNotPausedError):
            await root.resume(paused.run_id, tenant_id=TENANT)
        assert [await store.get_run(row.id) for row in rows] == before
        assert effects.counts() == counts
    finally:
        await root.close()


@pytest.mark.parametrize("token", ["wait", "other"])
async def test_unheld_resume_token_does_not_adopt_an_all_terminal_wait(store, token):
    root = lead(state_store=store)
    fanout = await parked(store, root, [Child(), Child()])
    pause_id = fanout.pause_id if token == "wait" else generate_ulid()
    ids = [fanout.run_id, *fanout.child_ids]
    before = [await store.get_run(run_id) for run_id in ids]
    try:
        with pytest.raises(PauseStatusMismatchError) as caught:
            await root.resume(fanout.run_id, tenant_id=ADOPTION_TENANT, expected_pause_id=pause_id)
        assert caught.value.reason == "pause_id"
        assert caught.value.current_pause_id is None
        assert [await store.get_run(run_id) for run_id in ids] == before
        assert root.provider.calls == 0 and root.subagents[0].provider.calls == 0
    finally:
        await root.close()


async def test_tokenless_resume_race_adopts_once(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent claims require Postgres")
    effects = FanoutEffects()
    root = lead(
        state_store=store,
        llm=effects.provider("root", ScriptedLLM([LLMResponse(text="done")])),
        tool_result_transform=effects.transform(),
    )
    fanout = await parked(store, root, [Child(), Child()])
    parent = await store.get_run(fanout.run_id)
    before = [await store.get_run(child) for child in fanout.child_ids]
    ready, arrivals = asyncio.Event(), 0
    original = _managed_fanout.settle_parent

    async def settle(*args, **kwargs):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 10)
        return await original(*args, **kwargs)

    monkeypatch.setattr(_managed_fanout, "settle_parent", settle)
    try:
        results = await asyncio.gather(
            root.resume(fanout.run_id, tenant_id=ADOPTION_TENANT),
            root.resume(fanout.run_id, tenant_id=ADOPTION_TENANT),
        )
        assert arrivals == 2 and any(result.status == RunStatus.SUCCESS for result in results)
        assert {result.run_id for result in results} == {fanout.run_id}
        assert effects.counts().provider_calls("root") == 1
        assert effects.counts().transform_calls() == 2
        assert root.subagents[0].provider.calls == 0
        assert [await store.get_run(child) for child in fanout.child_ids] == before
        after = await store.get_run(fanout.run_id)
        assert after.status == "success" and after.claim_epoch == parent.claim_epoch + 1
        rows = await store.get_tool_calls(fanout.run_id)
        assert len(rows) == 2 and all(row.outcome == "completed" for row in rows)
        assert len(await observations(engine, fanout.run_id)) == 2
        events = await store.get_run_events(fanout.run_id)
        assert sum(event.event_type == "run.completed" for event in events) == 1
    finally:
        await root.close()


if __name__ == "__main__":
    asyncio.run(crash_phase(*sys.argv[1:]))
