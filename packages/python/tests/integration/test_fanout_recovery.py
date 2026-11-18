"""Fan-out recovery through the public Agent and developer-called sweep."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, sweep
from dendrux.db.models import AgentRun, ToolCallRecord
from dendrux.errors import ManagedStateError, RunFencedError, RunNotReclaimableError
from dendrux.guardrails import PII
from dendrux.runtime import _managed_fanout, _reclaim
from dendrux.runtime._capabilities import fanout_support
from dendrux.runtime._managed_adoption import adopt_settled
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.stores.sql._sql_sweep import _collect_runs
from dendrux.types import (
    Limits,
    LLMResponse,
    PauseState,
    RunStatus,
    ToolCall,
    UsageStats,
    generate_ulid,
)
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.nlp_models import check_model
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_adoption import (
    TENANT,
    Child,
    CrashError,
    crash_on_completion,
    lead,
    node_of,
    observations,
    parked,
)
from tests.integration.test_fanout_store_races import batch, fanout_parent, nest

SAVED_CASES = (
    "saved-error",
    "saved-cancelled",
    "saved-max_iterations",
    "saved-oversized",
    "saved-mixed",
    "saved-gap",
    "saved-unknown",
    "saved-digestless",
    "saved-intent",
)


def write_phase(**payload):
    Path(os.environ["T6_PHASE"]).write_text(json.dumps(payload))


def saved_children(cut):
    if cut == "saved-mixed":
        return [Child(), Child(status="error"), Child(), Child()]
    first = (
        Child("x" * 256)
        if cut == "saved-oversized"
        else Child(status=cut.removeprefix("saved-"))
        if cut in SAVED_CASES[:3]
        else Child()
    )
    return [first, Child()]


pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


@pytest.mark.parametrize("cut", [1, 2])
async def test_claimed_adoption_reclaims_only_uncommitted_rows(store, engine, monkeypatch, cut):
    transformed = []

    async def transform(ctx):
        transformed.append(ctx.call_id)
        return ctx.result

    root = lead(
        state_store=store,
        tool_result_transform=transform,
        llm=ScriptedLLM([LLMResponse(text="done")]),
    )
    fanout = await parked(store, root, [Child(), Child()])
    with monkeypatch.context() as patch:
        crash_on_completion(patch, store, cut)
        with pytest.raises(CrashError):
            await adopt_settled(store, node_of(fanout, root))
    item = await expire(store, engine, fanout.run_id)
    try:
        result = await root.reclaim(fanout.run_id, tenant_id=TENANT, item=item)
        assert result.status == RunStatus.SUCCESS
        assert transformed == [c.id for c in fanout.calls[:cut]] + [
            c.id for c in fanout.calls[cut - 1 :]
        ]
        assert len(await observations(engine, fanout.run_id)) == 2
    finally:
        await root.close()


@pytest.mark.xfail(strict=True, reason="T5 frame setup must call _reclaim.configure_recovery_frame")
@pytest.mark.parametrize("first_width", [1, 2])
async def test_adopted_parent_can_fan_out_again(store, first_width):
    children = [
        Agent(
            name=f"c{i}",
            description="d",
            prompt="p",
            provider=ScriptedLLM([LLMResponse(text="ok")] * 2),
        )
        for i in range(2)
    ]
    calls = [ToolCall(c.name, {"task": "go"}) for c in children]
    root = Agent(
        name="lead",
        prompt="p",
        state_store=store,
        subagents=children,
        provider=ScriptedLLM(
            [
                LLMResponse(tool_calls=calls[:first_width]),
                LLMResponse(tool_calls=[replace(c, id=f"next-{c.id}") for c in calls]),
                LLMResponse(text="done"),
            ]
        ),
    )
    try:
        result = await root.run("go", tenant_id=TENANT)
        rows = await store.get_tool_calls(result.run_id, iteration_index=2)
        assert len(rows) == 2
        assert all(row.outcome == "completed" for row in rows)
        assert all(row.managed_child_run_id for row in rows)
    finally:
        await root.close()
        for child in children:
            await child.close()


async def test_sweep_accepts_valid_v4_marks(store, engine, caplog):
    root = lead()
    fanout = await parked(store, root, [Child(), Child()])
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == fanout.run_id)
            .values(
                status_changed_at=datetime.now(UTC) - timedelta(hours=2),
                updated_at=datetime.now(UTC) - timedelta(hours=2),
            )
        )
    try:
        result = await sweep(state_store=store, abandoned_waiting=timedelta(hours=1))
        assert result.abandoned_waiting == []
        assert not any(record.message == "managed_state_invalid" for record in caplog.records)
    finally:
        await root.close()


async def test_adoptable_scan_is_bounded_and_never_abandons(store, engine, monkeypatch):
    root = lead()
    waits = [await parked(store, root, [Child(), Child()]) for _ in range(3)]
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_([w.run_id for w in waits]))
            .values(
                status_changed_at=datetime.now(UTC) - timedelta(hours=2),
                updated_at=datetime.now(UTC) - timedelta(hours=2),
            )
        )
    reads = []
    original = store.get_managed_children

    async def read(parent, ids, *, tenant_id):
        assert tenant_id == TENANT and len(ids) <= 32
        reads.append(parent)
        return await original(parent, ids, tenant_id=tenant_id)

    monkeypatch.setattr(store, "get_managed_children", read)
    cursor, found = None, []
    try:
        for _ in range(5):
            adopted = []
            abandoned, cursor = await _collect_runs(
                store,
                None,
                waiting=True,
                batch_size=1,
                max_runs=1,
                cursor=cursor,
                adoptable_after=timedelta(hours=1),
                adoptable=adopted,
            )
            assert abandoned == [] and len(adopted) <= 1
            found.extend(adopted)
            if cursor is None:
                break
        assert sorted(found) == sorted(w.run_id for w in waits)
        assert len(reads) == 3
        assert all([(await store.get_run(w.run_id)).status == "paused" for w in waits])
    finally:
        await root.close()


async def test_reconnect_rereads_leaves_before_dispatch(store, monkeypatch):
    root = lead(state_store=store, llm=ScriptedLLM([LLMResponse(text="done")]))
    fanout = await parked(store, root, [Child(), Child()])
    node = node_of(fanout, root)
    stale = replace(node, children=tuple(replace(c, status="running") for c in node.children))

    async def forbidden(*args, **kwargs):
        pytest.fail("terminal child must never be reclaimed")

    monkeypatch.setattr(_reclaim, "reclaim", forbidden)
    try:
        result = await _managed_fanout.drive_node(store, stale, tenant_id=TENANT, fanout_tree=True)
        assert result.status == RunStatus.SUCCESS
    finally:
        await root.close()


@pytest.mark.parametrize("kind", ["live", "single_call", "mismatch"])
async def test_reconnect_leaves_only_not_owners(store, monkeypatch, kind):
    root = lead()
    fanout = await parked(store, root, [Child(status="running"), Child()])
    leaf = replace(
        node_of(fanout, root).children[0],
        status="running",
        loop="single_call" if kind == "single_call" else "react",
    )

    async def refused(*args, **kwargs):
        raise RunNotReclaimableError(
            leaf.run_id, "lease_fresh" if kind == "live" else "agent_mismatch"
        )

    if kind == "single_call":
        state = await store.get_loop_state(leaf.run_id)
        state["loop"] = "single_call"
        async with store._session_factory() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == leaf.run_id).values(loop_state=state)
            )
            await session.commit()
    monkeypatch.setattr(_reclaim, "reclaim", refused)
    try:
        if kind == "mismatch":
            with pytest.raises(RunNotReclaimableError, match="agent_mismatch"):
                await _managed_fanout.drive_node(store, leaf, tenant_id=TENANT, fanout_tree=True)
        else:
            result = await _managed_fanout.drive_node(
                store, leaf, tenant_id=TENANT, fanout_tree=True
            )
            assert result.status == RunStatus.RUNNING
        with pytest.raises(RunNotReclaimableError):
            await _managed_fanout.drive_node(store, leaf, tenant_id=TENANT, fanout_tree=False)
    finally:
        await root.close()


@pytest.mark.parametrize(
    "cut",
    [
        pytest.param(
            "unbound",
            marks=pytest.mark.xfail(
                strict=True,
                reason="T5 must configure the batch recovery frame before classification",
            ),
        ),
        "claim",
        "row-1",
        "row-2",
        "all",
        *SAVED_CASES,
    ],
)
async def test_fresh_process_adoption_cuts(store, engine, tmp_path, cut):
    effects, report = tmp_path / "effects.jsonl", tmp_path / "phase.json"
    env = {
        **os.environ,
        "T6_CRASH_URL": engine.url.render_as_string(hide_password=False),
        "T6_EFFECTS": str(effects),
        "T6_PHASE": str(report),
        "PYTHONPATH": "src:.",
    }
    crashed = subprocess.run(
        [sys.executable, "-m", __name__, "crash", cut],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert crashed.returncode == 17, crashed.stderr[-3000:]
    async with AsyncSession(engine) as session:
        root = (
            await session.execute(select(AgentRun).where(AgentRun.parent_run_id.is_(None)))
        ).scalar_one()
    ids = [p["call_id"] for p in root.loop_state.get("pending", [])]
    saved, invalid = cut in SAVED_CASES, cut in SAVED_CASES[5:]
    before_rows = await store.get_tool_calls(root.id, iteration_index=1)
    assert fanout_support(store).capable
    if cut == "claim":
        assert_claim_envelope(root, ids)
    before = FanoutEffects(effects).counts()
    assert before.provider_calls("root") == (0 if saved else 1)
    assert all(
        before.provider_calls(f"c{i}") == (0 if saved or cut == "unbound" else 1) for i in range(3)
    )
    allocated = json.loads(report.read_text())["child_ids"] if cut == "unbound" else []
    fresh = subprocess.run(
        [sys.executable, "-m", __name__, "recover", cut],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert fresh.returncode == 0, fresh.stderr[-3000:]
    after = FanoutEffects(effects).counts()
    if saved:
        n = len(saved_children(cut))
        assert json.loads(report.read_text()) == {
            "refused": invalid,
            "rows": n,
            "observations": 0 if invalid else n,
        }
        assert after.provider_calls("root") == (0 if invalid else 1)
        assert after.provider_calls("saved_child") == 0
        assert after.transform_calls() == (0 if invalid else 3 if cut == "saved-mixed" else 1)
        rows = await store.get_tool_calls(root.id, iteration_index=1)
        assert_saved_rows(cut, rows, before_rows, invalid)
        return
    assert json.loads(report.read_text()) == {"status": "success", "rows": 3, "observations": 3}
    assert after.provider_calls("root") == 2
    assert all(after.provider_calls(f"c{i}") == 1 for i in range(3))
    assert after.transform_calls() == 3
    if cut == "claim":
        assert await store.get_pii_mapping(root.id) == {
            f"<<EMAIL_ADDRESS_{i + 1}>>": f"fanout{i}.sentinel@example.com" for i in range(3)
        }
        seen = await observations(engine, root.id)
        assert all(
            f"<<EMAIL_ADDRESS_{i + 1}>>" in observation.content
            for i, observation in enumerate(seen)
        )
    if cut == "unbound":
        async with AsyncSession(engine) as session:
            created = (
                (
                    await session.execute(
                        select(AgentRun.id).where(AgentRun.parent_run_id == root.id)
                    )
                )
                .scalars()
                .all()
            )
        assert len(created) == 3 and set(created).isdisjoint(allocated)


def assert_saved_rows(cut, rows, before_rows, invalid):
    prefix = 2 if cut == "saved-mixed" else 1
    assert rows == before_rows if invalid else rows[:prefix] == before_rows[:prefix]
    if not invalid:
        assert [row.outcome for row in rows] == (
            ["completed", "failed", "completed", "completed"]
            if cut == "saved-mixed"
            else ["failed", "completed"]
        )


def assert_claim_envelope(root, ids):
    descriptor = root.loop_state["submission"]
    envelope = root.pause_data["submission"]
    assert descriptor == {
        "submission_id": envelope["submission_id"],
        "kind": "tool_results",
        "pause_id": root.pause_data["pause_id"],
        "payload_ref": {"pause_data_key": "submission"},
        "call_ids": ids,
        "reserved_order_index": root.pause_data["trace_order_offset"],
        "managed_fanout_adoption": {"v": 1},
    }
    assert envelope == {
        "submission_id": descriptor["submission_id"],
        "kind": "tool_results",
        "pause_id": descriptor["pause_id"],
        "call_ids": ids,
        "payload": {"managed_fanout_adoption": {"v": 1, "calls": ids}},
    }


def crash_children(effects, cut):
    return [
        Agent(
            name=f"c{i}",
            description="d",
            prompt="p",
            provider=effects.provider(
                f"c{i}",
                ScriptedLLM(
                    [
                        LLMResponse(
                            text=f"contact fanout{i}.sentinel@example.com"
                            if cut == "claim"
                            else "ok"
                        )
                    ]
                ),
            ),
        )
        for i in range(3)
    ]


async def _crash_phase(mode, cut):
    check_model()
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    effects = FanoutEffects(Path(os.environ["T6_EFFECTS"]))
    children = crash_children(effects, cut)
    responses = (
        [LLMResponse(tool_calls=[ToolCall(c.name, {"task": "go"}) for c in children])]
        if mode == "crash"
        else [LLMResponse(text="done")]
    )
    root_llm = ScriptedLLM(responses)
    root = Agent(
        name="lead",
        prompt="p",
        state_store=store,
        subagents=children,
        provider=effects.provider("root", root_llm),
        guardrails=[PII(action="redact")] if cut == "claim" else None,
        tool_result_transform=effects.transform(),
    )
    if mode == "crash":
        if cut == "unbound":

            async def unbound(invocation, **kwargs):
                Path(os.environ["T6_PHASE"]).write_text(
                    json.dumps({"child_ids": [b.child_run_id for b in invocation.branches]})
                )
                os._exit(17)

            _managed_fanout.start_fanout = unbound
        elif cut == "claim":
            original = store.submit_and_claim

            async def claim(*args, **kwargs):
                await original(*args, **kwargs)
                os._exit(17)

            store.submit_and_claim = claim
        else:
            original = store.complete_tool_call
            done = 0

            async def complete(*args, **kwargs):
                nonlocal done
                await original(*args, **kwargs)
                done += 1
                if done == (3 if cut == "all" else int(cut.split("-")[1])):
                    os._exit(17)

            store.complete_tool_call = complete
        await root.run("go", tenant_id=TENANT)
        raise RuntimeError("crash cut was not reached")
    async with AsyncSession(engine) as session:
        run_id = (
            await session.execute(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
        ).scalar_one()
    item = await expire(store, engine, run_id)
    result = await root.reclaim(run_id, tenant_id=TENANT, item=item)
    rows = await store.get_tool_calls(run_id, iteration_index=1)
    assert all(row.outcome == "completed" for row in rows)
    if cut == "claim":
        sent = json.dumps([message.content for message in root_llm.messages[0]])
        assert all(f"fanout{i}.sentinel@example.com" not in sent for i in range(3))
    write_phase(
        status=result.status.value,
        rows=len(rows),
        observations=len(await observations(engine, run_id)),
    )
    await root.close()
    for child in children:
        await child.close()
    await engine.dispose()


async def test_replaced_adopter_is_fenced_after_transform(store, engine):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def transform(ctx):
        seen.append(ctx.call_id)
        if len(seen) == 1:
            entered.set()
            await release.wait()
        return ctx.result

    root = lead(
        state_store=store,
        tool_result_transform=transform,
        llm=ScriptedLLM([LLMResponse(text="done")]),
    )
    fanout = await parked(store, root, [Child(), Child()])
    replacement_engine = create_async_engine(engine.url)
    replacement_store = store_for_engine(replacement_engine)
    replacement = lead(
        state_store=replacement_store,
        tool_result_transform=transform,
        llm=ScriptedLLM([LLMResponse(text="done")]),
    )
    owner = asyncio.create_task(adopt_settled(store, node_of(fanout, root)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        item = await expire(replacement_store, replacement_engine, fanout.run_id)
        result = await replacement.reclaim(fanout.run_id, tenant_id=TENANT, item=item)
        assert result.status == RunStatus.SUCCESS
        saved = await store.get_run(fanout.run_id)
        release.set()
        with pytest.raises(RunFencedError):
            await owner
        assert seen == [fanout.calls[0].id, *[c.id for c in fanout.calls]]
        assert len(await observations(engine, fanout.run_id)) == 2
        assert await store.get_run(fanout.run_id) == saved
        assert root._require_provider().calls == 0
        assert replacement._require_provider().calls == 1
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        await root.close()
        await replacement.close()
        await replacement_engine.dispose()


async def test_cancel_cascade_repeats_after_process_death(store, engine, tmp_path):
    report = tmp_path / "cancel.json"
    env = {
        **os.environ,
        "T6_CRASH_URL": engine.url.render_as_string(hide_password=False),
        "T6_PHASE": str(report),
        "PYTHONPATH": "src:.",
    }
    for mode, expected in [("cancel-crash", 17), ("cancel-recover", 0)]:
        result = subprocess.run(
            [sys.executable, "-m", __name__, mode, "cancel"],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == expected, result.stderr[-3000:]
    assert json.loads(report.read_text()) == {
        "statuses": ["cancelled"] * 3,
        "rows": 2,
        "observations": 0,
        "provider_calls": 0,
        "transforms": 0,
    }


async def _cancel_phase(mode):
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    effects = FanoutEffects()
    root = lead(
        state_store=store,
        llm=effects.provider("root", ScriptedLLM([])),
        tool_result_transform=effects.transform(),
    )
    if mode == "cancel-crash":
        fanout = await parked(store, root, [Child(status="running"), Child(status="running")])
        original = store.request_cancel

        async def request(*args, **kwargs):
            await original(*args, **kwargs)
            os._exit(17)

        store.request_cancel = request
        await root.cancel_run(fanout.run_id, tenant_id=TENANT)
        raise RuntimeError("cancel cut was not reached")
    async with AsyncSession(engine) as session:
        run_id = (
            await session.execute(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
        ).scalar_one()
    await root.cancel_run(run_id, tenant_id=TENANT)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).values(last_progress_at=datetime.now(UTC) - timedelta(hours=2))
        )
    await sweep(state_store=store, stale_running=timedelta(hours=1))
    async with AsyncSession(engine) as session:
        statuses = (await session.execute(select(AgentRun.status))).scalars().all()
    Path(os.environ["T6_PHASE"]).write_text(
        json.dumps(
            {
                "statuses": statuses,
                "rows": len(await store.get_tool_calls(run_id, iteration_index=1)),
                "observations": len(await observations(engine, run_id)),
                "provider_calls": sum(effects.counts().provider.values()),
                "transforms": effects.counts().transform_calls(),
            }
        )
    )
    await root.close()
    await engine.dispose()


async def test_nested_adoptable_scan_reports_root(store, engine):
    root, calls = await fanout_parent(store, 2)
    first = batch(root, calls)
    await store.bind_managed_children(first)
    for branch in first.children:
        parent = branch.child_run_id
        nested = batch(parent, await nest(store, parent, 2), root=root, path=("reviewer",))
        await store.bind_managed_children(nested)
        for child in nested.children:
            await store.finalize_run(
                child.child_run_id, status="success", answer="ok", claim_epoch=0
            )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).values(status_changed_at=datetime.now(UTC) - timedelta(hours=2))
        )

    class Results(list):
        membership_checks = 0

        def __contains__(self, value):
            self.membership_checks += 1
            return super().__contains__(value)

    adopted = Results()
    swept, cursor = await _collect_runs(
        store,
        None,
        waiting=True,
        batch_size=1,
        max_runs=10,
        adoptable=adopted,
        adoptable_after=timedelta(hours=1),
    )
    assert swept == [] and cursor is None and adopted == [root]
    assert adopted.membership_checks == 0
    assert (await store.get_run(parent)).status == "paused"


async def test_adoption_age_never_shortens_abandonment(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "lead", tenant_id=TENANT)
    pause = PauseState("lead", [], [], [], 0, 0, UsageStats(), pause_id=generate_ulid())
    await store.pause_run(run_id, status="paused", pause_data=pause.to_dict(), claim_epoch=0)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                status_changed_at=datetime.now(UTC) - timedelta(minutes=10),
                updated_at=datetime.now(UTC) - timedelta(minutes=10),
            )
        )
    adopted = []
    swept, _ = await _collect_runs(
        store,
        timedelta(hours=1),
        waiting=True,
        batch_size=1,
        max_runs=10,
        adoptable_after=timedelta(minutes=1),
        adoptable=adopted,
    )
    assert swept == [] and adopted == []
    assert (await store.get_run(run_id)).status == "paused"


@pytest.mark.parametrize(
    "frame_enabled",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="T5 must configure the batch recovery frame before classification",
            ),
        ),
    ],
)
async def test_unbound_fanout_reclaims_all_admitted_calls(
    store, engine, monkeypatch, frame_enabled
):

    class ProcessCut(BaseException):
        pass

    children = [
        Agent(
            name=f"c{i}",
            description="d",
            prompt="p",
            provider=ScriptedLLM([LLMResponse(text="ok")]),
        )
        for i in range(2)
    ]
    calls = [ToolCall(c.name, {"task": "go"}) for c in children]
    first = Agent(
        name="lead",
        prompt="p",
        state_store=store,
        subagents=children,
        provider=ScriptedLLM([LLMResponse(tool_calls=calls)]),
    )

    async def cut(*args, **kwargs):
        raise ProcessCut

    with monkeypatch.context() as patch:
        patch.setattr(_managed_fanout, "start_fanout", cut)
        with pytest.raises(ProcessCut):
            await first.run("go", tenant_id=TENANT)
    async with AsyncSession(engine) as session:
        run_id = (await session.execute(select(AgentRun.id))).scalar_one()
    state = await store.get_loop_state(run_id)
    assert state["managed_admitted"] == [call.id for call in calls]
    item = await expire(store, engine, run_id)
    fresh = Agent(
        name="lead",
        prompt="p",
        state_store=store,
        subagents=children,
        provider=ScriptedLLM([LLMResponse(text="done")]),
    )
    if not frame_enabled:
        original = _reclaim._recover_managed_calls

        async def disabled(agent, ctx, *args):
            ctx.recorder._managed_fanout_capable = False
            return await original(agent, ctx, *args)

        monkeypatch.setattr(_reclaim, "_recover_managed_calls", disabled)
    try:
        result = await fresh.reclaim(run_id, tenant_id=TENANT, item=item)
        assert result.status == RunStatus.SUCCESS
        rows = await store.get_tool_calls(run_id, iteration_index=1)
        assert len(rows) == 2
        assert all(r.outcome == ("completed" if frame_enabled else "failed") for r in rows)
        assert all(c._require_provider().calls == int(frame_enabled) for c in children)
    finally:
        await first.close()
        await fresh.close()
        for child in children:
            await child.close()


async def test_reconnect_joins_children_and_keeps_first_cancellation(store, monkeypatch):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owners = []

    async def leaf(*args, **kwargs):
        owners.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    root = lead()
    fanout = await parked(store, root, [Child(), Child()])
    monkeypatch.setattr(_managed_fanout, "drive_node", leaf)
    task = asyncio.create_task(
        _managed_fanout._reconnect_children(
            store, node_of(fanout, root), fanout_tree=True, notifier=None
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel("first")
        await asyncio.wait_for(cleaning.wait(), 5)
        task.cancel("second")
        release.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert error.value.args == ("first",)
        assert owners and all(owner.done() for owner in owners)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await root.close()


@pytest.mark.parametrize("corrupt", ["marker", "binding", "payload", "pending"])
async def test_corrupt_adoption_refuses_before_recovery_claim(store, engine, monkeypatch, corrupt):
    root = lead(state_store=store)
    fanout = await parked(store, root, [Child(), Child()])
    with monkeypatch.context() as patch:
        crash_on_completion(patch, store, 1)
        with pytest.raises(CrashError):
            await adopt_settled(store, node_of(fanout, root))
    state, raw = (
        await store.get_loop_state(fanout.run_id),
        await store.get_pause_state(fanout.run_id),
    )
    if corrupt == "marker":
        state["submission"]["managed_fanout_adoption"] = {"v": 2}
    elif corrupt == "binding":
        state["managed_wait"]["pause_id"] = "another-generation"
    elif corrupt == "payload":
        raw["submission"]["payload"]["managed_fanout_adoption"]["calls"].reverse()
    else:
        state["pending"].reverse()
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == fanout.run_id)
            .values(loop_state=state, pause_data=raw)
        )
    item = await expire(store, engine, fanout.run_id)
    before = await store.get_run(fanout.run_id)
    try:
        with pytest.raises(ManagedStateError) as error:
            await root.reclaim(fanout.run_id, tenant_id=TENANT, item=item)
        assert error.value.reason == "binding_conflict"
        after = await store.get_run(fanout.run_id)
        assert (after.claim_epoch, after.reclaim_attempts) == (
            before.claim_epoch,
            before.reclaim_attempts,
        )
        assert await observations(engine, fanout.run_id) == []
    finally:
        await root.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("capable", [False, True])
async def test_recovery_frame_configures_explicit_capability(store, streaming, capable):
    from dendrux.runtime._reclaim import configure_recovery_frame

    store.capabilities = replace(store.capabilities, managed_fanout=capable)
    recorder = SimpleNamespace()
    configure_recovery_frame(SimpleNamespace(recorder=recorder), store, streaming=streaming)
    assert recorder._managed_fanout_capable is (capable and not streaming)


@pytest.mark.parametrize(
    "case", ["running", "missing", "bad_header", "wrong_binding", "bad_pause", "young"]
)
async def test_adoptable_probe_refuses_unsettled_or_invalid_candidates(
    store, engine, monkeypatch, case
):

    root = lead()
    fanout = await parked(
        store, root, [Child(status="running" if case == "running" else "success"), Child()]
    )
    age = timedelta(minutes=10) if case == "young" else timedelta(hours=2)
    values = {"status_changed_at": datetime.now(UTC) - age, "updated_at": datetime.now(UTC) - age}
    if case == "bad_pause":
        values["pause_data"] = {"schema_version": 4, "pause_reason": "managed_wait"}
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == fanout.run_id).values(**values))
    original = store.get_managed_children

    async def read(*args, **kwargs):
        rows = await original(*args, **kwargs)
        if case == "missing":
            return rows[:-1]
        if case == "bad_header":
            rows[0] = replace(rows[0], header={"v": 1})
        if case == "wrong_binding":
            rows[0] = replace(rows[0], parent_run_id="another-parent")
        return rows

    monkeypatch.setattr(store, "get_managed_children", read)
    adopted = []
    try:
        swept, _ = await _collect_runs(
            store,
            timedelta(minutes=1) if case == "young" else None,
            waiting=True,
            batch_size=1,
            max_runs=10,
            adoptable_after=timedelta(hours=1),
            adoptable=adopted,
        )
        assert swept == [] and adopted == []
        assert (await store.get_run(fanout.run_id)).status == "paused"
    finally:
        await root.close()


async def _saved_phase(mode, cut):
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    effects = FanoutEffects(Path(os.environ["T6_EFFECTS"]))
    child = Agent(
        name="reviewer",
        description="Reviews one draft and reports.",
        prompt="You review.",
        provider=effects.provider("saved_child", ScriptedLLM([])),
    )
    root = Agent(
        name="lead",
        prompt="You lead.",
        state_store=store,
        subagents=[child],
        provider=effects.provider("root", ScriptedLLM([LLMResponse(text="done")])),
        tool_result_transform=effects.transform(),
        limits=Limits(max_tool_result_bytes=128),
        on_governance_change="raise",
    )
    invalid = cut in SAVED_CASES[5:]
    if mode == "crash":
        fanout = await parked(store, root, saved_children(cut))
        original = store.submit_and_claim if invalid else store.complete_tool_call
        committed = 0

        async def die(*args, **kwargs):
            nonlocal committed
            result = await original(*args, **kwargs)
            committed += 1
            if invalid:
                call = fanout.calls[1 if cut == "saved-gap" else 0]
                values = {"outcome": "unknown" if cut == "saved-unknown" else "completed"}
                if cut in ("saved-gap", "saved-digestless"):
                    values["content_digest"] = "a" * 64 if cut == "saved-gap" else None
                async with engine.begin() as connection:
                    await connection.execute(
                        update(ToolCallRecord)
                        .where(
                            ToolCallRecord.agent_run_id == fanout.run_id,
                            ToolCallRecord.tool_call_id == call.id,
                        )
                        .values(**values)
                    )
            if invalid or committed == (2 if cut == "saved-mixed" else 1):
                os._exit(17)
            return result

        if invalid:
            store.submit_and_claim = die
        else:
            store.complete_tool_call = die
        await adopt_settled(store, node_of(fanout, root))
        raise RuntimeError("saved-outcome cut was not reached")
    async with AsyncSession(engine) as session:
        run_id = (
            await session.execute(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
        ).scalar_one()
    item = await expire(store, engine, run_id)
    before = await store.get_run(run_id)
    try:
        result = await root.reclaim(run_id, tenant_id=TENANT, item=item)
        assert not invalid and result.status == RunStatus.SUCCESS
    except ManagedStateError as error:
        assert invalid and error.reason == "binding_conflict"
        after = await store.get_run(run_id)
        assert (after.claim_epoch, after.reclaim_attempts) == (
            before.claim_epoch,
            before.reclaim_attempts,
        )
    write_phase(
        refused=invalid,
        rows=len(await store.get_tool_calls(run_id, iteration_index=1)),
        observations=len(await observations(engine, run_id)),
    )
    await root.close()
    await child.close()
    await engine.dispose()


if __name__ == "__main__":
    if sys.argv[1].startswith("cancel-"):
        asyncio.run(_cancel_phase(sys.argv[1]))
    elif sys.argv[2] in SAVED_CASES:
        asyncio.run(_saved_phase(sys.argv[1], sys.argv[2]))
    else:
        asyncio.run(_crash_phase(sys.argv[1], sys.argv[2]))
