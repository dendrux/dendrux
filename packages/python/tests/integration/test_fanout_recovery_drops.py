"""Live and recovered managed batches bind only their admitted ordered calls."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import ManagedStateError, RunNotReclaimableError
from dendrux.loops import react
from dendrux.runtime import _managed_fanout, _reclaim_managed
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Limits, LLMResponse, RunStatus, ToolCall, ToolDecision, generate_ulid
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_adoption import (
    TENANT,
    Child,
    lead,
    node_of,
    observations,
    parked,
)
from tests.integration.test_fanout_recovery_fixes import install_binding_capture


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("kind", ["hook", "brief", "invalid"])
@pytest.mark.parametrize("survivors", [0, 1, 2])
async def test_recovered_drops_match_live(store, engine, tmp_path, kind, survivors):
    phase, effects = tmp_path / "phase.json", tmp_path / "effects.jsonl"
    phase.write_text(json.dumps({"ids": [generate_ulid() for _ in range(3)]}))
    env = {
        **os.environ,
        "PYTHONPATH": "src:.",
        "T6_CRASH_URL": engine.url.render_as_string(hide_password=False),
        "T6_EFFECTS": str(effects),
        "T6_PHASE": str(phase),
    }
    for mode, expected in (("crash", 17), ("live", 0), ("recover", 0)):
        result = subprocess.run(
            [sys.executable, "-m", __name__, mode, kind, str(survivors)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stderr[-3000:]
        counts = FanoutEffects(effects).counts()
        if mode == "crash":
            assert counts.provider_calls("recover:root") == 1
            assert counts.transform_calls() == 0
            assert counts.tool("recover", "hook") == 0
        else:
            assert counts.transform_calls() == survivors * (2 if mode == "recover" else 1)
    saved = json.loads(phase.read_text())
    expected_ids = [saved["ids"][i] for i in kept_indices(survivors)]
    assert (
        saved["live"]
        == saved["recover"]
        == {
            "bound": expected_ids,
            "version": None if survivors == 0 else 3 if survivors == 1 else 4,
            "children": survivors,
            "rows": survivors if kind == "hook" else 3,
            "observations": 3,
        }
    )
    counts = FanoutEffects(effects).counts()
    assert counts.transform_calls() == 2 * survivors
    for label in ("live", "recover"):
        assert counts.provider_calls(label + ":root") == 2
        assert counts.tool(label, "hook") == (survivors if kind == "invalid" else 3)
        assert [counts.provider_calls(f"{label}:c{i}") for i in range(3)] == [
            int(i in kept_indices(survivors)) for i in range(3)
        ]


def kept_indices(survivors):
    return [0, 2] if survivors == 2 else [2] if survivors == 1 else []


async def drop_agents(store, effects, plan, mode, kind, survivors):
    kept = kept_indices(survivors)
    label = "live" if mode == "live" else "recover"
    children = [
        Agent(
            name=f"c{i}",
            description="Reviews",
            prompt="Review",
            limits=Limits(max_input_bytes=8),
            provider=effects.provider(f"{label}:c{i}", ScriptedLLM([LLMResponse(text="ok")])),
        )
        for i in range(3)
    ]
    calls = [
        ToolCall(
            f"c{i}",
            {}
            if kind == "invalid" and i not in kept
            else {"task": "x" * 64 if kind == "brief" and i not in kept else "go"},
            id=call_id,
        )
        for i, call_id in enumerate(plan["ids"])
    ]

    async def hook(ctx):
        effects.record_tool(label, "hook")
        if kind == "hook" and ctx.tool_name not in {f"c{i}" for i in kept}:
            return ToolDecision.deny("policy")
        return None

    script = [LLMResponse(text="done")]
    if mode != "recover":
        script.insert(0, LLMResponse(tool_calls=calls))
    root = Agent(
        name="lead",
        prompt="Lead",
        state_store=store,
        subagents=children,
        provider=effects.provider(label + ":root", ScriptedLLM(script)),
        before_tool=hook,
        limits=Limits(max_parallel_subagents=4),
        tool_result_transform=effects.transform(),
    )
    return root, children


async def drop_phase(mode, kind, survivors):
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    path = Path(os.environ["T6_PHASE"])
    plan = json.loads(path.read_text())
    effects = FanoutEffects(Path(os.environ["T6_EFFECTS"]))
    root, children = await drop_agents(store, effects, plan, mode, kind, survivors)
    if mode == "crash":

        async def cut(calls, *, batch, **kwargs):
            plan["root"] = batch.run_id
            path.write_text(json.dumps(plan))
            os._exit(17)

        react._prepare_tool_calls = cut
        await root.run("go", tenant_id=TENANT)
        raise RuntimeError("preparation cut was not reached")
    saved = {"version": None, "bound": []}
    install_binding_capture(store, saved)
    if mode == "recover":
        run_id = plan["root"]
        result = await root.reclaim(
            run_id, tenant_id=TENANT, item=await expire(store, engine, run_id)
        )
    else:
        result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS, result.status.value
    async with AsyncSession(engine) as session:
        saved["children"] = len(
            (
                await session.scalars(
                    select(AgentRun.id).where(
                        AgentRun.parent_run_id == result.run_id, AgentRun.tenant_id == TENANT
                    )
                )
            ).all()
        )
    saved["rows"] = len(await store.get_tool_calls(result.run_id, iteration_index=1))
    saved["observations"] = len(await observations(engine, result.run_id))
    plan[mode] = saved
    path.write_text(json.dumps(plan))
    await root.close()
    for child in children:
        await child.close()
    await engine.dispose()


@pytest.mark.parametrize(
    "pending",
    [
        None,
        {},
        [None],
        ["x"],
        [{}],
        [{"call_id": "a"}],
        [{"call_id": [], "phase": "approved"}],
        [{"call_id": "", "phase": "approved"}],
        [{"call_id": "a", "phase": []}],
        [{"call_id": "a", "phase": "future"}],
        [{"call_id": "a", "phase": "approved"}, {"call_id": "a", "phase": "approved"}],
    ],
)
def test_malformed_pending_refuses_with_binding_conflict(pending):
    run_id = generate_ulid()
    found = _reclaim_managed._ManagedPending(ToolCall("c0", id="a"), False)
    with pytest.raises(ManagedStateError) as error:
        _reclaim_managed._surviving_managed(found, {"pending": pending}, run_id)
    assert error.value.reason == "binding_conflict" and error.value.run_id == run_id


@pytest.mark.parametrize(
    "admitted", [["a", "foreign", "c"], ["c", "a"], ["a"], ["a", "a", "c"], ["a", {}, "c"], []]
)
def test_list_admission_refuses_unordered_or_foreign_survivors(admitted):
    run_id = generate_ulid()
    calls = [ToolCall("c0", id=i) for i in ("a", "b", "c")]
    found = _reclaim_managed._ManagedPending.from_calls(calls, admitted=None, run_id=run_id)
    state = {
        "pending": [
            {"call_id": "a", "phase": "approved"},
            {"call_id": "b", "phase": "rejected"},
            {"call_id": "c", "phase": "approved"},
        ],
        "managed_admitted": admitted,
    }
    with pytest.raises(ManagedStateError) as error:
        _reclaim_managed._surviving_managed(found, state, run_id)
    assert error.value.reason == "binding_conflict" and error.value.run_id == run_id


@pytest.mark.parametrize(
    "held,allowed",
    [
        ([], False),
        (["a"], True),
        (["b"], True),
        (["a", "c"], True),
        (["b", "c"], True),
        (["c", "a"], False),
        (["a", "a"], False),
        (["foreign"], False),
        (["a", "b", "c"], True),
    ],
)
def test_handoff_accepts_only_nonempty_ordered_subsets(held, allowed):
    calls = [ToolCall("c0", id=i) for i in ("a", "b", "c")]
    found = _reclaim_managed._ManagedPending.from_calls(
        calls, admitted=None, run_id=generate_ulid()
    )
    assert _reclaim_managed._handed_off(found, [ToolCall("c0", id=i) for i in held]) is allowed
    assert _reclaim_managed._handed_off(None, calls) is False


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_false_reclaim_is_not_owner_only_in_fanout(store, monkeypatch):
    root = lead()
    fanout = await parked(store, root, [Child(status="running"), Child()])
    leaf = node_of(fanout, root).children[0]
    called = []

    async def lost_claim(run_id, **kwargs):
        called.append((run_id, kwargs["tenant_id"]))
        return False

    monkeypatch.setattr(store, "reclaim_run", lost_claim)
    before = await store.get_run(leaf.run_id)
    try:
        result = await _managed_fanout.drive_node(store, leaf, tenant_id=TENANT, fanout_tree=True)
        assert result.status == RunStatus.RUNNING
        assert await store.get_run(leaf.run_id) == before
        with pytest.raises(RunNotReclaimableError) as error:
            await _managed_fanout.drive_node(store, leaf, tenant_id=TENANT, fanout_tree=False)
        assert error.value.reason == "already_reclaimed"
        assert called == [(leaf.run_id, TENANT)] * 2
    finally:
        await root.close()


if __name__ == "__main__":
    asyncio.run(drop_phase(sys.argv[1], sys.argv[2], int(sys.argv[3])))
