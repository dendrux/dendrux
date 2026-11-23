"""A refused continuation preserves its entire workflow and its pause generation."""

import asyncio
from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.errors import GovernanceMismatchError, RunNotPausedError, RunNotReclaimableError
from dendrux.media import Image
from dendrux.runtime import _managed_fanout
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Limits, RunStatus, ToolResult
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import tool_calls
from tests.integration.test_fanout_routing import finisher
from tests.integration.test_h0_tenancy import database_snapshot
from tests.integration.test_managed_routing import (
    CHILD_TEXT,
    TENANT,
    ContextLLM,
    build,
    done,
    journey,
    lead,
    patch_questions,
    pausing_reviewer,
    published,
)

CASES = {
    "submit_tool_results": "client_tool",
    "submit_input": "input",
    "submit_resume": "paused",
    "submit_approval": "approval",
    "resume": "input",
    "resume_stream": "input",
}


def direct(store, case):
    agent = pausing_reviewer(store, case, state_store=store)
    llm = agent.provider
    llm.capabilities = replace(llm.capabilities, supports_multimodal=True)
    return agent


async def answer(agent, paused, entry, *, image=False):
    args = {"tenant_id": TENANT, "expected_pause_id": paused.pause_id}
    run_id = paused.run_id
    images = ()
    if image:
        png = Path(__file__).parents[1] / "fixtures/images/zebra42.png"
        images = (Image.from_bytes(png.read_bytes()),)
    user_input = ["draft two", *images] if images else "draft two"
    if entry == "submit_tool_results":
        call_id = paused.pending.tool_calls[0].id
        result = ToolResult("read_sheet", call_id, '"cells"', images=images)
        return await agent.submit_tool_results(run_id, [result], **args)
    if entry == "submit_input":
        return await agent.submit_input(run_id, user_input, **args)
    if entry == "submit_approval":
        return await agent.submit_approval(run_id, approved=True, **args)
    if entry == "submit_resume":
        return await agent.submit_resume(run_id, **args)
    if entry == "resume":
        return await agent.resume(run_id, user_input=user_input, **args)
    async with agent.resume_stream(run_id, user_input=user_input, **args) as stream:
        events = [event async for event in stream]
    return events[-1].run_result


@pytest.mark.parametrize("entry,case", CASES.items())
@pytest.mark.parametrize("managed", [False, True], ids=["direct", "managed"])
async def test_governance_precedes_continuation(store, engine, monkeypatch, managed, entry, case):
    patch_questions(monkeypatch)
    published.clear()
    original = build(store, case) if managed else direct(store, case)
    if managed and entry == "submit_approval":
        original._limits = Limits(max_parallel_subagents=1)
    paused = await original.run("go", tenant_id=TENANT)
    assert paused.pending and bool(paused.pending.origin_run_id) == managed
    changed = copy(original)
    changed._limits = Limits(
        max_parallel_subagents=4 if original._limits.max_parallel_subagents == 1 else 1
    )
    image = not managed and case in {"client_tool", "input"}
    before = await database_snapshot(engine)

    try:
        refused = await answer(changed, paused, entry, image=image)
    except GovernanceMismatchError as err:
        assert (err.run_id, err.changed) == (paused.run_id, ("limits",))
    else:
        # A failure during stream setup may be delivered in its final event.
        assert entry == "resume_stream"
        assert "different governance" in (refused.error or "")
    assert await database_snapshot(engine) == before
    assert published == []

    retried = await answer(original, paused, entry, image=image)
    assert (retried.status, retried.answer) == (
        RunStatus.SUCCESS,
        "root done" if managed else CHILD_TEXT,
    )
    if case == "approval":
        assert published == ["d"]


@pytest.mark.parametrize("shape", ["chain", "fanout", "custom_fanout"])
@pytest.mark.parametrize("stream", [False, True], ids=["submit", "stream"])
async def test_changed_intermediate_parent_preserves_leaf(
    store, engine, monkeypatch, shape, stream
):
    original, _ = journey(store)
    if shape != "chain":
        original = lead(
            store,
            ContextLLM(
                [
                    tool_calls(
                        ("reviewer", {"task": "review"}, "r"), ("sibling", {"task": "read"}, "s")
                    ),
                    done("root done"),
                ]
            ),
            original.subagents[0],
            finisher("sibling"),
        )
    published.clear()
    paused = await original.run("go", tenant_id=TENANT)
    assert paused.pending.agent_path == ["lead", "reviewer", "auditor"]
    if shape == "custom_fanout":
        project = store.get_managed_children

        async def legacy_projection(*args, **kwargs):
            rows = await project(*args, **kwargs)
            return [
                SimpleNamespace(**{k: v for k, v in vars(row).items() if k != "governance"})
                for row in rows
            ]

        monkeypatch.setattr(store, "get_managed_children", legacy_projection)
    changed = copy(original)
    middle = copy(original.subagents[0])
    middle._limits = Limits(max_parallel_subagents=1)
    changed._subagents = [middle, *original.subagents[1:]]
    args = {"tenant_id": TENANT, "expected_pause_id": paused.pause_id}
    before = await database_snapshot(engine)

    async def approve(agent):
        if not stream:
            return await agent.submit_approval(paused.run_id, approved=True, **args)
        async with agent.resume_stream(paused.run_id, **args) as resumed:
            events = [event async for event in resumed]
        return events[-1].run_result

    try:
        result = await approve(changed)
    except GovernanceMismatchError as err:
        assert err.changed == ("limits",)
    else:
        assert stream and "different governance" in (result.error or "")
    assert await database_snapshot(engine) == before
    assert published == []
    assert (await approve(original)).status == RunStatus.SUCCESS
    assert published == ["A1"]


async def test_pg_only_matching_controller_can_claim_approval(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("independent PostgreSQL sessions")
    published.clear()
    original = build(store, "approval")
    paused = await original.run("go", tenant_id=TENANT)
    second = create_async_engine(engine.url)
    try:
        changed = build(store_for_engine(second), "approval")
        changed._limits = Limits(max_parallel_subagents=1)
        before = await database_snapshot(engine)
        refused = await asyncio.gather(
            *(answer(changed, paused, "submit_approval") for _ in range(4)),
            return_exceptions=True,
        )
        assert all(isinstance(error, GovernanceMismatchError) for error in refused)
        assert await database_snapshot(engine) == before
        assert published == []

        winner, *others = await asyncio.gather(
            answer(original, paused, "submit_approval"),
            *(answer(changed, paused, "submit_approval") for _ in range(4)),
            return_exceptions=True,
        )
        assert winner.status == RunStatus.SUCCESS
        # A late reader may observe the winner has already consumed the pause.
        assert all(
            isinstance(error, (GovernanceMismatchError, RunNotPausedError)) for error in others
        )
        assert published == ["d"]
    finally:
        await second.dispose()


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_tokenless_resume_checks_all_reclaimable_children(store, engine, monkeypatch):
    class ProcessCut(BaseException):
        pass

    children = [finisher(name) for name in ("left", "right")]
    root = lead(
        store,
        ContextLLM(
            [
                tool_calls(*[(child.name, {"task": "read"}, child.name) for child in children]),
                done("root done"),
            ]
        ),
        *children,
    )
    run_ids = []

    def cut_after_bind(invocation, bound, *args, **kwargs):
        run_ids.extend(
            [invocation.parent_run_id, *(child.child_run_id for child in bound.children)]
        )
        raise ProcessCut

    with monkeypatch.context() as patch:
        patch.setattr(_managed_fanout, "FanoutRun", cut_after_bind)
        with pytest.raises(ProcessCut):
            await root.run("go", tenant_id=TENANT)
    root_id, left_id, right_id = run_ids
    for run_id in (left_id, right_id):
        await expire(store, engine, run_id)

    controller = copy(root)
    right = copy(children[1])
    right._limits = Limits(max_parallel_subagents=1)
    controller._subagents = [children[0], right]
    before = await database_snapshot(engine)
    with pytest.raises(RunNotReclaimableError) as refused:
        await controller.resume(root_id, tenant_id=TENANT)
    assert (refused.value.run_id, refused.value.reason) == (right_id, "governance_changed")
    assert await database_snapshot(engine) == before
    assert [child.provider.calls for child in children] == [0, 0]

    assert (await root.resume(root_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    assert [child.provider.calls for child in children] == [1, 1]
