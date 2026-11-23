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
from dendrux.skills import Skill
from dendrux.types import Limits, RunEventType, RunStatus, ToolResult
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
    if events[-1].error:
        assert [e.type for e in events] == [RunEventType.RUN_ERROR]
        assert events[0].run_result.run_id == run_id
    return events[-1].run_result


def with_skill(agent, body):
    changed = copy(agent)
    changed._explicit_skills = (
        [] if body is None else [Skill(name="review", description="Review", body=body)]
    )
    changed._loaded_skills = None
    return changed


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
        assert entry != "resume_stream"
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


@pytest.mark.parametrize("policy", ["governance", "skills"])
@pytest.mark.parametrize("shape", ["chain", "fanout", "custom_fanout"])
@pytest.mark.parametrize("stream", [False, True], ids=["submit", "stream"])
async def test_changed_intermediate_parent_preserves_leaf(
    store, engine, monkeypatch, shape, stream, policy
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
                SimpleNamespace(
                    **{
                        k: v
                        for k, v in vars(row).items()
                        if k not in {"governance", "skill_pins", "agent_name"}
                    }
                )
                for row in rows
            ]

        monkeypatch.setattr(store, "get_managed_children", legacy_projection)
    changed = copy(original)
    middle = copy(original.subagents[0])
    if policy == "skills":
        middle = with_skill(middle, "new")
    else:
        middle._limits = Limits(max_parallel_subagents=1)
    changed._subagents = [middle, *original.subagents[1:]]
    args = {"tenant_id": TENANT, "expected_pause_id": paused.pause_id}
    before = await database_snapshot(engine)

    async def approve(agent):
        if not stream:
            return await agent.submit_approval(paused.run_id, approved=True, **args)
        async with agent.resume_stream(paused.run_id, **args) as resumed:
            events = [event async for event in resumed]
        if events[-1].error:
            assert [e.type for e in events] == [RunEventType.RUN_ERROR]
            assert events[0].run_id == paused.run_id
        return events[-1].run_result

    try:
        result = await approve(changed)
    except GovernanceMismatchError as err:
        assert not stream and policy == "governance" and err.changed == ("limits",)
    except RunNotReclaimableError as err:
        assert not stream and policy == "skills" and err.reason == "skill_mismatch"
    else:
        assert (
            stream
            and ("skill_mismatch" if policy == "skills" else "different governance") in result.error
        )
    assert await database_snapshot(engine) == before
    assert published == []
    assert (await approve(original)).status == RunStatus.SUCCESS
    assert published == ["A1"]


@pytest.mark.parametrize("policy", ["governance", "skills"])
async def test_pg_only_matching_controller_can_claim_approval(store, engine, policy):
    if engine.dialect.name != "postgresql":
        pytest.skip("independent PostgreSQL sessions")
    published.clear()
    original = build(store, "approval")
    paused = await original.run("go", tenant_id=TENANT)
    second = create_async_engine(engine.url)
    try:
        changed = build(store_for_engine(second), "approval")
        expected = GovernanceMismatchError
        if policy == "skills":
            changed = with_skill(changed, "added")
            expected = RunNotReclaimableError
        else:
            changed._limits = Limits(max_parallel_subagents=1)
        before = await database_snapshot(engine)
        refused = await asyncio.gather(
            *(answer(changed, paused, "submit_approval") for _ in range(4)),
            return_exceptions=True,
        )
        assert all(isinstance(error, expected) for error in refused)
        assert await database_snapshot(engine) == before
        assert published == []

        winner, *others = await asyncio.gather(
            answer(original, paused, "submit_approval"),
            *(answer(changed, paused, "submit_approval") for _ in range(4)),
            return_exceptions=True,
        )
        assert winner.status == RunStatus.SUCCESS
        # A late reader may observe the winner has already consumed the pause.
        assert all(isinstance(error, (expected, RunNotPausedError)) for error in others)
        assert published == ["d"]
    finally:
        await second.dispose()


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("operation", ["resume", "reclaim"])
@pytest.mark.parametrize("policy", ["governance", "skills"])
async def test_tokenless_resume_checks_all_reclaimable_children(
    store, engine, monkeypatch, operation, policy
):
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
    if policy == "skills":
        right = with_skill(right, "added")
    else:
        right._limits = Limits(max_parallel_subagents=1)
    controller._subagents = [children[0], right]
    before = await database_snapshot(engine)
    with pytest.raises(RunNotReclaimableError) as refused:
        await getattr(controller, operation)(root_id, tenant_id=TENANT)
    assert (refused.value.run_id, refused.value.reason) == (
        right_id,
        "skill_mismatch" if policy == "skills" else "governance_changed",
    )
    assert await database_snapshot(engine) == before
    assert [child.provider.calls for child in children] == [0, 0]

    assert (await getattr(root, operation)(root_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    assert [child.provider.calls for child in children] == [1, 1]


@pytest.mark.parametrize("entry,case", CASES.items())
@pytest.mark.parametrize("new_body", ["revised", None], ids=["changed", "removed"])
async def test_changed_root_skill_preserves_child(
    store, engine, monkeypatch, entry, case, new_body
):
    patch_questions(monkeypatch)
    original = with_skill(build(store, case), "original")
    paused = await original.run("go", tenant_id=TENANT)
    changed = with_skill(original, new_body)
    before = await database_snapshot(engine)
    if entry == "resume_stream":
        refused = await answer(changed, paused, entry)
        assert "skill_mismatch" in refused.error
    else:
        with pytest.raises(RunNotReclaimableError) as refused:
            await answer(changed, paused, entry)
        assert (refused.value.run_id, refused.value.reason) == (paused.run_id, "skill_mismatch")
    assert await database_snapshot(engine) == before
    assert (await answer(original, paused, entry)).status == RunStatus.SUCCESS


@pytest.mark.parametrize("case", ["input", "client_tool"])
async def test_direct_stream_pinned_skill_refusal_precedes_image(store, engine, monkeypatch, case):
    patch_questions(monkeypatch)
    original = with_skill(direct(store, case), "original")
    transformed = []

    def transform(ctx):
        transformed.append(ctx)
        return "cells"

    if case == "client_tool":
        original._tool_result_transform = transform
    paused = await original.run("go", tenant_id=TENANT)
    png = Path(__file__).parents[1] / "fixtures/images/zebra42.png"
    image = Image.from_bytes(png.read_bytes())
    payload = {"user_input": ["draft two", image]}
    if case == "client_tool":
        call = paused.pending.tool_calls[0]
        payload = {"tool_results": [ToolResult(call.name, call.id, '"cells"', images=(image,))]}

    async def stream(agent):
        async with agent.resume_stream(
            paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id, **payload
        ) as resumed:
            return [e async for e in resumed]

    before = await database_snapshot(engine)
    refused = await stream(with_skill(original, "revised"))
    assert [e.type for e in refused] == [RunEventType.RUN_ERROR]
    assert refused[0].run_result.meta["error_type"] == "SkillMismatchError"
    assert transformed == []
    assert await database_snapshot(engine) == before
    compatible = with_skill(original, "original")
    compatible._explicit_skills.append(Skill(name="extra", description="Extra", body="new"))
    assert (await stream(compatible))[-1].run_result.status == RunStatus.SUCCESS
    assert len(transformed) == int(case == "client_tool")


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("operation", ["resume", "reclaim"])
async def test_tokenless_checks_parent_under_fanout(store, engine, operation):
    from dendrux import Agent
    from dendrux.runtime.submit import GOVERNANCE_KEY, governance_snapshot
    from tests.integration.test_fanout_routing import fan_out, root_agent, start_root

    a, b = finisher("a"), finisher("b")
    middle = Agent(
        name="middle",
        description="Delegates",
        prompt="work",
        provider=ContextLLM([done("joined")]),
        subagents=[a, b],
    )
    sibling = finisher("sibling")
    root = root_agent(store, middle, sibling)
    root_id = await start_root(store, root)
    top = await fan_out(store, root, root_id, [middle, sibling], run=(1,))
    middle_id = top.child(0)
    await store.update_run_meta(middle_id, {GOVERNANCE_KEY: governance_snapshot(middle)})
    inner = await fan_out(store, root, middle_id, [a, b], root_id=root_id, path=("middle",), run=())
    for child in inner.children:
        await expire(store, engine, child)
    changed = copy(root)
    changed_middle = copy(middle)
    changed_middle._limits = Limits(max_parallel_subagents=1)
    changed._subagents = [changed_middle, sibling]
    before = await database_snapshot(engine)
    with pytest.raises(GovernanceMismatchError) as refused:
        await getattr(changed, operation)(root_id, tenant_id=TENANT)
    assert refused.value.run_id == middle_id
    assert await database_snapshot(engine) == before
    assert [a.provider.calls, b.provider.calls] == [0, 0]
    assert (await getattr(root, operation)(root_id, tenant_id=TENANT)).status == RunStatus.SUCCESS


async def test_renamed_root_preserves_child_approval(store, engine):
    original = build(store, "approval")
    paused = await original.run("go", tenant_id=TENANT)
    other = copy(original)
    other.name = "other"
    before = await database_snapshot(engine)
    with pytest.raises(RunNotReclaimableError) as refused:
        await answer(other, paused, "submit_approval")
    assert refused.value.reason == "agent_mismatch"
    assert await database_snapshot(engine) == before
    assert (await answer(original, paused, "submit_approval")).status == RunStatus.SUCCESS


async def test_sql_projects_skills_and_distinguishes_absence_from_null(store):
    from tests.integration.test_fanout_store_races import batch, fanout_parent

    parent, calls = await fanout_parent(store, 4)
    request = batch(parent, calls)
    sections = (
        {},
        {"skills": {"pins": {}}},
        {"skills": {"pins": None}},
        {"skills": {"pins": {"demo": "saved"}, "session_ref": "ref"}},
    )
    children = tuple(
        replace(c, child_loop_state={**c.child_loop_state, **section})
        for c, section in zip(request.children, sections, strict=True)
    )
    await store.bind_managed_children(replace(request, children=children))
    rows = await store.get_managed_children(
        parent, [c.child_run_id for c in children], tenant_id=request.tenant_id
    )
    assert [getattr(row, "skill_pins", "unprojected") for row in rows] == [
        {},
        {},
        None,
        {"demo": "saved"},
    ]


async def test_malformed_manifest_stream_refusal_precedes_image(store, engine, monkeypatch):
    from dendrux.skills._session import META_KEY

    patch_questions(monkeypatch)
    agent = with_skill(direct(store, "input"), "original")
    paused = await agent.run("go", tenant_id=TENANT)
    before_meta = (await store.get_run(paused.run_id)).meta[META_KEY]
    await store.update_run_meta(paused.run_id, {META_KEY: {"version": 2}})
    before = await database_snapshot(engine)
    refused = await answer(agent, paused, "resume_stream", image=True)
    assert refused.meta["error_type"] == "ValueError"
    assert await database_snapshot(engine) == before
    await store.update_run_meta(paused.run_id, {META_KEY: before_meta})
    assert (await answer(agent, paused, "resume_stream", image=True)).status == RunStatus.SUCCESS
