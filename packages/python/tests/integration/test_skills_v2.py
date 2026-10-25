"""Runtime lifecycle acceptance matrix; runs on SQLite and PostgreSQL."""

from __future__ import annotations

import asyncio

import pytest

from dendrux import Agent, Skill, SkillExecutionResult, SkillExecutor, tool
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock
from dendrux.errors import SkillMismatchError, SkillMissingError
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, Role, RunStatus, ToolCall, ToolResult


class Executor(SkillExecutor):
    async def execute(self, request):
        return SkillExecutionResult(stdout="secret stdout")


def skill(body="secret body"):
    return Skill(
        name="demo",
        description="Demo",
        body=body,
        files={"ref.txt": "secret resource", "scripts/a.py": "secret script"},
    )


def agent(store, responses, **kwargs):
    return Agent(
        prompt="Test.",
        name="skills",
        provider=MockLLM(responses),
        state_store=store,
        skills=kwargs.pop("skills", [skill()]),
        skill_executor=kwargs.pop("skill_executor", Executor()),
        **kwargs,
    )


async def invoke(a, streaming=False, **kwargs):
    if streaming:
        stream = a.stream("raw input", **kwargs)
        async for _ in stream:
            pass
        return stream.result
    return await a.run("raw input", **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("contextual", [False, True])
async def test_explicit_wire_and_raw_trace(store, streaming, contextual):
    a = agent(store, [LLMResponse(text="done")])
    kwargs = {
        "activate_skills": ["demo", "demo"],
        "context": [
            ContextBlock(content="STABLE", placement="stable"),
            ContextBlock(content="DYNAMIC"),
        ],
    }
    if contextual:
        kwargs.update(
            history=[
                ChatMessage(role=Role.USER, content="old"),
                ChatMessage(role=Role.ASSISTANT, content="answer"),
            ],
        )
    result = await invoke(a, streaming, **kwargs)
    assert result.status == RunStatus.SUCCESS
    record = await store.get_run(result.run_id)
    assert record.meta["dendrux.skills"]["explicit"] == ["demo"]
    traces = await store.get_traces(result.run_id)
    user = next(t for t in traces if t.role == "user")
    assert user.content == "raw input"
    wire = user.meta["wire_content"]
    assert wire.count('<skill_content name="demo"') == 1
    assert wire.index("<skill_content") < wire.index("DYNAMIC") < wire.index("raw input")
    if contextual:
        assert "STABLE" in a._provider.call_history[0]["messages"][1].content
        assert "STABLE" not in wire
    else:
        assert wire.index("<skill_content") < wire.index("STABLE") < wire.index("DYNAMIC")
    assert "activate_skills" not in a._provider.call_history[0].get("kwargs", {})


@pytest.mark.parametrize("streaming", [False, True])
async def test_activation_then_read_in_same_response(store, streaming):
    a = agent(
        store,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="use_skill", params={"name": "demo"}),
                    ToolCall(
                        name="read_skill_resource", params={"skill": "demo", "path": "ref.txt"}
                    ),
                ]
            ),
            LLMResponse(text="done"),
        ],
        tool_result_transform=lambda ctx: "CORRUPTED",
    )
    result = await invoke(a, streaming)
    traces = await store.get_traces(result.run_id)
    read = next(
        t
        for t in traces
        if t.role == "tool" and (t.meta or {}).get("tool_name") == "read_skill_resource"
    )
    assert "secret resource" in read.content and "CORRUPTED" not in read.content
    events = await store.get_run_events(result.run_id)
    payload = repr([e.data for e in events if e.event_type.startswith("skill.")])
    assert all(
        secret not in payload
        for secret in ["secret body", "secret resource", "secret script", "secret stdout"]
    )


@pytest.mark.parametrize("activated,path", [(False, "scripts/a.py"), (True, "../escape")])
async def test_preflight_before_approval(store, activated, path):
    a = agent(
        store,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="run_skill_script", params={"skill": "demo", "path": path, "args": []}
                    )
                ]
            ),
            LLMResponse(text="done"),
        ],
        require_approval=["run_skill_script"],
    )
    result = await a.run("input", activate_skills=["demo"] if activated else [])
    assert result.status == RunStatus.SUCCESS
    events = await store.get_run_events(result.run_id)
    assert not any(e.event_type in ("approval.requested", "skill.script_requested") for e in events)
    results = await store.get_tool_calls(result.run_id)
    assert len(results) == 1 and not results[0].success


@tool(target="client")
async def client() -> str:
    return ""


@pytest.mark.parametrize(
    "path", ["resume", "resume_stream", "submit_tool_results", "submit_input", "approve", "reject"]
)
@pytest.mark.parametrize("change", ["missing", "mismatch", "success"])
async def test_continuation_matrix(store, path, change):
    approval = path in {"approve", "reject"}
    tc = ToolCall(
        name="run_skill_script" if approval else "client",
        params={"skill": "demo", "path": "scripts/a.py", "args": []} if approval else {},
    )
    a = agent(
        store,
        [LLMResponse(tool_calls=[tc])],
        tools=[client],
        require_approval=["run_skill_script"] if approval else None,
    )
    paused = await a.run("input", activate_skills=["demo"])
    if path == "submit_input":
        # A human-input pause uses the same durable state; construct its status
        # through the store API rather than relying on a provider-specific ask tool.
        await store.pause_run(
            paused.run_id,
            status=RunStatus.WAITING_HUMAN_INPUT.value,
            pause_data=await store.get_pause_state(paused.run_id),
        )
    before = (await store.get_run(paused.run_id)).status
    new = agent(
        store,
        [LLMResponse(text="done")],
        skills=[]
        if change == "missing"
        else [skill("changed" if change == "mismatch" else "secret body")],
        tools=[client],
    )
    pause = await store.get_pause_state(paused.run_id)
    results = [
        ToolResult(name="client", call_id=pause["pending_tool_calls"][0]["id"], payload='"ok"')
    ]

    async def proceed():
        if path == "resume_stream":
            stream = new.resume_stream(
                paused.run_id, tool_results=results, expected_pause_id=paused.pause_id
            )
            events = [e async for e in stream]
            if change != "success":
                assert events[-1].run_result.meta["error_type"] == (
                    "SkillMissingError" if change == "missing" else "SkillMismatchError"
                )
            return stream.result
        if path == "resume":
            return await new.resume(
                paused.run_id, tool_results=results, expected_pause_id=paused.pause_id
            )
        if path == "submit_tool_results":
            return await new.submit_tool_results(paused.run_id, results)
        if path == "submit_input":
            return await new.submit_input(paused.run_id, "answer")
        return await new.submit_approval(
            paused.run_id, approved=path == "approve", expected_pause_id=paused.pause_id
        )

    if change != "success" and path != "resume_stream":
        with pytest.raises(SkillMissingError if change == "missing" else SkillMismatchError):
            await proceed()
    else:
        result = await proceed()
        if change == "success":
            assert result.status == RunStatus.SUCCESS
    if change != "success":
        assert (await store.get_run(paused.run_id)).status == before


@pytest.mark.parametrize("change", ["deny", "no_executor"])
async def test_approved_execution_rechecks_policy(store, change):
    tc = ToolCall(
        name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
    )
    a = agent(store, [LLMResponse(tool_calls=[tc])], require_approval=["run_skill_script"])
    paused = await a.run("input", activate_skills=["demo"])
    kwargs = {"deny": ["run_skill_script"]} if change == "deny" else {"skill_executor": None}
    new = agent(store, [LLMResponse(text="done")], **kwargs)
    result = await new.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )
    assert result.status == RunStatus.SUCCESS
    traces = await store.get_traces(paused.run_id)
    assert next(t for t in traces if t.role == "tool").meta["is_error"]
    events = await store.get_run_events(paused.run_id)
    if change == "no_executor":
        assert any(
            e.event_type == "skill.script_failed" and e.data["error_type"] == "executor_unavailable"
            for e in events
        )
    else:
        assert any(e.event_type == "policy.denied" for e in events)


async def test_retry_rebuilds_activation_and_checks_tenant(store):
    from dendrux.errors import RunNotFoundError

    a = agent(store, [LLMResponse(text="done")])
    original = await a.run("raw", tenant_id="tenant", activate_skills=["demo"])
    new = agent(store, [LLMResponse(text="retried")], skills=[skill("updated body")])
    with pytest.raises(RunNotFoundError):
        await new.retry(original.run_id, tenant_id="other")
    result = await new.retry(original.run_id, tenant_id="tenant")
    assert result.status == RunStatus.SUCCESS
    messages = new._provider.call_history[0]["messages"]
    first = next(m for m in messages if m.role == Role.USER)
    assert first.content.count('<skill_content name="demo"') == 1
    assert "updated body" in first.content and "secret body" not in first.content


@pytest.mark.parametrize("path", ["batch", "stream", "submit"])
async def test_script_cancellation_finalizes_cancelled(store, path):
    class Blocking(SkillExecutor):
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = False

        async def execute(self, request):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    executor = Blocking()
    tc = ToolCall(
        name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
    )
    a = agent(
        store,
        [LLMResponse(tool_calls=[tc])],
        skill_executor=executor,
        require_approval=["run_skill_script"] if path == "submit" else None,
    )
    if path == "submit":
        paused = await a.run("input", activate_skills=["demo"])
        run_id = paused.run_id
        task = asyncio.create_task(
            a.submit_approval(run_id, approved=True, expected_pause_id=paused.pause_id)
        )
    elif path == "stream":
        stream = a.stream("input", activate_skills=["demo"])
        run_id = stream.run_id

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
    else:
        task = asyncio.create_task(a.run("input", activate_skills=["demo"]))
    await asyncio.wait_for(executor.started.wait(), 5)
    if path == "batch":
        run_id = a._provider.call_history[0]["run_id"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert executor.cancelled
    assert (await store.get_run(run_id)).status == RunStatus.CANCELLED.value


async def test_resume_pins_catalogue_and_reconstructs_activation(store):
    a = agent(
        store,
        [
            LLMResponse(tool_calls=[ToolCall(name="use_skill", params={"name": "demo"})]),
            LLMResponse(tool_calls=[ToolCall(name="client", params={})]),
        ],
        tools=[client],
    )
    paused = await a.run("input")
    new = agent(
        store,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="read_skill_resource", params={"skill": "demo", "path": "ref.txt"}
                    )
                ]
            ),
            LLMResponse(text="done"),
        ],
        skills=[skill(), Skill(name="extra", description="extra", body="extra")],
        tools=[client],
    )
    pause = await store.get_pause_state(paused.run_id)
    result = await new.submit_tool_results(
        paused.run_id,
        [ToolResult(name="client", call_id=pause["pending_tool_calls"][0]["id"], payload='"ok"')],
    )
    assert result.status == RunStatus.SUCCESS
    call = new._provider.call_history[0]
    assert "extra" not in call["messages"][0].content
    assert next(td for td in call["tools"] if td.name == "use_skill").parameters["properties"][
        "name"
    ]["enum"] == ["demo"]
    traces = await store.get_traces(paused.run_id)
    read = next(t for t in traces if (t.meta or {}).get("tool_name") == "read_skill_resource")
    assert "secret resource" in read.content and not read.meta.get("is_error")


async def test_approved_script_rechecks_activation(store):
    a = agent(
        store,
        [
            LLMResponse(tool_calls=[ToolCall(name="use_skill", params={"name": "demo"})]),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="run_skill_script",
                        params={"skill": "demo", "path": "scripts/a.py", "args": []},
                    )
                ]
            ),
        ],
        require_approval=["run_skill_script"],
    )
    paused = await a.run("input")
    state = await store.get_pause_state(paused.run_id)
    # Simulate a failed activation in authoritative durable history. Approval
    # cannot replace the preflight activation check on the reconstructed session.
    for message in state["history"]:
        if message["role"] == "tool" and message.get("name") == "use_skill":
            message.setdefault("meta", {})["is_error"] = True
    await store.pause_run(paused.run_id, status="waiting_approval", pause_data=state)
    new = agent(store, [LLMResponse(text="done")])
    await new.submit_approval(paused.run_id, approved=True, expected_pause_id=paused.pause_id)
    calls = await store.get_tool_calls(paused.run_id)
    script = next(c for c in calls if c.tool_name == "run_skill_script")
    assert not script.success and "not active" in script.error_message
