"""Skills v2 runtime acceptance contracts (written before runtime changes)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, Skill, SkillExecutionResult, SkillExecutor
from dendrux.loops.single import SingleCall
from dendrux.skills._session import SkillSession
from dendrux.types import Message, Role, ToolCall, compute_idempotency_fingerprint


def skill(name="demo", body="private body", files=None):
    return Skill(name=name, description="Example", body=body, files=files or {})


class Executor(SkillExecutor):
    def __init__(self, result=None, error=None):
        self.result = result or SkillExecutionResult(stdout="private output")
        self.error = error
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result


def session(*, executor=None, explicit=(), **kwargs):
    kwargs.setdefault("strict_tenancy", False)
    agent = Agent(
        prompt="Test.",
        skills=[skill(files={"ref.txt": "x" * 50001, "asset": b"bin", "scripts/a.py": "secret"})],
        skill_executor=executor,
        **kwargs,
    )
    return SkillSession.prepare(agent, explicit=explicit)


@pytest.mark.asyncio
async def test_activation_resource_and_events():
    s = session(executor=Executor())
    s.sink = AsyncMock()
    read = ToolCall(name="read_skill_resource", params={"skill": "demo", "path": "ref.txt"})
    assert not (await s.execute(read)).success
    tc = ToolCall(name="use_skill", params={"name": "demo"})
    assert "<skill_content" in (await s.execute(tc)).payload
    assert "already active" in (await s.execute(tc)).payload
    result = await s.execute(read)
    assert result.success and "showing 50000 of 50001" in result.payload
    events = [c.args for c in s.sink.call_args_list]
    assert sum(e[0] == "skill.activated" for e in events) == 1
    assert "private body" not in repr(events)


@pytest.mark.parametrize("path", ["asset", "scripts/a.py", "../ref.txt", "a\n", "missing"])
@pytest.mark.asyncio
async def test_invalid_reads(path):
    s = session(executor=Executor(), explicit=["demo"])
    assert not (
        await s.execute(
            ToolCall(name="read_skill_resource", params={"skill": "demo", "path": path})
        )
    ).success


@pytest.mark.parametrize(
    "result,error,success,event",
    [
        (SkillExecutionResult(), None, True, "skill.script_completed"),
        (SkillExecutionResult(exit_code=1), None, False, "skill.script_completed"),
        (None, RuntimeError("failed"), False, "skill.script_failed"),
    ],
)
@pytest.mark.asyncio
async def test_execution(result, error, success, event):
    executor = Executor(result, error)
    s = session(executor=executor, explicit=["demo"])
    s.run_id = "run"
    s.sink = AsyncMock()
    tc = ToolCall(
        name="run_skill_script",
        params={"skill": "demo", "path": "scripts/a.py", "args": ["secret arg"]},
    )
    assert (await s.execute(tc)).success is success
    assert s.sink.call_args.args[0] == event
    request = executor.requests[0]
    assert request.run_id == "run" and request.tool_call_id == tc.id
    assert request.files["SKILL.md"].startswith(b"---")
    await s.execute(tc)
    assert executor.requests[1].files is request.files
    assert "secret arg" not in repr(s.sink.call_args_list)


@pytest.mark.asyncio
async def test_timeout_and_cancel():
    class Slow(Executor):
        timeout_seconds = 0.001

        async def execute(self, request):
            await asyncio.Event().wait()

    s = session(executor=Slow(), explicit=["demo"])
    s.sink = AsyncMock()
    tc = ToolCall(
        name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
    )
    assert not (await s.execute(tc)).success
    assert s.sink.call_args.args[1]["error_type"] == "timeout"
    s.executor.timeout_seconds = 100
    task = asyncio.create_task(s.execute(tc))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_restore_uses_paired_successful_calls():
    s = session(executor=Executor())
    calls = [
        ToolCall(name="use_skill", params={"name": "demo"}, id="ok"),
        ToolCall(name="use_skill", params={"name": "other"}, id="bad"),
    ]
    s.restore(
        [
            Message(role=Role.ASSISTANT, content="", tool_calls=calls),
            Message(role=Role.TOOL, content="opaque", name="use_skill", call_id="ok"),
            Message(
                role=Role.TOOL,
                content="fake success",
                name="use_skill",
                call_id="bad",
                meta={"is_error": True},
            ),
        ]
    )
    assert s.activated == {"demo"}


def test_denied_catalogue_and_frozen_enums():
    s = session(executor=Executor(), deny=["use_skill"])
    assert not s.tool_defs()
    s = session(executor=Executor(), explicit=["demo"], deny=["use_skill"])
    assert {t.name for t in s.tool_defs()} == {"read_skill_resource", "run_skill_script"}
    assert all(t.parameters["properties"]["skill"]["enum"] == ["demo"] for t in s.tool_defs())


def test_singlecall_rejects_files_and_explicit_dedup():
    with pytest.raises(ValueError, match="SingleCall"):
        Agent(prompt="Test.", skills=[skill(files={"ref": "x"})], loop=SingleCall())
    a = Agent(prompt="Test.", skills=[skill()])
    s = SkillSession.prepare(a, explicit=["demo", "demo"])
    assert s.explicit == ["demo"]
    with pytest.raises(ValueError, match="missing"):
        SkillSession.prepare(a, explicit=["missing"])


def test_fingerprint_changes_for_each_skill_component():
    s = SkillSession.prepare(Agent(prompt="Test.", skills=[skill()]))
    base = compute_idempotency_fingerprint("a", "input", skills=s.fingerprint())
    for agent, explicit in [
        (Agent(prompt="Test.", skills=[skill()]), ["demo"]),
        (Agent(prompt="Test.", skills=[skill(body="changed")]), []),
        (Agent(prompt="Test.", skills=[skill(), skill("other")]), []),
    ]:
        other = SkillSession.prepare(agent, explicit=explicit)
        assert compute_idempotency_fingerprint("a", "input", skills=other.fingerprint()) != base


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_shared_agent_isolation(streaming):
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse

    class Concurrent(MockLLM):
        def __init__(self):
            super().__init__([])
            self.seen = []
            self.arrived = 0
            self.gate = asyncio.Event()

        async def complete(self, messages, tools=None, **kwargs):
            results = [m for m in messages if m.role == Role.TOOL]
            if results:
                self.seen.append((messages[1].content, results[-1].meta.get("is_error", False)))
                return LLMResponse(text="done")
            self.arrived += 1
            if self.arrived == 2:
                self.gate.set()
            await self.gate.wait()
            return LLMResponse(
                tool_calls=[
                    ToolCall(name="read_skill_resource", params={"skill": "demo", "path": "ref"})
                ]
            )

    provider = Concurrent()
    a = Agent(prompt="Test.", provider=provider, skills=[skill(files={"ref": "resource"})])

    async def run(explicit):
        if streaming:
            stream = a.stream("input", activate_skills=explicit)
            async for _ in stream:
                pass
        else:
            await a.run("input", activate_skills=explicit)

    await asyncio.gather(run(["demo"]), run([]))
    assert sorted(error for _, error in provider.seen) == [False, True]
    from dendrux.skills._session import current

    assert current() is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("denied", [False, True])
@pytest.mark.asyncio
async def test_invalid_activation_before_creation(streaming, denied):
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.runner import run, run_stream

    store = AsyncMock()
    a = Agent(prompt="Test.", skills=[skill()], deny_skills=["demo"] if denied else [])
    kwargs = dict(
        provider=MockLLM([]),
        user_input="input",
        state_store=store,
        activate_skills=["demo" if denied else "unknown"],
    )
    with pytest.raises(ValueError):
        if streaming:
            run_stream(a, **kwargs)
        else:
            await run(a, **kwargs)
    store.create_run.assert_not_called()


@pytest.mark.parametrize("name", ["use_skill", "read_skill_resource", "run_skill_script"])
def test_reserved_names_and_governance(name):
    from dendrux import tool

    def function():
        return "ok"

    function.__name__ = name
    with pytest.raises(ValueError, match="reserved"):
        Agent(prompt="Test.", skills=[skill()], tools=[tool()(function)])
    Agent(prompt="Test.", skills=[skill()], deny=[name])
    Agent(prompt="Test.", skills=[skill()], require_approval=[name])


def test_lazy_directory_warning_and_singlecall(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    s = skill(files={"scripts/a.py": "pass"})
    (root / "SKILL.md").write_text(s.to_markdown())
    (root / "scripts").mkdir()
    (root / "scripts/a.py").write_text("pass")
    with pytest.warns(UserWarning, match="demo"):
        a = Agent(prompt="Test.", skills=[s])
    assert "run_skill_script" not in {t.name for t in a.get_tool_defs()}
    a = Agent(prompt="Test.", skills_dir=tmp_path)
    with pytest.warns(UserWarning, match="demo"):
        SkillSession.prepare(a)
    a = Agent(prompt="Test.", skills_dir=tmp_path, loop=SingleCall())
    with pytest.raises(ValueError, match="SingleCall"):
        SkillSession.prepare(a)
    with pytest.raises(ValueError, match="SingleCall"):
        SkillSession.prepare(
            Agent(prompt="Test.", skills=[s], skill_executor=Executor()),
            loop=SingleCall(),
        )


@pytest.mark.parametrize("name", ["use_skill", "read_skill_resource", "run_skill_script"])
@pytest.mark.asyncio
async def test_mcp_reserved_names(name):
    from unittest.mock import MagicMock

    from dendrux.mcp._errors import MCPToolNameCollisionError
    from dendrux.mcp._server import MCPServer
    from dendrux.types import ToolDef

    source = MagicMock(spec=MCPServer)
    source.name = "source"
    source._discover = AsyncMock(
        return_value=[
            ToolDef(
                name=name,
                description="collision",
                parameters={"type": "object", "properties": {}},
                meta={"source_name": "source", "mcp_tool_name": name},
            )
        ]
    )
    source.close = AsyncMock()
    a = Agent(prompt="Test.", skills=[skill()], tool_sources=[source])
    with pytest.raises(MCPToolNameCollisionError, match="reserved"):
        await a.get_tool_lookups()


@pytest.mark.asyncio
async def test_output_caps_and_invalid_script_args():
    executor = Executor(SkillExecutionResult(stdout="x" * 20001, stderr="y" * 20002))
    s = session(executor=executor, explicit=["demo"])
    tc = ToolCall(
        name="run_skill_script", params={"skill": "demo", "path": "scripts/a.py", "args": []}
    )
    result = await s.execute(tc)
    import json

    data = json.loads(result.payload)
    assert data["truncated"] and len(data["stdout"]) == len(data["stderr"]) == 20000
    for args in [["x"] * 65, ["x" * 4097], [1], "argument"]:
        tc.params["args"] = args
        assert not (await s.execute(tc)).success
    assert len(executor.requests) == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_singlecall_activation_events(streaming):
    from dendrux.llm.mock import MockLLM
    from dendrux.loops.base import BaseNotifier
    from dendrux.types import LLMResponse

    class Sink(BaseNotifier):
        def __init__(self):
            self.events = []

        async def on_governance_event(
            self, run_id, event_type, iteration, data, correlation_id=None
        ):
            self.events.append((event_type, data))

    sink = Sink()
    a = Agent(
        prompt="Test.",
        provider=MockLLM([LLMResponse(text="done")]),
        loop=SingleCall(),
        skills=[skill(), skill("other")],
    )
    with pytest.warns(UserWarning, match="inlined"):
        if streaming:
            async for _ in a.stream("input", activate_skills=["demo"], notifier=sink):
                pass
        else:
            await a.run("input", activate_skills=["demo"], notifier=sink)
    activated = [d for e, d in sink.events if e == "skill.activated"]
    assert [(d["name"], d["origin"]) for d in activated] == [
        ("demo", "explicit"),
        ("other", "inlined"),
    ]
    messages = a._provider.call_history[0]["messages"]
    assert "<skill_content" not in messages[-1].content
