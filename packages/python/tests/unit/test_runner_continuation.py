"""Continuation payload, loop arguments and approval/stop ordering pins."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux import Agent, chat
from dendrux.llm.mock import MockLLM
from dendrux.loops import _helpers
from dendrux.loops.react import ReActLoop
from dendrux.runtime import _reclaim, _run_events, _run_images, _run_outcome, runner
from dendrux.types import (
    ImagePart,
    ImageRef,
    ImageViewState,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
    TextPart,
    ToolCall,
    ToolResult,
    UsageStats,
)
from tests.unit.test_runner_failure_save import client_read, observer
from tests.unit.test_runner_failure_save import store as store  # noqa: F401


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
@pytest.mark.parametrize("kind", ["results", "input", "recovering"])
async def test_resumed_payload_on_both_entries(store, monkeypatch, entry, kind):
    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("client_read", id="call")])])
    agent = Agent(
        name="payload",
        prompt="p",
        tools=[client_read],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        loop=ReActLoop(),
    )
    paused = await agent.run("go", tenant_id=None)
    # Isolate caller arguments and the real builder from claim/media/replay machinery.
    monkeypatch.setattr(
        runner,
        "_claim_resume",
        AsyncMock(
            side_effect=lambda store, run, raw, agent, expected, **kw: (store, expected, raw)
        ),
    )
    monkeypatch.setattr(runner, "_replay_resume_injected_history", AsyncMock())
    monkeypatch.setattr(_reclaim, "needs_resolution", AsyncMock(return_value=kind == "recovering"))
    monkeypatch.setattr(_reclaim, "resolve", AsyncMock(return_value=None))
    image = ImageRef("a" * 64, "image/png", 1, 1, 1)
    results = [
        ToolResult("client_read", "call", '"ok"', images=(image,)),
        ToolResult("client_read", "second", '"bad"', success=False),
    ]
    parts = (TextPart("answer"), ImagePart(image))

    # Image ingestion is pinned elsewhere; bypass blob writes for this payload unit.
    async def submission(
        store,
        agent,
        provider,
        run,
        pause,
        raw,
        reads,
        loop,
        skip,
        expected,
        pid,
        trs,
        text,
        user_parts,
        items,
        decisions,
    ):
        return store, expected, raw, trs, parts if text is not None else None

    monkeypatch.setattr(runner, "_resume_submission", submission)
    monkeypatch.setattr(
        runner,
        "_stream_resume_images",
        AsyncMock(
            return_value=(store, None, parts, False)
            if kind == "input"
            else (store, results, None, True)
        ),
    )
    if kind == "input":
        monkeypatch.setattr(chat, "split_user_content", lambda value: ("answer", (object(),)))
    monkeypatch.setattr(runner, "validate_result_names", Mock())
    monkeypatch.setattr(
        runner, "prepare_resume_results", AsyncMock(side_effect=lambda a, s, r, p, trs: trs)
    )
    # Text input's image count is pinned directly below; public input media still uses its own seam.
    captured = []
    original = runner._emit_event

    async def emit(store, run, event, data, **kwargs):
        if event == "run.resumed":
            captured.append(data)
        return await original(store, run, event, data, **kwargs)

    monkeypatch.setattr(_run_events, "_emit_event", emit)

    async def run(**kwargs):
        return RunResult(status=RunStatus.SUCCESS, answer="done", run_id=kwargs["run_id"])

    async def stream(**kwargs):
        yield RunEvent(type=RunEventType.RUN_COMPLETED, run_result=await run(**kwargs))

    monkeypatch.setattr(agent._loop, "run", run)
    monkeypatch.setattr(agent._loop, "run_stream", stream)
    kwargs = {"user_input": "answer"} if kind == "input" else {"tool_results": results}
    if entry == "resume":
        await agent.resume(paused.run_id, tenant_id=None, **kwargs)
    else:
        async for _ in agent.resume_stream(paused.run_id, tenant_id=None, **kwargs):
            pass
    expected = {"resumed_from": "waiting_human_input" if kind == "input" else "waiting_client_tool"}
    if kind == "results":
        expected["submitted_results"] = [
            dict(call_id="call", name="client_read", success=True, image_count=1),
            dict(call_id="second", name="client_read", success=False),
        ]
    elif kind == "input":
        expected["user_input"] = "answer"
        expected["image_count"] = 1
    assert captured == [expected]
    await agent.close()


@pytest.mark.parametrize("parts_present", [False, True])
async def test_prepare_input_image_count(store, parts_present):
    provider = MockLLM([])
    agent = Agent(name="input", prompt="p", provider=provider, strict_tenancy=False)
    run_id = "01J00000000000000000000000"
    await store.create_run(run_id, agent_name=agent.name, input_data={})
    pause = PauseState(agent.name, [], [], [], 0, 0, UsageStats())
    parts = (
        (TextPart("answer"), ImagePart(ImageRef("b" * 64, "image/png", 1, 1, 1)))
        if parts_present
        else None
    )
    ctx = await runner._prepare_resume(
        run_id,
        pause,
        state_store=store,
        agent=agent,
        provider=provider,
        expected_status="waiting_human_input",
        user_input="answer",
        user_parts=parts,
    )
    events = await store.get_run_events(run_id)
    data = next(e.data for e in events if e.event_type == "run.resumed")
    assert data == {
        "resumed_from": "waiting_human_input",
        "user_input": "answer",
        **({"image_count": 1} if parts_present else {}),
    }
    assert ctx.history[-1] == (
        Message.with_parts(Role.USER, parts) if parts else Message(Role.USER, "answer")
    )


async def test_resume_loop_kwargs_match_stream(store, monkeypatch):
    captured = {}
    contexts = {}
    image_runtime = object()
    original_images = runner._image_runtime
    original = runner._prepare_resume

    async def prepare(*args, **kwargs):
        ctx = await original(*args, **kwargs)
        ctx.pause_state.image_view = ImageViewState(epoch=8)
        ctx.pause_state.budget_fired = ["budget"]
        contexts[current] = ctx
        return ctx

    monkeypatch.setattr(runner, "_prepare_resume", prepare)
    for current in ("resume", "resume_stream"):
        provider = MockLLM([LLMResponse(tool_calls=[ToolCall("client_read", id="call")])])
        agent = Agent(
            name="kwargs",
            prompt="p",
            tools=[client_read],
            provider=provider,
            state_store=store,
            strict_tenancy=False,
            loop=ReActLoop(),
        )
        monkeypatch.setattr(_run_images, "_image_runtime", original_images)
        paused = await agent.run("go", tenant_id=None)
        monkeypatch.setattr(_run_images, "_image_runtime", lambda *args: image_runtime)

        async def run(*, budget_carry=None, _entry=current, **kwargs):
            captured[_entry] = dict(kwargs, budget_carry=budget_carry)
            return RunResult(status=RunStatus.SUCCESS, answer="done", run_id=kwargs["run_id"])

        async def stream(*, budget_carry=None, **kwargs):
            result = await run(budget_carry=budget_carry, **kwargs)
            yield RunEvent(type=RunEventType.RUN_COMPLETED, run_result=result)

        monkeypatch.setattr(agent._loop, "run", run)
        monkeypatch.setattr(agent._loop, "run_stream", stream)
        kwargs = dict(tool_results=[ToolResult("client_read", "call", '"ok"')], tenant_id=None)
        if current == "resume":
            await agent.resume(paused.run_id, **kwargs)
        else:
            async for _ in agent.resume_stream(paused.run_id, **kwargs):
                pass
        await agent.close()
    r, s = captured["resume"], captured["resume_stream"]
    assert r.keys() - s.keys() == {"initial_pii_mapping"}
    assert s.keys() - r.keys() == {"interrupt"}
    assert r["initial_pii_mapping"] is None
    assert s["interrupt"] is not None
    for entry, values in captured.items():
        ctx = contexts[entry]
        assert values["initial_history"] is ctx.history
        assert values["initial_steps"] is ctx.pause_state.steps
        assert values["initial_usage"] is ctx.pause_state.usage
        assert values["recorder"] is ctx.recorder and values["notifier"] is ctx.notifier
        assert values["strategy"] is ctx.resolved_strategy
        assert values["image_runtime"] is image_runtime
    per_run = {
        "agent",
        "provider",
        "strategy",
        "run_id",
        "recorder",
        "notifier",
        "state_store",
        "initial_pii_mapping",
        "interrupt",
    }
    assert {k: v for k, v in r.items() if k not in per_run} == {
        k: v for k, v in s.items() if k not in per_run
    }


@pytest.mark.parametrize(
    "status", [RunStatus.CANCELLED, RunStatus.PAUSED, RunStatus.WAITING_APPROVAL]
)
async def test_recovering_stream_stop_matches_helpers(store, monkeypatch, status):
    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("client_read", id="call")])])
    agent = Agent(
        name="stop",
        prompt="p",
        tools=[client_read],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
    )
    paused = await agent.run("go", tenant_id=None)
    monkeypatch.setattr(_reclaim, "needs_resolution", AsyncMock(return_value=True))
    result = RunResult(status=status, run_id=paused.run_id, answer="stopped", meta={"sentinel": 1})
    monkeypatch.setattr(_reclaim, "resolve", AsyncMock(return_value=result))
    calls = []

    async def persist(**kwargs):
        calls.append(("persist", kwargs["result"]))
        return replace(kwargs["result"], answer="persisted")

    async def recorder(*args):
        calls.append(("recorder", args[-1]))

    async def notifier(*args):
        calls.append(("notifier", args[-1]))

    monkeypatch.setattr(_run_outcome, "_persist_loop_outcome", persist)
    monkeypatch.setattr(_helpers, "record_run_finished", recorder)
    monkeypatch.setattr(_helpers, "notify_run_finished", notifier)
    events = [
        e
        async for e in agent.resume_stream(
            paused.run_id,
            tool_results=[ToolResult("client_read", "call", '"ok"')],
            tenant_id=None,
            notifier=observer(),
        )
    ]
    actual_calls = list(calls)
    assert [e.type for e in events] == [
        RunEventType.RUN_RESUMED,
        RunEventType.RUN_CANCELLED if status == RunStatus.CANCELLED else RunEventType.RUN_PAUSED,
    ]
    assert [name for name, _ in calls] == ["persist", "recorder", "notifier"]
    calls.clear()
    ctx = SimpleNamespace(
        recorder=object(), notifier=object(), pause_state=SimpleNamespace(provider_kwargs={})
    )
    expected = runner._stopped_stream_event(
        await runner._finish_resumed_result(store, paused.run_id, ctx, result)
    )
    assert events[-1] == expected
    assert calls == actual_calls
    await agent.close()


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
@pytest.mark.parametrize("mode", ["approve", "paused", "irrelevant", "reject"])
async def test_approval_batch_order_and_decisions(monkeypatch, entry, mode):
    pause = PauseState(
        "approval",
        [ToolCall("tool", id="call")],
        [],
        [],
        2,
        0,
        UsageStats(),
        pending_decisions={"call": {"approved": True}},
    )
    ctx = runner._ResumeContext(
        history=[],
        recorder=Mock(),
        notifier=Mock(),
        pause_state=pause,
        resolved_loop=ReActLoop(),
        resolved_strategy=Mock(),
    )
    calls = []
    decision_map = {"call": object()}

    async def validate(*args):
        calls.append("validate")
        return decision_map

    monkeypatch.setattr(runner, "_validated_decisions", validate)
    tool_result = ToolResult("tool", "call", '"ok"')
    approval = runner._ApprovalResume(
        [(pause.pending_tool_calls[0], tool_result)], [], {"decision": "approved"}, None
    )

    async def execute(*args, **kwargs):
        calls.append("execute")
        if mode == "approve":
            assert kwargs["decisions"] == (decision_map if entry == "resume" else None)
        else:
            assert kwargs["decisions"]["call"].approved
        if kwargs.get("on_work"):
            kwargs["on_work"]()
        return approval

    monkeypatch.setattr(runner, "_execute_approved_tools", execute)

    async def record(*args):
        calls.append("decided")

    monkeypatch.setattr(runner, "_record_approval_decided", record)

    def stop(*args):
        calls.append("stop")
        return None

    monkeypatch.setattr(runner, "_approval_stop_result", stop)
    expected = "paused" if mode == "paused" else "waiting_approval"
    results = [tool_result] if mode == "reject" else None
    text = "input" if mode == "irrelevant" else None
    if entry == "resume":
        await runner._resume_approval_work(
            Mock(),
            Mock(),
            "run",
            ctx,
            expected,
            results,
            text,
            [],
            None,
            lambda: calls.append("work"),
        )
    else:
        async for event in runner._stream_approval_events(
            Mock(), Mock(), "run", ctx, expected, results, text, None, {}
        ):
            assert event.type == RunEventType.TOOL_RESULT
            calls.append("yield")
    if mode == "irrelevant":
        assert calls == []
    elif mode == "reject":
        assert calls == ["decided"]
    else:
        assert calls == (["validate"] if mode == "approve" and entry == "resume" else []) + [
            "execute",
            "work" if entry == "resume" else "yield",
        ] + (["decided"] if mode == "approve" else []) + ["stop"]


async def test_closing_approval_results_does_not_record_decision(monkeypatch):
    tc = ToolCall("tool", id="call")
    pause = PauseState("approval", [tc], [], [], 2, 0, UsageStats())
    ctx = runner._ResumeContext(
        history=[],
        recorder=Mock(),
        notifier=Mock(),
        pause_state=pause,
        resolved_loop=ReActLoop(),
        resolved_strategy=Mock(),
    )
    approval = runner._ApprovalResume([(tc, ToolResult("tool", "call", '"ok"'))], [], {}, None)
    monkeypatch.setattr(runner, "_execute_approved_tools", AsyncMock(return_value=approval))
    decided = AsyncMock()
    monkeypatch.setattr(runner, "_record_approval_decided", decided)
    events = runner._stream_approval_events(
        Mock(), Mock(), "run", ctx, "waiting_approval", None, None, None, {}
    )
    assert (await anext(events)).type == RunEventType.TOOL_RESULT
    decided.assert_not_awaited()
    await events.aclose()
    decided.assert_not_awaited()
