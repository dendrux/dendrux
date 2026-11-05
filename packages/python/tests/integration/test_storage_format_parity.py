"""Public pause, trace and governance contracts agree across storage formats."""

import json

import pytest

from dendrux import Agent, SingleCall, tool
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.replay import rebuild_history
from dendrux.store import RunStore
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import AgentStep, Clarification, LLMResponse, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format


class Questions(NativeToolCalling):
    def parse_response(self, response):
        if (response.text or "").endswith("?"):
            return AgentStep(
                reasoning=None,
                action=Clarification(question=response.text, options=["first", "second"]),
                raw_response=response.text,
                meta={},
            )
        return super().parse_response(response)


async def _invoke(agent, path, run_id=None, **kwargs):
    if path in {"stream", "resume_stream"}:
        async with getattr(agent, path)(run_id or "go", tenant_id="t", **kwargs) as stream:
            events = [event async for event in stream]
        return events[-1].run_result
    return await getattr(agent, path)(run_id or "go", tenant_id="t", **kwargs)


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize(
    "entry,continuation",
    [
        ("run", "submit_input"),
        ("run", "resume"),
        ("stream", "resume_stream"),
        ("stream", "resume"),
        ("run", "resume_stream"),
    ],
)
async def test_question_survives_every_pause_read(
    store, engine, fmt, entry, continuation, monkeypatch
):
    await pin_storage_format(engine, fmt)
    monkeypatch.setattr("dendrux.runtime.runner.NativeToolCalling", Questions)
    agent = Agent(
        prompt="system",
        provider=MockLLM(
            [LLMResponse(text="Which?"), LLMResponse(text="Again?"), LLMResponse(text="done")]
        ),
        state_store=store,
    )
    result = await _invoke(agent, entry, strategy=Questions())
    for question in ("Which?", "Again?"):
        assert result.status == RunStatus.WAITING_HUMAN_INPUT
        pending = await RunStore(store).get_pending(result.run_id, tenant_id="t")
        saved = await agent.get_result(result.run_id, tenant_id="t")
        for value in (pending, result.pending, saved.pending):
            assert value.question == question
            assert value.options == ["first", "second"]
            assert value.to_dict()["question"] == question
        assert result.to_dict()["pending"]["options"] == ["first", "second"]
        kwargs = {"expected_pause_id": result.pause_id}
        kwargs["text" if continuation == "submit_input" else "user_input"] = "first"
        result = await _invoke(agent, continuation, result.run_id, **kwargs)
    assert result.status == RunStatus.SUCCESS


class Governance(BaseNotifier):
    def __init__(self):
        self.events = []

    async def on_governance_event(self, run_id, event_type, iteration, data, correlation_id=None):
        self.events.append((event_type, data))


@tool()
async def contact() -> str:
    return "other@example.com"


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
async def test_redaction_is_stored_notified_and_never_sent_raw(store, engine, fmt, single):
    await pin_storage_format(engine, fmt)
    provider = MockLLM(
        [LLMResponse(text="done")]
        if single
        else [LLMResponse(tool_calls=[ToolCall("contact")]), LLMResponse(text="done")]
    )
    notifier = Governance()
    agent = Agent(
        prompt="system",
        provider=provider,
        state_store=store,
        guardrails=[PII(action="redact")],
        **({"loop": SingleCall()} if single else {"tools": [contact]}),
    )
    result = await agent.run("me@example.com", tenant_id="t", notifier=notifier)
    assert result.status == RunStatus.SUCCESS
    events = await RunStore(store).get_events(result.run_id, tenant_id="t", limit=100)
    stored = [e.event_type for e in events if e.event_type.startswith("guardrail.")]
    notified = [kind for kind, _ in notifier.events if kind.startswith("guardrail.")]
    assert stored == notified
    assert stored.count("guardrail.redacted") == (1 if single else 2)
    wire = str(provider.wire_messages)
    assert "me@example.com" not in wire and "other@example.com" not in wire
    assert "<<EMAIL_ADDRESS_" in wire


@pytest.mark.parametrize("fmt", [1, 2])
async def test_block_location_contains_current_message_index(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    notifier = Governance()
    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("contact")])])
    result = await Agent(
        prompt="system",
        provider=provider,
        state_store=store,
        guardrails=[PII(action="block")],
        tools=[contact],
    ).run("go", tenant_id="t", notifier=notifier)
    assert result.status == RunStatus.ERROR
    assert "message 3 (role=tool, tool_call_id=" in result.error
    assert len(provider.wire_messages) == 1
    events = await RunStore(store).get_events(result.run_id, tenant_id="t", limit=100)
    blocked = [e.data for e in events if e.event_type == "guardrail.blocked"]
    assert "message 3 (role=tool, tool_call_id=" in json.dumps(blocked)
    assert "me@example.com" not in json.dumps(blocked)
    assert blocked == [data for kind, data in notifier.events if kind == "guardrail.blocked"]


async def test_public_traces_match_legacy_and_preserve_internal_rebuild(store, engine):
    views = []
    for fmt in (1, 2):
        await pin_storage_format(engine, fmt)
        result = await Agent(
            prompt="private system",
            provider=MockLLM(
                [LLMResponse(tool_calls=[ToolCall("contact", id="call")]), LLMResponse(text="done")]
            ),
            tools=[contact],
            state_store=store,
        ).run("go", tenant_id="t")
        reader = RunStore(store)
        rows = await reader.get_traces(result.run_id, tenant_id="t", limit=100)
        views.append([(r.role, r.content, r.order_index, r.meta) for r in rows])
        assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
        assert [r.order_index for r in rows] == list(range(4))
        for offset in range(5):
            page = await reader.get_traces(result.run_id, tenant_id="t", limit=1, offset=offset)
            assert [(r.role, r.content, r.order_index) for r in page] == [
                (r.role, r.content, r.order_index) for r in rows[offset : offset + 1]
            ]
        assert await reader.get_traces(result.run_id, tenant_id="other", limit=100) == []
        if fmt == 2:
            rebuilt = await rebuild_history(store, result.run_id)
            assert rebuilt.messages[0].role.value == "system"
            assert rebuilt.messages[0].content == "private system"
            assert rebuilt.steps[-1].raw_response is not None
    assert views[0] == views[1]


@pytest.mark.parametrize("hostile", [False, True])
async def test_pause_summary_bound_keeps_run_resumable(store, engine, hostile, caplog):
    await pin_storage_format(engine, 2)
    question = "Q" * (2_000_000 if hostile else 10_000) + "?"

    class LargeQuestion(Questions):
        def parse_response(self, response):
            step = super().parse_response(response)
            if isinstance(step.action, Clarification):
                step.action.options[:] = ["option" * 20] * (10_000 if hostile else 100)
            return step

    agent = Agent(
        prompt="system",
        provider=MockLLM([LLMResponse(text=question), LLMResponse(text="done")]),
        state_store=store,
    )
    result = await agent.run("go", tenant_id="t", strategy=LargeQuestion())
    assert result.status == RunStatus.WAITING_HUMAN_INPUT
    raw = await store.get_pause_state(result.run_id)
    assert len(json.dumps(raw["clarification"]).encode()) <= 65536
    if hostile:
        assert result.pending.question.endswith("[truncated]")
        assert "pause_summary_truncated" in caplog.text
    else:
        assert result.pending.question == question
        assert result.pending.options == ["option" * 20] * 100
    resumed = await agent.submit_input(
        result.run_id, "ok", tenant_id="t", expected_pause_id=result.pause_id
    )
    assert resumed.status == RunStatus.SUCCESS


async def test_rejected_tool_observations_keep_validation_order(store, engine):
    views = []
    for fmt in (1, 2):
        await pin_storage_format(engine, fmt)
        provider = MockLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall("contact", {"bad": True}, id="invalid"),
                        ToolCall("missing", id="unknown"),
                    ]
                ),
                LLMResponse(text="done"),
            ]
        )
        result = await Agent(
            prompt="system", provider=provider, tools=[contact], state_store=store
        ).run("go", tenant_id="t")
        assert result.status == RunStatus.SUCCESS
        rows = await RunStore(store).get_traces(result.run_id, tenant_id="t", limit=100)
        views.append(
            (
                [(m.role, m.content, m.name) for m in provider.wire_messages[-1]],
                [(r.role, r.content, r.order_index) for r in rows],
            )
        )
    assert views[0] == views[1]


async def test_pause_summary_limit_override_preserves_larger_question(store, engine):
    from dendrux.types import Limits

    await pin_storage_format(engine, 2)
    question = "Q" * 70000 + "?"
    result = await Agent(
        prompt="system",
        provider=MockLLM([LLMResponse(text=question)]),
        state_store=store,
        limits=Limits(max_pause_summary_bytes=80000),
    ).run("go", tenant_id="t", strategy=Questions())
    assert result.pending.question == question
    assert result.pending.options == ["first", "second"]


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("path", ["run", "stream"])
async def test_single_call_keeps_question_text_as_final_answer(store, engine, fmt, path):
    await pin_storage_format(engine, fmt)
    agent = Agent(
        prompt="system",
        provider=MockLLM([LLMResponse(text="Which?")]),
        loop=SingleCall(),
        state_store=store,
    )
    result = await _invoke(agent, path, strategy=Questions())
    assert result.status == RunStatus.SUCCESS
    assert result.answer == "Which?" and result.pending is None


async def test_question_larger_than_rebuild_budget_can_resume(store, engine):
    await pin_storage_format(engine, 2)
    question = "Q" * (17 * 1024 * 1024) + "?"
    agent = Agent(
        prompt="system",
        provider=MockLLM([LLMResponse(text=question), LLMResponse(text="done")]),
        state_store=store,
    )
    paused = await agent.run("go", tenant_id="t", strategy=Questions())
    assert paused.status == RunStatus.WAITING_HUMAN_INPUT
    result = await agent.submit_input(
        paused.run_id, "first", tenant_id="t", expected_pause_id=paused.pause_id
    )
    assert result.status == RunStatus.SUCCESS


@pytest.mark.parametrize("fmt", [1, 2])
async def test_pause_question_and_options_keep_model_pii_placeholders(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    question = "Send to <<EMAIL_ADDRESS_1>>?"

    class EmailQuestion(Questions):
        def parse_response(self, response):
            step = super().parse_response(response)
            if isinstance(step.action, Clarification):
                step.action.options[:] = ["<<EMAIL_ADDRESS_1>>", "cancel"]
            return step

    provider = MockLLM([LLMResponse(text=question), LLMResponse(text="done")])
    agent = Agent(
        prompt="system", provider=provider, state_store=store, guardrails=[PII(action="redact")]
    )
    paused = await agent.run("me@example.com", tenant_id="t", strategy=EmailQuestion())
    assert paused.pending.question == question
    assert paused.pending.options == ["<<EMAIL_ADDRESS_1>>", "cancel"]
    assert paused.answer == "Send to me@example.com?"
    raw = await store.get_pause_state(paused.run_id)
    saved = raw["clarification"] if fmt == 2 else raw["steps"][-1]["action"]
    assert saved["question"] == question and saved["options"] == paused.pending.options
    resumed = await agent.submit_input(
        paused.run_id, "me@example.com", tenant_id="t", expected_pause_id=paused.pause_id
    )
    assert resumed.status == RunStatus.SUCCESS
    assert "me@example.com" not in str(provider.wire_messages)


@pytest.mark.parametrize("entry", ["engine", "url"])
async def test_public_trace_factory_paths_share_projection(tmp_path, entry):
    from sqlalchemy.ext.asyncio import create_async_engine

    url = f"sqlite+aiosqlite:///{tmp_path / 'factory-traces.db'}"
    agent = Agent(
        prompt="private system", provider=MockLLM([LLMResponse(text="done")]), database_url=url
    )
    result = await agent.run("go", tenant_id="t")
    engine = create_async_engine(url)
    reader = RunStore.from_database_url(url) if entry == "url" else RunStore.from_engine(engine)
    try:
        rows = await reader.get_traces(result.run_id, tenant_id="t", limit=100)
        assert [(row.role, row.content, row.order_index, row.meta) for row in rows] == [
            ("user", "go", 0, {"iteration": 0}),
            ("assistant", "done", 1, {"phase": "final_answer", "iteration": 1}),
        ]
        page = await reader.get_traces(result.run_id, tenant_id="t", limit=1, offset=1)
        assert [(row.role, row.order_index) for row in page] == [("assistant", 1)]
        assert await reader.get_traces(result.run_id, tenant_id="other", limit=100) == []
    finally:
        await reader.close()
        await engine.dispose()
        await agent.close()
