import json

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime import persistence
from dendrux.runtime._clarification import clarification_summary
from dendrux.runtime._pending import _clarification_fields
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import AgentStep, Clarification, Limits, LLMResponse, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_storage_format_parity import Questions


@pytest.mark.parametrize("options", [7, "abc", {}])
def test_malformed_options_are_not_iterated(options):
    assert _clarification_fields(
        {"clarification": {"question": "q", "options": options}}, None, "input"
    ) == ("q", [])


@pytest.mark.parametrize("character,count", [("語", 12000), ("😀", 6000)])
@pytest.mark.parametrize("hostile", [False, True])
def test_summary_uses_utf8_budget(character, count, hostile):
    question = character * (count * 10 if hostile else count)
    summary = clarification_summary(question, ["yes"])
    assert len(json.dumps(summary, ensure_ascii=False).encode()) <= 65536
    if hostile:
        assert summary["question"].endswith("[truncated]")
        assert len(summary["question"].encode()) > 64000
    else:
        assert summary == {"question": question, "options": ["yes"]}


def test_display_budget_is_not_governance():
    a = Agent(prompt="p", provider=MockLLM([]))
    b = Agent(prompt="p", provider=MockLLM([]), limits=Limits(max_pause_summary_bytes=80000))
    assert governance_snapshot(a) == governance_snapshot(b)


async def test_options_truncation_preserves_model_turn_and_computes_once(
    store, engine, monkeypatch
):
    await pin_storage_format(engine, 2)
    calls = []
    original = persistence.clarification_summary

    def summarize(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(persistence, "clarification_summary", summarize)

    class Options(Questions):
        def parse_response(self, response):
            return AgentStep(
                reasoning=None,
                action=Clarification("Which?", ["x" * 100000]),
                raw_response=response.text,
            )

    provider = MockLLM(
        [LLMResponse(text="Context explaining the question. Which?"), LLMResponse(text="done")]
    )
    agent = Agent(prompt="p", provider=provider, state_store=store)
    paused = await agent.run("go", tenant_id="t", strategy=Options())
    assert len(calls) == 1
    assert paused.steps[-1].raw_response == "Context explaining the question. Which?"
    assert paused.pending.options[0].endswith("[truncated]")
    await agent.submit_input(paused.run_id, "yes", tenant_id="t", expected_pause_id=paused.pause_id)
    assert [m.content for m in provider.wire_messages[-1] if m.role.value == "assistant"] == [
        "Context explaining the question. Which?"
    ]


async def test_reclaim_preserves_display_override(store, engine, monkeypatch):
    await pin_storage_format(engine, 2)

    class Crash(BaseException):
        pass

    async def crash(*args, **kwargs):
        raise Crash()

    original = type(store).pause_with_event
    monkeypatch.setattr(type(store), "pause_with_event", crash)
    question = "Q" * 70000 + "?"
    provider = MockLLM([LLMResponse(text=question), LLMResponse(text="done")])
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        limits=Limits(max_pause_summary_bytes=80000),
    )
    with pytest.raises(Crash):
        await agent.run("go", tenant_id="t", strategy=Questions())
    monkeypatch.setattr(type(store), "pause_with_event", original)
    run = (await store.list_runs(limit=1))[0]
    item = await expire(store, engine, run.id)
    recovered = await agent.reclaim(run.id, tenant_id="t", item=item)
    assert recovered.pending.question == question
    assert recovered.pending.options == ["first", "second"]
    raised = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        limits=Limits(max_pause_summary_bytes=90000),
    )
    result = await raised.submit_input(
        run.id, "yes", tenant_id="t", expected_pause_id=recovered.pause_id
    )
    assert result.status == RunStatus.SUCCESS
