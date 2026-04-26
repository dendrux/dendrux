"""Return-path deanonymization for RunResult.answer.

The DB-stores-raw invariant says persisted answer is what the LLM produced
— which contains placeholders because the LLM only saw placeholders. The
runner reverses the mapping on the way back to the dev so chatbots get
"Nice to meet you, anmol!" instead of "<<PERSON_1>>" on result.answer.
"""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import LLMResponse, RunStatus, UsageStats


def _resp(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
    )


@pytest.fixture
def db_store(engine):
    return SQLAlchemyStateStore(engine)


class TestReturnPathDeanonymize:
    async def test_runresult_answer_is_deanonymized(self, db_store) -> None:
        """LLM sees placeholder, replies with placeholder; dev gets the original back."""
        llm = MockLLM([_resp("Nice to meet you, <<PERSON_1>>!")])
        agent = Agent(
            provider=llm,
            prompt="Friendly chatbot.",
            state_store=db_store,
            guardrails=[PII(engine="presidio")],
        )

        result = await agent.run("My name is Anmol Gautam")

        assert result.status == RunStatus.SUCCESS
        # The user-facing answer has the original name, not the placeholder.
        assert "Anmol Gautam" in result.answer
        assert "<<PERSON_1>>" not in result.answer

    async def test_db_row_keeps_raw_placeholder(self, db_store) -> None:
        """DB persists the LLM's raw output (placeholder); only the return value
        is deanonymized. Dashboard's audit view stays placeholder-faithful."""
        llm = MockLLM([_resp("Hello <<PERSON_1>>!")])
        agent = Agent(
            provider=llm,
            prompt="Greeter.",
            state_store=db_store,
            guardrails=[PII(engine="presidio")],
        )

        result = await agent.run("I am Anmol Gautam")
        assert result.status == RunStatus.SUCCESS

        persisted = await db_store.get_run(result.run_id)
        assert persisted is not None
        # DB still holds the placeholder version — the audit invariant.
        assert "<<PERSON_1>>" in (persisted.answer or "")
        assert "Anmol Gautam" not in (persisted.answer or "")

    async def test_unmapped_placeholder_emits_event_and_passes_through(self, db_store) -> None:
        """LLM hallucinates <<PERSON_99>> (no entry in mapping) → event + passthrough."""
        llm = MockLLM([_resp("Nice to meet you, <<PERSON_99>>!")])
        agent = Agent(
            provider=llm,
            prompt="Friendly chatbot.",
            state_store=db_store,
            guardrails=[PII(engine="presidio")],
        )

        result = await agent.run("My name is Anmol Gautam")
        assert result.status == RunStatus.SUCCESS

        # Hallucinated placeholder remains visible because we have no
        # original to substitute. PERSON_1 (the real one) deanonymized fine.
        assert "<<PERSON_99>>" in result.answer

        events = await db_store.get_run_events(result.run_id)
        unmapped_events = [e for e in events if e.event_type == "guardrail.unmapped_placeholder"]
        assert len(unmapped_events) == 1
        assert unmapped_events[0].data["placeholders"] == ["<<PERSON_99>>"]

    async def test_no_pii_guardrail_no_behavior_change(self, db_store) -> None:
        """Without PII configured, mapping is empty; answer flows through verbatim."""
        llm = MockLLM([_resp("Hello, world!")])
        agent = Agent(
            provider=llm,
            prompt="Greeter.",
            state_store=db_store,
        )

        result = await agent.run("Hi")
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Hello, world!"

        # No unmapped-placeholder events when no mapping exists.
        events = await db_store.get_run_events(result.run_id)
        assert not any(e.event_type == "guardrail.unmapped_placeholder" for e in events)

    async def test_multiple_entities_all_deanonymized(self, db_store) -> None:
        """Mapping with multiple entries — each one substituted."""
        llm = MockLLM([_resp("I'll email <<EMAIL_ADDRESS_1>> about <<PERSON_1>>'s account.")])
        agent = Agent(
            provider=llm,
            prompt="Helper.",
            state_store=db_store,
            guardrails=[PII(engine="presidio")],
        )

        result = await agent.run("Email jane@example.com about Anmol Gautam's account.")
        assert result.status == RunStatus.SUCCESS
        assert "jane@example.com" in result.answer
        assert "Anmol Gautam" in result.answer
        assert "<<" not in result.answer
