"""Audit totals and notifier serialization survive finding-sample truncation."""

import dataclasses
import json

import pytest
from sqlalchemy import select

from dendrux import Agent, tool
from dendrux.db.models import LLMInteraction
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.store import RunStore
from dendrux.types import LLMResponse, ToolCall
from tests._fixtures.storage import pin_storage_format


class SerializeMessages(BaseNotifier):
    def __init__(self):
        self.messages = []
        self.errors = []
        self.events = []

    async def on_message_appended(self, run_id, message, iteration):
        try:
            self.messages.append(json.dumps(dataclasses.asdict(message), default=str))
        except TypeError as error:
            self.errors.append(str(error))

    async def on_governance_event(self, run_id, event_type, iteration, data, correlation_id=None):
        self.events.append((event_type, data))


@tool()
async def other_entities() -> str:
    return "card 4111 1111 1111 1111 ssn 123-45-6789 call +1 415 555 2671"


@pytest.mark.parametrize("action,initial", [("redact", False), ("redact", True), ("block", True)])
async def test_audit_kinds_are_independent_of_finding_sample(store, engine, action, initial):
    views = []
    for version in (1, 2):
        await pin_storage_format(engine, version)
        text = " ".join(f"person{i}@example.com" for i in range(130))
        provider = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("other_entities", id="call")]),
                LLMResponse(text="done"),
            ]
            if not initial
            else [LLMResponse(text="done")]
        )
        notifier = SerializeMessages()
        if initial:
            text += " card 4111 1111 1111 1111 ssn 123-45-6789 call +1 415 555 2671"
        result = await Agent(
            prompt="p",
            provider=provider,
            tools=[other_entities],
            state_store=store,
            guardrails=[PII(action=action)],
        ).run(text, tenant_id="t", notifier=notifier)
        events = await RunStore(store).get_events(result.run_id, tenant_id="t", limit=50)
        audit = [(e.event_type, e.data) for e in events if e.event_type.startswith("guardrail.")]
        for _, data in audit:
            for field in ("entities", "kinds"):
                if field in data:
                    data[field] = sorted(data[field])
        views.append(audit)
        if action == "redact":
            assert audit[-2][1]["findings_count"] == 133
            assert len(audit[-2][1]["entities"]) == 4
            assert len(audit[-1][1]["entities"]) == 4
            for raw in ("@example.com", "4111 1111", "123-45-6789", "+1 415"):
                assert raw not in str(provider.wire_messages)
            async with engine.connect() as connection:
                samples = (
                    (
                        await connection.execute(
                            select(LLMInteraction.guardrail_findings).where(
                                LLMInteraction.agent_run_id == result.run_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            assert len(samples[-1]["incoming"]) == (133 if version == 1 else 128)
        else:
            assert len(audit[-1][1]["kinds"]) == 4
            assert sum(audit[-1][1]["counts"].values()) == 133
        assert not notifier.errors
        assert [
            (kind, data) for kind, data in notifier.events if kind.startswith("guardrail.")
        ] == audit
    assert views[0] == views[1]
