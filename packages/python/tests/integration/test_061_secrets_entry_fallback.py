"""Discarded entry projections cannot contribute reversible state."""

import pytest
from sqlalchemy import select

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock
from dendrux.db.models import Base
from dendrux.guardrails import PII
from dendrux.guardrails._protocol import Finding
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse, RunStatus
from tests._fixtures.storage import pin_storage_format

REJECTED = "blocked.person@example.com"
ALLOWED = "allowed.person@example.com"


class ContextualBlock:
    action = "block"

    async def scan(self, text):
        if "ADDITIONAL CONTEXT:" in text and REJECTED in text:
            start = text.index(REJECTED)
            return [Finding("SECRET", start, start + len(REJECTED), 1.0, REJECTED)]
        return []


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("earlier", [False, True])
async def test_entry_fallback_discards_piece_mappings(store, engine, fmt, single, earlier):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = Agent(
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        guardrails=[ContextualBlock(), PII(action="redact", restore="allowlisted")],
        loop=SingleCall() if single else None,
    )
    try:
        result = await agent.run(
            REJECTED,
            history=[ChatMessage.user(ALLOWED), ChatMessage.assistant("okay")] if earlier else None,
            context=[ContextBlock("restricted account")],
            tenant_id="t",
        )
        assert result.status == RunStatus.ERROR and provider._call_count == 0
        assert REJECTED not in repr(result)
        mapping = await store.get_pii_mapping(result.run_id)
        assert REJECTED not in repr(mapping)
        assert (ALLOWED in mapping.values()) is earlier
        async with engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert REJECTED not in repr(rows), table.name
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
async def test_entry_redaction_still_restores_allowed_values(store, engine, fmt, single):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        guardrails=[PII(action="redact")],
        loop=SingleCall() if single else None,
    )
    try:
        result = await agent.run(ALLOWED, tenant_id="t")
        assert result.status == RunStatus.SUCCESS and provider._call_count == 1
        assert ALLOWED in result.meta["pii_mapping"].values()
        assert ALLOWED in (await store.get_pii_mapping(result.run_id)).values()
    finally:
        await agent.close()
