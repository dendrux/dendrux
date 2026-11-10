"""Fallback restores the state that belongs to the retained projection."""

from dendrux.context_blocks import ContextBlock
from dendrux.guardrails import PII, GuardrailEngine
from dendrux.runtime._entry_guardrails import _EntryProjection
from dendrux.types import Message, Role
from tests.integration.test_061_secrets_entry_fallback import ALLOWED, REJECTED, ContextualBlock


async def test_discarded_projection_restores_origins_counters_and_cache():
    engine = GuardrailEngine([ContextualBlock(), PII(action="redact", restore="allowlisted")])
    await engine.project_incoming_message(
        Message(Role.TOOL, ALLOWED, name="read_value", call_id="call")
    )
    expected_mapping = engine._pii_mapping.copy()
    expected_origins = {key: value.copy() for key, value in engine._tool_origins.items()}
    expected_cache = engine._projected.copy()
    expected_bytes = engine._projection_text_bytes
    expected_counters = engine._counters.copy()
    entry = _EntryProjection(engine)
    await entry.prepare(
        [], [ContextBlock("restricted account")], f"{ALLOWED} {REJECTED}", None, None
    )
    assert entry.error is not None
    assert engine._pii_mapping == expected_mapping
    assert engine._reverse == {value: key for key, value in expected_mapping.items()}
    assert engine._trusted == set()
    assert engine._tool_origins == expected_origins
    assert engine._counters == expected_counters
    assert engine._projected == expected_cache
    assert engine._projection_text_bytes == expected_bytes
    projected, _, error = await engine.project_incoming_message(Message(Role.USER, REJECTED))
    assert error is None and projected.content == "<<EMAIL_ADDRESS_2>>"
    assert engine._pii_mapping[projected.content] == REJECTED
