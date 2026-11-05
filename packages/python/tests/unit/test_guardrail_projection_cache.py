"""Cached message projections retain governance evidence at the provider boundary."""

import pytest

from dendrux.guardrails import PII
from dendrux.guardrails._engine import GuardrailEngine
from dendrux.types import Message, Role, TextPart
from tests.unit.test_guardrail_message_parts import _IMAGE


@pytest.mark.parametrize("parts", [False, True])
async def test_cached_redaction_keeps_change_signal_and_message_envelope(parts):
    engine = GuardrailEngine([PII(action="redact")])
    msg = Message(
        role=Role.USER,
        content="a@example.com",
        parts=(TextPart("a@example.com"), _IMAGE) if parts else None,
        meta={"marker": "kept"},
    )
    clean, findings, block = await engine.project_incoming_message(msg)
    again, cached, cached_block = await engine.project_incoming_message(clean, index=2)
    assert clean is not msg and again is not clean
    assert again == clean
    assert findings == cached and block is cached_block is None
    assert "a@example.com" not in again.content


@pytest.mark.parametrize("action", ["warn", "block"])
async def test_cache_uses_current_location_without_claiming_redaction(action):
    engine = GuardrailEngine([PII(action=action)])
    msg = Message(role=Role.TOOL, content="a@example.com", name="lookup", call_id="call")
    clean, _, _ = await engine.project_incoming_message(msg)
    again, _, block = await engine.project_incoming_message(clean, index=7)
    assert again is clean
    if action == "block":
        assert "message 7 (role=tool, tool_call_id=call)" in block
    else:
        assert block is None
