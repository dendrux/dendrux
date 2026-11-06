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


@pytest.mark.parametrize("count", [20, 200])
async def test_projection_retains_true_count_with_bounded_evidence(count):
    engine = GuardrailEngine([PII(action="redact")])
    msg = Message(Role.USER, " ".join(f"person{i}@example.com" for i in range(count)))
    clean, _, _ = await engine.project_incoming_message(msg)
    again, findings, _ = await engine.project_incoming_message(clean, index=1)
    assert again is not clean
    assert getattr(findings, "total_count", len(findings)) == count
    assert len(findings) <= 128
    assert all(not finding.text for finding in findings)


async def test_redaction_evidence_survives_projection_cache_eviction():
    from dataclasses import replace

    from dendrux.types import _message_to_dict

    engine = GuardrailEngine([PII(action="redact")])
    messages = []
    for i in range(300):
        clean, _, _ = await engine.project_incoming_message(
            Message(Role.USER, f"person{i}@example.com")
        )
        messages.append(clean)
    clean, _, _ = await engine.project_incoming_message(messages[0])
    again, findings, _ = await engine.project_incoming_message(clean, index=1)
    assert again is not clean and len(findings) == 1
    assert "projection" not in str(_message_to_dict(again))
    changed = replace(clean, content="new@example.com")
    projected, _, _ = await engine.project_incoming_message(changed, index=1)
    assert "new@example.com" not in projected.content


async def test_capped_block_summary_keeps_entity_totals():
    from dendrux.loops.react import _blocked_data

    engine = GuardrailEngine([PII(action="block")])
    message = Message(Role.USER, " ".join(f"person{i}@example.com" for i in range(200)))
    _, findings, blocked = await engine.project_incoming_message(message, index=3)
    data = _blocked_data("incoming", blocked, findings, "message 3 (role=user)")
    assert data["counts"] == {"EMAIL_ADDRESS": 200}
