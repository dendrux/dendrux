"""Projection evidence stays outside the public Message dataclass contract."""

import copy
import dataclasses
import inspect
import pickle

import pytest

from dendrux.guardrails import PII
from dendrux.guardrails._engine import GuardrailEngine
from dendrux.types import Message, Role


@pytest.mark.parametrize("text", ["plain", "person@example.com"])
@pytest.mark.parametrize("action", ["warn", "block", "redact"])
async def test_projection_preserves_dataclass_contract_and_caller_object(text, action):
    message = Message(Role.USER, text)
    before = vars(message).copy()
    projected, _, _ = await GuardrailEngine([PII(action=action)]).project_incoming_message(message)
    assert vars(message) == before
    assert "_guardrail_projection" not in {f.name for f in dataclasses.fields(Message)}
    assert "_guardrail_projection" not in inspect.signature(Message).parameters
    assert "_guardrail_projection" not in dataclasses.asdict(projected)
    assert len(dataclasses.astuple(projected)) == len(dataclasses.fields(Message))
    assert "_guardrail_projection" not in repr(projected)
    assert projected == Message(Role.USER, projected.content)


@pytest.mark.parametrize(
    "clone", [copy.deepcopy, lambda message: pickle.loads(pickle.dumps(message))]
)
async def test_copied_projection_rescans(clone, monkeypatch):
    engine = GuardrailEngine([PII(action="warn")])
    message, _, _ = await engine.project_incoming_message(Message(Role.USER, "person@example.com"))
    copied = clone(message)
    scans = []
    original = engine.scan_incoming_message

    async def scan(*args, **kwargs):
        scans.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(engine, "scan_incoming_message", scan)
    await engine.project_incoming_message(copied)
    assert scans == [1]


async def test_projection_from_warn_engine_is_not_trusted_by_redact_engine():
    warn = GuardrailEngine([PII(action="warn")])
    redact = GuardrailEngine([PII(action="redact")])
    message, _, _ = await warn.project_incoming_message(Message(Role.USER, "person@example.com"))
    clean, findings, blocked = await redact.project_incoming_message(message)
    assert "person@example.com" not in clean.content
    assert len(findings) == 1 and blocked is None


@pytest.mark.parametrize("kinds", [100, 1000])
def test_entity_aggregate_is_bounded_without_losing_total(kinds):
    from dendrux.guardrails._engine import _ProjectionFindings
    from dendrux.guardrails._protocol import UNSCANNABLE, Finding

    findings = [Finding(f"KIND_{i}", 0, 1, 1.0, "") for i in range(kinds)]
    counts = _ProjectionFindings(findings).entity_counts
    assert sum(counts.values()) == kinds
    assert len(counts) <= 129
    if kinds == 100:
        assert len(counts) == 100 and UNSCANNABLE not in counts
    else:
        assert counts[UNSCANNABLE] == kinds - 128


async def test_changed_message_does_not_reuse_projection_evidence():
    engine = GuardrailEngine([PII(action="redact")])
    projected, _, _ = await engine.project_incoming_message(Message(Role.USER, "first@example.com"))
    changed = copy.copy(projected)
    object.__setattr__(changed, "content", "second@example.com")
    cleaned, findings, blocked = await engine.project_incoming_message(changed)
    assert "second@example.com" not in cleaned.content
    assert len(findings) == 1 and blocked is None
