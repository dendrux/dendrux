"""Rejected message projections contain only disposable, one-way replacements."""

import asyncio
import copy
from dataclasses import asdict, replace

import pytest

from dendrux.guardrails import PII, GuardrailEngine, SecretDetection
from dendrux.guardrails._protocol import Finding
from dendrux.types import ImagePart, ImageRef, Message, Role, TextPart, ToolCall, _message_to_dict

SECRET = "AKIAIOSFODNN7EXAMPLE"
MAIL = "person@example.com"
IMAGE = ImagePart(ImageRef("0" * 64, "image/png", 1, 1, 1))


@pytest.mark.parametrize("parts", [False, True])
@pytest.mark.parametrize("role", [Role.USER, Role.TOOL, Role.SYSTEM, Role.ASSISTANT])
async def test_061_block_projection_removes_every_copy(parts, role):
    if parts and role not in (Role.USER, Role.TOOL):
        pytest.skip("only user/tool messages support image parts")
    engine = GuardrailEngine([SecretDetection(action="block")])
    kwargs = {"name": "lookup", "call_id": "call"} if role == Role.TOOL else {}
    text = SECRET + "\n\n" + SECRET if parts else SECRET
    original = Message(
        role,
        text,
        **kwargs,
        parts=(TextPart(SECRET), IMAGE, TextPart(SECRET)) if parts else None,
        meta={"wire_content": SECRET, "stable_context": [SECRET], "nested": {SECRET: SECRET}},
        source=SECRET,
        reasoning=SECRET,
        reasoning_blocks=[{"text": SECRET}],
        tool_calls=[ToolCall("lookup", {"value": SECRET}, raw_arguments=SECRET, parse_error=SECRET)]
        if role == Role.ASSISTANT
        else None,
    )
    before = asdict(original)
    projected, findings, blocked = await engine.project_incoming_message(original)
    assert blocked is not None
    assert SECRET not in repr(asdict(projected))
    assert SECRET not in repr(_message_to_dict(projected))
    assert all(not f.text and f.start == f.end == 0 for f in findings)
    assert engine.get_pii_mapping() == {}
    assert projected.content.startswith("[blocked:")
    assert projected.images == original.images
    assert asdict(original) == before


async def test_rejected_projection_discards_new_mapping_and_origin_changes():
    engine = GuardrailEngine([PII(action="redact", restore="allowlisted"), SecretDetection()])
    await engine.project_incoming_message(Message(Role.TOOL, MAIL, name="lookup", call_id="old"))
    before = engine.get_pii_mapping()
    original = Message(Role.USER, f"{MAIL} new@example.com {SECRET}")
    _, findings, blocked = await engine.project_incoming_message(original)
    assert blocked and findings
    assert engine.get_pii_mapping() == before
    assert "new@example.com" not in repr(vars(engine))


class CounterGuard:
    action = "block"

    def __init__(self):
        self.calls = 0

    async def scan(self, text):
        self.calls += 1
        return [Finding("SECRET", 0, len(text), 1.0, text)] if SECRET in text else []


async def test_trusted_block_handoff_scans_once_but_placeholder_spelling_is_not_provenance():
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    projected, _, first = await engine.project_incoming_message(Message(Role.USER, SECRET))
    again, _, second = await engine.project_incoming_message(projected, index=2)
    assert first and second and guard.calls == 1
    assert again.content == projected.content
    fresh = Message(Role.USER, projected.content)
    _, _, blocked = await engine.project_incoming_message(fresh)
    assert blocked is None and guard.calls == 2


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_projection_failure_rolls_back_mapping_without_exception_text(failure):
    class Failing:
        action = "block"

        async def scan(self, text):
            raise failure() if failure is asyncio.CancelledError else failure(SECRET)

    engine = GuardrailEngine([PII(action="redact"), Failing()])
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await engine.project_incoming_message(Message(Role.USER, MAIL))
    else:
        projected, findings, blocked = await engine.project_incoming_message(
            Message(Role.USER, MAIL)
        )
        assert blocked and "UNSCANNABLE" in blocked
        assert SECRET not in repr((projected, findings, blocked))
        assert MAIL not in projected.content
    assert engine.get_pii_mapping() == {}


async def test_mutated_duplicate_fields_are_removed_on_trusted_handoff():
    engine = GuardrailEngine([SecretDetection()])
    projected, _, _ = await engine.project_incoming_message(Message(Role.USER, SECRET))
    projected.meta["wire_content"] = SECRET
    again, _, blocked = await engine.project_incoming_message(projected)
    assert blocked and SECRET not in repr(asdict(again))


@pytest.mark.parametrize("clone", [copy.copy, lambda msg: replace(msg, content=SECRET + " ")])
async def test_changed_content_cannot_reuse_clean_verdict(clone):
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    original, _, _ = await engine.project_incoming_message(Message(Role.USER, "clean"))
    changed = clone(original)
    object.__setattr__(changed, "content", SECRET + " ")
    projected, _, blocked = await engine.project_incoming_message(changed)
    assert blocked and SECRET not in projected.content and guard.calls == 2


async def test_incoming_assistant_params_cross_the_same_block_boundary():
    engine = GuardrailEngine([SecretDetection()])
    original = Message(Role.ASSISTANT, "clean", tool_calls=[ToolCall("lookup", {"key": SECRET})])
    projected, findings, blocked = await engine.project_incoming_message(original)
    assert blocked and findings
    assert SECRET not in repr(asdict(projected))


@pytest.mark.parametrize("outgoing", [False, True])
@pytest.mark.parametrize("action", ["block", "redact", "warn"])
async def test_projection_scan_limits_keep_existing_action_policy(monkeypatch, outgoing, action):
    from dendrux.guardrails import _engine

    monkeypatch.setattr(_engine, "MAX_SCAN_CHARS", 32)
    engine = GuardrailEngine([SecretDetection(action=action)])
    message = Message(Role.USER, SECRET * 3)
    project = engine._project_outgoing_message if outgoing else engine.project_incoming_message
    clean, findings, blocked = await project(message)
    assert findings
    if action == "warn":
        assert blocked is None and clean == message
    else:
        assert blocked and "UNSCANNABLE" in blocked
        assert SECRET not in clean.content and engine.get_pii_mapping() == {}


@pytest.mark.parametrize("params", [False, True])
async def test_outgoing_projection_removes_text_and_params_together(params):
    engine = GuardrailEngine([SecretDetection()])
    message = Message(
        Role.ASSISTANT,
        "clean" if params else SECRET,
        tool_calls=[ToolCall("lookup", {"key": SECRET if params else "clean"})],
        meta={"duplicate": SECRET},
        reasoning=SECRET,
    )
    clean, findings, blocked = await engine._project_outgoing_message(message)
    assert blocked and findings
    assert SECRET not in repr(asdict(clean))
    assert engine._projected_block(clean) == blocked
    assert engine._projected_block(message) is None
    assert engine._projected_block(replace(clean, content="changed")) is None
    assert GuardrailEngine([SecretDetection()])._projected_block(clean) is None
    _, _, repeated = await engine.project_incoming_message(clean)
    assert repeated == blocked


async def test_outgoing_unblocked_redaction_retains_detection_only_contract():
    engine = GuardrailEngine([PII(action="redact")])
    message = Message(Role.ASSISTANT, MAIL)
    clean, findings, blocked = await engine._project_outgoing_message(message)
    assert clean is message and findings and blocked is None
    assert engine.get_pii_mapping() == {}
    incoming, _, blocked = await engine.project_incoming_message(clean)
    assert blocked is None and MAIL not in incoming.content
    assert MAIL in engine.get_pii_mapping().values()


async def test_mapping_capacity_rejection_rolls_back_earlier_guardrail(monkeypatch):
    from dendrux.guardrails import _engine

    monkeypatch.setattr(_engine, "MAX_MAPPING_ENTRIES", 1)
    engine = GuardrailEngine([PII(action="redact"), SecretDetection(action="redact")])
    clean, _, blocked = await engine.project_incoming_message(
        Message(Role.USER, MAIL + " " + SECRET)
    )
    assert blocked and SECRET not in clean.content and MAIL not in clean.content
    assert engine.get_pii_mapping() == {} and engine._counters == {}


async def test_large_projection_and_trusted_tool_handoff_scan_once():
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    # Ten times a 100 KiB tool result, below the established scan cap.
    message = Message(Role.TOOL, "x" * (100 * 1024 * 10), name="lookup", call_id="call")
    clean, _, blocked = await engine.project_incoming_message(message)
    again, _, repeated = await engine.project_incoming_message(clean)
    assert clean is again is message and blocked is repeated is None and guard.calls == 1


async def test_block_evidence_survives_cache_eviction_without_retaining_raw_text():
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    blocked, _, _ = await engine.project_incoming_message(Message(Role.USER, SECRET))
    for i in range(300):
        await engine.project_incoming_message(Message(Role.USER, str(i)))
    count = guard.calls
    again, findings, reason = await engine.project_incoming_message(blocked, index=7)
    assert reason and findings and guard.calls == count and again.content == blocked.content
    assert len(engine._projected) <= 256 and SECRET not in repr(vars(engine))


async def test_cancelled_scan_does_not_rollback_another_projection():
    started, release = asyncio.Event(), asyncio.Event()

    class WaitOnce:
        action = "block"
        first = True

        async def scan(self, text):
            if self.first:
                self.first = False
                started.set()
                await release.wait()
            return []

    engine = GuardrailEngine([PII(action="redact"), WaitOnce()])
    first = asyncio.create_task(engine.project_incoming_message(Message(Role.USER, MAIL)))
    await started.wait()
    second = asyncio.create_task(
        engine.project_incoming_message(Message(Role.USER, "other@example.com"))
    )
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    result, _, blocked = await second
    assert blocked is None and result.content == "<<EMAIL_ADDRESS_1>>"
    assert engine.get_pii_mapping() == {"<<EMAIL_ADDRESS_1>>": "other@example.com"}


async def test_mutated_tool_params_cannot_reuse_clean_text_verdict():
    engine = GuardrailEngine([SecretDetection()])
    message = Message(Role.ASSISTANT, "clean", tool_calls=[ToolCall("lookup", {"key": "clean"})])
    clean, _, blocked = await engine.project_incoming_message(message)
    assert blocked is None
    clean.tool_calls[0].params["key"] = SECRET
    projected, _, blocked = await engine.project_incoming_message(clean)
    assert blocked and SECRET not in repr(asdict(projected))


async def test_low_level_message_boundary_retains_trusted_block_verdict():
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    projected, _, _ = await engine.project_incoming_message(Message(Role.USER, SECRET))
    clean, findings, blocked = await engine.scan_incoming_message(projected, index=3)
    assert blocked and findings and clean == projected and guard.calls == 1


async def test_fencing_propagates_without_committing_mapping_changes():
    from dendrux.errors import RunFencedError

    fenced = RunFencedError("run", claim_epoch=1)

    class Fenced:
        action = "block"

        async def scan(self, text):
            raise fenced

    engine = GuardrailEngine([PII(action="redact"), Fenced()])
    with pytest.raises(RunFencedError) as caught:
        await engine.project_incoming_message(Message(Role.USER, MAIL))
    assert caught.value is fenced and engine.get_pii_mapping() == {}


async def test_tool_observation_handoff_preserves_block_without_reintroducing_metadata():
    from dendrux.loops._helpers import project_message

    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    raw = Message(Role.TOOL, SECRET, name="lookup", call_id="call", meta={"wire_content": SECRET})
    projected, _, _ = await engine.project_incoming_message(raw)

    class Recorder:
        guardrail_engine = engine

        def take_observation(self, call_id):
            assert call_id == "call"
            return projected

    handed = await project_message(Recorder(), raw)
    assert engine._projected_block(handed)
    assert SECRET not in repr(asdict(handed)) and guard.calls == 1


async def test_parameter_findings_do_not_cross_into_another_message():
    guard = CounterGuard()
    guard.action = "warn"
    engine = GuardrailEngine([guard])
    message = Message(Role.ASSISTANT, "clean", tool_calls=[ToolCall("lookup", {"key": SECRET})])
    _, findings, blocked = await engine.project_incoming_message(message)
    assert findings and blocked is None
    _, findings, blocked = await engine.project_incoming_message(Message(Role.ASSISTANT, "clean"))
    assert findings == [] and blocked is None


async def test_unchanged_tool_parameters_reuse_the_trusted_scan():
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    message = Message(Role.ASSISTANT, "clean", tool_calls=[ToolCall("lookup", {"key": "clean"})])
    projected, _, blocked = await engine.project_incoming_message(message)
    count = guard.calls
    _, _, repeated = await engine.project_incoming_message(projected)
    assert blocked is repeated is None and guard.calls == count


@pytest.mark.parametrize("fenced", [False, True])
async def test_parameter_key_failure_is_safe_and_preserves_fencing(fenced):
    from dendrux.errors import RunFencedError

    error = RunFencedError("run", claim_epoch=1) if fenced else ValueError(SECRET)

    class Broken(dict):
        def items(self):
            raise error

    engine = GuardrailEngine([SecretDetection()])
    message = Message(Role.ASSISTANT, "clean", tool_calls=[ToolCall("lookup", Broken())])
    if fenced:
        with pytest.raises(RunFencedError) as caught:
            await engine.project_incoming_message(message)
        assert caught.value is error
    else:
        projected, findings, blocked = await engine.project_incoming_message(message)
        assert findings and blocked and "UNSCANNABLE" in blocked
        assert SECRET not in repr((projected, findings, blocked))
    assert engine.get_pii_mapping() == {}


@pytest.mark.parametrize("cyclic", [False, True])
async def test_parameter_key_uses_bounded_scan_input(cyclic):
    guard = CounterGuard()
    engine = GuardrailEngine([guard])
    params = {"text": "x" * (100 * 1024 * 10)}
    if cyclic:
        params["cycle"] = params
    message = Message(Role.ASSISTANT, "clean", tool_calls=[ToolCall("lookup", params)])
    projected, _, blocked = await engine.project_incoming_message(message)
    if cyclic:
        assert blocked and "UNSCANNABLE" in blocked
        assert projected.tool_calls[0].params != params
    else:
        assert blocked is None
        before = guard.calls
        await engine.project_incoming_message(projected)
        assert guard.calls == before
