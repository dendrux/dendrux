"""Fresh text projections share their verdict with the loop."""

from collections import Counter

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse, RunStatus


@pytest.mark.parametrize("single", [False, True])
async def test_ephemeral_entry_scans_unchanged_text_once(single):
    counts = Counter()

    class Guard:
        action = "block"

        async def scan(self, text):
            counts[text] += 1
            return []

    agent = Agent(
        prompt="system",
        provider=MockLLM([LLMResponse(text="answer")]),
        guardrails=[Guard()],
        loop=SingleCall() if single else None,
    )
    try:
        result = await agent.run("input")
        assert result.status == RunStatus.SUCCESS
        assert counts == {"system": 1, "input": 1, "answer": 1}
    finally:
        await agent.close()


@pytest.mark.parametrize("operation", ["update", "read"])
async def test_custom_block_mark_failure_has_safe_error(operation):
    from dendrux.errors import PersistenceError
    from dendrux.runtime._blocked_content import _blocked_evidence

    class Store:
        async def update_run_meta(self, run_id, meta):
            if operation == "update":
                raise ValueError("PRIVATE_MARK_ERROR")

        async def get_run(self, run_id):
            raise ValueError("PRIVATE_MARK_ERROR")

    with pytest.raises(PersistenceError) as caught:
        async with _blocked_evidence(Store(), "run"):
            pytest.fail("unsafe marker must stop evidence")
    assert "PRIVATE_MARK_ERROR" not in str(caught.value)
    assert caught.value.__context__ is None


def test_caller_metadata_cannot_certify_custom_store_entry():
    from dendrux.loops.react import ReActLoop
    from dendrux.runtime.runner import _new_run_metadata
    from dendrux.skills._session import SkillSession

    agent = Agent(prompt="system", provider=MockLLM([LLMResponse(text="answer")]))
    loop = ReActLoop()
    metadata, _ = _new_run_metadata(
        agent,
        loop,
        SkillSession.prepare(agent, loop=loop),
        {"dendrux.blocked": True, "keep": "link"},
        None,
        user_input="clean",
    )
    assert "dendrux.blocked" not in metadata
    assert metadata["keep"] == "link"


@pytest.mark.parametrize("parts", [False, True])
async def test_entry_reuses_allowed_redaction_before_provider(parts):
    from dendrux.guardrails import PII, GuardrailEngine
    from dendrux.runtime._entry_guardrails import _EntryProjection
    from dendrux.types import ImagePart, ImageRef, Message, Role, TextPart

    counts = Counter()
    pii = PII(action="redact")

    class Counting:
        action = "redact"

        async def scan(self, text):
            counts[text] += 1
            return await pii.scan(text)

    engine = GuardrailEngine([Counting()])
    entry = _EntryProjection(engine)
    image = ImagePart(ImageRef("0" * 64, "image/png", 1, 1, 1))
    raw = "person@example.com"
    text, _ = await entry.text_items(raw, (raw, image) if parts else (), Role.USER, 1)
    assert text == raw  # allowed request identity stays distinct
    message = Message(Role.USER, raw, parts=(TextPart(raw), image) if parts else None)
    projected, _, blocked = await engine.project_incoming_message(message, index=1)
    assert blocked is None and raw not in projected.content
    assert counts[raw] == 1
    assert projected.images == message.images


@pytest.mark.parametrize("hostile", [False, True])
async def test_redaction_handoff_cache_has_bounded_text_and_realistic_capacity(hostile):
    from dendrux.guardrails import GuardrailEngine
    from dendrux.guardrails._engine import _PROJECTION_TEXT_BYTES
    from dendrux.guardrails._protocol import Finding
    from dendrux.types import Message, Role

    calls = 0

    class Redact:
        action = "redact"

        async def scan(self, text):
            nonlocal calls
            calls += 1
            return [Finding("TEXT", 0, 1, 1.0, text[0])]

    engine = GuardrailEngine([Redact()])
    # Ten times the 32-Ki-character realistic harness; hostile input exceeds retention.
    count, length = (1, _PROJECTION_TEXT_BYTES + 1) if hostile else (100, 3277)
    messages = [Message(Role.USER, str(i) + "x" * length) for i in range(count)]
    for message in messages:
        await engine.project_incoming_message(message)
    assert engine._projection_text_bytes <= _PROJECTION_TEXT_BYTES
    before = calls
    for message in messages:
        projected, _, blocked = await engine.project_incoming_message(message)
        assert blocked is None and projected.content.startswith("<<TEXT_")
    assert calls == before + 1 if hostile else calls == before
    assert engine._projection_text_bytes <= _PROJECTION_TEXT_BYTES


@pytest.mark.parametrize("hostile", [False, True])
async def test_complete_allowed_findings_share_the_cache_budget(hostile):
    from dendrux.guardrails import GuardrailEngine
    from dendrux.guardrails._engine import _PROJECTION_TEXT_BYTES
    from dendrux.guardrails._protocol import Finding
    from dendrux.types import Message, Role

    count = 20000 if hostile else 200  # ten times a 20-finding ordinary input
    calls = 0

    class Warn:
        action = "warn"

        async def scan(self, text):
            nonlocal calls
            calls += 1
            return [Finding("TEXT", i, i + 1, 1.0, "x") for i in range(count)]

    engine = GuardrailEngine([Warn()])
    engine._retain_allowed_findings = True
    message = Message(Role.USER, "x" * count)
    for _ in range(2):
        _, findings, blocked = await engine.project_incoming_message(message)
        assert blocked is None and len(findings) == count
        assert all(not finding.text for finding in findings)
        assert engine._projection_text_bytes <= _PROJECTION_TEXT_BYTES
    assert calls == (2 if hostile else 1)


@pytest.mark.parametrize("single", [False, True])
async def test_legacy_loop_projection_keeps_full_allowed_audit_sample(single):
    from types import SimpleNamespace

    from dendrux.guardrails import PII, GuardrailEngine
    from dendrux.loops.react import _incoming_project
    from dendrux.loops.single import _scan_single_incoming
    from dendrux.types import Message, Role

    engine = GuardrailEngine([PII(action="redact")])
    message = Message(Role.USER, " ".join(f"person{i}@example.com" for i in range(200)))
    if single:
        findings, result = await _scan_single_incoming(
            engine, [message], recorder=None, notifier=None, run_id="r", warnings=[]
        )
        assert result is None
    else:
        _, findings, blocked, _ = await _incoming_project(
            SimpleNamespace(recorder=None), engine, message, 1
        )
        assert blocked is None
    assert len(findings) == 200
    assert all(not finding.text for finding in findings)


@pytest.mark.parametrize("params", [False, True])
async def test_clean_outgoing_projection_is_reused_before_recording(params):
    from dendrux.guardrails import GuardrailEngine
    from dendrux.loops._helpers import project_response
    from dendrux.types import Message, Role, ToolCall

    calls = Counter()

    class Allow:
        action = "block"

        async def scan(self, text):
            calls[text] += 1
            return []

    engine = GuardrailEngine([Allow()])
    response = LLMResponse(
        text="" if params else "done",
        tool_calls=[ToolCall("search", params={"query": "hello"})] if params else None,
    )
    _, _, blocked = await project_response(engine, response)
    assert blocked is None
    before = calls.copy()
    await engine.project_incoming_message(
        Message(Role.ASSISTANT, response.text or "", tool_calls=response.tool_calls)
    )
    assert calls == before
