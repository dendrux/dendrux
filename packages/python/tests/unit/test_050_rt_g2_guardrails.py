"""Guardrails scan the text parts of one message as the single text they project to."""

from __future__ import annotations

from dendrux.guardrails import PII, GuardrailEngine, SecretDetection
from dendrux.types import ImagePart, ImageRef, Message, Role, TextPart

_TOKEN = "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5zA7b"
_EMAIL = "alice@example.com"
_IMAGE = ImagePart(ImageRef("0" * 64, "image/png", 1, 1, 1))


def _msg(role: Role, *parts: object, **kw: object) -> Message:
    return Message.with_parts(role, tuple(parts), **kw)  # type: ignore[arg-type]


class TestContextAcrossTextParts:
    async def test_secret_split_across_text_parts_is_blocked_like_the_joined_text(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        msg = _msg(Role.USER, TextPart("aws secret:"), TextPart(_TOKEN), _IMAGE)
        _, joined_findings, joined_block = await engine.scan_incoming(msg.content)
        assert joined_block is not None
        _, findings, block = await engine.scan_incoming_message(msg)
        assert block is not None
        assert len(findings) == len(joined_findings)
        assert _TOKEN not in block

    async def test_tool_result_parts_are_scanned_the_same_way(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        msg = _msg(
            Role.TOOL,
            TextPart("here is the aws secret key"),
            TextPart(_TOKEN),
            _IMAGE,
            name="lookup",
            call_id="c1",
        )
        _, _, block = await engine.scan_incoming_message(msg, index=3)
        assert block is not None
        assert "message 3" in block
        assert "tool_call_id=c1" in block

    async def test_text_parts_separated_by_an_image_still_read_as_one_text(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        msg = _msg(Role.USER, TextPart("aws secret:"), _IMAGE, TextPart(_TOKEN))
        _, _, block = await engine.scan_incoming_message(msg)
        assert block is not None

    async def test_redact_replaces_the_token_in_its_own_part(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="redact")])
        msg = _msg(Role.USER, TextPart("aws secret:"), TextPart(_TOKEN), _IMAGE)
        out, findings, block = await engine.scan_incoming_message(msg)
        assert block is None
        assert findings
        assert out.parts is not None
        assert out.parts[0] == TextPart("aws secret:")
        assert isinstance(out.parts[1], TextPart)
        assert out.parts[1].text.startswith("<<AWS_SECRET_KEY_")
        assert out.parts[2] == _IMAGE
        assert _TOKEN not in out.content
        assert _TOKEN in engine.get_pii_mapping().values()

    async def test_match_spanning_two_parts_leaves_none_of_its_text(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="redact")])
        msg = _msg(Role.USER, TextPart("my api_key:"), TextPart(_TOKEN), _IMAGE)
        out, _, block = await engine.scan_incoming_message(msg)
        assert block is None
        assert out.parts is not None
        assert _TOKEN not in out.content
        assert all(_TOKEN not in p.text for p in out.parts if isinstance(p, TextPart))
        assert out.content == "\n\n".join(p.text for p in out.parts if isinstance(p, TextPart))
        # Every redacted piece restores to exactly the text it replaced.
        restored = out.content
        for placeholder, original in engine.get_pii_mapping().items():
            restored = restored.replace(placeholder, original)
        assert _TOKEN in restored

    async def test_warn_reports_without_changing_parts(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="warn")])
        msg = _msg(Role.USER, TextPart("aws secret:"), TextPart(_TOKEN), _IMAGE)
        out, findings, block = await engine.scan_incoming_message(msg)
        assert block is None
        assert findings
        assert out is msg


class TestRedactionOffsetsUnchanged:
    async def test_each_part_keeps_its_own_findings_and_numbering(self) -> None:
        engine = GuardrailEngine([PII(action="redact")])
        msg = _msg(
            Role.USER,
            TextPart(f"first {_EMAIL} then"),
            _IMAGE,
            TextPart(f"bob@example.org and {_EMAIL}"),
        )
        out, findings, block = await engine.scan_incoming_message(msg)
        assert block is None
        assert len(findings) == 3
        assert out.parts is not None
        assert isinstance(out.parts[0], TextPart)
        assert isinstance(out.parts[2], TextPart)
        assert out.parts[0].text == "first <<EMAIL_ADDRESS_1>> then"
        assert out.parts[2].text == "<<EMAIL_ADDRESS_2>> and <<EMAIL_ADDRESS_1>>"
        assert out.parts[1] == _IMAGE

    async def test_single_text_part_matches_the_text_only_scan(self) -> None:
        text = f"mail {_EMAIL}, and {_EMAIL}"
        engine_a = GuardrailEngine([PII(action="redact")])
        engine_b = GuardrailEngine([PII(action="redact")])
        plain, _, _ = await engine_a.scan_incoming(text)
        out, _, _ = await engine_b.scan_incoming_message(_msg(Role.USER, TextPart(text), _IMAGE))
        assert out.parts is not None
        assert isinstance(out.parts[0], TextPart)
        assert out.parts[0].text == plain

    async def test_second_guardrail_sees_the_first_guardrails_redaction(self) -> None:
        engine = GuardrailEngine([PII(action="redact"), SecretDetection(action="block")])
        msg = _msg(Role.USER, TextPart(f"mail {_EMAIL}"), TextPart("fine"), _IMAGE)
        out, _, block = await engine.scan_incoming_message(msg)
        assert block is None
        assert _EMAIL not in out.content

    async def test_joined_text_over_the_scan_limit_fails_closed(self) -> None:
        from dendrux.guardrails import _engine as engine_mod

        limit = engine_mod.MAX_SCAN_CHARS
        engine = GuardrailEngine([SecretDetection(action="block")])
        half = "x " * (limit // 4)
        msg = _msg(Role.USER, TextPart(half), TextPart(half), TextPart("y" * 8), _IMAGE)
        assert all(len(p.text) < limit for p in msg.parts or () if isinstance(p, TextPart))
        _, _, block = await engine.scan_incoming_message(msg)
        assert block is not None
        assert "scan limit" in block


class TestContextAcrossToolCallParams:
    async def test_secret_whose_context_is_a_sibling_param_is_blocked(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        params = {"label": "aws secret:", "value": _TOKEN}
        _, joined_block = await engine.scan_outgoing(f"{params['label']} {_TOKEN}")
        assert joined_block is not None
        _, block = await engine.scan_outgoing("ok", [params])
        assert block is not None
        assert "tool call params (call 0)" in block
        assert _TOKEN not in block

    async def test_nested_sibling_context_is_blocked(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        params = {"a": {"note": "my aws secret"}, "b": [{"v": _TOKEN}]}
        _, block = await engine.scan_outgoing("ok", [params])
        assert block is not None

    async def test_plain_params_still_pass(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block"), PII(action="block")])
        findings, block = await engine.scan_outgoing(
            "ok", [{"city": "Oslo", "n": 3, "sha": "a" * 40}]
        )
        assert block is None
        assert findings == []

    async def test_findings_are_not_double_counted(self) -> None:
        engine = GuardrailEngine([PII(action="warn")])
        findings, block = await engine.scan_outgoing("ok", [{"a": _EMAIL, "b": "x"}])
        assert block is None
        assert len(findings) == 1
