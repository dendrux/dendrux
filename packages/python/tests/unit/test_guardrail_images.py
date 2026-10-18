"""Track D1: image guardrails (§5.11, S-03) and parts-aware text guardrails."""

from __future__ import annotations

from typing import Any, Literal

from dendrux.agent import Agent
from dendrux.guardrails import PII, Finding, GuardrailEngine, ImageGuardrail, ImageScanInput
from dendrux.loops._images import apply_delta
from dendrux.loops.react import ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import ImageViewState, LLMResponse, Message, Role, RunStatus, TextPart
from tests.unit.test_loop_images import (
    ImageLLM,
    RecordingRecorder,
    make_png,
    make_runtime,
    prep,
    store_image,
    tool_msg,
    user_msg,
)

RAW = make_png(seed=42)


class NsfwScan:
    """Image-only guardrail with a configurable verdict."""

    guardrail_id = "acme.NsfwScan"

    def __init__(
        self,
        action: Literal["redact", "block", "warn"] = "redact",
        *,
        flag: bool = True,
        version: str = "1",
    ) -> None:
        self.action = action
        self.flag = flag
        self.guardrail_version = version
        self.calls: list[ImageScanInput] = []

    async def scan_image(self, image: ImageScanInput) -> list[Finding]:
        self.calls.append(image)
        assert image.data == RAW
        return [Finding("NSFW", 0, 0, 0.9, "")] if self.flag else []


async def setup() -> tuple[Any, Any]:
    rt = make_runtime()
    part = await store_image(rt, RAW)
    return rt, part


def test_protocol_and_repr_hide_bytes() -> None:
    assert isinstance(NsfwScan(), ImageGuardrail)
    assert not isinstance(PII(), ImageGuardrail)
    scan = ImageScanInput(image_id="img_x", ref=None, origin="user", tool_name=None, data=RAW)  # type: ignore[arg-type]
    assert "PNG" not in repr(scan)


async def test_redact_verdict_placeholder_and_state_event() -> None:
    rt, part = await setup()
    guard = NsfwScan("redact")
    out = await prep([tool_msg("shot", "c1", part)], rt, guardrails=[guard])
    assert len(guard.calls) == 1 and guard.calls[0].origin == "tool"
    assert guard.calls[0].tool_name == "shot"
    assert out.wire[0].content == f'"ok"\n\n[image {part.image_id} removed by a guardrail]'
    key = ("t:acme", part.ref.sha256, "acme.NsfwScan", "1", "tool", "shot")
    assert out.delta.guardrail_verdicts[key].action == "redact"
    assert out.delta.guardrail_verdicts[key].entities == ("NSFW",)
    assert out.guardrail_findings == {
        "count": 1,
        "scanned": True,
        "verdicts": [
            {"image_id": part.image_id, "guardrail_id": "acme.NsfwScan", "action": "redact"}
        ],
    }
    types = [e.event_type for e in out.events]
    assert types == ["image.degraded", "image.state", "guardrail.detected"]
    state_evt = out.events[1].data
    assert state_evt["guardrail_verdicts"] == [
        {
            "sha256": part.ref.sha256,
            "guardrail_id": "acme.NsfwScan",
            "guardrail_version": "1",
            "origin": "tool",
            "tool_name": "shot",
            "action": "redact",
            "entities": ["NSFW"],
        }
    ]


async def test_verdict_reused_from_state_and_rescan_on_version_bump() -> None:
    rt, part = await setup()
    guard = NsfwScan("warn", flag=False)
    state = ImageViewState()
    out = await prep([user_msg(part)], rt, guardrails=[guard], state=state)
    apply_delta(state, out.delta)
    assert len(guard.calls) == 1
    out2 = await prep([user_msg(part)], rt, guardrails=[guard], state=state)
    assert len(guard.calls) == 1 and out2.delta.guardrail_verdicts == {}
    bumped = NsfwScan("warn", flag=False, version="2")
    await prep([user_msg(part)], rt, guardrails=[bumped], state=state)
    assert len(bumped.calls) == 1
    # Different origin = different key: same image as a tool result is scanned again.
    await prep([tool_msg("shot", "c", part)], rt, guardrails=[guard], state=state)
    assert len(guard.calls) == 2


async def test_warn_emits_detected_and_keeps_image() -> None:
    rt, part = await setup()
    out = await prep([user_msg(part)], rt, guardrails=[NsfwScan("warn")])
    assert out.slots[0].visible
    (detected,) = [e.data for e in out.events if e.event_type == "guardrail.detected"]
    assert detected == {"direction": "incoming_image", "findings_count": 1, "entities": ["NSFW"]}


async def test_block_verdict_ends_run_with_error() -> None:
    rt, part = await setup()
    llm = ImageLLM([LLMResponse(text="never")])
    rec = RecordingRecorder()
    agent = Agent(provider=llm, prompt="p", guardrails=[NsfwScan("block")])  # type: ignore[list-item]
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="x",
        initial_history=[user_msg("see", part)],
        recorder=rec,
        image_runtime=rt,
    )
    assert result.status == RunStatus.ERROR and "acme.NsfwScan" in (result.error or "")
    assert llm.calls == []
    (blocked,) = rec.of("guardrail.blocked")
    assert blocked["direction"] == "incoming_image"


async def test_unscanned_record_once_per_run() -> None:
    rt, part = await setup()
    llm = ImageLLM([LLMResponse(text="a")])
    rec = RecordingRecorder()
    agent = Agent(provider=llm, prompt="p", guardrails=[PII()])
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="x",
        initial_history=[user_msg("see", part)],
        recorder=rec,
        image_runtime=rt,
    )
    assert result.status == RunStatus.SUCCESS
    assert rec.of("guardrail.image_unscanned") == [{"count": 1}]
    assert rec.completed[0]["guardrail_findings"]["images"] == {"count": 1, "scanned": False}
    state = ImageViewState(warned=["guardrail.image_unscanned"])
    out = await prep([user_msg(part)], rt, guardrails=[PII()], state=state)
    assert out.guardrail_findings == {"count": 1, "scanned": False}
    assert not [e for e in out.events if e.event_type == "guardrail.image_unscanned"]


async def test_text_guardrail_redacts_text_parts_keeps_images() -> None:
    rt, part = await setup()
    engine = GuardrailEngine([PII(action="redact")])
    msg = Message.with_parts(Role.USER, [TextPart("mail bob@example.com"), part])
    new, findings, block = await engine.scan_incoming_message(msg)
    assert block is None and findings
    assert new is not msg and new.images == (part,)
    assert "bob@example.com" not in new.content and new.content == new.parts[0].text  # type: ignore[index, union-attr]
    plain = Message(role=Role.USER, content="hello")
    same, _, _ = await engine.scan_incoming_message(plain)
    assert same is plain


async def test_image_only_guardrail_skipped_by_text_scans() -> None:
    engine = GuardrailEngine([NsfwScan()])  # type: ignore[list-item]
    text, findings, block = await engine.scan_incoming("anything")
    assert (text, findings, block) == ("anything", [], None)
    assert await engine.scan_outgoing("x", [{"a": "b"}]) == ([], None)


async def test_loop_parts_aware_redaction_reaches_wire() -> None:
    rt, part = await setup()
    llm = ImageLLM([LLMResponse(text="ok")])
    agent = Agent(provider=llm, prompt="p", guardrails=[PII(action="redact")])
    await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="x",
        initial_history=[user_msg("mail bob@example.com", part)],
        image_runtime=rt,
    )
    wire_user = llm.calls[0][1]
    assert "bob@example.com" not in wire_user.content and wire_user.images[0].data == RAW


async def test_object_without_scan_methods_keeps_v030_failure() -> None:
    """P2-2: a misnamed guardrail still fails loudly; only image-only guardrails are skipped."""
    import pytest

    class Misnamed:
        action = "redact"

        async def scna(self, text: str) -> list[Finding]:
            return []

    engine = GuardrailEngine([Misnamed()])  # type: ignore[list-item]
    with pytest.raises(AttributeError):
        await engine.scan_incoming("secret")
