"""request view (prepare_request), loop commit protocol, tool images, rejection.

The module also hosts the in-test fakes shared by the other D1 test files:
an in-memory ``BlobCatalog`` + ``BlobResolver``, a PNG generator, an
image-capable scripted provider (batch + stream) and a recording recorder.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import zlib
from dataclasses import replace
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import (
    BlobIntegrityError,
    BlobNotFoundError,
    ImageLimitError,
    ImageRejectedError,
    UnsupportedModalityError,
)
from dendrux.llm.base import LLMProvider
from dendrux.loops._images import (
    apply_delta,
    learn_image_unsupported,
    pause_image_view,
    prepare_request,
)
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop, _execute_tool
from dendrux.loops.single import SingleCall
from dendrux.media import (
    BlobInfo,
    Image,
    ImageDropPolicy,
    ImagePolicy,
    ImageRuntime,
    ToolOutput,
    ingest,
    inspect_image,
)
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.tools._lookups import build_tool_lookups
from dendrux.types import (
    ImagePart,
    ImageViewState,
    LLMResponse,
    Message,
    ProviderCapabilities,
    Role,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    TextPart,
    ToolCall,
    ToolDecision,
    ToolResult,
    ToolTarget,
)

# ---------------------------------------------------------------------------
# Fakes (shared with test_drop_policy / test_guardrail_images / test_mcp_images)
# ---------------------------------------------------------------------------


def make_png(width: int = 4, height: int = 2, seed: int = 0) -> bytes:
    """A valid RGB PNG; ``seed`` changes the pixels (and so the sha256)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    pixel = bytes([seed % 256, (seed // 256) % 256, 7])
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class FakeCatalog:
    """Dict-backed ``BlobCatalog`` (DB storage only) — enough for ingest + prepare."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], BlobInfo] = {}
        self.data: dict[tuple[str, str], bytes] = {}
        self.get_current_calls: list[tuple[str, tuple[str, ...]]] = []
        self.writes = 0

    async def get_current(self, scope: str, sha256s: Any) -> dict[str, BlobInfo]:
        self.get_current_calls.append((scope, tuple(sha256s)))
        return {
            sha: self.rows[(scope, sha)]
            for sha in sha256s
            if (scope, sha) in self.rows
            and self.rows[(scope, sha)].state in ("live", "purging", "purged")
        }

    async def touch_live(self, scope: str, sha256s: Any) -> dict[str, BlobInfo]:
        return {
            sha: self.rows[(scope, sha)]
            for sha in sha256s
            if (scope, sha) in self.rows and self.rows[(scope, sha)].state == "live"
        }

    async def get_row(self, scope: str, sha256: str) -> BlobInfo | None:
        return self.rows.get((scope, sha256))

    async def insert_row(
        self,
        scope: str,
        sha256: str,
        info: BlobInfo,
        *,
        state: str,
        generation: str,
        data: bytes | None,
    ) -> bool:
        if (scope, sha256) in self.rows:
            return False
        self.writes += 1
        self.rows[(scope, sha256)] = replace(info, state=state, generation=generation)  # type: ignore[arg-type]
        if data is not None:
            self.data[(scope, sha256)] = data
        return True

    async def promote(self, scope: str, sha256: str, generation: str) -> bool:  # pragma: no cover
        return False

    async def adopt_pending(self, *a: Any) -> bool:  # pragma: no cover
        return False

    async def revive(self, *a: Any, **k: Any) -> bool:  # pragma: no cover
        return False

    async def mark_deleting(self, *a: Any, **k: Any) -> str | None:  # pragma: no cover
        return None

    async def delete_row(self, *a: Any) -> bool:  # pragma: no cover
        return False

    async def mark_purging(self, *a: Any) -> str | None:  # pragma: no cover
        return None

    async def finish_purge(self, *a: Any) -> None:  # pragma: no cover
        return None

    async def stuck_rows(self, *a: Any) -> list[BlobInfo]:  # pragma: no cover
        return []

    async def read_blob_data(self, key: Any) -> bytes | None:  # pragma: no cover
        return self.data.get((key.tenant_scope, key.sha256))

    async def add_blob_refs(self, *a: Any) -> None:  # pragma: no cover
        return None

    async def run_references(self, *a: Any) -> bool:  # pragma: no cover
        return False

    async def blob_references(self, *a: Any) -> list[Any]:  # pragma: no cover
        return []

    async def delete_orphan_edges(self, limit: int) -> int:  # pragma: no cover
        return 0

    # --- test helpers ---
    def purge(self, scope: str, sha256: str) -> None:
        self.rows[(scope, sha256)] = replace(self.rows[(scope, sha256)], state="purged")
        self.data.pop((scope, sha256), None)


class FakeResolver:
    """``BlobResolver`` over :class:`FakeCatalog`; records ``(scope, sha)`` reads."""

    def __init__(self, catalog: FakeCatalog) -> None:
        self.catalog = catalog
        self.reads: list[tuple[str, str]] = []

    async def read(self, scope: str, info: BlobInfo) -> bytes:
        self.reads.append((scope, info.sha256))
        data = self.catalog.data.get((scope, info.sha256))
        if data is None:
            raise BlobNotFoundError("img_" + info.sha256[:12])
        return data


def make_runtime(
    policy: ImagePolicy | None = None,
    *,
    scope: str = "t:acme",
    catalog: FakeCatalog | None = None,
) -> ImageRuntime:
    cat = catalog or FakeCatalog()
    return ImageRuntime(
        policy=policy or ImagePolicy(),
        catalog=cat,  # type: ignore[arg-type]
        blob_store=None,
        resolver=FakeResolver(cat),
        tenant_scope=scope,
        run_id="run-1",
    )


async def store_image(runtime: ImageRuntime, data: bytes, *, detail: Any = None) -> ImagePart:
    (part,) = await ingest(
        [Image.from_bytes(data, detail=detail)],
        policy=runtime.policy,
        runtime=runtime,
        origin="user",
    )
    return part


class StreamScript:
    """A scripted stream: emit text deltas, then raise ``error`` or finish with ``response``."""

    def __init__(
        self,
        deltas: list[str] | None = None,
        *,
        error: BaseException | None = None,
        response: LLMResponse | None = None,
    ) -> None:
        self.deltas = deltas or []
        self.error = error
        self.response = response


class _ScriptedStream:
    def __init__(self, owner: ImageLLM, item: Any) -> None:
        self.owner = owner
        self.closed = False
        self.events: list[StreamEvent] = []
        self.error: BaseException | None = None
        if isinstance(item, BaseException):
            self.error = item
        elif isinstance(item, StreamScript):
            self.events = [
                StreamEvent(type=StreamEventType.TEXT_DELTA, text=d) for d in item.deltas
            ]
            self.error = item.error
            if item.response is not None:
                self.events.append(StreamEvent(type=StreamEventType.DONE, raw=item.response))
        else:
            if item.text:
                self.events.append(StreamEvent(type=StreamEventType.TEXT_DELTA, text=item.text))
            for tc in item.tool_calls or []:
                self.events.append(
                    StreamEvent(type=StreamEventType.TOOL_USE_END, tool_call=tc, tool_name=tc.name)
                )
            self.events.append(StreamEvent(type=StreamEventType.DONE, raw=item))

    def __aiter__(self) -> _ScriptedStream:
        return self

    async def __anext__(self) -> StreamEvent:
        if self.events:
            await asyncio.sleep(0)
            return self.events.pop(0)
        if self.error is not None:
            err, self.error = self.error, None
            raise err
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.owner.log.append("aclose")
        self.owner.aclosed += 1


class ImageLLM(LLMProvider):
    """Scripted provider that accepts image parts and records the wire messages."""

    capabilities = ProviderCapabilities(
        supports_native_tools=True, supports_tool_call_ids=True, supports_multimodal=True
    )

    def __init__(
        self,
        script: list[Any],
        *,
        support: str = "yes",
        rule: tuple[int, int] | None = None,
        thinking: bool = False,
        model: str = "claude-test",
        base_url: str = "https://api.test",
    ) -> None:
        self.script = list(script)
        self.support = support
        self.rule = rule
        self._thinking = thinking
        self._model = model
        self.base_url = base_url
        self.calls: list[list[Message]] = []
        self.support_calls = 0
        self.aclosed = 0
        self.log: list[str] = []
        self.on_open: Any = None
        self.keep_calls = True

    @property
    def model(self) -> str:
        return self._model

    @property
    def thinking_enabled(self) -> bool:
        return self._thinking

    async def image_input_support(self, model: str) -> str:
        self.support_calls += 1
        return self.support

    def many_image_rule(self, model: str) -> tuple[int, int] | None:
        return self.rule

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        return ("anthropic", self.base_url, model)

    def _next(self, messages: list[Message]) -> Any:
        if self.keep_calls:
            self.calls.append(list(messages))
        return self.script.pop(0)

    async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> LLMResponse:
        item = self._next(messages)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, LLMResponse)
        return item

    def complete_stream(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:  # type: ignore[override]
        self.log.append("open")
        if self.on_open is not None:
            self.on_open()
        return _ScriptedStream(self, self._next(messages))


class RecordingRecorder(BaseRecorder):
    """Records governance events, LLM records and failures."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.started: list[list[Message] | None] = []
        self.completed: list[dict[str, Any]] = []
        self.failed: list[BaseException] = []
        self.messages: list[Message] = []

    async def on_governance_event(
        self, run_id: str, event_type: str, iteration: int, data: dict[str, Any], **kw: Any
    ) -> None:
        self.events.append((event_type, data))

    async def on_llm_call_started(self, run_id: str, iteration: int, **kw: Any) -> None:
        self.started.append(kw.get("semantic_messages"))

    async def on_llm_call_completed(
        self, run_id: str, response: LLMResponse, iteration: int, **kw: Any
    ) -> None:
        self.completed.append(kw)

    async def on_llm_call_failed(
        self, run_id: str, iteration: int, error: BaseException, **kw: Any
    ) -> None:
        self.failed.append(error)

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        self.messages.append(message)

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [d for t, d in self.events if t == event_type]


class Control:
    def __init__(self, cancel: bool = False, pause: bool = False) -> None:
        self.cancel_requested = cancel
        self.pause_requested = pause
        self.has_pending_notes = False


class ControlStore:
    """Minimal state store for the loop's run-control reads."""

    def __init__(self) -> None:
        self.reads = 0
        self.cancel_after: int | None = None
        self.pause_after: int | None = None

    async def get_run_control(self, run_id: str) -> Control:
        self.reads += 1
        cancel = self.cancel_after is not None and self.reads > self.cancel_after
        pause = self.pause_after is not None and self.reads > self.pause_after
        return Control(cancel=cancel, pause=pause)


def user_msg(*parts: Any) -> Message:
    built = [TextPart(p) if isinstance(p, str) else p for p in parts]
    return Message.with_parts(Role.USER, built)


def tool_msg(name: str, call_id: str, *images: ImagePart, payload: str = '"ok"') -> Message:
    return Message.with_parts(Role.TOOL, [TextPart(payload), *images], name=name, call_id=call_id)


def assistant(text: str = "", call_ids: list[str] | None = None) -> Message:
    calls = [ToolCall(name="shot", params={}, id=cid) for cid in call_ids or []] or None
    return Message(role=Role.ASSISTANT, content=text, tool_calls=calls)


async def prep(
    messages: list[Message],
    runtime: ImageRuntime,
    provider: LLMProvider | None = None,
    *,
    state: ImageViewState | None = None,
    guardrails: Any = (),
    force: Any = (),
) -> Any:
    return await prepare_request(
        messages,
        state=state or ImageViewState(),
        policy=runtime.policy,
        provider=provider or ImageLLM([]),
        model="claude-test",
        runtime=runtime,
        guardrails=guardrails,
        iteration=1,
        force_tombstones=force,
    )


def leak_free(obj: Any, raw: bytes) -> bool:
    """No bytes, base64 prefix or data URL anywhere in ``obj``'s JSON/repr."""
    import base64

    text = json.dumps(obj, default=repr)
    b64 = base64.b64encode(raw).decode()[:16]
    return b64 not in text and "data:image" not in text and repr(raw)[2:30] not in text


# ---------------------------------------------------------------------------
# prepare_request: fast path, purity, step order
# ---------------------------------------------------------------------------


async def test_no_images_is_identity_and_empty() -> None:
    msgs = [Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="hi")]
    rt = make_runtime()
    out = await prep(msgs, rt)
    assert out.semantic is msgs and out.wire is msgs
    assert out.slots == () and out.events == () and out.block_error is None
    assert out.delta.placeholders == {} and out.guardrail_findings is None
    assert rt.catalog.get_current_calls == []  # type: ignore[union-attr]


async def test_prepare_is_pure_and_builds_wire_and_semantic() -> None:
    rt = make_runtime()
    raw = make_png(seed=1)
    part = await store_image(rt, raw, detail="low")
    msgs = [Message(role=Role.SYSTEM, content="s"), user_msg("look", part)]
    state = ImageViewState()
    snapshot = state.to_dict()
    out = await prep(msgs, rt, state=state)
    assert state.to_dict() == snapshot  # never mutated
    assert out.new_slots == (0,)
    (slot,) = out.slots
    assert slot.visible and slot.origin == "user" and slot.message_index == 1
    sem_img = out.semantic[1].images[0]
    wire_img = out.wire[1].images[0]
    assert sem_img.data is None and wire_img.data == raw
    assert sem_img.detail == "low" == wire_img.detail
    assert out.stats.count == 1 and out.stats.bytes == len(raw)
    assert out.stats.estimated_tokens is not None
    assert out.events == () and out.delta.placeholders == {}
    assert rt.catalog.writes == 1  # type: ignore[union-attr] — only the ingest wrote


async def test_missing_row_is_typed_not_found() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=2))
    other = make_runtime(scope="t:other")  # same bytes, other tenant: absent there
    with pytest.raises(BlobNotFoundError):
        await prep([user_msg(part)], other)


async def test_descriptor_mismatch_caught_before_policy() -> None:
    rt = make_runtime(ImagePolicy(max_image_bytes=10))  # would also fail policy
    rt_ingest = make_runtime(catalog=rt.catalog)  # type: ignore[arg-type]
    part = await store_image(rt_ingest, make_png(seed=3))
    bad = ImagePart(replace(part.ref, width=part.ref.width + 1))
    with pytest.raises(BlobIntegrityError) as err:
        await prep([user_msg(bad)], rt)
    assert err.value.check == "descriptor"


async def test_history_policy_placeholder_vs_new_image_error() -> None:
    loose = make_runtime()
    part = await store_image(loose, make_png(seed=4))
    strict = make_runtime(ImagePolicy(max_long_side=2), catalog=loose.catalog)  # type: ignore[arg-type]
    msgs = [user_msg("a", part)]
    with pytest.raises(ImageLimitError):
        await prep(msgs, strict)  # new image -> typed error
    out = await prep(msgs, strict, state=ImageViewState(sent_through=1))
    assert out.semantic[0].parts is None
    assert (
        out.semantic[0].content
        == f"a\n\n[image {part.image_id} not shown: it exceeds the current image policy]"
    )
    assert out.delta.placeholders == {0: "policy"}
    assert [e.data for e in out.events if e.event_type == "image.degraded"] == [
        {"reason": "policy", "count": 1, "slots": [0]}
    ]


# ---------------------------------------------------------------------------
# Capability: new vs history (§5.6, N-6)
# ---------------------------------------------------------------------------


async def test_capability_no_new_image_raises() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=5))
    with pytest.raises(UnsupportedModalityError) as err:
        await prep([user_msg(part)], rt, ImageLLM([], support="no"))
    assert err.value.source == "capability" and err.value.image_count == 1


async def test_capability_no_history_image_placeholder() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=6))
    out = await prep(
        [user_msg("q", part)], rt, ImageLLM([], support="no"), state=ImageViewState(sent_through=1)
    )
    assert out.delta.placeholders == {0: "unsupported"}
    assert "this model does not accept images" in out.wire[0].content
    assert out.wire[0].parts is None


async def test_unknown_capability_sends_unless_error_mode() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=7))
    out = await prep([user_msg(part)], rt, ImageLLM([], support="unknown"))
    assert out.slots[0].visible
    strict = make_runtime(ImagePolicy(unknown_capability="error"), catalog=rt.catalog)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedModalityError):
        await prep([user_msg(part)], strict, ImageLLM([], support="unknown"))


async def test_capability_override_beats_provider() -> None:
    rt = make_runtime(ImagePolicy(capability_overrides={"claude-test": "yes"}))
    part = await store_image(rt, make_png(seed=8))
    llm = ImageLLM([], support="no")
    out = await prep([user_msg(part)], rt, llm)
    assert out.slots[0].visible and llm.support_calls == 0


async def test_placeholder_mode_and_describe_callable() -> None:
    rt = make_runtime(ImagePolicy(on_unsupported="placeholder"))
    part = await store_image(rt, make_png(seed=9))
    out = await prep([user_msg(part)], rt, ImageLLM([], support="no"))
    assert out.delta.placeholders == {0: "unsupported"}

    seen: list[str] = []

    async def describe(req: Any) -> str:
        seen.append(req.image_id)
        assert req.data.startswith(b"\x89PNG")
        return "  a red square  "

    rt2 = make_runtime(ImagePolicy(on_unsupported=describe), catalog=rt.catalog)  # type: ignore[arg-type]
    out2 = await prep(
        [user_msg(part), tool_msg("shot", "c1", part)], rt2, ImageLLM([], support="no")
    )
    assert seen == [part.image_id]  # once per ImageKey
    assert out2.wire[0].content == f"[image {part.image_id} (described): a red square]"
    assert out2.delta.described == {("t:acme", part.ref.sha256): "a red square"}
    state = ImageViewState()
    apply_delta(state, out2.delta)
    seen.clear()
    await prep([user_msg(part)], rt2, ImageLLM([], support="no"), state=state)
    assert seen == []  # state.described read first
    (state_event,) = [e.data for e in out2.events if e.event_type == "image.state"]
    assert state_event["described"] == [{"sha256": part.ref.sha256, "text": "a red square"}]


async def test_learned_no_is_per_provider_instance() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=10))
    a = ImageLLM([], support="unknown", base_url="https://a")
    b = ImageLLM([], support="unknown", base_url="https://b")
    learn_image_unsupported(a, "claude-test")
    out_a = await prep([user_msg(part)], rt, a, state=ImageViewState(sent_through=1))
    out_b = await prep([user_msg(part)], rt, b, state=ImageViewState(sent_through=1))
    assert out_a.delta.placeholders == {0: "unsupported"}
    assert out_b.slots[0].visible


# ---------------------------------------------------------------------------
# Precedence, stickiness, purge (N-7, N-9)
# ---------------------------------------------------------------------------


async def test_sticky_placeholder_and_degraded_once_per_slot() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=11))
    msgs = [user_msg("q"[0], part)]
    state = ImageViewState(sent_through=1)
    out1 = await prep(msgs, rt, state=state, force=[("t:acme", part.ref.sha256)])
    assert out1.delta.placeholders == {0: "rejected"}
    apply_delta(state, out1.delta)
    # Now the model loses image support: stays "rejected" (sticky), no new event.
    out2 = await prep(msgs, rt, ImageLLM([], support="no"), state=state)
    assert out2.slots[0].placeholder == "rejected"
    assert out2.delta.placeholders == {} and out2.events == ()
    assert out2.semantic[0].content == out1.semantic[0].content  # byte-stable


async def test_purged_replaces_sticky_reason() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=12))
    state = ImageViewState(sent_through=1, placeholders={0: "drop"})
    rt.catalog.purge("t:acme", part.ref.sha256)  # type: ignore[union-attr]
    out = await prep([user_msg(part)], rt, state=state)
    assert out.delta.placeholders == {0: "purged"}
    assert out.wire[0].content == f"[image {part.image_id} was deleted]"
    apply_delta(state, out.delta)
    assert state.placeholders == {0: "purged"}


async def test_mixed_parts_keep_order_in_place() -> None:
    rt = make_runtime()
    a = await store_image(rt, make_png(seed=13))
    b = await store_image(rt, make_png(seed=14))
    msg = tool_msg("shot", "c1", a, b)
    out = await prep(
        [msg], rt, force=[("t:acme", a.ref.sha256)], state=ImageViewState(sent_through=2)
    )
    wire = out.wire[0]
    assert wire.parts is not None and len(wire.parts) == 3
    assert isinstance(wire.parts[1], TextPart) and "rejected" in wire.parts[1].text
    assert isinstance(wire.parts[2], ImagePart) and wire.parts[2].data is not None
    assert wire.name == "shot" and wire.call_id == "c1"


async def test_many_image_rule() -> None:
    rt = make_runtime()
    parts = [await store_image(rt, make_png(seed=100 + i)) for i in range(20)]
    big = await store_image(rt, make_png(width=2400, height=1, seed=99))
    msgs = [user_msg("hi", *parts[:10]), user_msg(*parts[10:], big)]
    with pytest.raises(ImageLimitError) as err:
        await prep(msgs, rt, ImageLLM([], rule=(20, 2000)))
    assert err.value.scope == "many_image_dimension" and err.value.actual == 2400
    out = await prep(msgs, rt, ImageLLM([], rule=None))  # e.g. OpenAI: sent unchanged
    assert out.stats.count == 21


async def test_many_image_rule_forced_prune_with_drop_policy() -> None:
    rt = make_runtime(ImagePolicy(drop=ImageDropPolicy(keep_recent=3, interval=25)))
    big = await store_image(rt, make_png(width=2400, height=1, seed=98))
    msgs: list[Message] = [Message(role=Role.USER, content="start")]
    for i in range(21):
        part = big if i == 0 else await store_image(rt, make_png(seed=200 + i))
        msgs.append(assistant(call_ids=[f"c{i}"]))
        msgs.append(tool_msg("shot", f"c{i}", part))
    msgs.append(assistant("ok"))
    out = await prep(msgs, rt, ImageLLM([], rule=(20, 2000)), state=ImageViewState(sent_through=21))
    assert out.stats.count == 3
    assert out.delta.dropped_through == 18 and out.delta.epoch == 1
    assert [e.event_type for e in out.events][:2] == ["image.view_changed", "image.degraded"]


# ---------------------------------------------------------------------------
# Tool execution: ToolOutput, transform, substitute, strategy guard
# ---------------------------------------------------------------------------


PNG_A = make_png(seed=21)


@tool()
async def shot() -> Any:
    """Take a screenshot."""
    return ToolOutput("Captured.", images=[Image.from_bytes(PNG_A)])


@tool()
async def shot_at(target: str, nested: list[str]) -> Any:
    """Take a screenshot of a target."""
    return ToolOutput("Captured.", images=[Image.from_bytes(PNG_A)])


@tool()
async def bad_shot() -> Any:
    """Return an image the wrong way."""
    return {"img": Image.from_bytes(PNG_A)}


@tool()
async def svg_shot() -> Any:
    """Return an unsupported image."""
    return ToolOutput("x", images=[Image.from_bytes(b"<svg></svg>")])


@tool(target=ToolTarget.CLIENT)
async def client_shot() -> str:
    """Client screenshot."""
    return ""


def _lookups(*fns: Any) -> Any:
    return build_tool_lookups(list(fns))


async def test_execute_tool_tooloutput_ingests_and_serialises() -> None:
    rt = make_runtime()
    lookups = _lookups(shot)
    result = await _execute_tool(
        ToolCall(name="shot", params={}, id="c1"), lookups, image_runtime=rt
    )
    assert result.success and result.payload == '"Captured."'
    (img,) = result.images
    assert isinstance(img, ImagePart) and img.ref.sha256 == hashlib.sha256(PNG_A).hexdigest()
    plain = await _execute_tool(ToolCall(name="shot", params={}, id="c2"), lookups)
    assert not plain.success and "loop" in (plain.error or "")


async def test_execute_tool_strict_errors_and_hint() -> None:
    rt = make_runtime()
    lookups = _lookups(svg_shot, bad_shot)
    svg = await _execute_tool(
        ToolCall(name="svg_shot", params={}, id="c1"), lookups, image_runtime=rt
    )
    assert not svg.success and json.loads(svg.payload)["error"]
    bad = await _execute_tool(
        ToolCall(name="bad_shot", params={}, id="c2"), lookups, image_runtime=rt
    )
    assert not bad.success and "return ToolOutput(result, images=[...])" in (bad.error or "")


async def test_transform_sees_tooloutput_and_can_drop_images() -> None:
    rt = make_runtime()
    seen: list[Any] = []

    def drop(ctx: Any) -> Any:
        seen.append(ctx.result)
        return ctx.result.result  # plain value -> no images

    result = await _execute_tool(
        ToolCall(name="shot", params={}, id="c1"), _lookups(shot), drop, image_runtime=rt
    )
    assert isinstance(seen[0], ToolOutput)
    assert result.images == () and result.payload == '"Captured."'


async def test_react_tool_image_reaches_wire_and_pause_carries_view() -> None:
    rt = make_runtime()
    llm = ImageLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="shot", params={}, provider_tool_call_id="p1")]),
            LLMResponse(
                tool_calls=[ToolCall(name="client_shot", params={}, provider_tool_call_id="p2")]
            ),
        ]
    )
    agent = Agent(provider=llm, tools=[shot, client_shot], prompt="p")
    rec = RecordingRecorder()
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=rec,
        image_runtime=rt,
    )
    assert result.status == RunStatus.WAITING_CLIENT_TOOL
    wire_tool = llm.calls[1][-1]
    assert wire_tool.role == Role.TOOL and wire_tool.images[0].data == PNG_A
    sem_tool = rec.started[1][-1]  # type: ignore[index]
    assert sem_tool.images[0].data is None
    pause = result.meta["pause_state"]
    assert pause.image_view is not None and pause.image_view.sent_through == 1
    assert leak_free(pause.to_dict(), PNG_A)
    assert leak_free(rec.events, PNG_A)


async def test_text_only_pause_has_no_image_view() -> None:
    llm = ImageLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="client_shot", params={}, provider_tool_call_id="p")]
            )
        ]
    )
    agent = Agent(provider=llm, tools=[client_shot], prompt="p")
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        image_runtime=make_runtime(),
        image_view_state=ImageViewState(),
    )
    assert result.meta["pause_state"].image_view is None
    assert "image_view" not in result.meta["pause_state"].to_dict()
    assert pause_image_view(ImageViewState(), [Message(role=Role.USER, content="x")]) is None


async def test_strategy_dropping_images_raises() -> None:
    class Lossy(NativeToolCalling):
        def format_tool_result(self, result: ToolResult) -> Message:
            return Message(
                role=Role.TOOL, content=result.payload, name=result.name, call_id=result.call_id
            )

    llm = ImageLLM(
        [LLMResponse(tool_calls=[ToolCall(name="shot", params={}, provider_tool_call_id="p")])]
    )
    agent = Agent(provider=llm, tools=[shot], prompt="p")
    with pytest.raises(TypeError, match="strategy dropped image parts"):
        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=Lossy(),
            user_input="go",
            image_runtime=make_runtime(),
        )


async def test_before_tool_substitute_tooloutput_and_ctx_images() -> None:
    rt = make_runtime()
    prior = await store_image(rt, make_png(seed=31))
    contexts: list[Any] = []

    def hook(ctx: Any) -> Any:
        contexts.append(ctx)
        return ToolDecision.substitute(ToolOutput({"ok": 1}, images=[Image.from_bytes(PNG_A)]))

    llm = ImageLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="shot_at",
                        params={"target": prior.image_id, "nested": [prior.image_id + "x"]},
                        provider_tool_call_id="p",
                    )
                ]
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(provider=llm, tools=[shot_at], prompt="p", before_tool=hook)
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        initial_history=[user_msg("see", prior)],
        image_runtime=rt,
    )
    assert result.status == RunStatus.SUCCESS
    assert contexts[0].images == (prior.ref,)
    tool_wire = llm.calls[1][-1]
    assert tool_wire.content == '{"ok": 1}' and tool_wire.images[0].data == PNG_A


async def test_before_tool_substitute_image_failure_fails_closed() -> None:
    def hook(ctx: Any) -> Any:
        return ToolDecision.substitute(ToolOutput("x", images=[Image.from_bytes(b"nope")]))

    llm = ImageLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="shot", params={}, provider_tool_call_id="p")]),
            LLMResponse(text="d"),
        ]
    )
    rec = RecordingRecorder()
    agent = Agent(provider=llm, tools=[shot], prompt="p", before_tool=hook)
    await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=rec,
        image_runtime=make_runtime(),
    )
    assert rec.of("policy.hook_error")[0]["error_type"] == "InvalidImageError"
    assert "denied" in llm.calls[1][-1].content


# ---------------------------------------------------------------------------
# Rejection (§5.7, N-10, N-15)
# ---------------------------------------------------------------------------


def rejected(*ids: str, located: bool = False) -> ImageRejectedError:
    return ImageRejectedError(
        provider="ImageLLM", model="claude-test", image_ids=ids, located=located, status_code=400
    )


def tool_turn(i: int) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name="shot", params={}, provider_tool_call_id=f"p{i}")])


async def _run(
    llm: ImageLLM,
    *,
    tools: list[Any] | None = None,
    history: list[Message] | None = None,
    rt: ImageRuntime | None = None,
    state: ImageViewState | None = None,
    store: Any = None,
    rec: RecordingRecorder | None = None,
) -> Any:
    agent = Agent(provider=llm, tools=tools if tools is not None else [shot], prompt="p")
    return await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        initial_history=history,
        recorder=rec,
        image_runtime=rt or make_runtime(),
        image_view_state=state,
        state_store=store,
    )


async def _history(rt: ImageRuntime, *, tools: int, users: int) -> list[Message]:
    msgs: list[Message] = []
    user_parts = [await store_image(rt, make_png(seed=300 + i)) for i in range(users)]
    msgs.append(user_msg("q", *user_parts) if user_parts else Message(role=Role.USER, content="q"))
    for i in range(tools):
        msgs.append(assistant(call_ids=[f"h{i}"]))
        msgs.append(tool_msg("shot", f"h{i}", await store_image(rt, make_png(seed=400 + i))))
    return msgs


async def test_located_rejection_tombstones_one_and_retries() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=3, users=0)
    target = history[4].images[0]
    llm = ImageLLM([rejected(target.image_id, located=True), LLMResponse(text="ok")])
    rec = RecordingRecorder()
    state = ImageViewState()
    result = await _run(llm, history=history, rt=rt, rec=rec, state=state)
    assert result.status == RunStatus.SUCCESS
    assert state.rejected == [("t:acme", target.ref.sha256)]
    events = rec.of("image.rejected")
    assert len(events) == 1 and events[0]["retried"] is True and events[0]["located"] is True
    assert len(llm.calls) == 2 and len(rec.failed) == 1
    second_wire = llm.calls[1][5]  # system + history[4]
    assert "was rejected by the model provider" in second_wire.content


async def test_unlocated_with_new_images_tombstones_all_new() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=3, users=0)
    llm = ImageLLM([rejected(), LLMResponse(text="ok")])
    state = ImageViewState(sent_through=1)  # slots 1, 2 are new
    await _run(llm, history=history, rt=rt, state=state)
    assert state.rejected == [("t:acme", m.images[0].ref.sha256) for m in history[4::2]]


async def test_unlocated_history_only_tombstones_tool_images_not_user() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=2, users=1)
    llm = ImageLLM([rejected(), LLMResponse(text="ok")])
    state = ImageViewState(sent_through=3)
    result = await _run(llm, history=history, rt=rt, state=state)
    assert result.status == RunStatus.SUCCESS
    tool_keys = [("t:acme", m.images[0].ref.sha256) for m in history[2::2]]
    assert state.rejected == tool_keys
    assert llm.calls[1][1].images  # the user image is still sent


async def test_unlocated_only_user_history_raises_without_tombstone() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=0, users=1)
    llm = ImageLLM([rejected()])
    state = ImageViewState(sent_through=1)
    with pytest.raises(ImageRejectedError) as err:
        await _run(llm, history=history, rt=rt, state=state)
    assert err.value.located is False and err.value.retried is False
    assert err.value.__cause__ is None and err.value.__context__ is None
    assert state.rejected == []


async def test_user_image_rejection_never_retried() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=500))
    llm = ImageLLM([rejected(part.image_id, located=True)])
    state = ImageViewState()
    with pytest.raises(ImageRejectedError) as err:
        await _run(llm, history=[user_msg("what", part)], rt=rt, state=state)
    assert err.value.image_ids == (part.image_id,) and err.value.retried is False
    assert state.rejected == [("t:acme", part.ref.sha256)] and len(llm.calls) == 1


async def test_double_rejection_errors_retried_true() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=2, users=0)
    llm = ImageLLM([rejected(), rejected()])
    with pytest.raises(ImageRejectedError) as err:
        await _run(llm, history=history, rt=rt)
    assert err.value.retried is True and len(llm.calls) == 2


async def test_cancel_before_retry_honoured() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    llm = ImageLLM([rejected(), LLMResponse(text="never")])
    store = ControlStore()
    store.cancel_after = 1  # top-of-iteration read passes, the pre-retry read cancels
    state = ImageViewState()
    result = await _run(llm, history=history, rt=rt, state=state, store=store)
    assert result.status == RunStatus.CANCELLED and len(llm.calls) == 1
    assert state.rejected  # tombstone durable anyway


async def test_pause_before_retry_carries_tombstone() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    llm = ImageLLM([rejected()])
    store = ControlStore()
    store.pause_after = 1
    result = await _run(llm, history=history, rt=rt, store=store)
    assert result.status == RunStatus.PAUSED
    view = result.meta["pause_state"].image_view
    assert view is not None and len(view.rejected) == 1


async def test_unsupported_history_retry_learns_and_degrades() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=1)
    err = UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")
    llm = ImageLLM([err, LLMResponse(text="ok")], support="unknown")
    rec = RecordingRecorder()
    result = await _run(llm, history=history, rt=rt, state=ImageViewState(sent_through=2), rec=rec)
    assert result.status == RunStatus.SUCCESS
    assert not any(m.images for m in llm.calls[1])
    assert rec.of("image.degraded")[0]["reason"] == "unsupported"


async def test_unsupported_with_new_images_propagates() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    err = UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")
    llm = ImageLLM([err], support="unknown")
    with pytest.raises(UnsupportedModalityError):
        await _run(llm, history=history, rt=rt)


async def _stream(
    llm: ImageLLM, history: list[Message], rt: ImageRuntime, state: ImageViewState
) -> list[Any]:
    agent = Agent(provider=llm, tools=[shot], prompt="p")
    events = []
    async for ev in ReActLoop().run_stream(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        initial_history=history,
        image_runtime=rt,
        image_view_state=state,
    ):
        events.append(ev)
    return events


async def test_stream_retry_closes_first_stream() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    llm = ImageLLM([StreamScript(error=rejected()), LLMResponse(text="fine")])
    state = ImageViewState()
    events = await _stream(llm, history, rt, state)
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert len(llm.calls) == 2 and llm.aclosed >= 2  # first stream closed before the retry
    assert len(state.rejected) == 1


async def test_stream_rejection_after_delta_no_retry() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    llm = ImageLLM([StreamScript(["partial"], error=rejected())])
    state = ImageViewState()
    with pytest.raises(ImageRejectedError) as err:
        await _stream(llm, history, rt, state)
    assert err.value.retried is False and len(llm.calls) == 1
    assert len(state.rejected) == 1  # durable; a later retry() never resends it


async def test_single_call_user_image_rejection_never_retries() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=600))
    llm = ImageLLM([rejected()])
    agent = Agent(provider=llm, prompt="p", loop=SingleCall())
    with pytest.raises(ImageRejectedError):
        await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="x",
            initial_history=[user_msg("what", part)],
            image_runtime=rt,
        )
    assert len(llm.calls) == 1


async def test_single_call_stream_sends_user_image() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=601))
    llm = ImageLLM([LLMResponse(text="red")])
    agent = Agent(provider=llm, prompt="p", loop=SingleCall())
    events = [
        ev
        async for ev in SingleCall().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="x",
            initial_history=[user_msg("what", part)],
            image_runtime=rt,
        )
    ]
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert llm.calls[0][1].images[0].data is not None


async def test_text_only_errors_keep_v030_path() -> None:
    llm = ImageLLM([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        await _run(llm, tools=[], history=[Message(role=Role.USER, content="hi")])


# ---------------------------------------------------------------------------
# Explicit runtime threading: tenants never mix (§8 shared interfaces)
# ---------------------------------------------------------------------------


def _tenant_tool(tag: int) -> Any:
    @tool(parallel=True)
    async def snap(n: int) -> Any:
        """Snapshot."""
        await asyncio.sleep(0)
        return ToolOutput(n, images=[Image.from_bytes(make_png(seed=1000 + tag * 10 + n))])

    return snap


async def test_interleaved_streams_different_tenants() -> None:
    catalog = FakeCatalog()
    rts = [make_runtime(scope=s, catalog=catalog) for s in ("t:a", "t:b")]
    snap = _tenant_tool(0)
    streams = []
    llms = []
    for rt in rts:
        llm = ImageLLM(
            [
                LLMResponse(
                    tool_calls=[ToolCall(name="snap", params={"n": 1}, provider_tool_call_id="p")]
                ),
                LLMResponse(text="done"),
            ]
        )
        llms.append(llm)
        agent = Agent(provider=llm, tools=[snap], prompt="p")
        streams.append(
            ReActLoop().run_stream(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="go",
                image_runtime=rt,
            )
        )
    done = [False, False]
    while not all(done):
        for i, stream in enumerate(streams):
            if done[i]:
                continue
            try:
                await stream.__anext__()
            except StopAsyncIteration:
                done[i] = True
    sha = hashlib.sha256(make_png(seed=1001)).hexdigest()
    assert ("t:a", sha) in catalog.rows and ("t:b", sha) in catalog.rows
    for rt, llm in zip(rts, llms, strict=True):
        assert {s for s, _ in rt.resolver.reads} == {rt.tenant_scope}  # type: ignore[union-attr]
        assert llm.calls[1][-1].images[0].data == make_png(seed=1001)


async def test_eight_concurrent_runs_two_tenants_parallel_tools() -> None:
    catalog = FakeCatalog()

    async def one(i: int) -> tuple[ImageRuntime, Any]:
        rt = make_runtime(scope=f"t:{i % 2}", catalog=catalog)
        snap = _tenant_tool(i)
        llm = ImageLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="snap", params={"n": 1}, provider_tool_call_id="a"),
                        ToolCall(name="snap", params={"n": 2}, provider_tool_call_id="b"),
                    ]
                ),
                LLMResponse(text="ok"),
            ]
        )
        agent = Agent(provider=llm, tools=[snap], prompt="p")
        res = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            image_runtime=rt,
        )
        return rt, (res, llm)

    outs = await asyncio.gather(*(one(i) for i in range(8)))
    for i, (rt, (res, llm)) in enumerate(outs):
        assert res.status == RunStatus.SUCCESS
        for n in (1, 2):
            sha = hashlib.sha256(make_png(seed=1000 + i * 10 + n)).hexdigest()
            assert (rt.tenant_scope, sha) in catalog.rows
            assert (f"t:{(i + 1) % 2}", sha) not in catalog.rows
        assert {s for s, _ in rt.resolver.reads} == {rt.tenant_scope}  # type: ignore[union-attr]
        assert [m.images[0].data for m in llm.calls[1][-2:]] == [
            make_png(seed=1000 + i * 10 + n) for n in (1, 2)
        ]


async def test_inspect_fixture_png_valid() -> None:
    header = inspect_image(make_png(width=7, height=3))
    assert (header.width, header.height, header.media_type) == (7, 3, "image/png")


async def test_commit_order_events_before_llm_started_then_sent() -> None:
    """Commit protocol: delta applied, events recorded, then llm started; sent after success."""
    log: list[str] = []

    class OrderRecorder(RecordingRecorder):
        async def on_governance_event(
            self, run_id: str, event_type: str, *a: Any, **kw: Any
        ) -> None:
            log.append(event_type)

        async def on_llm_call_started(self, run_id: str, iteration: int, **kw: Any) -> None:
            log.append("llm.started")

        async def on_llm_call_completed(self, *a: Any, **kw: Any) -> None:
            log.append("llm.completed")

    rt = make_runtime()
    part = await store_image(rt, make_png(seed=800))
    state = ImageViewState(sent_through=1)
    llm = ImageLLM([LLMResponse(text="ok")], support="no")
    await _run(
        llm,
        tools=[],
        history=[user_msg("q", part)],
        rt=rt,
        state=state,
        rec=OrderRecorder(),
    )
    assert log == ["image.degraded", "image.state", "llm.started", "llm.completed"]
    assert state.placeholders == {0: "unsupported"} and state.sent_through == 1


async def test_release_wire_drops_bytes() -> None:
    from dendrux.loops._images import ImageCallSite, release_wire

    rt = make_runtime()
    part = await store_image(rt, make_png(seed=810))
    msgs = [user_msg("q", part)]
    site = ImageCallSite(
        state=ImageViewState(),
        policy=rt.policy,
        runtime=rt,
        provider=ImageLLM([]),
        model="claude-test",
        recorder=None,
        notifier=None,
        run_id="r",
        warnings=[],
    )
    prepared = await site.prepare(msgs, iteration=1)
    assert prepared.wire[0].images[0].data is not None
    released = site.mark_sent(prepared)
    assert released.wire is released.semantic and site.state.sent_through == 1
    assert all(p.data is None for m in released.wire for p in m.images)
    plain = await site.prepare([Message(role=Role.USER, content="x")], iteration=2)
    assert release_wire(plain) is plain


# ---------------------------------------------------------------------------
# Image wire cleanup
# ---------------------------------------------------------------------------


async def test_no_wire_bytes_retained_after_multi_iteration_run() -> None:
    """P1-1 regression (reviewer probe): no ImagePart with bytes survives the call."""
    import gc

    import dendrux.types as core_types

    seeds = (5001, 5002, 5003)
    marks = {make_png(seed=s) for s in seeds}

    def live() -> int:
        gc.collect()
        return sum(
            1
            for o in gc.get_objects()
            if isinstance(o, ImagePart) and o.data is not None and o.data in marks
        )

    during: list[int] = []

    @tool()
    async def snap(n: int) -> Any:
        """Snap."""
        during.append(live())
        return ToolOutput(n, images=[Image.from_bytes(make_png(seed=5000 + n))])

    llm = ImageLLM(
        [
            *(
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="snap", params={"n": n}, provider_tool_call_id=f"p{n}")
                    ]
                )
                for n in (1, 2, 3)
            ),
            LLMResponse(text="done"),
        ]
    )
    llm.keep_calls = False
    agent = Agent(provider=llm, tools=[snap], prompt="p")
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        image_runtime=make_runtime(),
    )
    assert result.status == RunStatus.SUCCESS
    assert during == [0, 0, 0]
    assert live() == 0
    pinned = sum(
        1
        for parts, _ in core_types._PROJECTION_MEMO.values()
        for p in parts
        if isinstance(p, ImagePart) and p.data is not None
    )
    assert pinned == 0


async def test_identical_bytes_user_history_never_tombstoned() -> None:
    """P2-1: a tool image with the same bytes as a user history image."""
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=5100))
    history = [user_msg("see", part), assistant(call_ids=["c"]), tool_msg("shot", "c", part)]
    llm = ImageLLM([rejected()])
    state = ImageViewState(sent_through=2)
    with pytest.raises(ImageRejectedError) as err:
        await _run(llm, history=history, rt=rt, state=state)
    assert err.value.located is False and state.rejected == [] and len(llm.calls) == 1
    # New tool image with those bytes: still never tombstoned through the user copy.
    llm2 = ImageLLM([rejected()])
    state2 = ImageViewState(sent_through=1)
    with pytest.raises(ImageRejectedError):
        await _run(llm2, history=history, rt=rt, state=state2)
    assert state2.rejected == []


async def test_stream_in_process_cancel_before_retry() -> None:
    """P2-3: interrupt set while the rejected stream is in flight → no second call."""
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    llm = ImageLLM([StreamScript(error=rejected()), LLMResponse(text="never")])
    interrupt = asyncio.Event()
    llm.on_open = interrupt.set
    agent = Agent(provider=llm, tools=[shot], prompt="p")
    events = [
        ev
        async for ev in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            initial_history=history,
            image_runtime=rt,
            interrupt=interrupt,
        )
    ]
    assert events[-1].type == RunEventType.RUN_CANCELLED
    assert llm.log == ["open", "aclose"]


async def test_single_stream_in_process_cancel_before_retry() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=5200))
    err = UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")
    llm = ImageLLM([StreamScript(error=err), LLMResponse(text="never")], support="unknown")
    interrupt = asyncio.Event()
    llm.on_open = interrupt.set
    agent = Agent(provider=llm, prompt="p", loop=SingleCall())
    events = [
        ev
        async for ev in SingleCall().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="x",
            initial_history=[user_msg("q", part)],
            image_runtime=rt,
            image_view_state=ImageViewState(sent_through=1),
            interrupt=interrupt,
        )
    ]
    assert events[-1].type == RunEventType.RUN_CANCELLED and llm.log == ["open", "aclose"]


async def _stream_run(
    llm: ImageLLM, history: list[Message], rt: ImageRuntime, **kw: Any
) -> list[Any]:
    agent = Agent(provider=llm, tools=[shot], prompt="p")
    return [
        ev
        async for ev in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            initial_history=history,
            image_runtime=rt,
            **kw,
        )
    ]


async def test_stream_store_cancel_and_pause_before_retry() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    store = ControlStore()
    store.cancel_after = 1
    llm = ImageLLM([StreamScript(error=rejected()), LLMResponse(text="never")])
    events = await _stream_run(llm, history, rt, state_store=store)
    assert events[-1].type == RunEventType.RUN_CANCELLED and len(llm.calls) == 1
    store2 = ControlStore()
    store2.pause_after = 1
    llm2 = ImageLLM([StreamScript(error=rejected())])
    events2 = await _stream_run(llm2, history, rt, state_store=store2)
    assert events2[-1].type == RunEventType.RUN_PAUSED
    assert len(events2[-1].run_result.meta["pause_state"].image_view.rejected) == 1


async def test_stream_close_before_reopen_order() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=0)
    llm = ImageLLM([StreamScript(error=rejected()), LLMResponse(text="ok")])
    events = await _stream_run(llm, history, rt)
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert llm.log == ["open", "aclose", "open", "aclose"]


async def test_stream_unsupported_history_retry() -> None:
    rt = make_runtime()
    history = await _history(rt, tools=1, users=1)
    err = UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")
    llm = ImageLLM([StreamScript(error=err), LLMResponse(text="ok")], support="unknown")
    events = await _stream_run(llm, history, rt, image_view_state=ImageViewState(sent_through=2))
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert llm.log == ["open", "aclose", "open", "aclose"]
    assert not any(m.images for m in llm.calls[1])


async def test_single_stream_unsupported_history_retry() -> None:
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=5300))
    err = UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")
    llm = ImageLLM([StreamScript(error=err), LLMResponse(text="ok")], support="unknown")
    agent = Agent(provider=llm, prompt="p", loop=SingleCall())
    events = [
        ev
        async for ev in SingleCall().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="x",
            initial_history=[user_msg("q", part)],
            image_runtime=rt,
            image_view_state=ImageViewState(sent_through=1),
        )
    ]
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert llm.log == ["open", "aclose", "open", "aclose"]
    assert "not shown: this model does not accept images" in llm.calls[1][1].content


async def test_learned_no_for_unhashable_provider() -> None:
    """P2-4: an eq-dataclass provider (unhashable) still learns "no"."""
    from dataclasses import dataclass

    @dataclass
    class EqLLM(LLMProvider):
        name: str = "eq"
        capabilities = ProviderCapabilities(supports_multimodal=True)

        @property
        def model(self) -> str:
            return "m"

        async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
            raise AssertionError

    provider = EqLLM()
    with pytest.raises(TypeError):
        hash(provider)
    rt = make_runtime()
    part = await store_image(rt, make_png(seed=5400))
    before = await prep([user_msg(part)], rt, provider, state=ImageViewState(sent_through=1))
    assert before.slots[0].visible
    learn_image_unsupported(provider, "claude-test")
    after = await prep([user_msg(part)], rt, provider, state=ImageViewState(sent_through=1))
    assert after.delta.placeholders == {0: "unsupported"}


async def test_ingest_timeout_not_reported_as_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import dendrux.loops.react as react_mod

    async def slow(*a: Any, **k: Any) -> Any:
        raise TimeoutError

    monkeypatch.setattr(react_mod, "_ingest_tool_images", slow)
    result = await _execute_tool(
        ToolCall(name="shot", params={}, id="c"), _lookups(shot), image_runtime=make_runtime()
    )
    assert not result.success
    assert result.error == "Image ingest for tool 'shot' timed out"


async def test_identical_bytes_two_tenants_concurrently() -> None:
    catalog = FakeCatalog()
    same = make_png(seed=5500)

    @tool()
    async def snap() -> Any:
        """Snap."""
        await asyncio.sleep(0)
        return ToolOutput("x", images=[Image.from_bytes(same)])

    async def one(scope: str) -> ImageRuntime:
        rt = make_runtime(scope=scope, catalog=catalog)
        llm = ImageLLM(
            [
                LLMResponse(
                    tool_calls=[ToolCall(name="snap", params={}, provider_tool_call_id="p")]
                ),
                LLMResponse(text="ok"),
            ]
        )
        agent = Agent(provider=llm, tools=[snap], prompt="p")
        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            image_runtime=rt,
        )
        return rt

    rts = await asyncio.gather(*(one(f"t:{i % 2}") for i in range(8)))
    sha = hashlib.sha256(same).hexdigest()
    assert {k for k in catalog.rows if k[1] == sha} == {("t:0", sha), ("t:1", sha)}
    for rt in rts:
        assert {s for s, _ in rt.resolver.reads} == {rt.tenant_scope}  # type: ignore[union-attr]
