"""Core image types: ImageRef, parts, Message.parts, tool results, view state, hashing."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from dendrux.errors import InvalidImageError
from dendrux.types import (
    AgentStep,
    Finish,
    GovernanceEventType,
    GuardrailVerdict,
    ImagePart,
    ImageRef,
    ImageViewState,
    Message,
    PauseState,
    Role,
    TextPart,
    ToolCall,
    ToolCallContext,
    ToolResult,
    ToolTarget,
    UsageStats,
    _message_from_dict,
    _message_to_dict,
    compute_idempotency_fingerprint,
    content_part_from_dict,
    content_part_to_dict,
    parts_digest,
    tool_result_message,
)

SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()


def ref(sha: str = SHA_A, **kw: Any) -> ImageRef:
    base: dict[str, Any] = {
        "sha256": sha,
        "media_type": "image/png",
        "byte_length": 5321,
        "width": 400,
        "height": 200,
    }
    base.update(kw)
    return ImageRef(**base)


def img(sha: str = SHA_A, detail: Any = None) -> ImagePart:
    return ImagePart(ref=ref(sha), detail=detail)


# ---------------------------------------------------------------------------
# ImageRef / parts
# ---------------------------------------------------------------------------


class TestImageRef:
    def test_image_id_and_descriptor(self) -> None:
        r = ref()
        assert r.image_id == "img_" + SHA_A[:12]
        assert r.to_dict() == {
            "type": "image",
            "v": 1,
            "kind": "blob",
            "sha256": SHA_A,
            "media_type": "image/png",
            "bytes": 5321,
            "width": 400,
            "height": 200,
        }
        assert len(json.dumps(r.to_dict())) <= 512

    def test_round_trip(self) -> None:
        r = ref()
        assert ImageRef.from_dict(r.to_dict()) == r
        no_type = {k: v for k, v in r.to_dict().items() if k != "type"}
        assert ImageRef.from_dict(no_type) == r

    @pytest.mark.parametrize("bad", [{"v": 2}, {"v": None}, {"kind": "url"}, {"kind": None}])
    def test_wrong_version_or_kind(self, bad: dict[str, Any]) -> None:
        d = ref().to_dict()
        d.update(bad)
        with pytest.raises(InvalidImageError) as ei:
            ImageRef.from_dict(d)
        assert ei.value.reason == "descriptor_version"
        assert ei.value.__context__ is None

    @pytest.mark.parametrize("drop", ["sha256", "media_type", "bytes", "width", "height"])
    def test_missing_field(self, drop: str) -> None:
        d = ref().to_dict()
        del d[drop]
        with pytest.raises(InvalidImageError) as ei:
            ImageRef.from_dict(d)
        assert ei.value.reason == "descriptor_mismatch"
        assert ei.value.__context__ is None and ei.value.__cause__ is None

    def test_not_a_dict_and_wrong_type(self) -> None:
        with pytest.raises(InvalidImageError):
            ImageRef.from_dict([])  # type: ignore[arg-type]
        d = ref().to_dict()
        d["type"] = "text"
        with pytest.raises(InvalidImageError) as ei:
            ImageRef.from_dict(d)
        assert ei.value.reason == "descriptor_mismatch"

    @pytest.mark.parametrize(
        "kw",
        [
            {"sha256": "ABC"},
            {"sha256": SHA_A.upper()},
            {"media_type": ""},
            {"byte_length": 0},
            {"width": -1},
            {"height": True},
            {"kind": "url"},
        ],
    )
    def test_validation(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ref(**kw)


class TestParts:
    def test_text_part_non_empty(self) -> None:
        with pytest.raises(ValueError):
            TextPart("")
        with pytest.raises(ValueError):
            TextPart(None)  # type: ignore[arg-type]

    def test_image_part_detail_and_data(self) -> None:
        with pytest.raises(ValueError):
            ImagePart(ref=ref(), detail="max")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ImagePart(ref=ref().to_dict())  # type: ignore[arg-type]
        a = ImagePart(ref=ref(), data=b"\x89PNG secret")
        b = ImagePart(ref=ref())
        assert a == b
        assert "secret" not in repr(a) and "data" not in repr(a)
        assert a.image_id == ref().image_id

    def test_part_dicts(self) -> None:
        assert content_part_to_dict(TextPart("hi")) == {"type": "text", "text": "hi"}
        d = content_part_to_dict(img(detail="low"))
        assert d["detail"] == "low" and d["sha256"] == SHA_A
        assert "detail" not in content_part_to_dict(img())
        for part in (TextPart("hi"), img(), img(detail="high")):
            assert content_part_from_dict(content_part_to_dict(part)) == part
        with pytest.raises(TypeError):
            content_part_to_dict("x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Message.parts
# ---------------------------------------------------------------------------


class TestMessageParts:
    def test_with_parts_projection(self) -> None:
        m = Message.with_parts(Role.USER, [TextPart("a"), img(), TextPart("b")])
        assert m.content == "a\n\nb"
        assert isinstance(m.parts, tuple)
        assert m.images == (img(),)
        image_only = Message.with_parts(Role.USER, [img(), img(SHA_B)])
        assert image_only.content == ""
        assert len(image_only.images) == 2

    def test_text_only_unchanged(self) -> None:
        m = Message(Role.USER, "hi")
        assert m.parts is None and m.images == ()

    def test_tool_message_parts(self) -> None:
        m = Message.with_parts(Role.TOOL, [TextPart('"ok"'), img()], name="t", call_id="c")
        assert m.kind == "tool_result" and m.content == '"ok"'

    @pytest.mark.parametrize("role", [Role.ASSISTANT, Role.SYSTEM])
    def test_parts_only_user_or_tool(self, role: Role) -> None:
        with pytest.raises(ValueError, match="cannot have parts"):
            Message.with_parts(role, [img()])

    def test_parts_need_an_image(self) -> None:
        with pytest.raises(ValueError, match="at least one ImagePart"):
            Message.with_parts(Role.USER, [TextPart("only text")])
        with pytest.raises(ValueError):
            Message(Role.USER, "", parts=())

    def test_bad_part_types(self) -> None:
        with pytest.raises(TypeError):
            Message.with_parts(Role.USER, ["text", img()])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            Message(Role.USER, "", parts=[img(), "x"])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            Message(Role.USER, "", parts="abc")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            Message.with_parts(Role.USER, "abc")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="computes content"):
            Message.with_parts(Role.USER, [img()], content="x")

    def test_content_must_match_projection(self) -> None:
        with pytest.raises(ValueError, match="text projection"):
            Message(Role.USER, "wrong", parts=(TextPart("a"), img()))
        m = Message.with_parts(Role.USER, [TextPart("a"), img()])
        with pytest.raises(ValueError):
            replace(m, content="stale")
        # Equal (not identical) content is accepted.
        ok = Message(Role.USER, "".join(["a"]), parts=(TextPart("a"), img()))
        assert ok.content == "a"

    def test_list_parts_normalised(self) -> None:
        m = Message(Role.USER, "a", parts=[TextPart("a"), img()])  # type: ignore[arg-type]
        assert isinstance(m.parts, tuple)

    def test_replace_keeps_parts_identity(self) -> None:
        m = Message.with_parts(Role.USER, [TextPart("a"), img()])
        m2 = replace(m, meta={"x": 1})
        assert m2.parts is m.parts and m2.content is m.content
        assert m2.meta == {"x": 1}

    def test_projection_memo_by_identity(self) -> None:
        from dendrux import types as t

        parts = (TextPart("hello"), img())
        p1 = t._text_projection(parts)
        p2 = t._text_projection(parts)
        assert p1 is p2
        equal_copy = tuple(list(parts))
        assert equal_copy is not parts
        assert t._text_projection(equal_copy) == p1
        for i in range(2000):
            t._text_projection((TextPart(f"x{i}"), img()))
        assert len(t._PROJECTION_MEMO) <= t._PROJECTION_MEMO_MAX

    def test_equality_and_copies(self) -> None:
        m = Message.with_parts(Role.USER, [TextPart("a"), img(detail="low")])
        assert m == Message.with_parts(Role.USER, [TextPart("a"), img(detail="low")])
        assert m != Message.with_parts(Role.USER, [TextPart("a"), img()])
        assert pickle.loads(pickle.dumps(m)) == m
        assert copy.deepcopy(m) == m

    def test_init_signature_matches_fields(self) -> None:
        params = list(inspect.signature(Message.__init__).parameters)[1:]
        assert params == [f.name for f in dataclasses.fields(Message)]
        m = Message(Role.USER, "hi", None, None, None, {"k": 1}, "chat", "stable")
        assert m.meta == {"k": 1} and m.placement == "stable"

    def test_meta_default_semantics(self) -> None:
        a, b = Message(Role.USER, "x"), Message(Role.USER, "y")
        assert a.meta == {} and a.meta is not b.meta
        # An explicit None is stored as given, exactly as the v0.3.0 dataclass did.
        assert Message(Role.USER, "x", meta=None).meta is None  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        m = Message(Role.USER, "x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# v0.3.0 Message replica for the hot-path benchmark (verbatim __post_init__)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _V030Message:
    role: Role
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    kind: str = "chat"
    placement: str = "dynamic"
    source: str | None = None
    reasoning: str | None = None
    reasoning_blocks: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.placement not in ("stable", "dynamic"):
            raise ValueError(f"placement must be 'stable' or 'dynamic', got {self.placement!r}")
        if self.kind == "chat":
            if self.role == Role.TOOL:
                object.__setattr__(self, "kind", "tool_result")
            elif self.role == Role.SYSTEM:
                object.__setattr__(self, "kind", "system")
        if self.role == Role.TOOL:
            if not self.name:
                raise ValueError("TOOL messages require name")
            if not self.call_id:
                raise ValueError("TOOL messages require call_id")
            if self.tool_calls is not None:
                raise ValueError("TOOL messages cannot have tool_calls")
        elif self.role == Role.ASSISTANT:
            if self.call_id is not None:
                raise ValueError("ASSISTANT messages cannot have call_id")
            if self.name is not None:
                raise ValueError("ASSISTANT messages cannot have name")
        else:
            if self.tool_calls is not None:
                raise ValueError(f"{self.role.value.upper()} messages cannot have tool_calls")
            if self.call_id is not None:
                raise ValueError(f"{self.role.value.upper()} messages cannot have call_id")
            if self.name is not None:
                raise ValueError(f"{self.role.value.upper()} messages cannot have name")


_BENCH = """
import sys, timeit
from dataclasses import replace
sys.path.insert(0, {root!r})
from tests.unit.test_types_images import _V030Message
from dendrux.types import Message, Role

def best(fn):
    return min(timeit.repeat(fn, number=20000, repeat=7))

old_msg, new_msg = _V030Message(Role.USER, "hello"), Message(Role.USER, "hello")
for _ in range(3):
    ratios = (
        best(lambda: Message(Role.USER, "hello")) / best(lambda: _V030Message(Role.USER, "hello")),
        best(lambda: Message(Role.TOOL, "x", name="n", call_id="c"))
        / best(lambda: _V030Message(Role.TOOL, "x", name="n", call_id="c")),
        best(lambda: replace(new_msg, meta={{"a": 1}}))
        / best(lambda: replace(old_msg, meta={{"a": 1}})),
    )
    if all(r <= 1.05 for r in ratios):
        print("ok", ratios)
        break
else:
    print("slow", ratios)
    sys.exit(1)
"""


def test_text_only_hot_path_within_5_percent_of_v030() -> None:
    """Text-only construct and replace cost at most 5% more than v0.3.0.

    Runs in a fresh interpreter so a coverage tracer does not distort timings.
    """
    root = str(Path(__file__).resolve().parents[2])
    proc = subprocess.run(
        [sys.executable, "-c", _BENCH.format(root=root)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# ToolResult images / notes and the TOOL-message builder
# ---------------------------------------------------------------------------


class TestToolResultMessage:
    def test_v030_shape(self) -> None:
        tr = ToolResult(name="t", call_id="c", payload='"ok"')
        m = tool_result_message(tr)
        assert m == Message(role=Role.TOOL, content='"ok"', name="t", call_id="c", meta={})
        assert m.parts is None
        err = tool_result_message(ToolResult(name="t", call_id="c", payload="{}", success=False))
        assert err.meta == {"is_error": True}

    def test_images(self) -> None:
        tr = ToolResult(name="t", call_id="c", payload='"Captured."', images=[img(), ref(SHA_B)])  # type: ignore[arg-type]
        assert isinstance(tr.images, tuple)
        m = tool_result_message(tr)
        assert m.parts == (TextPart('"Captured."'), img(), ImagePart(ref(SHA_B)))
        assert m.content == '"Captured."' and m.name == "t" and m.call_id == "c"

    def test_empty_payload_with_images(self) -> None:
        m = tool_result_message(ToolResult(name="t", call_id="c", payload="", images=(img(),)))
        assert m.parts == (img(),) and m.content == ""

    def test_raw_image_rejected(self) -> None:
        from dendrux.media import Image

        tr = ToolResult(name="t", call_id="c", payload="1", images=(Image.from_bytes(b"x"),))
        with pytest.raises(TypeError, match="ingested"):
            tool_result_message(tr)

    def test_notes_without_images(self) -> None:
        tr = ToolResult(
            name="t",
            call_id="c",
            payload='"x"',
            notes=["[n1]", "[n2]"],  # type: ignore[arg-type]
        )
        assert tr.notes == ("[n1]", "[n2]")
        m = tool_result_message(tr)
        assert m.parts is None
        assert m.content == '"x"\n\n[n1]\n[n2]'

    def test_notes_with_images(self) -> None:
        tr = ToolResult(name="t", call_id="c", payload='"x"', images=(img(),), notes=("[n1]",))
        m = tool_result_message(tr)
        assert m.parts == (TextPart('"x"'), TextPart("[n1]"), img())
        assert m.content == '"x"\n\n[n1]'

    def test_notes_not_compared(self) -> None:
        a = ToolResult(name="t", call_id="c", payload="1", notes=("a",))
        assert a == ToolResult(name="t", call_id="c", payload="1")

    def test_tool_call_context_images_default(self) -> None:
        ctx = ToolCallContext(
            run_id="r",
            call_id="c",
            tool_name="t",
            params={},
            target=ToolTarget.SERVER,
            source="local",
            meta={},
            iteration=1,
        )
        assert ctx.images == ()


# ---------------------------------------------------------------------------
# ImageViewState / PauseState
# ---------------------------------------------------------------------------


def _view(order: int) -> ImageViewState:
    rejected = [("t:b", SHA_B), ("t:a", SHA_A)]
    described = {("t:b", SHA_B): "b text", ("t:a", SHA_A): "a text"}
    verdicts = {
        ("t:a", SHA_A, "g2", "1", "tool", "shot"): GuardrailVerdict("redact", ("FACE",)),
        ("t:a", SHA_A, "g1", "", "user", ""): GuardrailVerdict("pass"),
    }
    placeholders = {4: "drop", 0: "purged", 2: "rejected"}
    if order:
        rejected.reverse()
        described = dict(reversed(described.items()))
        verdicts = dict(reversed(verdicts.items()))
        placeholders = dict(reversed(placeholders.items()))
    return ImageViewState(
        epoch=2,
        dropped_through=5,
        sent_through=7,
        rejected=rejected,
        described=described,
        guardrail_verdicts=verdicts,
        placeholders=placeholders,
        warned=["drop_with_thinking"],
    )


class TestImageViewState:
    def test_round_trip(self) -> None:
        v = _view(0)
        back = ImageViewState.from_dict(json.loads(json.dumps(v.to_dict())))
        assert back.to_dict() == v.to_dict()
        assert back.placeholders == {0: "purged", 2: "rejected", 4: "drop"}
        assert back.guardrail_verdicts[("t:a", SHA_A, "g2", "1", "tool", "shot")] == (
            GuardrailVerdict("redact", ("FACE",))
        )
        assert set(back.rejected) == {("t:a", SHA_A), ("t:b", SHA_B)}

    def test_deterministic_sort(self) -> None:
        a, b = _view(0).to_dict(), _view(1).to_dict()
        assert json.dumps(a) == json.dumps(b)
        assert [p["slot"] for p in a["placeholders"]] == [0, 2, 4]
        assert [r["tenant_scope"] for r in a["rejected"]] == ["t:a", "t:b"]

    def test_shape(self) -> None:
        d = ImageViewState(sent_through=1).to_dict()
        assert d == {
            "epoch": 0,
            "dropped_through": 0,
            "sent_through": 1,
            "rejected": [],
            "described": [],
            "guardrail_verdicts": [],
            "placeholders": [],
            "warned": [],
        }
        assert list(_view(0).to_dict()) == list(d)

    def test_from_dict_first_wins_and_dedup(self) -> None:
        d = {
            "rejected": [{"tenant_scope": "-", "sha256": SHA_A}] * 2,
            "described": [
                {"tenant_scope": "-", "sha256": SHA_A, "text": "first"},
                {"tenant_scope": "-", "sha256": SHA_A, "text": "second"},
            ],
            "placeholders": [{"slot": 1, "reason": "drop"}, {"slot": 1, "reason": "purged"}],
            "warned": ["w", "w"],
        }
        v = ImageViewState.from_dict(d)
        assert v.rejected == [("-", SHA_A)]
        assert v.described == {("-", SHA_A): "first"}
        assert v.placeholders == {1: "drop"}
        assert v.warned == ["w"]
        assert v.epoch == 0 and v.sent_through == 0

    def test_verdict_entities_tuple(self) -> None:
        assert GuardrailVerdict("warn", ["A"]).entities == ("A",)  # type: ignore[arg-type]


def _pause(**kw: Any) -> PauseState:
    return PauseState(
        agent_name="a",
        pending_tool_calls=[ToolCall(name="t", id="01TC")],
        history=[Message(Role.USER, "hi")],
        steps=[AgentStep(reasoning=None, action=Finish(answer="x"))],
        iteration=1,
        trace_order_offset=2,
        usage=UsageStats(),
        **kw,
    )


class TestPauseState:
    def test_without_image_view_no_new_keys(self) -> None:
        d = _pause().to_dict()
        assert "image_view" not in d
        assert "parts" not in json.dumps(d)
        assert PauseState.from_dict(d).image_view is None

    def test_with_image_view(self) -> None:
        p = _pause(image_view=_view(0))
        d = json.loads(json.dumps(p.to_dict()))
        assert d["image_view"] == _view(0).to_dict()
        back = PauseState.from_dict(d)
        assert back.image_view is not None
        assert back.image_view.to_dict() == _view(0).to_dict()

    def test_history_parts_round_trip(self) -> None:
        user = Message.with_parts(Role.USER, [TextPart("look"), img(detail="low")])
        tool = Message.with_parts(
            Role.TOOL, [TextPart('"Captured."'), img(SHA_B)], name="shot", call_id="C1"
        )
        p = _pause()
        p.history = [user, tool]
        d = json.loads(json.dumps(p.to_dict()))
        assert d["history"][1] == {
            "role": "tool",
            "content": '"Captured."',
            "name": "shot",
            "call_id": "C1",
            "parts": [{"type": "text", "text": '"Captured."'}, ref(SHA_B).to_dict()],
        }
        back = PauseState.from_dict(d)
        assert back.history == [user, tool]
        assert _message_from_dict(_message_to_dict(user)) == user

    def test_v030_pause_json_loads(self) -> None:
        legacy = {
            "agent_name": "a",
            "pending_tool_calls": [],
            "pending_targets": {},
            "history": [{"role": "user", "content": "hi"}],
            "steps": [],
            "iteration": 0,
            "trace_order_offset": 0,
            "usage": {},
        }
        p = PauseState.from_dict(legacy)
        assert p.image_view is None and p.history[0].parts is None


# ---------------------------------------------------------------------------
# Digests and fingerprints (U-09)
# ---------------------------------------------------------------------------


class TestHashing:
    def test_parts_digest_golden(self) -> None:
        parts = [TextPart("What does this say?"), img(detail="low"), ImagePart(ref(SHA_B))]
        assert parts_digest(parts) == [
            ["t", "What does this say?"],
            ["i", "blob", SHA_A, "image/png", "low"],
            ["i", "blob", SHA_B, "image/png", None],
        ]
        with pytest.raises(TypeError):
            parts_digest(["x"])  # type: ignore[list-item]

    def test_text_fingerprints_unchanged_from_v030(self) -> None:
        # Values computed with the v0.3.0 tag's compute_idempotency_fingerprint.
        assert (
            compute_idempotency_fingerprint("agent", "hello")
            == "fbc558a0e36cbc071861f0799266e8f13c0dd44c344c4f67321f33098fd8ee73"
        )
        assert (
            compute_idempotency_fingerprint(
                "agent",
                "hello",
                history=[Message(Role.USER, "hi"), Message(Role.ASSISTANT, "yo")],
                output_type_name="Out",
            )
            == "4daea32d5cb692f2ac6a02ba2dfc2997987aa33e4f623f9e86ab76fd34c3415a"
        )

    def test_parts_fingerprint_golden(self) -> None:
        parts = [TextPart("What does this say?"), img()]
        fp = compute_idempotency_fingerprint("agent", "What does this say?", user_input_parts=parts)
        expected = hashlib.sha256(
            json.dumps(
                {
                    "agent_name": "agent",
                    "user_input": "What does this say?",
                    "user_input_parts": [
                        ["t", "What does this say?"],
                        ["i", "blob", SHA_A, "image/png", None],
                    ],
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        assert fp == expected
        assert fp != compute_idempotency_fingerprint("agent", "What does this say?")

    @pytest.mark.parametrize(
        "variant",
        [
            [img(), TextPart("q")],  # order
            [TextPart("q"), img(SHA_B)],  # hash
            [TextPart("q"), ImagePart(ref(media_type="image/jpeg"))],  # media type
            [TextPart("q"), img(detail="high")],  # provider-visible option
        ],
    )
    def test_parts_fingerprint_sensitive(self, variant: list[Any]) -> None:
        base = compute_idempotency_fingerprint("a", "q", user_input_parts=[TextPart("q"), img()])
        assert compute_idempotency_fingerprint("a", "q", user_input_parts=variant) != base

    def test_wire_only_data_does_not_affect_hash(self) -> None:
        a = [TextPart("q"), ImagePart(ref(), data=b"bytes-one")]
        b = [TextPart("q"), ImagePart(ref(), data=b"bytes-two")]
        assert compute_idempotency_fingerprint(
            "a", "q", user_input_parts=a
        ) == compute_idempotency_fingerprint("a", "q", user_input_parts=b)

    def test_history_with_parts(self) -> None:
        text_hist = [Message(Role.USER, "hi")]
        parts_hist = [Message.with_parts(Role.USER, [TextPart("hi"), img()])]
        a = compute_idempotency_fingerprint("a", "q", history=text_hist)
        b = compute_idempotency_fingerprint("a", "q", history=parts_hist)
        assert a != b
        expected = hashlib.sha256(
            json.dumps(
                {
                    "agent_name": "a",
                    "user_input": "q",
                    "history": [
                        {
                            "role": "user",
                            "content": "hi",
                            "parts": [["t", "hi"], ["i", "blob", SHA_A, "image/png", None]],
                        }
                    ],
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        assert b == expected


# ---------------------------------------------------------------------------
# Layering, events
# ---------------------------------------------------------------------------


def test_types_does_not_import_media() -> None:
    """``dendrux.types`` never imports ``dendrux.media`` (or anything else in dendrux).

    ``import dendrux.types`` runs ``dendrux/__init__.py`` first, which exports
    ``Image`` (and imports the agent stack), so the package is stubbed to load
    ``types.py`` on its own: this checks the module's own import graph.
    """
    code = (
        "import importlib.util, sys, types\n"
        "spec = importlib.util.find_spec('dendrux')\n"
        "pkg = types.ModuleType('dendrux')\n"
        "pkg.__path__ = list(spec.submodule_search_locations)\n"
        "sys.modules['dendrux'] = pkg\n"
        "import dendrux.types\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('dendrux.'))\n"
        "assert loaded == ['dendrux.types'], loaded\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_image_governance_event_members() -> None:
    assert GovernanceEventType.IMAGE_REJECTED == "image.rejected"
    assert GovernanceEventType.IMAGE_STATE == "image.state"
    assert GovernanceEventType.IMAGE_VIEW_CHANGED == "image.view_changed"
    assert GovernanceEventType.IMAGE_DEGRADED == "image.degraded"
    assert GovernanceEventType.IMAGE_WARNING == "image.warning"
    assert GovernanceEventType.GUARDRAIL_IMAGE_UNSCANNED == "guardrail.image_unscanned"


def test_top_level_exports() -> None:
    import dendrux
    from dendrux import media

    for name in ("Image", "ImageRef", "ImagePart", "TextPart", "ToolOutput"):
        assert name in dendrux.__all__ and hasattr(dendrux, name)
    assert dendrux.Image is media.Image


# ---------------------------------------------------------------------------
# Fix round (review-A): P1 memo retention, error pickling, small robustness items
# ---------------------------------------------------------------------------


def test_projection_memo_never_pins_wire_bytes() -> None:
    import gc
    import tracemalloc

    from dendrux import types as t

    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        messages = [
            Message.with_parts(
                Role.USER,
                [TextPart(f"q{i}"), ImagePart(ref(), data=bytes([i % 256]) * (4 * 1024 * 1024))],
            )
            for i in range(50)
        ]
        held, _ = tracemalloc.get_traced_memory()
        assert held - baseline > 150 * 1024 * 1024
        del messages
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert retained - baseline < 4 * 1024 * 1024  # less than one image survives
    assert not any(
        isinstance(p, ImagePart) and p.data is not None
        for parts, _ in t._PROJECTION_MEMO.values()
        for p in parts
    )


def test_semantic_parts_still_memoised() -> None:
    from dendrux import types as t

    parts = (TextPart("memo me"), img())
    Message(Role.USER, "memo me", parts=parts)
    assert t._PROJECTION_MEMO[id(parts)][0] is parts


def _all_image_errors() -> list[BaseException]:
    from dendrux import errors as e

    return [
        e.InvalidImageError("corrupt", index=1, media_type="image/png", byte_length=9, detail="d"),
        e.ImageLimitError("image_bytes", limit=1, actual=2, index=0),
        e.ImageFetchError("http_status", host="h.example", status_code=404),
        e.UnsupportedModalityError(
            provider="openai",
            model="gpt-3.5-turbo",
            image_count=2,
            origin="user",
            source="capability",
            status_code=400,
            provider_code="x",
        ),
        e.ImageRejectedError(
            provider="anthropic",
            model="claude-x",
            image_ids=("img_0123456789ab",),
            located=True,
            retried=True,
            status_code=400,
            provider_code="invalid_request_error",
        ),
        e.ProviderError(
            provider="openai",
            model="m",
            status_code=500,
            provider_code="c",
            error_type="t",
            message="boom",
        ),
        e.BlobNotFoundError("img_0123456789ab"),
        e.BlobIntegrityError("img_0123456789ab", "sha256"),
        e.ImagePersistenceUnsupportedError("state_store", store_type="Custom"),
        e.BlobContentionError("img_0123456789ab"),
    ]


@pytest.mark.parametrize("err", _all_image_errors(), ids=lambda x: type(x).__name__)
def test_image_errors_pickle_and_copy(err: BaseException) -> None:
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err), copy.deepcopy(err)):
        assert type(clone) is type(err)
        assert str(clone) == str(err)
        assert vars(clone) == vars(err)


def test_invalid_persisted_detail_is_descriptor_mismatch() -> None:
    d = content_part_to_dict(img())
    d["detail"] = "ultra"
    with pytest.raises(InvalidImageError) as ei:
        content_part_from_dict(d)
    assert ei.value.reason == "descriptor_mismatch"


def test_tool_result_rejects_string_sequences() -> None:
    with pytest.raises(TypeError):
        ToolResult(name="t", call_id="c", payload="1", images="abc")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ToolResult(name="t", call_id="c", payload="1", notes="note")  # type: ignore[arg-type]


def test_message_signature_shows_factory_for_meta() -> None:
    assert repr(inspect.signature(Message).parameters["meta"].default) == "<factory>"


def test_from_base64_non_ascii_bytes_has_no_chained_input() -> None:
    from dendrux.media import Image

    with pytest.raises(TypeError) as ei:
        Image.from_base64(b"\xff\xfe secret")  # type: ignore[arg-type]
    assert ei.value.__context__ is None and ei.value.__cause__ is None
