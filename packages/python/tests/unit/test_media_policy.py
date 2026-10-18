"""Image inputs, policies, placeholders, token estimates and the in-memory blob store."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from dendrux.errors import ImageLimitError
from dendrux.media import (
    PLACEHOLDER_PRECEDENCE,
    PLACEHOLDERS,
    BlobKey,
    BlobStore,
    Image,
    ImageDropPolicy,
    ImagePolicy,
    ImageRuntime,
    InMemoryBlobStore,
    ToolOutput,
    UrlPolicy,
    estimate_image_tokens,
    image_id,
    placeholder,
    placeholder_rank,
    tenant_scope_for,
)
from dendrux.types import ImagePart, ImageRef

SHA = hashlib.sha256(b"x").hexdigest()


def ref(w: int = 400, h: int = 200) -> ImageRef:
    return ImageRef(SHA, "image/png", 100, w, h)


# ---------------------------------------------------------------------------
# Image / ToolOutput
# ---------------------------------------------------------------------------


class TestImage:
    def test_constructors(self) -> None:
        b = Image.from_bytes(bytearray(b"abc"), media_type="IMAGE/JPG; q=1", detail="low")
        assert (b.source, b.data, b.media_type, b.detail) == ("bytes", b"abc", "image/jpeg", "low")
        p = Image.from_path("/tmp/x.png")
        assert p.source == "path" and p.path == "/tmp/x.png"
        s = Image.from_base64(b"aGk=")  # type: ignore[arg-type]
        assert s.text == "aGk="
        u = Image.from_url("https://e.x/a.png", detail="high")
        assert u.source == "url" and u.media_type is None

    @pytest.mark.parametrize(
        "call",
        [
            lambda: Image.from_bytes("str"),  # type: ignore[arg-type]
            lambda: Image.from_base64(123),  # type: ignore[arg-type]
            lambda: Image.from_base64(b"\xff\xfe"),  # type: ignore[arg-type]
            lambda: Image.from_url(None),  # type: ignore[arg-type]
            lambda: Image.from_path(b"/x"),  # type: ignore[arg-type]
            lambda: Image.from_bytes(b"x", media_type=5),  # type: ignore[arg-type]
        ],
    )
    def test_type_errors(self, call: Any) -> None:
        with pytest.raises(TypeError):
            call()

    def test_value_errors(self) -> None:
        with pytest.raises(ValueError):
            Image.from_bytes(b"x", detail="max")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            Image.from_bytes(b"x", media_type=" ")
        with pytest.raises(ValueError):
            Image(source="bytes")
        with pytest.raises(ValueError):
            Image(source="bytes", data=b"x", url="https://x")
        with pytest.raises(ValueError):
            Image(source="file", data=b"x")  # type: ignore[arg-type]

    def test_repr(self) -> None:
        assert repr(Image.from_bytes(b"x" * 5321)) == (
            "Image(source='bytes', bytes=5321, media_type=None)"
        )
        assert repr(Image.from_path("/secret/dir/a.png", media_type="image/png")) == (
            "Image(source='path', media_type='image/png')"
        )
        assert repr(Image.from_base64("aGk=")) == "Image(source='base64', chars=4, media_type=None)"
        assert repr(Image.from_url("https://u:p@cdn.example/a?sig=1", detail="low")) == (
            "Image(source='url', host='cdn.example', detail='low')"
        )
        assert str(Image.from_url("::bad::")) == "Image(source='url', host=None)"

    def test_tool_output(self) -> None:
        r = ImageRef(SHA, "image/png", 1, 1, 1)
        out = ToolOutput("Captured.", images=[Image.from_bytes(b"x"), r, ImagePart(r)])
        assert isinstance(out.images, tuple) and len(out.images) == 3
        assert ToolOutput().result is None and ToolOutput().images == ()
        with pytest.raises(TypeError):
            ToolOutput("x", images=["not an image"])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            ToolOutput("x", images=Image.from_bytes(b"x"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_defaults(self) -> None:
        p = ImagePolicy()
        assert (p.max_image_bytes, p.max_source_bytes) == (5 * 1024 * 1024, 25 * 1024 * 1024)
        assert (p.max_long_side, p.max_pixels) == (8000, 64_000_000)
        assert (p.max_images_per_message, p.max_images_per_request) == (20, 100)
        assert p.max_request_image_bytes == 20 * 1024 * 1024
        assert p.allowed_media_types == ("image/png", "image/jpeg", "image/gif", "image/webp")
        assert p.allow_animated is False and p.on_unsupported == "error"
        assert p.unknown_capability == "send" and p.drop is None
        assert p.urls == UrlPolicy() and UrlPolicy().allowed_schemes == ("https",)

    @pytest.mark.parametrize(
        "kw",
        [
            {"max_image_bytes": 0},
            {"max_pixels": -1},
            {"max_images_per_request": True},
            {"max_estimated_image_tokens_per_request": 0},
            {"allowed_media_types": ("image/bmp",)},
            {"allowed_media_types": ()},
            {"allowed_media_types": "image/png"},
            {"on_unsupported": "drop"},
            {"unknown_capability": "maybe"},
            {"capability_overrides": {"gpt-x": "perhaps"}},
            {"drop": ImageDropPolicy(keep_recent=50, interval=60)},
            {"drop": "yes"},
            {"urls": "https"},
            {"allow_animated": 1},
            {"ingest_hook": lambda r: None},
            {"ingest_hook": lambda r: None, "ingest_hook_version": ""},
            {"ingest_hook": lambda r: None, "ingest_hook_version": "   "},
            {"ingest_hook": lambda r: None, "ingest_hook_version": " v1"},
            {"ingest_hook": lambda r: None, "ingest_hook_version": "v" * 101},
            {"ingest_hook": "not callable", "ingest_hook_version": "v1"},
        ],
    )
    def test_invalid(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ImagePolicy(**kw)

    def test_valid_options(self) -> None:
        p = ImagePolicy(
            allowed_media_types=["IMAGE/PNG"],  # type: ignore[arg-type]
            on_unsupported=lambda req: "desc",
            capability_overrides={"m": "yes"},
            drop=ImageDropPolicy(keep_recent=3, interval=25),
            ingest_hook=lambda r: None,
            ingest_hook_version="v" * 100,
        )
        assert p.allowed_media_types == ("image/png",)

    @pytest.mark.parametrize(
        "kw", [{"keep_recent": 0}, {"interval": 0}, {"scope": "user"}, {"keep_recent": 1.5}]
    )
    def test_drop_policy(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ImageDropPolicy(**kw)
        assert ImageDropPolicy().scope == "tool_results"


# ---------------------------------------------------------------------------
# Ids and placeholders (exact strings, §5.9)
# ---------------------------------------------------------------------------


def test_placeholders_exact() -> None:
    iid = image_id(SHA)
    assert iid == "img_" + SHA[:12]
    assert placeholder("drop", iid) == f"[image {iid} omitted from context]"
    assert placeholder("rejected", iid) == (
        f"[image {iid} was rejected by the model provider and is not shown]"
    )
    assert placeholder("unsupported", iid) == (
        f"[image {iid} not shown: this model does not accept images]"
    )
    assert placeholder("described", iid, "a red sign") == f"[image {iid} (described): a red sign]"
    assert placeholder("purged", iid) == f"[image {iid} was deleted]"
    assert placeholder("policy", iid) == (
        f"[image {iid} not shown: it exceeds the current image policy]"
    )
    assert placeholder("guardrail", iid) == f"[image {iid} removed by a guardrail]"
    assert set(PLACEHOLDERS) == {
        "drop",
        "rejected",
        "unsupported",
        "described",
        "purged",
        "policy",
        "guardrail",
    }
    with pytest.raises(ValueError):
        placeholder("described", iid)


def test_placeholder_precedence() -> None:
    order = ["purged", "policy", "unsupported", "rejected", "drop", "guardrail"]
    assert [placeholder_rank(r) for r in order] == sorted(placeholder_rank(r) for r in order)
    assert placeholder_rank("unsupported") == placeholder_rank("described")
    assert {r for group in PLACEHOLDER_PRECEDENCE for r in group} == set(PLACEHOLDERS)


# ---------------------------------------------------------------------------
# Token estimates (§5.13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "model", "w", "h", "detail", "expected"),
    [
        ("anthropic", "claude-sonnet-4-6", 400, 200, None, 120),
        # 4032x3024 -> 1568x1176 = 56*42 = 2352 > 1568 tokens -> scaled to fit the cap.
        ("anthropic", "claude-sonnet-4-5", 4032, 3024, None, 1564),
        ("anthropic", "claude-haiku-4-5", 1568, 1568, None, 39 * 39),
        ("anthropic", "claude-opus-4-7", 400, 200, None, 120),
        ("anthropic", "claude-opus-4-7", 4032, 3024, None, 4661),
        ("anthropic", "claude-sonnet-5", 2576, 1000, None, 92 * 36),
        ("openai", "gpt-4o", 400, 200, "low", 85),
        ("openai", "gpt-4o", 1024, 1024, "high", 85 + 170 * 4),
        ("openai", "gpt-4o", 4096, 8192, None, 85 + 170 * 6),
        ("openai", "gpt-4o-mini", 400, 200, None, 2833 + 5667),
        ("openai", "gpt-4.1", 512, 512, "auto", 85 + 170),
        ("openai", "gpt-5-mini", 400, 200, None, 110),
        ("openai", "gpt-4.1-mini", 400, 200, None, 148),
        ("openai", "gpt-5", 4032, 3024, None, None),
        ("openrouter", "anthropic/claude-sonnet-4.6", 400, 200, None, 120),
        ("openrouter", "~anthropic/claude-haiku-4.5", 400, 200, None, 120),
        ("openrouter", "openai/gpt-5-mini", 400, 200, None, 110),
        ("openrouter", "deepseek/deepseek-chat", 400, 200, None, None),
        ("openrouter", "noslash", 400, 200, None, None),
        ("openai", "o4-mini", 400, 200, None, None),
        ("anthropic", "not-claude", 400, 200, None, None),
        ("gemini", "gemini-2.5", 400, 200, None, None),
    ],
)
def test_estimate_image_tokens(
    family: str, model: str, w: int, h: int, detail: Any, expected: int | None
) -> None:
    got = estimate_image_tokens(ref(w, h), provider_family=family, model=model, detail=detail)
    if expected is None and model == "gpt-5":
        # Large images on patch models are scaled into the 1536-patch budget.
        assert got is not None and got <= int(1536 * 1.2) + 1
        return
    assert got == expected


def test_anthropic_cap_never_exceeded() -> None:
    for w, h in [(8000, 8000), (8000, 10), (1, 8000), (3000, 2999)]:
        got = estimate_image_tokens(ref(w, h), provider_family="anthropic", model="claude-x-4")
        assert got is not None and 0 < got <= 1568


# ---------------------------------------------------------------------------
# Blob store, keys, runtime
# ---------------------------------------------------------------------------


async def test_in_memory_blob_store() -> None:
    store = InMemoryBlobStore(max_bytes=10)
    assert isinstance(store, BlobStore)
    key = BlobKey("t:acme/1", SHA, "01G")
    assert key.object_name == f"t%3Aacme%2F1/{SHA}/01G"
    await store.put(key, b"12345", media_type="image/png")
    await store.put(key, b"123456", media_type="image/png")  # idempotent overwrite
    assert await store.get(key) == b"123456" and store.total_bytes == 6
    with pytest.raises(ImageLimitError) as ei:
        await store.put(BlobKey("-", SHA, "02G"), b"12345", media_type="image/png")
    assert ei.value.scope == "store_bytes" and ei.value.limit == 10
    await store.delete(key)
    await store.delete(key)
    assert await store.get(key) is None and store.total_bytes == 0
    assert [op for op, _ in store.calls] == ["put", "put", "get", "put", "delete", "delete", "get"]
    with pytest.raises(ValueError):
        InMemoryBlobStore(max_bytes=0)


def test_runtime_and_scope() -> None:
    assert tenant_scope_for(None) == "-" and tenant_scope_for("acme") == "t:acme"
    rt = ImageRuntime(ImagePolicy(), None, None, None, "-")
    assert rt.storage == "db" and rt.run_id is None
    assert ImageRuntime(ImagePolicy(), None, InMemoryBlobStore(), None, "-").storage == "external"
