"""The ingest pipeline: phase 1 validation, phase 2 blob lifecycle, lenient/collect modes."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest

from dendrux.errors import (
    BlobContentionError,
    BlobNotFoundError,
    ImageError,
    ImageFetchError,
    ImageLimitError,
    ImagePersistenceUnsupportedError,
    InvalidImageError,
)
from dendrux.media import (
    BlobInfo,
    BlobKey,
    Image,
    ImageIngestRequest,
    ImagePolicy,
    ImageRuntime,
    InMemoryBlobStore,
    UrlPolicy,
    ingest,
    tenant_scope_for,
)
from dendrux.types import ImagePart, ImageRef

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "images"
PNG = (FIXTURES / "zebra42.png").read_bytes()
KIWI = (FIXTURES / "kiwi77.png").read_bytes()
JPG = (FIXTURES / "zebra42.jpg").read_bytes()
GIF_ANIM = (FIXTURES / "animated.gif").read_bytes()
SHA = hashlib.sha256(PNG).hexdigest()


def _tiny_png() -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


TINY = _tiny_png()


class FakeCatalog:
    """In-memory BlobCatalog with the §3.5 conditional-statement semantics.

    ``hooks[name]`` runs before the named method executes (fault/race injection).
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], BlobInfo] = {}
        self.data: dict[tuple[str, str], bytes | None] = {}
        self.calls: list[str] = []
        self.hooks: dict[str, Callable[[], Any]] = {}

    async def _hook(self, name: str) -> None:
        self.calls.append(name)
        hook = self.hooks.pop(name, None)
        if hook is not None:
            result = hook()
            if hasattr(result, "__await__"):
                await result

    def _set(self, scope: str, sha: str, **kw: Any) -> None:
        now = datetime.now(UTC)
        self.rows[(scope, sha)] = replace(
            self.rows[(scope, sha)], last_used_at=now, state_changed_at=now, **kw
        )

    async def get_current(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]:
        await self._hook("get_current")
        return {
            s: self.rows[(scope, s)]
            for s in sha256s
            if (scope, s) in self.rows
            and self.rows[(scope, s)].state in ("live", "purging", "purged")
        }

    async def touch_live(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]:
        await self._hook("touch_live")
        out = {}
        for s in sha256s:
            row = self.rows.get((scope, s))
            if row is not None and row.state == "live":
                self.rows[(scope, s)] = replace(row, last_used_at=datetime.now(UTC))
                out[s] = self.rows[(scope, s)]
        return out

    async def get_row(self, scope: str, sha256: str) -> BlobInfo | None:
        await self._hook("get_row")
        return self.rows.get((scope, sha256))

    async def insert_row(
        self,
        scope: str,
        sha256: str,
        info: BlobInfo,
        *,
        state: Literal["pending", "live"],
        generation: str,
        data: bytes | None,
    ) -> bool:
        await self._hook("insert_row")
        if (scope, sha256) in self.rows:
            return False
        self.rows[(scope, sha256)] = replace(info, state=state, generation=generation)
        self.data[(scope, sha256)] = data
        return True

    async def promote(self, scope: str, sha256: str, generation: str) -> bool:
        await self._hook("promote")
        row = self.rows.get((scope, sha256))
        if row is None or row.state != "pending" or row.generation != generation:
            return False
        self._set(scope, sha256, state="live")
        return True

    async def adopt_pending(
        self, scope: str, sha256: str, old_generation: str, new_generation: str
    ) -> bool:
        await self._hook("adopt_pending")
        row = self.rows.get((scope, sha256))
        if row is None or row.state != "pending" or row.generation != old_generation:
            return False
        self._set(scope, sha256, generation=new_generation)
        return True

    async def revive(
        self,
        scope: str,
        sha256: str,
        old_generation: str,
        new_generation: str,
        *,
        to_state: Literal["pending", "live"],
        data: bytes | None,
    ) -> bool:
        await self._hook("revive")
        row = self.rows.get((scope, sha256))
        if row is None or row.state != "deleting" or row.generation != old_generation:
            return False
        storage = "db" if data is not None else "external"
        self._set(scope, sha256, state=to_state, generation=new_generation, storage=storage)
        self.data[(scope, sha256)] = data
        return True

    async def mark_purging(self, scope: str, sha256: str, reason: str | None) -> str | None:
        row = self.rows.get((scope, sha256))
        if row is None or row.state not in ("pending", "live", "deleting"):
            return None
        self._set(scope, sha256, state="purging", purge_reason=reason)
        self.data[(scope, sha256)] = None
        return row.generation

    async def mark_deleting(
        self, scope: str, sha256: str, cutoff: datetime, *, stale_pending_cutoff: datetime
    ) -> str | None:
        await self._hook("mark_deleting")
        row = self.rows.get((scope, sha256))
        if row is None:
            return None
        changed = row.state_changed_at or row.last_used_at
        if (row.state == "live" and row.last_used_at < cutoff) or (
            row.state == "pending" and changed < stale_pending_cutoff
        ):
            self._set(scope, sha256, state="deleting")
            return row.generation
        return None

    async def delete_row(self, scope: str, sha256: str, generation: str) -> bool:
        await self._hook("delete_row")
        row = self.rows.get((scope, sha256))
        if row is None or row.state != "deleting" or row.generation != generation:
            return False
        del self.rows[(scope, sha256)]
        self.data.pop((scope, sha256), None)
        return True

    # Unused by ingest; present so the fake satisfies the protocol shape.

    async def finish_purge(self, *a: Any, **k: Any) -> None:
        raise NotImplementedError

    async def stuck_rows(self, *a: Any, **k: Any) -> list[BlobInfo]:
        raise NotImplementedError

    async def read_blob_data(self, key: BlobKey) -> bytes | None:
        return self.data.get((key.tenant_scope, key.sha256))

    async def add_blob_refs(self, *a: Any, **k: Any) -> None:
        raise NotImplementedError

    async def run_references(self, *a: Any, **k: Any) -> bool:
        raise NotImplementedError

    async def blob_references(self, *a: Any, **k: Any) -> list[Any]:
        raise NotImplementedError

    async def delete_orphan_edges(self, *a: Any, **k: Any) -> int:
        raise NotImplementedError

    def seed(self, scope: str, data: bytes, *, state: str = "live", **kw: Any) -> BlobInfo:
        from dendrux.media import inspect_image

        h = inspect_image(data)
        now = datetime.now(UTC)
        info = BlobInfo(
            sha256=hashlib.sha256(data).hexdigest(),
            media_type=h.media_type,
            byte_length=len(data),
            width=h.width,
            height=h.height,
            frame_count=h.frame_count,
            storage=kw.pop("storage", "db"),
            state=state,  # type: ignore[arg-type]
            generation=kw.pop("generation", "01SEED"),
            created_at=now,
            last_used_at=now,
            state_changed_at=kw.pop("state_changed_at", now),
            **kw,
        )
        self.rows[(scope, info.sha256)] = info
        self.data[(scope, info.sha256)] = data if info.storage == "db" else None
        return info


def runtime(
    catalog: Any = None,
    *,
    policy: ImagePolicy | None = None,
    store: InMemoryBlobStore | None = None,
    tenant: str | None = "acme",
) -> ImageRuntime:
    return ImageRuntime(
        policy=policy or ImagePolicy(),
        catalog=FakeCatalog() if catalog is None else catalog,
        blob_store=store,
        resolver=None,
        tenant_scope=tenant_scope_for(tenant),
        run_id=None,
    )


async def run_ingest(inputs: Sequence[Any], rt: ImageRuntime, **kw: Any) -> tuple[Any, ...]:
    return await ingest(inputs, policy=rt.policy, runtime=rt, origin=kw.pop("origin", "user"), **kw)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


async def test_bytes_into_db_storage() -> None:
    cat = FakeCatalog()
    rt = runtime(cat)
    (part,) = await run_ingest([Image.from_bytes(PNG, detail="low")], rt)
    assert part == ImagePart(
        ImageRef(sha256=SHA, media_type="image/png", byte_length=len(PNG), width=400, height=200),
        detail="low",
    )
    row = cat.rows[("t:acme", SHA)]
    assert row.state == "live" and row.storage == "db" and row.frame_count == 1
    assert row.ingest_transform is None and len(row.generation) == 26
    assert cat.data[("t:acme", SHA)] == PNG


async def test_reuse_and_dedup_within_tenant_only() -> None:
    cat = FakeCatalog()
    a, b = await run_ingest([Image.from_bytes(PNG), Image.from_bytes(PNG)], runtime(cat))
    assert a == b and cat.calls.count("insert_row") == 1
    gen = cat.rows[("t:acme", SHA)].generation
    await run_ingest([Image.from_bytes(PNG)], runtime(cat))
    assert cat.calls.count("insert_row") == 1 and cat.rows[("t:acme", SHA)].generation == gen
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, tenant="other"))
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, tenant=None))
    assert set(cat.rows) == {("t:acme", SHA), ("t:other", SHA), ("-", SHA)}
    one_byte = PNG[:-1] + bytes([PNG[-1] ^ 1])
    assert hashlib.sha256(one_byte).hexdigest() != SHA


async def test_external_storage() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()
    (part,) = await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    row = cat.rows[("t:acme", SHA)]
    assert row.state == "live" and row.storage == "external"
    assert cat.data[("t:acme", SHA)] is None
    name = BlobKey("t:acme", SHA, row.generation).object_name
    assert store.calls == [("put", name)]
    assert await store.get(BlobKey("t:acme", SHA, row.generation)) == PNG
    assert part.ref.sha256 == SHA


async def test_all_sources(tmp_path: Path) -> None:
    p = tmp_path / "z.png"
    p.write_bytes(PNG)

    async def fetcher(url: str, max_bytes: int) -> bytes:
        return KIWI

    policy = ImagePolicy(urls=UrlPolicy(fetcher=fetcher))
    rt = runtime(policy=policy)
    parts = await run_ingest(
        [
            Image.from_path(p),
            Image.from_base64(base64.b64encode(PNG).decode()),
            Image.from_base64("data:image/png;base64," + base64.b64encode(PNG).decode()),
            Image.from_url("https://cdn.example/k.png"),
            Image.from_bytes(JPG, media_type="image/jpg"),
        ],
        rt,
    )
    assert [x.ref.sha256 for x in parts[:3]] == [SHA] * 3
    assert parts[3].ref.sha256 == hashlib.sha256(KIWI).hexdigest()
    assert parts[4].ref.media_type == "image/jpeg"


async def test_large_phone_photo_accepted_with_defaults() -> None:
    photo = (FIXTURES / "photo_4032x3024.jpg").read_bytes()
    (part,) = await run_ingest([Image.from_bytes(photo)], runtime())
    assert (part.ref.width, part.ref.height) == (4032, 3024)


async def test_empty_inputs_and_type_errors() -> None:
    assert await run_ingest([], runtime()) == ()
    with pytest.raises(TypeError):
        await run_ingest(Image.from_bytes(PNG), runtime())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await run_ingest(["text"], runtime())


async def test_no_catalog() -> None:
    rt = ImageRuntime(
        policy=ImagePolicy(), catalog=None, blob_store=None, resolver=None, tenant_scope="-"
    )
    with pytest.raises(ImagePersistenceUnsupportedError) as ei:
        await run_ingest([Image.from_bytes(PNG)], rt)
    assert ei.value.reason == "state_store"


# ---------------------------------------------------------------------------
# Refs are claims
# ---------------------------------------------------------------------------


async def test_refs_canonicalised_from_catalog() -> None:
    cat = FakeCatalog()
    info = cat.seed("t:acme", PNG)
    ref = info.to_ref()
    a, b = await run_ingest([ref, ImagePart(ref, detail="high", data=b"ignored")], runtime(cat))
    assert a == ImagePart(ref) and b == ImagePart(ref, detail="high")
    assert b.data is None
    assert "insert_row" not in cat.calls


@pytest.mark.parametrize(
    "field", [{"media_type": "image/jpeg"}, {"byte_length": 1}, {"width": 1}, {"height": 1}]
)
async def test_descriptor_mismatch(field: dict[str, Any]) -> None:
    cat = FakeCatalog()
    ref = replace(cat.seed("t:acme", PNG).to_ref(), **field)
    with pytest.raises(InvalidImageError) as ei:
        await run_ingest([ref], runtime(cat))
    assert ei.value.reason == "descriptor_mismatch"


async def test_ref_absent_or_other_tenant_is_not_found() -> None:
    cat = FakeCatalog()
    ref = cat.seed("t:other", PNG).to_ref()
    with pytest.raises(BlobNotFoundError) as e1:
        await run_ingest([ref], runtime(cat))
    cat2 = FakeCatalog()
    with pytest.raises(BlobNotFoundError) as e2:
        await run_ingest([ref], runtime(cat2))
    assert str(e1.value) == str(e2.value)


@pytest.mark.parametrize("state", ["pending", "deleting"])
async def test_ref_to_invisible_row_is_not_found(state: str) -> None:
    cat = FakeCatalog()
    ref = cat.seed("t:acme", PNG, state=state).to_ref()
    with pytest.raises(BlobNotFoundError):
        await run_ingest([ref], runtime(cat))


@pytest.mark.parametrize("state", ["purging", "purged"])
async def test_ref_to_purged(state: str) -> None:
    cat = FakeCatalog()
    ref = cat.seed("t:acme", PNG, state=state).to_ref()
    with pytest.raises(InvalidImageError) as ei:
        await run_ingest([ref], runtime(cat))
    assert ei.value.reason == "purged"


async def test_ref_rechecked_against_current_policy() -> None:
    cat = FakeCatalog()
    ref = cat.seed("t:acme", PNG).to_ref()
    with pytest.raises(ImageLimitError) as ei:
        await run_ingest([ref], runtime(cat, policy=ImagePolicy(max_long_side=300)))
    assert ei.value.scope == "image_side"


# ---------------------------------------------------------------------------
# Validation and caps (U-10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "policy", "kind", "value"),
    [
        (Image.from_bytes(b""), None, "reason", "empty"),
        (Image.from_bytes(b"BM" + b"\0" * 64), None, "reason", "unsupported_type"),
        (
            Image.from_bytes(JPG),
            ImagePolicy(allowed_media_types=("image/png",)),
            "reason",
            "unsupported_type",
        ),
        (Image.from_bytes(PNG, media_type="image/jpeg"), None, "reason", "type_mismatch"),
        (
            Image.from_base64("data:image/gif;base64," + base64.b64encode(PNG).decode()),
            None,
            "reason",
            "type_mismatch",
        ),
        (
            Image.from_base64(
                "data:image/png;base64," + base64.b64encode(PNG).decode(), media_type="image/jpeg"
            ),
            None,
            "reason",
            "type_mismatch",
        ),
        (Image.from_bytes(PNG[:-12]), None, "reason", "corrupt"),
        (Image.from_bytes(GIF_ANIM), None, "reason", "animated"),
        (Image.from_base64("aGV sbG8="), None, "reason", "bad_base64"),
        (Image.from_path("/definitely/missing.png"), None, "reason", "path"),
        (Image.from_bytes(PNG), ImagePolicy(max_image_bytes=100), "scope", "image_bytes"),
        (Image.from_bytes(PNG), ImagePolicy(max_source_bytes=100), "scope", "source_bytes"),
        (Image.from_bytes(PNG), ImagePolicy(max_long_side=399), "scope", "image_side"),
        (Image.from_bytes(PNG), ImagePolicy(max_pixels=400 * 200 - 1), "scope", "image_pixels"),
    ],
)
async def test_phase1_rejections(
    image: Image, policy: ImagePolicy | None, kind: str, value: str
) -> None:
    cat = FakeCatalog()
    with pytest.raises(ImageError) as ei:
        await run_ingest([Image.from_bytes(TINY), image], runtime(cat, policy=policy))
    assert getattr(ei.value, kind) == value
    assert getattr(ei.value, "index", 1) == 1
    assert ei.value.__context__ is None and ei.value.__cause__ is None
    assert cat.rows == {}  # caps before storage: the valid first input was not stored


async def test_animated_allowed() -> None:
    cat = FakeCatalog()
    await run_ingest(
        [Image.from_bytes(GIF_ANIM)], runtime(cat, policy=ImagePolicy(allow_animated=True))
    )
    (row,) = cat.rows.values()
    assert row.frame_count == 2


async def test_message_count_checked_before_reading() -> None:
    policy = ImagePolicy(max_images_per_message=2)
    inputs = [Image.from_path(f"/missing/{i}.png") for i in range(3)]
    with pytest.raises(ImageLimitError) as ei:
        await run_ingest(inputs, runtime(policy=policy))
    assert (ei.value.scope, ei.value.limit, ei.value.actual) == ("message_count", 2, 3)


async def test_url_fetch_error_propagates() -> None:
    with pytest.raises(ImageFetchError) as ei:
        await run_ingest(
            [Image.from_url("https://x.example/a.png")],
            runtime(policy=ImagePolicy(urls=UrlPolicy(fetch=False))),
        )
    assert ei.value.reason == "disabled"


# ---------------------------------------------------------------------------
# Ingest hook
# ---------------------------------------------------------------------------


async def test_hook_replaces_bytes_and_records_version() -> None:
    seen: list[ImageIngestRequest] = []

    def hook(req: ImageIngestRequest) -> bytes:
        seen.append(req)
        return JPG

    cat = FakeCatalog()
    policy = ImagePolicy(ingest_hook=hook, ingest_hook_version="to-jpeg-1")
    (part,) = await run_ingest(
        [Image.from_bytes(PNG, media_type="image/png")],
        runtime(cat, policy=policy),
        origin="tool",
        tool_name="shot",
    )
    assert (
        part.ref.media_type == "image/jpeg" and part.ref.sha256 == hashlib.sha256(JPG).hexdigest()
    )
    assert cat.rows[("t:acme", part.ref.sha256)].ingest_transform == "to-jpeg-1"
    req = seen[0]
    assert (req.declared_media_type, req.origin, req.tool_name) == ("image/png", "tool", "shot")
    assert req.max_output_bytes == policy.max_image_bytes and req.data == PNG
    assert "\\x89" not in repr(req) and "data=" not in repr(req)


async def test_async_hook_none_keeps_bytes() -> None:
    async def hook(req: ImageIngestRequest) -> None:
        return None

    cat = FakeCatalog()
    await run_ingest(
        [Image.from_bytes(PNG)],
        runtime(cat, policy=ImagePolicy(ingest_hook=hook, ingest_hook_version="noop")),
    )
    assert cat.rows[("t:acme", SHA)].ingest_transform is None


async def test_hook_output_checks() -> None:
    big = ImagePolicy(
        max_image_bytes=len(PNG),
        ingest_hook=lambda r: b"x" * (len(PNG) + 1),
        ingest_hook_version="v",
    )
    with pytest.raises(ImageLimitError, match="image_bytes"):
        await run_ingest([Image.from_bytes(PNG)], runtime(policy=big))
    bad = ImagePolicy(ingest_hook=lambda r: "nope", ingest_hook_version="v")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError):
        await run_ingest([Image.from_bytes(PNG)], runtime(policy=bad))
    empty = ImagePolicy(ingest_hook=lambda r: b"", ingest_hook_version="v")
    with pytest.raises(InvalidImageError, match="empty"):
        await run_ingest([Image.from_bytes(PNG)], runtime(policy=empty))


# ---------------------------------------------------------------------------
# Lenient / collect_errors / before_store
# ---------------------------------------------------------------------------


async def test_lenient_collect_errors() -> None:
    cat = FakeCatalog()
    wrapped = base64.encodebytes(PNG).decode()
    results = await run_ingest(
        [
            Image.from_base64(wrapped, media_type="image/jpeg"),  # MIME + wrong declared type
            Image.from_base64("!!!!"),
            Image.from_bytes(GIF_ANIM),
            Image.from_base64(base64.b64encode(KIWI).decode()),
        ],
        runtime(cat),
        origin="mcp",
        lenient=True,
        collect_errors=True,
    )
    assert isinstance(results[0], ImagePart) and results[0].ref.media_type == "image/png"
    assert isinstance(results[1], InvalidImageError) and results[1].reason == "bad_base64"
    assert isinstance(results[2], InvalidImageError) and results[2].reason == "animated"
    assert results[1].__traceback__ is None and results[1].__context__ is None
    assert isinstance(results[3], ImagePart)
    assert len(cat.rows) == 2


async def test_strict_mode_raises_first_error() -> None:
    with pytest.raises(InvalidImageError, match="bad_base64"):
        await run_ingest([Image.from_base64("!!!!"), Image.from_bytes(GIF_ANIM)], runtime())


async def test_collect_errors_phase2_purged_in_position() -> None:
    cat = FakeCatalog()
    cat.seed("t:acme", PNG, state="purged")
    results = await run_ingest(
        [Image.from_bytes(KIWI), Image.from_bytes(PNG)], runtime(cat), collect_errors=True
    )
    assert isinstance(results[0], ImagePart)
    assert isinstance(results[1], InvalidImageError) and results[1].reason == "purged"


async def test_before_store_runs_between_phases() -> None:
    cat = FakeCatalog()
    order: list[str] = []

    async def check() -> None:
        order.append(f"check rows={len(cat.rows)}")
        raise RuntimeError("model has no vision")

    with pytest.raises(RuntimeError, match="no vision"):
        await run_ingest([Image.from_bytes(PNG)], runtime(cat), before_store=check)
    assert order == ["check rows=0"] and cat.rows == {}

    called: list[int] = []

    async def never() -> None:
        called.append(1)

    with pytest.raises(InvalidImageError):
        await run_ingest([Image.from_bytes(b"x")], runtime(cat), before_store=never)
    assert called == []


# ---------------------------------------------------------------------------
# Phase 2 lifecycle branches (§5.2 step 9)
# ---------------------------------------------------------------------------


async def test_fresh_pending_finished_with_same_generation() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()
    cat.seed("t:acme", PNG, state="pending", storage="external", generation="01GPEND")
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    row = cat.rows[("t:acme", SHA)]
    assert row.state == "live" and row.generation == "01GPEND"
    assert store.object_names() == [BlobKey("t:acme", SHA, "01GPEND").object_name]


async def test_stale_pending_adopted_after_deleting_old_object() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()
    old = BlobKey("t:acme", SHA, "01GOLD")
    await store.put(old, PNG, media_type="image/png")
    cat.seed(
        "t:acme",
        PNG,
        state="pending",
        storage="external",
        generation="01GOLD",
        state_changed_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    row = cat.rows[("t:acme", SHA)]
    assert row.state == "live" and row.generation != "01GOLD"
    new = BlobKey("t:acme", SHA, row.generation)
    assert store.calls[1:] == [("delete", old.object_name), ("put", new.object_name)]
    assert "adopt_pending" in cat.calls
    assert store.object_names() == [BlobKey("t:acme", SHA, row.generation).object_name]


async def test_pending_without_blob_store() -> None:
    cat = FakeCatalog()
    cat.seed("t:acme", PNG, state="pending", storage="external")
    with pytest.raises(ImagePersistenceUnsupportedError) as ei:
        await run_ingest([Image.from_bytes(PNG)], runtime(cat))
    assert ei.value.reason == "blob_store_not_configured"


async def test_deleting_external_revived_with_new_generation() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()
    old = BlobKey("t:acme", SHA, "01GDEL")
    await store.put(old, PNG, media_type="image/png")
    cat.seed("t:acme", PNG, state="deleting", storage="external", generation="01GDEL")
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    row = cat.rows[("t:acme", SHA)]
    assert row.state == "live" and row.generation != "01GDEL"
    assert store.object_names() == [BlobKey("t:acme", SHA, row.generation).object_name]
    assert store.calls[1] == ("delete", old.object_name)  # old object removed before revive


async def test_deleting_db_revived_live() -> None:
    cat = FakeCatalog()
    cat.seed("t:acme", PNG, state="deleting", generation="01GDEL")
    cat.data[("t:acme", SHA)] = None
    await run_ingest([Image.from_bytes(PNG)], runtime(cat))
    row = cat.rows[("t:acme", SHA)]
    assert row.state == "live" and row.generation != "01GDEL" and row.storage == "db"
    assert cat.data[("t:acme", SHA)] == PNG


@pytest.mark.parametrize("state", ["purging", "purged"])
async def test_reingest_of_purged_hash_rejected(state: str) -> None:
    cat = FakeCatalog()
    cat.seed("t:acme", PNG, state=state)
    with pytest.raises(InvalidImageError) as ei:
        await run_ingest([Image.from_bytes(PNG)], runtime(cat))
    assert ei.value.reason == "purged"


async def test_purge_between_put_and_promote_wins_and_object_is_compensated() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()

    async def purge() -> None:
        await cat.mark_purging("t:acme", SHA, "erasure")

    cat.hooks["promote"] = purge
    with pytest.raises(InvalidImageError) as ei:
        await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    assert ei.value.reason == "purged"
    assert store.object_names() == []  # our object was deleted
    assert cat.rows[("t:acme", SHA)].state == "purging"


async def test_concurrent_promote_by_other_ingest_keeps_object() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()

    async def other_promotes() -> None:
        row = cat.rows[("t:acme", SHA)]
        cat._set("t:acme", SHA, state="live", generation=row.generation)

    cat.hooks["promote"] = other_promotes
    (part,) = await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    row = cat.rows[("t:acme", SHA)]
    assert part.ref.sha256 == SHA and row.state == "live"
    assert store.object_names() == [BlobKey("t:acme", SHA, row.generation).object_name]


async def test_lost_insert_race_reuses_winner() -> None:
    cat = FakeCatalog()

    def other_inserts() -> None:
        cat.seed("t:acme", PNG)

    cat.hooks["insert_row"] = other_inserts
    (part,) = await run_ingest([Image.from_bytes(PNG)], runtime(cat))
    assert part.ref.sha256 == SHA
    assert cat.rows[("t:acme", SHA)].generation == "01SEED"


async def test_contention_exhausted() -> None:
    class AlwaysLoses(FakeCatalog):
        async def insert_row(self, *a: Any, **k: Any) -> bool:
            return False

    with pytest.raises(BlobContentionError) as ei:
        await run_ingest([Image.from_bytes(PNG)], runtime(AlwaysLoses()))
    assert ei.value.image_id == "img_" + SHA[:12]


# ---------------------------------------------------------------------------
# Determinism across processes (U-02) and leak scan (S-06)
# ---------------------------------------------------------------------------

_DETERMINISM_SNIPPET = """
import asyncio, base64, hashlib, json, sys
sys.path.insert(0, {root!r})
from tests.unit.test_media_ingest import FakeCatalog, runtime, run_ingest, FIXTURES
from dendrux.media import Image

async def main():
    out = {{}}
    names = ["zebra42.png", "zebra42.jpg", "zebra42.gif", "zebra42.webp"]
    for name in names + ["zebra42_lossless.webp"]:
        data = (FIXTURES / name).read_bytes()
        (part,) = await run_ingest([Image.from_bytes(data)], runtime(FakeCatalog()))
        b64 = base64.b64encode(data).decode("ascii")
        out[name] = [part.ref.to_dict(), hashlib.sha256(b64.encode()).hexdigest()]
    print(json.dumps(out, sort_keys=True))

asyncio.run(main())
"""


def test_two_process_determinism() -> None:
    root = str(Path(__file__).resolve().parents[2])
    code = _DETERMINISM_SNIPPET.format(root=root)
    runs = [
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        for _ in range(2)
    ]
    first, second = (json.loads(r.stdout) for r in runs)
    assert first == second
    zebra = first["zebra42.png"][0]
    assert zebra == {
        "type": "image",
        "v": 1,
        "kind": "blob",
        "sha256": SHA,
        "media_type": "image/png",
        "bytes": len(PNG),
        "width": 400,
        "height": 200,
    }
    assert {v[0]["media_type"] for v in first.values()} == {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }


async def test_repr_and_error_leak_scan(tmp_path: Path) -> None:
    from dendrux.media import ToolOutput

    secret_path = tmp_path / "secret-dir" / "z.png"
    secret_path.parent.mkdir()
    secret_path.write_bytes(PNG)
    b64 = base64.b64encode(PNG).decode()
    objects: list[Any] = [
        Image.from_bytes(PNG),
        Image.from_path(secret_path),
        Image.from_base64(b64),
        Image.from_base64("data:image/png;base64," + b64),
        Image.from_url("https://user:pw@cdn.example/a.png?sig=TOPSECRET"),
        ToolOutput("ok", images=[Image.from_bytes(PNG)]),
        ImagePart(ImageRef(SHA, "image/png", len(PNG), 400, 200), data=PNG),
        ImageIngestRequest(
            data=PNG, declared_media_type=None, origin="user", tool_name=None, max_output_bytes=1
        ),
    ]
    errors: list[BaseException] = []
    for bad in (
        Image.from_bytes(PNG, media_type="image/gif"),
        Image.from_path(tmp_path / "secret-dir" / "missing.png"),
        Image.from_base64("data:image/png;base64," + b64[:-1] + "*"),
        Image.from_bytes(GIF_ANIM),
    ):
        try:
            await run_ingest([bad], runtime())
        except ImageError as exc:
            errors.append(exc)
    try:
        await run_ingest(
            [Image.from_url("https://user:pw@cdn.example/a.png?sig=TOPSECRET")],
            runtime(policy=ImagePolicy(urls=UrlPolicy(allowed_hosts=("other.example",)))),
        )
    except ImageError as exc:
        errors.append(exc)
    assert len(errors) == 5
    texts = [repr(o) for o in objects] + [str(o) for o in objects]
    texts += [str(e) for e in errors] + [repr(e) for e in errors]
    texts += [repr(vars(e)) for e in errors]
    forbidden = [
        b64[:40],
        "iVBORw0K",
        "data:image",
        "secret-dir",
        str(tmp_path),
        "TOPSECRET",
        "pw@",
        "\\x89PNG",
        PNG[:16].decode("latin-1"),
    ]
    for text in texts:
        for f in forbidden:
            assert f not in text, (f, text)
    for e in errors:
        assert e.__cause__ is None and e.__context__ is None


# ---------------------------------------------------------------------------
# Fix round (review-A): P2-1 .. P2-8
# ---------------------------------------------------------------------------


async def test_byte_cap_checked_before_parse_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from dendrux.media import _ingest

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("parsed an over-cap input")

    monkeypatch.setattr(_ingest, "_parse_and_hash", boom)
    monkeypatch.setattr(_ingest, "sniff_media_type", boom)
    with pytest.raises(ImageLimitError) as ei:
        await run_ingest(
            [Image.from_bytes(b"\xff\xd8\xff" + b"\xff\x00" * 600)],
            runtime(policy=ImagePolicy(max_image_bytes=1000)),
        )
    assert ei.value.scope == "image_bytes"


async def test_large_inputs_parsed_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from dendrux.media import _ingest

    threads: list[int] = []
    real = _ingest._parse_and_hash

    def spy(data: bytes, index: int) -> Any:
        threads.append(threading.get_ident())
        return real(data, index)

    monkeypatch.setattr(_ingest, "_parse_and_hash", spy)
    monkeypatch.setattr(_ingest, "_THREAD_THRESHOLD", 2000)
    assert len(PNG) < 2000 < len(JPG)
    await run_ingest([Image.from_bytes(PNG), Image.from_bytes(JPG)], runtime())
    main = threading.get_ident()
    assert threads[0] == main and threads[1] != main


async def test_deleting_revive_lost_to_gc_revert_restores_object() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()
    old = BlobKey("t:acme", SHA, "01GDEL")
    await store.put(old, PNG, media_type="image/png")
    cat.seed("t:acme", PNG, state="deleting", storage="external", generation="01GDEL")

    def gc_reverts() -> None:  # GC's edge re-check: deleting -> live, same generation
        cat._set("t:acme", SHA, state="live")

    cat.hooks["revive"] = gc_reverts
    (part,) = await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    row = cat.rows[("t:acme", SHA)]
    assert part.ref.sha256 == SHA and row.state == "live" and row.generation == "01GDEL"
    assert await store.get(old) == PNG  # the object we deleted was put back


async def test_stale_pending_adopt_lost_to_promote_restores_object() -> None:
    cat, store = FakeCatalog(), InMemoryBlobStore()
    old = BlobKey("t:acme", SHA, "01GOLD")
    await store.put(old, PNG, media_type="image/png")
    cat.seed(
        "t:acme",
        PNG,
        state="pending",
        storage="external",
        generation="01GOLD",
        state_changed_at=datetime.now(UTC) - timedelta(minutes=5),
    )

    def slow_ingest_promotes() -> None:
        cat._set("t:acme", SHA, state="live")

    cat.hooks["adopt_pending"] = slow_ingest_promotes
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    assert cat.rows[("t:acme", SHA)].generation == "01GOLD"
    assert await store.get(old) == PNG


async def test_lenient_collect_count_cap_per_input() -> None:
    cat = FakeCatalog()
    policy = ImagePolicy(max_images_per_message=2)
    images = [Image.from_bytes(PNG), Image.from_bytes(KIWI), Image.from_bytes(TINY)]
    results = await run_ingest(
        images, runtime(cat, policy=policy), lenient=True, collect_errors=True
    )
    assert isinstance(results[0], ImagePart) and isinstance(results[1], ImagePart)
    err = results[2]
    assert isinstance(err, ImageLimitError)
    assert (err.scope, err.limit, err.actual, err.index) == ("message_count", 2, 3, 2)
    assert len(cat.rows) == 2
    with pytest.raises(ImageLimitError):
        await run_ingest(images, runtime(policy=policy), collect_errors=True)
    with pytest.raises(ImageLimitError):
        await run_ingest(images, runtime(policy=policy), lenient=True)


@pytest.mark.parametrize("declared", ["", ";", "application/octet-stream", "image/jpeg", "png"])
async def test_lenient_trusts_sniff_over_odd_declared_type(declared: str) -> None:
    b64 = base64.b64encode(PNG).decode()
    (part,) = await run_ingest(
        [Image.from_base64(b64, media_type=declared)], runtime(), lenient=True, collect_errors=True
    )
    assert isinstance(part, ImagePart) and part.ref.media_type == "image/png"


class _FailingStore(InMemoryBlobStore):
    def __init__(self, fail_times: int, exc: Exception) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.exc = exc

    async def put(self, key: BlobKey, data: bytes, *, media_type: str) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            self.calls.append(("put-failed", key.object_name))
            raise self.exc
        await super().put(key, data, media_type=media_type)


async def test_put_failure_removes_own_pending_row() -> None:
    cat = FakeCatalog()
    store = _FailingStore(1, ConnectionError("s3 down"))
    with pytest.raises(ConnectionError):
        await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    assert cat.rows == {} and store.object_names() == []
    assert "mark_deleting" in cat.calls and "delete_row" in cat.calls


async def test_store_full_runs_one_gc_pass_then_retries() -> None:
    cat = FakeCatalog()
    full = ImageLimitError("store_bytes", limit=10, actual=20)
    store = _FailingStore(1, full)
    gc_calls: list[int] = []

    async def collect() -> None:
        gc_calls.append(1)

    rt = replace(runtime(cat, store=store), collect_garbage=collect)
    (part,) = await run_ingest([Image.from_bytes(PNG)], rt)
    assert gc_calls == [1] and part.ref.sha256 == SHA
    assert cat.rows[("t:acme", SHA)].state == "live"

    still_full = _FailingStore(2, ImageLimitError("store_bytes", limit=10, actual=20))
    cat2 = FakeCatalog()
    rt2 = replace(runtime(cat2, store=still_full), collect_garbage=collect)
    with pytest.raises(ImageLimitError) as ei:
        await run_ingest([Image.from_bytes(PNG)], rt2)
    assert ei.value.scope == "store_bytes" and cat2.rows == {}

    cat3 = FakeCatalog()
    with pytest.raises(ImageLimitError):  # no GC wired: raise immediately
        await run_ingest(
            [Image.from_bytes(PNG)],
            runtime(
                cat3, store=_FailingStore(1, ImageLimitError("store_bytes", limit=1, actual=2))
            ),
        )
    assert cat3.rows == {}


async def test_stale_pending_uses_catalog_clock() -> None:
    class SkewedCatalog(FakeCatalog):
        """DB clock 5 minutes behind the host: the pending row is fresh by the DB clock."""

        async def current_time(self) -> datetime:
            return datetime.now(UTC) - timedelta(minutes=5)

    cat, store = SkewedCatalog(), InMemoryBlobStore()
    db_now = datetime.now(UTC) - timedelta(minutes=5)
    cat.seed(
        "t:acme",
        PNG,
        state="pending",
        storage="external",
        generation="01GP",
        state_changed_at=db_now,
    )
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, store=store))
    assert "adopt_pending" not in cat.calls
    assert cat.rows[("t:acme", SHA)].generation == "01GP"


async def test_validated_repr_has_no_bytes() -> None:
    from dendrux.media._ingest import _validate_bytes

    v = await _validate_bytes(
        Image.from_bytes(PNG),
        policy=ImagePolicy(),
        origin="user",
        tool_name=None,
        index=0,
        lenient=False,
    )
    assert "data=" not in repr(v) and "\\x89" not in repr(v)


async def test_identity_hook_is_not_a_transform() -> None:
    policy = ImagePolicy(ingest_hook=lambda req: bytes(req.data), ingest_hook_version="v1")
    cat = FakeCatalog()
    await run_ingest([Image.from_bytes(PNG)], runtime(cat, policy=policy))
    assert cat.rows[("t:acme", SHA)].ingest_transform is None
    with pytest.raises(InvalidImageError) as ei:
        await run_ingest([Image.from_bytes(PNG, media_type="image/jpeg")], runtime(policy=policy))
    assert ei.value.reason == "type_mismatch"
