"""The one ingest pipeline: image input -> canonical ImagePart backed by a stored blob."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, overload

from dendrux.errors import (
    BlobContentionError,
    BlobNotFoundError,
    ImageError,
    ImageLimitError,
    ImagePersistenceUnsupportedError,
    InvalidImageError,
)
from dendrux.media._fetch import fetch_url
from dendrux.media._ids import image_id
from dendrux.media._image import Image, _norm_media_type
from dendrux.media._policy import ImageIngestRequest
from dendrux.media._read import check_bytes_source, decode_base64_bounded, read_path_bounded
from dendrux.media._sniff import ImageHeader, inspect_image, sniff_media_type
from dendrux.media._store import BlobInfo, BlobKey, BlobState, BlobStorage
from dendrux.types import ImageDetail, ImagePart, ImageRef, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from dendrux.media._catalog import BlobCatalog
    from dendrux.media._image import ImageInput
    from dendrux.media._policy import ImagePolicy, Origin
    from dendrux.media._runtime import ImageRuntime
    from dendrux.media._store import BlobStore

_MAX_PASSES = 3
# A pending row older than this belongs to a crashed ingest and is adopted.
PENDING_STALE_AFTER = timedelta(seconds=60)


@dataclass(frozen=True)
class _Validated:
    """Phase-1 output for one byte-backed input."""

    index: int
    data: bytes
    header: ImageHeader
    sha256: str
    transform: str | None
    detail: ImageDetail | None


def check_image_policy(
    policy: ImagePolicy,
    *,
    media_type: str,
    frame_count: int,
    byte_length: int,
    width: int,
    height: int,
    index: int | None = None,
) -> None:
    """Per-image policy checks on header/catalog metadata (type, animation, caps).

    Raises:
        InvalidImageError: ``"unsupported_type"`` or ``"animated"``.
        ImageLimitError: ``"image_bytes"``, ``"image_side"`` or ``"image_pixels"``.
    """
    if media_type not in policy.allowed_media_types:
        raise InvalidImageError(
            "unsupported_type", index=index, media_type=media_type, byte_length=byte_length
        )
    if frame_count > 1 and not policy.allow_animated:
        raise InvalidImageError(
            "animated", index=index, media_type=media_type, byte_length=byte_length
        )
    if byte_length > policy.max_image_bytes:
        raise ImageLimitError(
            "image_bytes", limit=policy.max_image_bytes, actual=byte_length, index=index
        )
    long_side = max(width, height)
    if long_side > policy.max_long_side:
        raise ImageLimitError(
            "image_side", limit=policy.max_long_side, actual=long_side, index=index
        )
    if width * height > policy.max_pixels:
        raise ImageLimitError(
            "image_pixels", limit=policy.max_pixels, actual=width * height, index=index
        )


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


async def _obtain(
    image: Image, policy: ImagePolicy, index: int, *, lenient: bool
) -> tuple[bytes, str | None]:
    """Bounded read of one byte source; returns (bytes, declared media type)."""
    declared = image.media_type
    if image.source == "bytes":
        assert image.data is not None
        data = check_bytes_source(image.data, policy.max_source_bytes, index=index)
    elif image.source == "path":
        assert image.path is not None
        data = await read_path_bounded(image.path, policy.max_source_bytes, index=index)
    elif image.source == "base64":
        assert image.text is not None
        data, url_type = decode_base64_bounded(
            image.text, policy.max_source_bytes, index=index, lenient=lenient
        )
        url_type = _norm_media_type(url_type)
        if url_type is not None:
            if declared is not None and declared != url_type and not lenient:
                raise InvalidImageError("type_mismatch", index=index, media_type=url_type)
            declared = declared or url_type
    else:
        assert image.url is not None
        data = await fetch_url(image.url, policy=policy.urls, max_bytes=policy.max_source_bytes)
        declared = None
    if not data:
        raise InvalidImageError("empty", index=index, byte_length=0)
    return data, declared


async def _run_hook(
    policy: ImagePolicy,
    data: bytes,
    declared: str | None,
    origin: Origin,
    tool_name: str | None,
    index: int,
) -> tuple[bytes, str | None]:
    hook = policy.ingest_hook
    if hook is None:
        return data, None
    out = hook(
        ImageIngestRequest(
            data=data,
            declared_media_type=declared,
            origin=origin,
            tool_name=tool_name,
            max_output_bytes=policy.max_image_bytes,
        )
    )
    if inspect.isawaitable(out):
        out = await out
    if out is None:
        return data, None
    if not isinstance(out, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"ImagePolicy.ingest_hook must return bytes or None, got {type(out).__name__}"
        )
    new = bytes(out)
    if len(new) > policy.max_image_bytes:
        raise ImageLimitError(
            "image_bytes", limit=policy.max_image_bytes, actual=len(new), index=index
        )
    if not new:
        raise InvalidImageError("empty", index=index, byte_length=0)
    return new, policy.ingest_hook_version


async def _validate_bytes(
    image: Image,
    *,
    policy: ImagePolicy,
    origin: Origin,
    tool_name: str | None,
    index: int,
    lenient: bool,
) -> _Validated:
    data, declared = await _obtain(image, policy, index, lenient=lenient)
    data, transform = await _run_hook(policy, data, declared, origin, tool_name, index)
    sniffed = sniff_media_type(data)
    if sniffed is None or sniffed not in policy.allowed_media_types:
        raise InvalidImageError(
            "unsupported_type", index=index, media_type=sniffed, byte_length=len(data)
        )
    # A hook may legitimately change the format (the declared type describes its
    # input), and lenient ingest trusts the sniffed type over the declared one.
    if not lenient and transform is None and declared is not None and declared != sniffed:
        raise InvalidImageError(
            "type_mismatch", index=index, media_type=sniffed, byte_length=len(data)
        )
    header = inspect_image(data, index=index)
    check_image_policy(
        policy,
        media_type=header.media_type,
        frame_count=header.frame_count,
        byte_length=len(data),
        width=header.width,
        height=header.height,
        index=index,
    )
    return _Validated(
        index=index,
        data=data,
        header=header,
        sha256=hashlib.sha256(data).hexdigest(),
        transform=transform,
        detail=image.detail,
    )


async def _lookup_refs(
    refs: list[tuple[int, ImageRef, ImageDetail | None]], *, catalog: BlobCatalog, scope: str
) -> tuple[dict[str, BlobInfo], dict[str, BlobInfo]]:
    shas = list(dict.fromkeys(ref.sha256 for _, ref, _ in refs))
    live = dict(await catalog.touch_live(scope, shas))
    missing = [sha for sha in shas if sha not in live]
    current = dict(await catalog.get_current(scope, missing)) if missing else {}
    return live, current


def _canonical_ref(
    index: int,
    ref: ImageRef,
    detail: ImageDetail | None,
    *,
    live: dict[str, BlobInfo],
    current: dict[str, BlobInfo],
    policy: ImagePolicy,
) -> ImagePart:
    row = live.get(ref.sha256)
    if row is None:
        cur = current.get(ref.sha256)
        if cur is not None and cur.state in ("purging", "purged"):
            raise InvalidImageError("purged", index=index, media_type=cur.media_type)
        if cur is None or cur.state != "live":
            raise BlobNotFoundError(image_id(ref.sha256))
        row = cur
    if (ref.media_type, ref.byte_length, ref.width, ref.height) != (
        row.media_type,
        row.byte_length,
        row.width,
        row.height,
    ):
        raise InvalidImageError("descriptor_mismatch", index=index)
    check_image_policy(
        policy,
        media_type=row.media_type,
        frame_count=row.frame_count,
        byte_length=row.byte_length,
        width=row.width,
        height=row.height,
        index=index,
    )
    return ImagePart(ref=row.to_ref(), detail=detail)


# ---------------------------------------------------------------------------
# Phase 2: blob lifecycle (one unique sha at a time)
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _new_info(
    item: _Validated, *, storage: BlobStorage, state: BlobState, generation: str
) -> BlobInfo:
    now = _now()
    return BlobInfo(
        sha256=item.sha256,
        media_type=item.header.media_type,
        byte_length=len(item.data),
        width=item.header.width,
        height=item.header.height,
        frame_count=item.header.frame_count,
        storage=storage,
        state=state,
        generation=generation,
        created_at=now,
        last_used_at=now,
        ingest_transform=item.transform,
        state_changed_at=now,
    )


def _is_stale_pending(row: BlobInfo) -> bool:
    changed = row.state_changed_at or row.last_used_at
    if changed.tzinfo is None:
        changed = changed.replace(tzinfo=UTC)
    return _now() - changed > PENDING_STALE_AFTER


def _require_store(runtime: ImageRuntime) -> BlobStore:
    if runtime.blob_store is None:
        raise ImagePersistenceUnsupportedError("blob_store_not_configured")
    return runtime.blob_store


async def _put_and_promote(
    runtime: ImageRuntime, catalog: BlobCatalog, item: _Validated, generation: str
) -> bool:
    """Put ``key(generation)`` then promote it; on a lost promote, compensate.

    Compensation: if the row no longer names ``generation`` in state
    ``pending|live`` (purged, adopted, revived or deleted meanwhile), the
    object this call wrote is deleted so it never outlives the row naming it.
    """
    scope, sha = runtime.tenant_scope, item.sha256
    store = _require_store(runtime)
    key = BlobKey(scope, sha, generation)
    await store.put(key, item.data, media_type=item.header.media_type)
    if await catalog.promote(scope, sha, generation):
        return True
    row = await catalog.get_row(scope, sha)
    if row is None or row.generation != generation or row.state not in ("pending", "live"):
        await store.delete(key)
    return False


async def _store_blob(runtime: ImageRuntime, catalog: BlobCatalog, item: _Validated) -> BlobInfo:
    """Reuse, create, finish/adopt a pending row or revive a deleting one."""
    scope, sha = runtime.tenant_scope, item.sha256
    external = runtime.blob_store is not None
    hit = await catalog.touch_live(scope, [sha])
    if sha in hit:
        return hit[sha]
    for _ in range(_MAX_PASSES):
        row = await catalog.get_row(scope, sha)
        if row is None:
            gen = generate_ulid()
            if not external:
                info = _new_info(item, storage="db", state="live", generation=gen)
                if await catalog.insert_row(
                    scope, sha, info, state="live", generation=gen, data=item.data
                ):
                    return info
                continue
            info = _new_info(item, storage="external", state="pending", generation=gen)
            if await catalog.insert_row(
                scope, sha, info, state="pending", generation=gen, data=None
            ) and await _put_and_promote(runtime, catalog, item, gen):
                return replace(info, state="live")
            continue
        if row.state == "live":
            hit = await catalog.touch_live(scope, [sha])
            if sha in hit:
                return hit[sha]
            continue
        if row.state in ("purging", "purged"):
            raise InvalidImageError("purged", index=item.index, media_type=row.media_type)
        if row.state == "pending":
            if not _is_stale_pending(row):
                # Another ingest is in flight: same bytes, same key (idempotent).
                if await _put_and_promote(runtime, catalog, item, row.generation):
                    return replace(row, state="live")
                continue
            # A crashed ingest: delete its object before the row stops naming it.
            store = _require_store(runtime)
            await store.delete(BlobKey(scope, sha, row.generation))
            gen = generate_ulid()
            if await catalog.adopt_pending(
                scope, sha, row.generation, gen
            ) and await _put_and_promote(runtime, catalog, item, gen):
                return _new_info(item, storage="external", state="live", generation=gen)
            continue
        if row.state == "deleting":
            if row.storage == "external":
                await _require_store(runtime).delete(BlobKey(scope, sha, row.generation))
            gen = generate_ulid()
            if not external:
                if await catalog.revive(
                    scope, sha, row.generation, gen, to_state="live", data=item.data
                ):
                    return _new_info(item, storage="db", state="live", generation=gen)
                continue
            if await catalog.revive(
                scope, sha, row.generation, gen, to_state="pending", data=None
            ) and await _put_and_promote(runtime, catalog, item, gen):
                return _new_info(item, storage="external", state="live", generation=gen)
            continue
    raise BlobContentionError(image_id(sha))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def _detach(err: ImageError) -> ImageError:
    # Returned (not raised) errors must not pin frames holding image bytes.
    err.__traceback__ = None
    err.__context__ = None
    err.__cause__ = None
    return err


@overload
async def ingest(
    inputs: Sequence[ImageInput],
    *,
    policy: ImagePolicy,
    runtime: ImageRuntime,
    origin: Origin,
    tool_name: str | None = ...,
    lenient: bool = ...,
    collect_errors: Literal[False] = ...,
    before_store: Callable[[], Awaitable[None]] | None = ...,
) -> tuple[ImagePart, ...]: ...


@overload
async def ingest(
    inputs: Sequence[ImageInput],
    *,
    policy: ImagePolicy,
    runtime: ImageRuntime,
    origin: Origin,
    tool_name: str | None = ...,
    lenient: bool = ...,
    collect_errors: Literal[True],
    before_store: Callable[[], Awaitable[None]] | None = ...,
) -> tuple[ImagePart | ImageError, ...]: ...


async def ingest(
    inputs: Sequence[ImageInput],
    *,
    policy: ImagePolicy,
    runtime: ImageRuntime,
    origin: Origin,
    tool_name: str | None = None,
    lenient: bool = False,
    collect_errors: bool = False,
    before_store: Callable[[], Awaitable[None]] | None = None,
) -> tuple[ImagePart, ...] | tuple[ImagePart | ImageError, ...]:
    """Validate, cap and store image inputs; return canonical parts in input order.

    Phase 1 (validation; no blob writes) runs for every input before phase 2
    (blob lifecycle). Caller-supplied ``ImageRef`` / ``ImagePart`` values are
    claims: they are checked against the tenant catalog and rebuilt from it.

    ``lenient`` (MCP results) strips ASCII whitespace from base64 and trusts the
    sniffed type over a declared one. ``collect_errors`` returns each failing
    input's :class:`~dendrux.errors.ImageError` in its position instead of
    raising; only successful inputs are stored. ``before_store`` runs after
    phase 1 succeeded and before any blob row or object is written (the entry
    points' capability check).

    Raises:
        InvalidImageError, ImageLimitError, ImageFetchError (unless collected),
        BlobNotFoundError, BlobContentionError, ImagePersistenceUnsupportedError.
    """
    if isinstance(inputs, (str, bytes, Image, ImageRef, ImagePart)):
        raise TypeError("ingest() takes a sequence of image inputs")
    items = tuple(inputs)
    for item in items:
        if not isinstance(item, (Image, ImageRef, ImagePart)):
            raise TypeError(
                f"image inputs must be Image, ImageRef or ImagePart, got {type(item).__name__}"
            )
    if not items:
        return ()
    if len(items) > policy.max_images_per_message:
        raise ImageLimitError(
            "message_count", limit=policy.max_images_per_message, actual=len(items)
        )
    catalog = runtime.catalog
    if catalog is None:
        raise ImagePersistenceUnsupportedError("state_store")

    results: dict[int, ImagePart | ImageError] = {}

    # Phase 1: validate every input; no blob rows written.
    refs: list[tuple[int, ImageRef, ImageDetail | None]] = []
    byte_items: list[tuple[int, Image]] = []
    for index, item in enumerate(items):
        if isinstance(item, ImagePart):
            refs.append((index, item.ref, item.detail))
        elif isinstance(item, ImageRef):
            refs.append((index, item, None))
        else:
            byte_items.append((index, item))
    if refs:
        live, current = await _lookup_refs(refs, catalog=catalog, scope=runtime.tenant_scope)
        for index, ref, detail in refs:
            try:
                results[index] = _canonical_ref(
                    index, ref, detail, live=live, current=current, policy=policy
                )
            except ImageError as exc:
                if not collect_errors:
                    raise
                results[index] = _detach(exc)
    validated: list[_Validated] = []
    for index, img in byte_items:
        try:
            validated.append(
                await _validate_bytes(
                    img,
                    policy=policy,
                    origin=origin,
                    tool_name=tool_name,
                    index=index,
                    lenient=lenient,
                )
            )
        except ImageError as exc:
            if not collect_errors:
                raise
            results[index] = _detach(exc)

    if before_store is not None:
        await before_store()

    # Phase 2: blob lifecycle per unique sha, in input order.
    stored: dict[str, BlobInfo | ImageError] = {}
    for v in validated:
        if v.sha256 not in stored:
            try:
                stored[v.sha256] = await _store_blob(runtime, catalog, v)
            except ImageError as exc:
                if not collect_errors:
                    raise
                stored[v.sha256] = _detach(exc)
        outcome = stored[v.sha256]
        results[v.index] = (
            outcome if isinstance(outcome, ImageError) else ImagePart(outcome.to_ref(), v.detail)
        )
    return tuple(results[i] for i in range(len(items)))
