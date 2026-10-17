"""The one ingest pipeline: image input -> canonical ImagePart backed by a stored blob."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.media._image import ImageInput
    from dendrux.media._policy import ImagePolicy, Origin
    from dendrux.media._runtime import ImageRuntime
    from dendrux.types import ImagePart


async def ingest(
    inputs: Sequence[ImageInput],
    *,
    policy: ImagePolicy,
    runtime: ImageRuntime,
    origin: Origin,
    tool_name: str | None = None,
) -> tuple[ImagePart, ...]:
    """Validate, cap and store image inputs; return canonical parts in input order.

    Phase 1 (validation, no blob writes) runs for every input before phase 2
    (blob lifecycle). Caller-supplied ``ImageRef`` / ``ImagePart`` values are
    checked against the tenant catalog and rebuilt from it.

    Raises:
        InvalidImageError, ImageLimitError, ImageFetchError, BlobNotFoundError,
        ImagePersistenceUnsupportedError.
    """
    raise NotImplementedError
