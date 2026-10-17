"""Image token estimates (informational; provider totals stay authoritative)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.types import ImageDetail, ImageRef


def estimate_image_tokens(
    ref: ImageRef,
    *,
    provider_family: str,
    model: str,
    detail: ImageDetail | None = None,
) -> int | None:
    """Estimated input tokens for one image, or ``None`` when unknown (never 0).

    ``provider_family`` is ``"anthropic"``, ``"openai"`` or ``"openrouter"``
    (family taken from the slug prefix).
    """
    raise NotImplementedError
