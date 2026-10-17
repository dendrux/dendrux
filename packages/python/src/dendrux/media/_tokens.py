"""Image token estimates (informational; provider totals stay authoritative)."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.types import ImageDetail, ImageRef

# Anthropic high-resolution tier: Opus 4.7 / 4.8 and every Claude 5 model
# ("claude-<family>-5..."; "claude-sonnet-4-5" is Claude 4.5, standard tier).
_ANTHROPIC_HIGH_RES = re.compile(r"^claude-(opus-4-[78]|[a-z]+-5)(?:$|[-.@:])")
_ANTHROPIC_STANDARD = (1568, 1568)  # (max long edge px, max tokens)
_ANTHROPIC_HIGH = (2576, 4784)

# OpenAI tile models: (base tokens, per-tile tokens). Order matters: most specific first.
_OPENAI_TILE: tuple[tuple[str, tuple[int, int]], ...] = (
    ("gpt-4o-mini", (2833, 5667)),
    ("gpt-4o", (85, 170)),
    ("chatgpt-4o", (85, 170)),
    ("gpt-4.1", (85, 170)),
    ("o1", (85, 170)),
    ("o3", (85, 170)),
)
# OpenAI patch models: multiplier. Checked before tile models.
_OPENAI_PATCH: tuple[tuple[str, float], ...] = (
    ("gpt-4.1-mini", 1.62),
    ("gpt-4.1-nano", 2.46),
    ("gpt-5", 1.2),
    ("gpt-6", 1.2),
)
_PATCH_BUDGET = 1536


def _anthropic_tokens(width: int, height: int, model: str) -> int:
    max_side, max_tokens = (
        _ANTHROPIC_HIGH if _ANTHROPIC_HIGH_RES.match(model) else _ANTHROPIC_STANDARD
    )
    w, h = width, height
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / long_side
        w, h = max(1, math.floor(w * scale)), max(1, math.floor(h * scale))
    tokens = math.ceil(w / 28) * math.ceil(h / 28)
    scale = math.sqrt(max_tokens / tokens) if tokens > max_tokens else 1.0
    base_w, base_h = w, h
    while tokens > max_tokens:
        w, h = max(1, math.floor(base_w * scale)), max(1, math.floor(base_h * scale))
        tokens = math.ceil(w / 28) * math.ceil(h / 28)
        scale *= 0.99
    return tokens


def _openai_tile_tokens(
    width: int, height: int, base: int, per_tile: int, detail: str | None
) -> int:
    if detail == "low":
        return base
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        w, h = w * scale, h * scale
    if min(w, h) > 768:
        scale = 768 / min(w, h)
        w, h = w * scale, h * scale
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return base + per_tile * tiles


def _openai_patch_tokens(width: int, height: int, multiplier: float) -> int:
    w, h = float(width), float(height)
    patches = math.ceil(w / 32) * math.ceil(h / 32)
    if patches > _PATCH_BUDGET:
        scale = math.sqrt(_PATCH_BUDGET * 32 * 32 / (w * h))
        w, h = w * scale, h * scale
        # Fit whole patches along the narrower dimension, as the vendor does.
        scale_w = math.floor(w / 32) / (w / 32)
        scale_h = math.floor(h / 32) / (h / 32)
        scale = min(scale_w, scale_h)
        w, h = w * scale, h * scale
        patches = math.ceil(w / 32) * math.ceil(h / 32)
    return math.ceil(patches * multiplier)


def _openai_tokens(width: int, height: int, model: str, detail: str | None) -> int | None:
    for prefix, multiplier in _OPENAI_PATCH:
        if model.startswith(prefix):
            return _openai_patch_tokens(width, height, multiplier)
    for prefix, (base, per_tile) in _OPENAI_TILE:
        if model == prefix or model.startswith(prefix + "-"):
            return _openai_tile_tokens(width, height, base, per_tile, detail)
    return None


def estimate_image_tokens(
    ref: ImageRef,
    *,
    provider_family: str,
    model: str,
    detail: ImageDetail | None = None,
) -> int | None:
    """Estimated input tokens for one image, or ``None`` when unknown (never 0).

    ``provider_family`` is ``"anthropic"``, ``"openai"`` or ``"openrouter"``;
    for OpenRouter the family comes from the slug prefix (``anthropic/``,
    ``openai/``, optionally with a leading ``~``).
    """
    family = provider_family.lower()
    name = model.strip().lower()
    if family == "openrouter":
        slug = name.lstrip("~")
        vendor, _, rest = slug.partition("/")
        if not rest:
            return None
        family, name = vendor, rest.split(":", 1)[0]
        if family == "anthropic":
            # OpenRouter slugs use dots in versions ("claude-opus-4.7").
            name = name.replace(".", "-")
    if family == "anthropic":
        if not name.startswith("claude-"):
            return None
        return _anthropic_tokens(ref.width, ref.height, name)
    if family == "openai":
        return _openai_tokens(ref.width, ref.height, name, detail)
    return None
