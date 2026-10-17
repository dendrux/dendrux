"""Image ids and the exact placeholder strings used in request views."""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

IMAGE_ID_PREFIX = "img_"
IMAGE_ID_RE = re.compile(r"img_[0-9a-f]{12}")

PlaceholderReason: TypeAlias = Literal[
    "drop", "rejected", "unsupported", "described", "purged", "policy", "guardrail"
]

PLACEHOLDER_DROP = "[image {id} omitted from context]"
PLACEHOLDER_REJECTED = "[image {id} was rejected by the model provider and is not shown]"
PLACEHOLDER_UNSUPPORTED = "[image {id} not shown: this model does not accept images]"
PLACEHOLDER_DESCRIBED = "[image {id} (described): {text}]"
PLACEHOLDER_PURGED = "[image {id} was deleted]"
PLACEHOLDER_POLICY = "[image {id} not shown: it exceeds the current image policy]"
PLACEHOLDER_GUARDRAIL = "[image {id} removed by a guardrail]"

PLACEHOLDERS: dict[str, str] = {
    "drop": PLACEHOLDER_DROP,
    "rejected": PLACEHOLDER_REJECTED,
    "unsupported": PLACEHOLDER_UNSUPPORTED,
    "described": PLACEHOLDER_DESCRIBED,
    "purged": PLACEHOLDER_PURGED,
    "policy": PLACEHOLDER_POLICY,
    "guardrail": PLACEHOLDER_GUARDRAIL,
}


# Frozen precedence (highest first), equal to the prepare_request step order.
# "unsupported" and "described" share a rank; ImagePolicy.on_unsupported picks one.
PLACEHOLDER_PRECEDENCE: tuple[tuple[str, ...], ...] = (
    ("purged",),
    ("policy",),
    ("unsupported", "described"),
    ("rejected",),
    ("drop",),
    ("guardrail",),
)
_RANK = {reason: rank for rank, group in enumerate(PLACEHOLDER_PRECEDENCE) for reason in group}


def placeholder_rank(reason: str) -> int:
    """Rank of a placeholder reason in :data:`PLACEHOLDER_PRECEDENCE` (0 = highest)."""
    return _RANK[reason]


def image_id(sha256: str) -> str:
    """Display id of an image: ``"img_" + sha256[:12]``. Never a state key."""
    return IMAGE_ID_PREFIX + sha256[:12]


def placeholder(reason: PlaceholderReason, image_id: str, text: str | None = None) -> str:
    """Render the placeholder for ``reason``. ``text`` is required for ``"described"``."""
    template = PLACEHOLDERS[reason]
    if reason == "described":
        if text is None:
            raise ValueError("the 'described' placeholder needs text")
        return template.format(id=image_id, text=text)
    return template.format(id=image_id)
