"""Types of the per-request image view (``loops/_images.py: prepare_request``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dendrux.types import GuardrailVerdict, GuardrailVerdictKey, ImageKey, Message


@dataclass(frozen=True)
class SlotInfo:
    """One image slot of the request view."""

    slot: int
    message_index: int
    part_index: int
    key: ImageKey
    image_id: str
    origin: Literal["user", "tool"]
    tool_name: str | None
    visible: bool
    placeholder: str | None


@dataclass(frozen=True)
class ImageStats:
    """Visible-image totals of one request."""

    count: int
    bytes: int
    estimated_tokens: int | None


@dataclass(frozen=True)
class ImageViewDelta:
    """State changes a request computed; applied by the caller, never by ``prepare_request``."""

    epoch: int | None = None
    dropped_through: int | None = None
    described: Mapping[ImageKey, str] = field(default_factory=dict)
    guardrail_verdicts: Mapping[GuardrailVerdictKey, GuardrailVerdict] = field(default_factory=dict)
    warned: tuple[str, ...] = ()


@dataclass(frozen=True)
class GovernanceEvent:
    """A governance event spec to record and notify, in order."""

    event_type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class PreparedRequest:
    """Result of preparing one provider request's image view."""

    semantic: list[Message]
    wire: list[Message]
    slots: tuple[SlotInfo, ...]
    new_slots: tuple[int, ...]
    stats: ImageStats
    guardrail_findings: dict[str, Any] | None
    events: tuple[GovernanceEvent, ...]
    delta: ImageViewDelta
    block_error: str | None
