"""Guardrail protocol and core types.

Guardrails scan text crossing the LLM boundary. The framework owns
the action logic (redact/block/warn) — guardrails only detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dendrux.types import ImageRef


@dataclass(frozen=True)
class Pattern:
    """A named regex pattern for PII/secret detection.

    Used by PII(extra_patterns=[...]) to extend built-in patterns
    with domain-specific detectors.

    Args:
        name: Entity type name (e.g. "EMPLOYEE_ID"). Becomes the
            placeholder prefix: <<EMPLOYEE_ID_1>>.
        regex: Raw regex pattern string.
    """

    name: str
    regex: str


UNSCANNABLE = "UNSCANNABLE"
"""Entity type of a finding the engine raises when content could not be scanned.

Raised when text or tool-call params exceed the scan size or nesting limits.
Under ``block`` and ``redact`` it blocks the run; under ``warn`` it is only
reported (``guardrail.detected``).
"""


@dataclass(frozen=True)
class Finding:
    """A single detection result from a guardrail scan.

    Framework uses these to apply actions (redact/block/warn).
    Guardrails produce them; framework consumes them.

    Args:
        entity_type: What was found (e.g. "EMAIL", "AWS_ACCESS_KEY").
        start: Character offset in the scanned text.
        end: Character offset (exclusive).
        score: Confidence score (0.0-1.0). Regex patterns use 1.0.
        text: The matched text content. Kept for redaction only; it is
            excluded from ``repr`` and never copied into events, errors or logs.
    """

    entity_type: str
    start: int
    end: int
    score: float
    text: str = field(repr=False)


@runtime_checkable
class Guardrail(Protocol):
    """Protocol for content guardrails.

    Guardrails detect findings in text. The framework applies actions:
      - redact: replace findings with <<TYPE_N>> placeholders
      - block: terminate the run with an error
      - warn: log the finding, continue unchanged

    scan() is async to support LLM-as-judge implementations that
    call a local model for evaluation. Regex/Presidio scanners
    simply don't await anything inside their async scan().
    """

    action: Literal["redact", "block", "warn"]

    async def scan(self, text: str) -> list[Finding]:
        """Detect findings in text. Framework handles the action."""
        ...


@dataclass(frozen=True)
class ImageScanInput:
    """One image handed to an :class:`ImageGuardrail`.

    ``data`` holds the verified image bytes; it is never shown in ``repr``.
    ``origin`` is ``"user"`` for user-input images and ``"tool"`` for
    tool-result images; ``tool_name`` names the tool for the latter.
    """

    image_id: str
    ref: ImageRef
    origin: Literal["user", "tool"]
    tool_name: str | None
    data: bytes = field(repr=False)


@runtime_checkable
class ImageGuardrail(Protocol):
    """Opt-in protocol for guardrails that scan images.

    Each image is scanned once per (tenant, image, guardrail id, guardrail
    version, origin, tool name); the verdict is recorded and reused on resume,
    retry and replay. Set ``guardrail_id`` / ``guardrail_version`` attributes
    to control that identity (defaults: the class path and ``""``); changing
    either triggers a rescan. No findings means ``pass``; otherwise the
    framework applies ``action``: ``redact`` replaces the image with a
    placeholder, ``block`` ends the run, ``warn`` emits ``guardrail.detected``.
    ``Finding`` offsets are unused for images (use ``0, 0``).

    A guardrail may implement both ``scan`` (text) and ``scan_image``.
    """

    action: Literal["redact", "block", "warn"]

    async def scan_image(self, image: ImageScanInput) -> list[Finding]:
        """Detect findings in one image. Framework handles the action."""
        ...


def guardrail_identity(guardrail: object) -> tuple[str, str]:
    """``(guardrail_id, guardrail_version)`` used in image verdict keys."""
    kind = type(guardrail)
    gid = getattr(guardrail, "guardrail_id", None)
    if not isinstance(gid, str) or not gid:
        gid = f"{kind.__module__}.{kind.__qualname__}"
    version = getattr(guardrail, "guardrail_version", "")
    return gid, "" if version is None else str(version)


def scans_text(guardrail: object) -> bool:
    """Whether ``guardrail`` implements the text ``scan`` method."""
    return callable(getattr(guardrail, "scan", None))


def scans_images(guardrail: object) -> bool:
    """Whether ``guardrail`` implements ``scan_image``."""
    return callable(getattr(guardrail, "scan_image", None))
