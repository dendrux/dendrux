"""Core types for Dendrux.

These dataclasses define the shapes of data flowing through the agent loop.
Every module in Dendrux speaks this common language.

Data flow:
    User input → Agent → Strategy → LLM → LLMResponse → parse → AgentStep
                                                              ↓
                                                    ToolCall? → execute → ToolResult
                                                    Finish?   → RunResult
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine

    from dendrux.errors import ProviderError
    from dendrux.media import Image

from ulid import ULID

logger = logging.getLogger(__name__)

# ``Message.meta`` key naming the model that produced an ASSISTANT message's
# ``reasoning_blocks``. Providers replay the blocks only to that same model.
REASONING_MODEL_META_KEY = "reasoning_model"

# ``Message.meta`` flag on messages replayed from an earlier run via
# ``agent.run(runs=...)``. Providers use it to opt replayed reasoning into
# vendor-side prefix-binding safety nets.
REPLAYED_META_KEY = "dendrux.replayed"


def generate_ulid() -> str:
    """Generate a new ULID string for Dendrux-owned correlation IDs."""
    return str(ULID())


class Role(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """Agent wants to execute a tool.

    Carries both a Dendrux-owned ID (stable across pause/resume/replay)
    and the provider's native ID (for building API requests back).

    When the model's arguments were not a valid JSON object, ``params`` is
    ``{}``, ``parse_error`` holds a short description of the problem and
    ``raw_arguments`` the original string. The loop rejects such a call
    without running the tool. ``raw_arguments`` is in-memory only and is
    never persisted; persisted records carry ``parse_error`` and the raw
    length instead.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=generate_ulid)
    provider_tool_call_id: str | None = None
    raw_arguments: str | None = field(default=None, repr=False)
    parse_error: str | None = None


class _MetaUnset:
    """Sentinel for Message.meta's default (keeps dataclass default_factory semantics)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<factory>"


_META_UNSET: Any = _MetaUnset()
_ROLE_TOOL = Role.TOOL
_ROLE_SYSTEM = Role.SYSTEM
_ROLE_ASSISTANT = Role.ASSISTANT


# ------------------------------------------------------------------
# Image content parts (0.4.0)
# ------------------------------------------------------------------

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
ImageDetail: TypeAlias = Literal["auto", "low", "high"]
_DETAILS = ("auto", "low", "high")


@dataclass(frozen=True)
class ImageRef:
    """Immutable descriptor of a stored image blob. Never contains bytes.

    ``sha256`` is the lowercase hex digest of the exact stored bytes;
    ``media_type``, ``width`` and ``height`` were read from the image header at
    ingest. ``kind`` is always ``"blob"`` in this version.
    """

    sha256: str
    media_type: str
    byte_length: int
    width: int
    height: int
    kind: Literal["blob"] = "blob"

    def __post_init__(self) -> None:
        if self.kind != "blob":
            raise ValueError(f"ImageRef.kind must be 'blob', got {self.kind!r}")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("ImageRef.sha256 must be 64 lowercase hex characters")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("ImageRef.media_type must be a non-empty string")
        for name in ("byte_length", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ImageRef.{name} must be a positive int")

    @property
    def image_id(self) -> str:
        """Short display id shown to models and in errors: ``"img_" + sha256[:12]``."""
        return "img_" + self.sha256[:12]

    def to_dict(self) -> dict[str, Any]:
        """The persisted image descriptor (``{"type": "image", "v": 1, ...}``)."""
        return {
            "type": "image",
            "v": 1,
            "kind": self.kind,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "bytes": self.byte_length,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageRef:
        """Rebuild from :meth:`to_dict` output.

        Raises:
            InvalidImageError: ``reason="descriptor_version"`` when ``v`` is not 1
                or ``kind`` is not ``"blob"``; ``"descriptor_mismatch"`` when a
                field is missing or malformed.
        """
        from dendrux.errors import InvalidImageError

        if not isinstance(d, dict) or d.get("v") != 1 or d.get("kind") != "blob":
            raise InvalidImageError("descriptor_version")
        if "type" in d and d["type"] != "image":
            raise InvalidImageError("descriptor_mismatch")
        try:
            return cls(
                sha256=d["sha256"],
                media_type=d["media_type"],
                byte_length=d["bytes"],
                width=d["width"],
                height=d["height"],
            )
        except (KeyError, TypeError, ValueError):
            pass
        raise InvalidImageError("descriptor_mismatch")


@dataclass(frozen=True)
class TextPart:
    """A text content part of a multimodal message. ``text`` is non-empty."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("TextPart.text must be a non-empty string")


@dataclass(frozen=True)
class ImagePart:
    """An image content part: a stored blob reference plus an optional ``detail``.

    ``detail`` is provider-visible (OpenAI family only). ``data`` holds verified
    bytes only between request preparation and the provider call; it is never
    serialised, compared or shown in ``repr``.
    """

    ref: ImageRef
    detail: ImageDetail | None = None
    data: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ImageRef):
            raise TypeError("ImagePart.ref must be an ImageRef")
        if self.detail is not None and self.detail not in _DETAILS:
            raise ValueError(f"ImagePart.detail must be one of {_DETAILS} or None")

    @property
    def image_id(self) -> str:
        """Short display id of the referenced image."""
        return self.ref.image_id


ContentPart: TypeAlias = TextPart | ImagePart
ImageKey: TypeAlias = tuple[str, str]
"""Image identity: ``(tenant_scope, sha256)``."""


def content_part_to_dict(part: ContentPart) -> dict[str, Any]:
    """Serialise one content part to its persisted dict shape."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        d = part.ref.to_dict()
        if part.detail is not None:
            d["detail"] = part.detail
        return d
    raise TypeError(f"Not a content part: {type(part).__name__}")


def content_part_from_dict(d: dict[str, Any]) -> ContentPart:
    """Rebuild one content part from :func:`content_part_to_dict` output."""
    if d.get("type") == "text":
        return TextPart(text=d["text"])
    ref = ImageRef.from_dict(d)
    detail = d.get("detail")
    if detail is not None and detail not in _DETAILS:
        from dendrux.errors import InvalidImageError

        raise InvalidImageError("descriptor_mismatch")
    return ImagePart(ref=ref, detail=detail)


def parts_digest(parts: Sequence[ContentPart]) -> list[list[Any]]:
    """Provider-visible identity of an ordered parts list, for hashing.

    Text -> ``["t", text]``; image -> ``["i", "blob", sha256, media_type, detail]``.
    Storage location, generation, process and base64 wrapping never enter.
    """
    out: list[list[Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            out.append(["t", part.text])
        elif isinstance(part, ImagePart):
            out.append(["i", part.ref.kind, part.ref.sha256, part.ref.media_type, part.detail])
        else:
            raise TypeError(f"Not a content part: {type(part).__name__}")
    return out


# Identity-keyed memo of parts -> text projection. Entries hold the parts tuple,
# so an id cannot be reused while its entry lives; ``entry[0] is parts`` validates.
# Parts carrying wire bytes (``ImagePart.data``) are never memoised, so the memo
# can never keep image bytes alive past the provider call.
_PROJECTION_MEMO: dict[int, tuple[tuple[ContentPart, ...], str]] = {}
_PROJECTION_MEMO_MAX = 1024


def _text_projection(parts: tuple[ContentPart, ...]) -> str:
    entry = _PROJECTION_MEMO.get(id(parts))
    if entry is not None and entry[0] is parts:
        return entry[1]
    projection = "\n\n".join(p.text for p in parts if isinstance(p, TextPart))
    if any(isinstance(p, ImagePart) and p.data is not None for p in parts):
        return projection
    if len(_PROJECTION_MEMO) >= _PROJECTION_MEMO_MAX:
        with contextlib.suppress(StopIteration, RuntimeError):
            _PROJECTION_MEMO.pop(next(iter(_PROJECTION_MEMO)), None)
    _PROJECTION_MEMO[id(parts)] = (parts, projection)
    return projection


@dataclass(frozen=True, init=False)
class Message:
    """A single message in the conversation.

    This is the universal format — providers convert to/from their native
    formats internally. Developers and Dendrux internals only deal with Message.

    Role-dependent fields:
        tool_calls: Present on ASSISTANT messages when the LLM called tools
                    (NativeToolCalling strategy). None for text-only turns
                    and all PromptBasedReAct turns.
        call_id:    Present on TOOL messages. References ToolCall.id (Dendrux ULID)
                    from the corresponding ASSISTANT message.
        name:       Present on TOOL messages only. Cached convenience field for
                    debugging/logging — call_id is the authoritative identity.

    Origin envelope (context-blocks foundation):
        kind:       Block identity. Defaults from role (TOOL -> "tool_result",
                    SYSTEM -> "system", else "chat"). Dev-supplied context blocks
                    set their own opaque kind; Dendrux carries it uninterpreted.
        placement:  Cache intent, "stable" | "dynamic". Only meaningful for
                    context-origin messages (drives the stable/dynamic assembly
                    split); ignored for history/scratchpad, which are positioned
                    by the assembly order.
        source:     Optional free-form provenance label (citations/telemetry).

    Multimodal content (USER and TOOL only):
        parts:      Ordered text and image parts, or None for text-only messages.
                    A parts message always has at least one ImagePart, and
                    ``content`` is its text projection (text parts joined by
                    blank lines). Build one with :meth:`with_parts`.
    """

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    kind: str = "chat"
    placement: str = "dynamic"
    source: str | None = None
    # Reasoning the model surfaced on an ASSISTANT turn: ``reasoning`` is the
    # human-readable summary (never raw CoT); ``reasoning_blocks`` are opaque
    # provider artifacts replayed verbatim across tool-call iterations.
    reasoning: str | None = None
    reasoning_blocks: list[Any] | None = None
    parts: tuple[ContentPart, ...] | None = None

    def __init__(
        self,
        role: Role,
        content: str,
        name: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        call_id: str | None = None,
        meta: dict[str, Any] = _META_UNSET,
        kind: str = "chat",
        placement: str = "dynamic",
        source: str | None = None,
        reasoning: str | None = None,
        reasoning_blocks: list[Any] | None = None,
        parts: tuple[ContentPart, ...] | None = None,
    ) -> None:
        # Hand-written (dataclass init=False) so the 13-field frozen message is
        # built with one dict update instead of one object.__setattr__ per field.
        self.__dict__.update(
            role=role,
            content=content,
            name=name,
            tool_calls=tool_calls,
            call_id=call_id,
            meta={} if meta is _META_UNSET else meta,
            kind=kind,
            placement=placement,
            source=source,
            reasoning=reasoning,
            reasoning_blocks=reasoning_blocks,
            parts=parts,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.parts is not None:
            self._validate_parts()
        # Hot path: locals and module-level role constants keep text-only
        # construction as cheap as v0.3.0 despite the extra field.
        placement = self.placement
        if placement != "dynamic" and placement != "stable":
            raise ValueError(f"placement must be 'stable' or 'dynamic', got {placement!r}")
        role = self.role
        # Derive kind from role when left at the generic default. A TOOL message
        # is always a tool_result; a SYSTEM message is always system context.
        if self.kind == "chat":
            if role == _ROLE_TOOL:
                object.__setattr__(self, "kind", "tool_result")
            elif role == _ROLE_SYSTEM:
                object.__setattr__(self, "kind", "system")
        if role == _ROLE_TOOL:
            if not self.name:
                raise ValueError("TOOL messages require name")
            if not self.call_id:
                raise ValueError("TOOL messages require call_id")
            if self.tool_calls is not None:
                raise ValueError("TOOL messages cannot have tool_calls")
        elif role == _ROLE_ASSISTANT:
            if self.call_id is not None:
                raise ValueError("ASSISTANT messages cannot have call_id")
            if self.name is not None:
                raise ValueError("ASSISTANT messages cannot have name")
        else:
            if self.tool_calls is not None:
                raise ValueError(f"{role.value.upper()} messages cannot have tool_calls")
            if self.call_id is not None:
                raise ValueError(f"{role.value.upper()} messages cannot have call_id")
            if self.name is not None:
                raise ValueError(f"{role.value.upper()} messages cannot have name")

    def _validate_parts(self) -> None:
        parts = self.parts
        assert parts is not None
        if self.role not in (Role.USER, Role.TOOL):
            raise ValueError(f"{self.role.value.upper()} messages cannot have parts")
        if type(parts) is not tuple:
            if isinstance(parts, (str, bytes)):
                raise TypeError("Message.parts must be a sequence of content parts")
            parts = tuple(parts)
            object.__setattr__(self, "parts", parts)
        has_image = False
        for part in parts:
            if isinstance(part, ImagePart):
                has_image = True
            elif not isinstance(part, TextPart):
                raise TypeError(
                    f"Message.parts items must be TextPart or ImagePart, got {type(part).__name__}"
                )
        if not has_image:
            raise ValueError(
                "Message.parts must contain at least one ImagePart; use content= for text-only"
            )
        projection = _text_projection(parts)
        if self.content is not projection and self.content != projection:
            raise ValueError(
                "Message.content must equal the text projection of parts; use Message.with_parts()"
            )

    @classmethod
    def with_parts(cls, role: Role, parts: Sequence[ContentPart], **kwargs: Any) -> Message:
        """Build a multimodal message; ``content`` is computed from ``parts``."""
        if "content" in kwargs:
            raise TypeError("with_parts() computes content; do not pass content=")
        if isinstance(parts, (str, bytes)):
            raise TypeError("parts must be a sequence of content parts")
        parts_t = tuple(parts)
        for part in parts_t:
            if not isinstance(part, (TextPart, ImagePart)):
                raise TypeError(
                    f"parts items must be TextPart or ImagePart, got {type(part).__name__}"
                )
        return cls(role=role, content=_text_projection(parts_t), parts=parts_t, **kwargs)

    @property
    def images(self) -> tuple[ImagePart, ...]:
        """Image parts of this message, in order (empty for text-only messages)."""
        if self.parts is None:
            return ()
        return tuple(p for p in self.parts if isinstance(p, ImagePart))


@dataclass(frozen=True)
class Finish:
    """Agent is done — has a final answer."""

    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Clarification:
    """Agent needs input from a human before continuing."""

    question: str
    options: list[str] = field(default_factory=list)


Action = ToolCall | Finish | Clarification


@dataclass(frozen=True)
class AgentStep:
    """The output of one ReAct iteration.

    A strategy parses the LLM response into this. The loop consumes it.
    This is the boundary between "talking to the LLM" and "executing actions."
    """

    reasoning: str | None
    action: Action
    raw_response: str | None = None  # Original LLM text (for tracing)
    meta: dict[str, Any] = field(default_factory=dict)  # extra_fields land here


class ToolTarget(StrEnum):
    """Where a tool executes.

    Only SERVER and CLIENT are implemented. HUMAN and AGENT are reserved
    for future use — the loop currently treats them as CLIENT (pauses).
    """

    SERVER = "server"  # Runs on the backend (default)
    CLIENT = "client"  # Shipped to client for execution
    # Reserved — not yet implemented. Loop treats as CLIENT today.
    HUMAN = "human"
    AGENT = "agent"


@dataclass(frozen=True)
class ToolDef:
    """Definition of a registered tool.

    Created by the @tool decorator. Used by strategies to describe tools
    to the LLM, and by the executor to dispatch tool calls.

    ``strict`` opts into OpenAI strict function calling. Normalization
    happens once, at decoration time: when ``strict`` is True, ``parameters``
    already holds the strict-mode schema (every property required, objects
    closed with ``additionalProperties: false``), so every provider sees the
    same stored schema. Strict tools are sent strict on OpenAI and Anthropic
    (Anthropic: schema normalized; falls back to non-strict when the schema or
    request limits don't allow it).
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for params
    target: ToolTarget = ToolTarget.SERVER
    parallel: bool = True  # Governs concurrent execution scheduling
    max_calls_per_run: int | None = None  # Enforced by ReActLoop — graceful limit
    timeout_seconds: float = 120.0  # Enforced by ReActLoop via asyncio.wait_for
    has_explicit_timeout: bool = False  # True when developer set timeout_seconds explicitly
    meta: dict[str, Any] = field(default_factory=dict)  # MCP source info, annotations, etc.
    strict: bool = False
    # PII restore allowlist (``PII(restore="allowlisted")``): False, True (every
    # entity) or a frozenset of entity names. Ignored under ``restore="all"``.
    receives_pii: bool | frozenset[str] = False


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool.

    name is a cached convenience field — call_id is the authoritative
    identity for correlation (call_id → ToolCall.id → ToolCall.name).
    """

    name: str
    call_id: str  # References ToolCall.id (Dendrux-owned)
    payload: str  # Always JSON string — serialized once by execution engine
    success: bool = True
    error: str | None = None
    duration_ms: int = 0
    # Runtime annotations persisted on the tool_calls row (e.g. ``substituted``).
    meta: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    # Images attached to the result. Client tools pass Image / ImageRef /
    # ImagePart; after ingest these are canonical ImageParts.
    images: tuple[Image | ImageRef | ImagePart, ...] = ()
    # Per-image ingest notes (lenient MCP results): rendered as text after the payload.
    notes: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if type(self.images) is not tuple:
            if isinstance(self.images, (str, bytes, bytearray)):
                raise TypeError("ToolResult.images must be a sequence of images")
            object.__setattr__(self, "images", tuple(self.images))
        if type(self.notes) is not tuple:
            if isinstance(self.notes, (str, bytes)):
                raise TypeError("ToolResult.notes must be a sequence of strings")
            object.__setattr__(self, "notes", tuple(self.notes))


def tool_result_message(result: ToolResult) -> Message:
    """Build the TOOL message for a tool result.

    Without images or notes this is exactly the v0.3.0 shape (``content=payload``).
    With images: ``Message.with_parts(TOOL, (TextPart(payload), *TextPart(note), *image_parts))``.
    Notes without images: text-only, ``content = payload + "\n\n" + "\n".join(notes)``.
    ``result.images`` must already be ingested (``ImagePart`` or ``ImageRef``).
    """
    meta: dict[str, Any] = {"is_error": True} if not result.success else {}
    if not result.images:
        content = result.payload
        if result.notes:
            content = result.payload + "\n\n" + "\n".join(result.notes)
        return Message(
            role=Role.TOOL,
            content=content,
            name=result.name,
            call_id=result.call_id,
            meta=meta,
        )
    parts: list[ContentPart] = []
    if result.payload:
        parts.append(TextPart(result.payload))
    parts.extend(TextPart(note) for note in result.notes)
    for img in result.images:
        if isinstance(img, ImagePart):
            parts.append(img)
        elif isinstance(img, ImageRef):
            parts.append(ImagePart(ref=img))
        else:
            raise TypeError(
                "tool_result_message() needs ingested images (ImagePart or ImageRef); "
                f"got {type(img).__name__}"
            )
    return Message.with_parts(Role.TOOL, parts, name=result.name, call_id=result.call_id, meta=meta)


@dataclass(frozen=True)
class ToolResultContext:
    """What a ``tool_result_transform`` sees for one successful tool call.

    ``result`` is the raw Python value the tool returned, before JSON
    serialization. ``meta`` is the tool's ``ToolDef.meta`` — for MCP
    tools it carries ``source_name``, ``namespace``, ``mcp_tool_name``, etc.;
    for local ``@tool`` functions it is empty.
    """

    tool_name: str
    call_id: str
    params: dict[str, Any]
    result: Any
    target: ToolTarget
    meta: dict[str, Any]


ToolResultTransform: TypeAlias = Callable[[ToolResultContext], Any | Awaitable[Any]]
"""Hook that rewrites a server tool's result before it is persisted or sent to the model.

The return value ALWAYS replaces the result — return ``ctx.result`` to leave
it unchanged (``None`` is a legitimate new result, not "unchanged"). Sync and
async callables are both accepted. See ``Agent(tool_result_transform=...)``.
"""


@dataclass(frozen=True)
class ToolCallContext:
    """What a ``before_tool`` policy hook sees for one tool call.

    ``params`` and ``meta`` are deep copies taken for the hook: mutating
    them never changes what executes. ``params`` are deanonymized (real
    values, not guardrail placeholders). ``source`` is ``"local"`` for
    ``@tool`` functions, ``"mcp"`` for MCP tools and ``"skill"`` for the
    built-in skill tools. ``tenant_id`` is the run's tenant (``None`` when
    unset or when the run is not persisted). ``images`` are the run's images
    whose ``img_...`` id appears as a whole string value in ``params``.
    """

    run_id: str
    call_id: str
    tool_name: str
    params: dict[str, Any]
    target: ToolTarget
    source: Literal["local", "mcp", "skill"]
    meta: dict[str, Any]
    iteration: int
    tenant_id: str | None = None
    images: tuple[ImageRef, ...] = ()


@dataclass(frozen=True)
class ToolDecision:
    """A ``before_tool`` hook's verdict for one tool call.

    Build one with :meth:`allow`, :meth:`deny`, :meth:`require_approval`
    or :meth:`substitute`. Returning ``None`` from the hook means allow.
    """

    kind: Literal["allow", "deny", "require_approval", "substitute"]
    reason: str | None = None
    result: Any = None

    @classmethod
    def allow(cls) -> ToolDecision:
        """Run the tool normally."""
        return cls(kind="allow")

    @classmethod
    def deny(cls, reason: str) -> ToolDecision:
        """Do not run the tool; the model receives ``{"denied": reason}``."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("ToolDecision.deny() requires a non-empty reason string.")
        return cls(kind="deny", reason=reason)

    @classmethod
    def require_approval(cls, reason: str | None = None) -> ToolDecision:
        """Pause the batch for human approval (server-target tools only)."""
        return cls(kind="require_approval", reason=reason)

    @classmethod
    def substitute(cls, result: Any) -> ToolDecision:
        """Do not run the tool; ``result`` (JSON-serializable) is the tool output."""
        return cls(kind="substitute", result=result)


BeforeToolHook: TypeAlias = Callable[
    [ToolCallContext], "ToolDecision | None | Awaitable[ToolDecision | None]"
]
"""Policy hook evaluated before every tool call. See ``Agent(before_tool=...)``."""


@dataclass(frozen=True)
class ApprovalDecision:
    """A human decision for one approval-gated tool call.

    Used with ``Agent.submit_approval(run_id, decisions={call_id: ...})``.
    """

    approved: bool
    reason: str | None = None

    @classmethod
    def approve(cls) -> ApprovalDecision:
        """Run the call."""
        return cls(approved=True)

    @classmethod
    def deny(cls, reason: str | None = None) -> ApprovalDecision:
        """Skip the call; the model receives ``{"denied": reason}``."""
        if reason is not None and not isinstance(reason, str):
            raise ValueError("ApprovalDecision.deny() reason must be a string or None.")
        return cls(approved=False, reason=reason)


StopReason: TypeAlias = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "content_filter",
    "refusal",
    "pause_turn",
    "other",
]
"""Why an LLM call stopped, normalized across providers.

- ``"end_turn"``: natural finish.
- ``"tool_use"``: stopped to call tools.
- ``"max_tokens"``: hit the output-token limit (the output is cut off).
- ``"stop_sequence"``: hit a stop sequence.
- ``"content_filter"``: the provider filtered the output.
- ``"refusal"``: the model refused (Anthropic ``"refusal"``).
- ``"pause_turn"``: Anthropic server-tool pause.
- ``"other"``: any other value the provider reported.

``None`` (on :class:`LLMResponse` / :class:`RunResult`) means the provider
did not report a stop reason; Dendrux never guesses one.
"""


@dataclass
class UsageStats:
    """Token usage from an LLM call.

    ``input_tokens`` always means fresh (non-cached) input tokens across
    all providers. OpenAI variants normalize this by subtracting cached
    from prompt_tokens; Anthropic returns it that way natively.

    ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` are
    ``None`` when the provider did not report the field (e.g.
    OpenAI-compatible backends without prompt_tokens_details), or an int
    (possibly 0) when the provider did report it. This distinction lets
    downstream tooling tell "didn't report" apart from "reported zero".

    ``reasoning_tokens`` is the count of internal reasoning/thinking tokens
    the model billed within ``output_tokens`` (Anthropic thinking, OpenAI
    reasoning). ``None`` when the provider did not report it.

    ``cost_usd`` is ``None`` when nothing priced the call: the provider did
    not report a cost and no :class:`~dendrux.pricing.PriceTable` covered the
    model. A run total is ``None`` as soon as any step is unpriced, so a
    partial sum never reads as the whole run's spend. ``cost_source`` says
    where a non-``None`` cost came from: ``"provider"`` (reported on the
    response, e.g. OpenRouter), ``"table"`` (computed from the developer's
    price table), or ``"mixed"`` (a run total whose steps differ).

    ``usage_reported`` is false when token counts were unavailable. Such
    usage cannot be priced by a table, even though its counters default to
    zero. Explicitly constructed usage defaults to reported; providers must
    set this flag to false when the API omits usage.

    ``cost_unknown`` tracks whether an aggregate includes an unpriced call,
    independently of token counts. It survives pause/resume serialization.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_source: str | None = None
    usage_reported: bool = True
    cost_unknown: bool = False


BudgetAction: TypeAlias = Literal["warn", "pause", "cancel"]
_BUDGET_ACTIONS = ("warn", "pause", "cancel")


@dataclass(frozen=True)
class Budget:
    """Token and/or dollar budget for a run and, by default, its delegated children.

    Usage is checked after every LLM response and at the top of every
    iteration. ``budget.threshold`` fires once per ``(metric, fraction)``
    when a ``warn_at`` fraction is first crossed and ``budget.exceeded``
    once per metric when its limit is reached; every payload carries
    ``metric`` (``"tokens"`` or ``"cost_usd"``), ``used`` and ``max``. Fired
    keys survive pause/resume, so a resumed run does not re-fire them.

    ``on_exceed`` decides what happens once a limit is reached:

    - ``"warn"`` (default): events only; the run continues.
    - ``"pause"``: durable pause (``RunStatus.PAUSED``, ``pause_reason="budget"``).
    - ``"cancel"``: ``RunStatus.CANCELLED`` with ``failure_reason="budget_exceeded"``.

    Pause and cancel take effect at the next top-of-iteration checkpoint,
    after the tools of the response that crossed the limit have run, so a
    run can overshoot by up to one iteration. A response that finishes the
    run still completes. Cancellation and requested pauses take precedence
    over a budget pause at the same checkpoint. ``SingleCall`` has no
    checkpoint: pause/cancel degrade to warn there.

    Resuming a budget pause succeeds only if the resuming Agent's limits are
    strictly above the current live usage; otherwise the run pauses again
    immediately with the same reason.

    With a state store, usage is read live from recorded LLM calls; with
    ``include_children=True`` it is the root-inclusive rollup of the run and
    all its delegated descendants. The cost limit counts priced calls only:
    unpriced calls (no provider cost and no ``PriceTable`` entry) emit one
    ``budget.unpriced`` event per run and are otherwise ignored.

    Args:
        max_tokens: Token limit (input + output, > 0), or None.
        warn_at: Fractions of each limit at which ``budget.threshold``
            fires. Each fraction must be in (0, 1) exclusive.
            Default: (0.5, 0.75, 0.9).
        max_cost_usd: Dollar limit (> 0), or None.
        on_exceed: ``"warn"``, ``"pause"`` or ``"cancel"``.
        include_children: Count delegated descendant runs' usage.
        count_cache_tokens: Count cache read and write tokens toward
            ``max_tokens``. ``None`` (default) means not chosen.

    Raises:
        ValueError: If neither limit is set, a limit is not positive, a
            ``warn_at`` fraction is outside (0, 1), or ``on_exceed`` is unknown.
    """

    max_tokens: int | None = None
    warn_at: tuple[float, ...] = (0.5, 0.75, 0.9)
    max_cost_usd: float | None = None
    on_exceed: BudgetAction = "warn"
    include_children: bool = True
    count_cache_tokens: bool | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is None and self.max_cost_usd is None:
            raise ValueError("Budget needs max_tokens and/or max_cost_usd.")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError(f"Budget max_tokens must be > 0, got {self.max_tokens}.")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not (0 < self.max_cost_usd < float("inf"))
        ):
            raise ValueError(f"Budget max_cost_usd must be > 0, got {self.max_cost_usd}.")
        for f in self.warn_at:
            if not (0 < f < 1):
                raise ValueError(f"Budget warn_at fractions must be in (0, 1) exclusive, got {f}.")
        if self.on_exceed not in _BUDGET_ACTIONS:
            raise ValueError(
                f"Budget on_exceed must be one of {_BUDGET_ACTIONS}, got {self.on_exceed!r}."
            )
        if not isinstance(self.include_children, bool):
            raise ValueError("Budget include_children must be a bool.")
        if self.count_cache_tokens is not None and not isinstance(self.count_cache_tokens, bool):
            raise ValueError("Budget count_cache_tokens must be a bool or None.")


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Anthropic, OpenAI, or any other provider all normalize to this.
    Strategies consume this — they never touch provider-specific APIs.

    ``stop_reason`` is the normalized reason the call stopped
    (:data:`StopReason`), ``None`` when the provider did not report one;
    ``raw_stop_reason`` keeps the vendor's own string.
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw: Any = None  # Full provider response for debugging
    usage: UsageStats = field(default_factory=lambda: UsageStats(usage_reported=False))
    # Adapter-boundary payloads — set by each provider, persisted as opaque JSON.
    # provider_request: the exact kwargs sent to the vendor API (e.g. Anthropic api_kwargs).
    # provider_response: the raw vendor response dict (e.g. response.model_dump()).
    provider_request: dict[str, Any] | None = None
    provider_response: dict[str, Any] | None = None
    # The model actually used for this call. Honors a per-call ``model=`` override;
    # falls back to the provider's configured model when a provider leaves it unset.
    model: str | None = None
    # Reasoning/thinking the model surfaced (summary text — never raw CoT).
    # None unless thinking was enabled and a summary/display was requested.
    reasoning: str | None = None
    # Opaque provider reasoning artifacts to replay verbatim for multi-turn
    # tool use (Anthropic thinking blocks + signature, OpenAI reasoning items).
    # Stored and round-tripped as-is; never interpreted by Dendrux.
    reasoning_blocks: list[Any] | None = None
    # Why the call stopped, normalized (see ``StopReason``); None when the
    # provider did not report one. ``raw_stop_reason`` is the vendor string
    # as received (e.g. Anthropic ``"max_tokens"``, OpenAI ``"length"``).
    stop_reason: StopReason | None = None
    raw_stop_reason: str | None = None


@dataclass(frozen=True)
class ProviderCapabilities:
    """What an LLM provider can do.

    Layers above check these flags instead of using isinstance.
    Strategy selection, streaming decisions, and tool handling
    all key off capabilities, not provider type.
    """

    supports_native_tools: bool = False
    supports_tool_call_ids: bool = False
    supports_streaming: bool = False
    supports_streaming_tool_deltas: bool = False
    supports_thinking: bool = False
    supports_multimodal: bool = False
    supports_system_prompt: bool = True
    supports_parallel_tool_calls: bool = False
    supports_structured_output: bool = False
    max_context_tokens: int | None = None


class RunStatus(StrEnum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_CLIENT_TOOL = "waiting_client_tool"
    WAITING_HUMAN_INPUT = "waiting_human_input"
    WAITING_APPROVAL = "waiting_approval"
    # Durable cooperative pause requested via ``Agent.request_pause``.
    PAUSED = "paused"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    MAX_ITERATIONS = "max_iterations"


class GovernanceEventType(StrEnum):
    """Governance event types emitted via on_governance_event().

    Use these instead of raw strings for autocomplete and typo prevention:

        from dendrux.types import GovernanceEventType

        if event_type == GovernanceEventType.POLICY_DENIED:
            ...

    StrEnum — compares equal to the string value, so existing code
    using raw strings continues to work without changes.
    """

    # Wave 1: Tool deny
    POLICY_DENIED = "policy.denied"

    # Wave 2: Approval (HITL)
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"

    # before_tool policy hook
    POLICY_SUBSTITUTED = "policy.substituted"
    POLICY_HOOK_ERROR = "policy.hook_error"
    POLICY_HOOK_WARNING = "policy.hook_warning"

    # Wave 3: Budget
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"
    BUDGET_UNPRICED = "budget.unpriced"

    # Wave 4: Guardrails
    GUARDRAIL_DETECTED = "guardrail.detected"
    GUARDRAIL_REDACTED = "guardrail.redacted"
    GUARDRAIL_BLOCKED = "guardrail.blocked"
    GUARDRAIL_UNMAPPED_PLACEHOLDER = "guardrail.unmapped_placeholder"
    # PII(restore="allowlisted") kept placeholders for a non-allowlisted tool (counts only)
    GUARDRAIL_RESTORE_SKIPPED = "guardrail.restore_skipped"

    # Init events: Skills + MCP
    SKILL_CATALOG_RESOLVED = "skill.catalog_resolved"
    SKILL_ACTIVATED = "skill.activated"
    SKILL_ACTIVATION_FAILED = "skill.activation_failed"
    SKILL_RESOURCE_READ = "skill.resource_read"
    SKILL_SCRIPT_REQUESTED = "skill.script_requested"
    SKILL_SCRIPT_COMPLETED = "skill.script_completed"
    SKILL_SCRIPT_FAILED = "skill.script_failed"
    MCP_CONNECTED = "mcp.connected"
    MCP_ERROR = "mcp.error"
    MCP_TOOLS_SKIPPED = "mcp.tools_skipped"

    # Provider transport: vendor-SDK retry observability
    PROVIDER_RETRY = "provider.retry"

    # Run control: inbox note delivered to the model (payload: note_id only)
    INBOX_DELIVERED = "inbox.delivered"

    # Images (payloads never contain bytes)
    IMAGE_REJECTED = "image.rejected"
    IMAGE_STATE = "image.state"
    IMAGE_VIEW_CHANGED = "image.view_changed"
    IMAGE_DEGRADED = "image.degraded"
    IMAGE_WARNING = "image.warning"
    GUARDRAIL_IMAGE_UNSCANNED = "guardrail.image_unscanned"

    # Resume with governance different from the run's start (digests only)
    GOVERNANCE_CHANGED = "governance.changed"


_MIB = 1024 * 1024


@dataclass(frozen=True)
class Limits:
    """Size and count caps for one Agent (``Agent(limits=Limits(...))``).

    Args:
        max_tool_result_bytes: Largest tool result (local, MCP or client)
            passed to the model, in bytes. Default 1 MiB.
        max_input_bytes: Largest UTF-8 text input to a run or submit, in
            bytes. Images have their own caps (``ImagePolicy``). Default 1 MiB.
        max_metadata_bytes: Largest run ``metadata`` as encoded JSON, in
            bytes. Default 64 KiB.
        max_notes_per_run: Most notes ``submit_note`` accepts for one run.
            Default 1000.

    Raises:
        ValueError: If a cap is not a positive integer.
    """

    max_tool_result_bytes: int = _MIB
    max_input_bytes: int = _MIB
    max_metadata_bytes: int = 64 * 1024
    max_notes_per_run: int = 1000

    def __post_init__(self) -> None:
        for name in (
            "max_tool_result_bytes",
            "max_input_bytes",
            "max_metadata_bytes",
            "max_notes_per_run",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Limits {name} must be a positive int, got {value!r}.")


@dataclass(frozen=True)
class ErrorInfo:
    """Structured, safe description of a provider failure that ended a run.

    Built from a :class:`dendrux.errors.ProviderError`. Every field is a code,
    number or name: ``safe_message`` is the error's fixed template, never
    vendor text. Persisted under ``agent_runs.meta["dendrux.error_info"]``.

    Attributes:
        category: The error's ``category`` (``"rate_limit"``, ``"timeout"``, ...).
        retryable: Whether repeating the same request may succeed.
        safe_message: ``"{provider} {category} (HTTP {status}, code {code})"``.
        error_type: The exception class name.
        provider: Provider name, when known.
        model: Model name, when known.
        status_code: HTTP status, when there was one.
        provider_code: The provider's error code when it looks like a code
            (``ProviderError.safe_provider_code``), else None.
        timeout_reason: Why a deadline fired (``"call_deadline"``,
            ``"stream_idle"``, ...), when one did.
    """

    category: str
    retryable: bool
    safe_message: str
    error_type: str | None = None
    provider: str | None = None
    model: str | None = None
    status_code: int | None = None
    provider_code: str | None = None
    timeout_reason: str | None = None

    @classmethod
    def from_error(cls, exc: ProviderError) -> ErrorInfo:
        """Build from a provider error, reading only its safe fields."""
        return cls(
            category=exc.category,
            retryable=exc.retryable,
            safe_message=exc.safe_message,
            error_type=type(exc).__name__,
            provider=exc.provider,
            model=exc.model,
            status_code=exc.status_code,
            provider_code=exc.safe_provider_code,
            timeout_reason=exc.timeout_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for persistence and events."""
        return {
            "category": self.category,
            "retryable": self.retryable,
            "safe_message": self.safe_message,
            "error_type": self.error_type,
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "provider_code": self.provider_code,
            "timeout_reason": self.timeout_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorInfo:
        """Rebuild from :meth:`to_dict` output.

        Unknown keys are ignored, and a field of the wrong type reads as its
        default, since stored run meta is not trusted input.
        """

        def _text(key: str) -> str | None:
            value = data.get(key)
            return value if isinstance(value, str) else None

        status = data.get("status_code")
        retryable = data.get("retryable")
        return cls(
            category=_text("category") or "unknown",
            retryable=retryable if isinstance(retryable, bool) else False,
            safe_message=_text("safe_message") or "",
            error_type=_text("error_type"),
            provider=_text("provider"),
            model=_text("model"),
            status_code=status
            if isinstance(status, int) and not isinstance(status, bool)
            else None,
            provider_code=_text("provider_code"),
            timeout_reason=_text("timeout_reason"),
        )


@dataclass
class RunResult:
    """The outcome of an agent run.

    On success, ``answer`` contains the final answer turn, not all assistant
    text produced across tool calls. ``commentary`` lists, in order, the
    non-empty assistant text of every step that ended in tool calls (progress
    narration such as "Let me look that up"). It is empty for SingleCall and
    for idempotent cached results. Persisted traces tag the same split with
    ``phase`` (``RunStore.get_traces()``).

    On stream interruption via cancel_run(), the partial answer contains
    only the current text buffer, which resets at each TOOL_RESULT. Closing
    a stream early persists that buffer as the run's answer. It can be None
    even if earlier turns produced text. Cancelling the coroutine awaiting run()
    re-raises CancelledError rather than returning a RunResult; it does not
    capture in-flight model text. ``Agent.cancel_run`` from the same process
    instead makes run() return CANCELLED with ``meta["interrupted"]=True`` and
    ``meta["partial_output"]=False``. Paused runs may carry a clarification
    question in ``answer``. ``meta["undelivered_notes"]`` lists inbox note ids
    a finished run never delivered (only present when there are any).

    ``stop_reason`` is the normalized stop reason (:data:`StopReason`) of the
    LLM call that produced the final answer. It is set on SUCCESS only, and
    ``None`` for other statuses, when the provider did not report one, and
    for idempotent cached results of runs finished before 0.4.1.
    ``truncated`` is True when that call hit the output-token limit
    (``stop_reason == "max_tokens"``): the run still ends SUCCESS, but the
    answer is cut off.

    ``error_info`` describes a provider failure that ended the run (ERROR
    only). ``pause_id`` identifies the current pause of a paused run.
    """

    run_id: str
    status: RunStatus
    answer: str | None = None
    output: Any = None
    steps: list[AgentStep] = field(default_factory=list)
    iteration_count: int = 0
    usage: UsageStats = field(default_factory=UsageStats)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    commentary: list[str] = field(default_factory=list)
    replay: ReplaySummary | None = None
    stop_reason: StopReason | None = None
    error_info: ErrorInfo | None = None
    pause_id: str | None = None

    @property
    def truncated(self) -> bool:
        """True when the final answer was cut off at the output-token limit."""
        return self.stop_reason == "max_tokens"


PauseRequestOutcome: TypeAlias = Literal[
    "requested", "already_waiting", "already_paused", "terminal"
]


@dataclass(frozen=True)
class PauseOutcome:
    """Result of :meth:`Agent.request_pause`.

    ``status_before`` / ``status_after`` are the root run's status when the
    request started and after it was recorded. ``affected`` lists every run
    the request touched as ``(run_id, outcome)``, deepest descendants first
    and the root last:

    - ``"requested"``: the run was RUNNING; it pauses at its next checkpoint.
    - ``"already_waiting"``: WAITING_*; already durable, nothing latched.
    - ``"already_paused"``: already PAUSED.
    - ``"terminal"``: already finished; nothing to pause.

    Returning is not confirmation that a run paused: ``run.paused`` in the
    event log (status ``paused``) confirms it.
    """

    run_id: str
    status_before: RunStatus
    status_after: RunStatus
    affected: list[tuple[str, PauseRequestOutcome]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplaySummary:
    """What ``agent.run(runs=...)`` replayed from earlier runs, and what it dropped.

    ``runs_full`` / ``runs_text`` list the run ids replayed in full (text,
    tool calls, results) and as question + answer only. ``reasoning_kept`` /
    ``reasoning_dropped`` list full-replay runs whose stored reasoning blocks
    were sent / withheld (runs with no reasoning appear in neither).
    ``warnings`` are human-readable notes: which run, what was dropped, why.
    """

    runs_full: list[str] = field(default_factory=list)
    runs_text: list[str] = field(default_factory=list)
    reasoning_kept: list[str] = field(default_factory=list)
    reasoning_dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (the shape mirrored into ``agent_runs.meta["replay"]``)."""
        return {
            "runs_full": list(self.runs_full),
            "runs_text": list(self.runs_text),
            "reasoning_kept": list(self.reasoning_kept),
            "reasoning_dropped": list(self.reasoning_dropped),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplaySummary:
        """Rebuild from :meth:`to_dict` output; unknown keys are ignored."""
        return cls(
            runs_full=list(data.get("runs_full") or []),
            runs_text=list(data.get("runs_text") or []),
            reasoning_kept=list(data.get("reasoning_kept") or []),
            reasoning_dropped=list(data.get("reasoning_dropped") or []),
            warnings=list(data.get("warnings") or []),
        )


@dataclass(frozen=True)
class GuardrailVerdict:
    """A recorded image-guardrail verdict for one image."""

    action: Literal["pass", "redact", "block", "warn"]
    entities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.entities) is not tuple:
            object.__setattr__(self, "entities", tuple(self.entities))


GuardrailVerdictKey: TypeAlias = tuple[str, str, str, str, str, str]
"""``(tenant_scope, sha256, guardrail_id, guardrail_version, origin, tool_name or "")``."""


@dataclass
class ImageViewState:
    """Per-run image view bookkeeping, persisted in ``PauseState.image_view``.

    ``sent_through``: slots below it count as history (sent or seeded).
    ``dropped_through``: eligible slots below it render the drop placeholder.
    ``rejected`` only grows; ``described`` and ``guardrail_verdicts`` are
    first-wins; ``placeholders`` maps a slot to the placeholder reason it took
    the first time (sticky; only ever changes to ``"purged"``); ``warned``
    lists warning keys already emitted.
    """

    epoch: int = 0
    dropped_through: int = 0
    sent_through: int = 0
    rejected: list[ImageKey] = field(default_factory=list)
    described: dict[ImageKey, str] = field(default_factory=dict)
    guardrail_verdicts: dict[GuardrailVerdictKey, GuardrailVerdict] = field(default_factory=dict)
    placeholders: dict[int, str] = field(default_factory=dict)
    warned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Deterministic JSON-safe dict (lists of objects, sorted)."""
        rejected = sorted({(str(s), str(h)) for s, h in self.rejected})
        verdicts = sorted(self.guardrail_verdicts.items(), key=lambda kv: kv[0])
        return {
            "epoch": self.epoch,
            "dropped_through": self.dropped_through,
            "sent_through": self.sent_through,
            "rejected": [{"tenant_scope": s, "sha256": h} for s, h in rejected],
            "described": [
                {"tenant_scope": s, "sha256": h, "text": text}
                for (s, h), text in sorted(self.described.items(), key=lambda kv: kv[0])
            ],
            "guardrail_verdicts": [
                {
                    "tenant_scope": k[0],
                    "sha256": k[1],
                    "guardrail_id": k[2],
                    "guardrail_version": k[3],
                    "origin": k[4],
                    "tool_name": k[5],
                    "action": v.action,
                    "entities": list(v.entities),
                }
                for k, v in verdicts
            ],
            "placeholders": [
                {"slot": slot, "reason": reason}
                for slot, reason in sorted(self.placeholders.items())
            ],
            "warned": sorted(set(self.warned)),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageViewState:
        """Rebuild from :meth:`to_dict` output."""
        rejected: list[ImageKey] = []
        seen: set[ImageKey] = set()
        for item in d.get("rejected") or []:
            key = (str(item["tenant_scope"]), str(item["sha256"]))
            if key not in seen:
                seen.add(key)
                rejected.append(key)
        described: dict[ImageKey, str] = {}
        for item in d.get("described") or []:
            described.setdefault((str(item["tenant_scope"]), str(item["sha256"])), item["text"])
        verdicts: dict[GuardrailVerdictKey, GuardrailVerdict] = {}
        for item in d.get("guardrail_verdicts") or []:
            vkey: GuardrailVerdictKey = (
                str(item["tenant_scope"]),
                str(item["sha256"]),
                str(item["guardrail_id"]),
                str(item.get("guardrail_version") or ""),
                str(item["origin"]),
                str(item.get("tool_name") or ""),
            )
            verdicts.setdefault(
                vkey,
                GuardrailVerdict(action=item["action"], entities=tuple(item.get("entities") or ())),
            )
        placeholders: dict[int, str] = {}
        for item in d.get("placeholders") or []:
            placeholders.setdefault(int(item["slot"]), str(item["reason"]))
        return cls(
            epoch=int(d.get("epoch") or 0),
            dropped_through=int(d.get("dropped_through") or 0),
            sent_through=int(d.get("sent_through") or 0),
            rejected=rejected,
            described=described,
            guardrail_verdicts=verdicts,
            placeholders=placeholders,
            warned=list(dict.fromkeys(str(w) for w in d.get("warned") or [])),
        )


@dataclass
class PauseState:
    """Everything needed to resume a paused agent run.

    This is execution state, not observability state. It is persisted
    raw in pause_data (JSON column) and cleared on finalize — the DB
    is ground truth across traces, tool_calls, and pause_data alike.
    Guardrail redaction (placeholders) applies only at the LLM-call
    boundary; the bijection lives on agent_runs.pii_mapping for audit
    replay.
    """

    agent_name: str
    pending_tool_calls: list[ToolCall]
    history: list[Message]
    steps: list[AgentStep]
    iteration: int
    trace_order_offset: int
    usage: UsageStats
    pending_targets: dict[str, str] = field(default_factory=dict)  # tool_call_id → target
    # Approval-gated call ids at pause time: {call_id: {"source", "reason"}}.
    # None on non-approval pauses and on pauses persisted before 0.3.0.
    approval_gates: dict[str, dict[str, str | None]] | None = None
    # Why a PAUSED run paused ("requested" or "budget"); None for WAITING_* pauses.
    pause_reason: str | None = None
    # Free-text reason given to ``Agent.request_pause(reason=...)``.
    pause_message: str | None = None
    # Decisions already made for pending calls of a pause that interrupted an
    # approval batch: {call_id: {"approved": bool, "reason": str | None}}.
    # Resume executes them without asking for approval again.
    pending_decisions: dict[str, dict[str, Any]] | None = None
    # Budget events already emitted, as (metric, threshold); threshold 1.0 is
    # "exceeded". Persisted on every pause so a resumed run does not re-fire.
    budget_fired: list[tuple[str, float]] | None = None
    # Whether ``budget.unpriced`` was already emitted for this run.
    budget_unpriced_reported: bool = False
    # Limits a budget pause hit: [{"metric", "used", "max"}]. Budget pauses only.
    budget_exceeded: list[dict[str, Any]] | None = None
    # Own priced spend and unpriced-call count, persisted when the run total
    # is unknown (mixed priced/unpriced) so stores without live rollups keep it.
    budget_known_cost_usd: float | None = None
    budget_unpriced_calls: int = 0
    # Image view bookkeeping; None until a run has seen an image.
    image_view: ImageViewState | None = None
    # Identifier of this pause (ULID); None on pauses written before 0.5.0.
    pause_id: str | None = None
    # Pause-state format version; None on pauses written before 0.5.0.
    schema_version: int | None = 1

    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for DB storage.

        Validates that the output is JSON-serializable. Raises TypeError
        if any field (e.g. custom meta values) cannot be serialized.
        """
        import json

        d = {
            "agent_name": self.agent_name,
            "pending_tool_calls": [_tool_call_to_dict(tc) for tc in self.pending_tool_calls],
            "pending_targets": self.pending_targets,
            "history": [_message_to_dict(m) for m in self.history],
            "steps": [_step_to_dict(s) for s in self.steps],
            "iteration": self.iteration,
            "trace_order_offset": self.trace_order_offset,
            "usage": _usage_to_dict(self.usage),
        }
        if self.approval_gates is not None:
            d["approval_gates"] = self.approval_gates
        if self.pause_reason is not None:
            d["pause_reason"] = self.pause_reason
        if self.pause_message is not None:
            d["pause_message"] = self.pause_message
        if self.pending_decisions is not None:
            d["pending_decisions"] = self.pending_decisions
        if self.budget_fired:
            d["budget_fired"] = [[metric, threshold] for metric, threshold in self.budget_fired]
        if self.budget_unpriced_reported:
            d["budget_unpriced_reported"] = True
        if self.budget_exceeded is not None:
            d["budget_exceeded"] = self.budget_exceeded
        if self.budget_known_cost_usd is not None:
            d["budget_known_cost_usd"] = self.budget_known_cost_usd
        if self.budget_unpriced_calls:
            d["budget_unpriced_calls"] = self.budget_unpriced_calls
        if self.image_view is not None:
            d["image_view"] = self.image_view.to_dict()
        if self.pause_id is not None:
            d["pause_id"] = self.pause_id
        if self.schema_version is not None:
            d["schema_version"] = self.schema_version
        d["provider_kwargs"] = {
            k: v
            for k, v in self.provider_kwargs.items()
            if k in {"model", "max_tokens", "temperature", "thinking", "effort"}
        }
        # Validate JSON safety — catch non-serializable meta values early
        try:
            json.dumps(d)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"PauseState contains non-JSON-serializable data: {e}. "
                f"Ensure all Message.meta and AgentStep.meta values are JSON-safe."
            ) from e
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PauseState:
        """Deserialize from a dict (loaded from DB)."""
        return cls(
            agent_name=data["agent_name"],
            pending_tool_calls=[
                _tool_call_from_dict(tc) for tc in data.get("pending_tool_calls", [])
            ],
            pending_targets=data.get("pending_targets", {}),
            history=[_message_from_dict(m) for m in data.get("history", [])],
            steps=[_step_from_dict(s) for s in data.get("steps", [])],
            iteration=data.get("iteration", 0),
            trace_order_offset=data.get("trace_order_offset", 0),
            usage=_usage_from_dict(data.get("usage", {})),
            approval_gates=data.get("approval_gates"),
            pause_reason=data.get("pause_reason"),
            pause_message=data.get("pause_message"),
            pending_decisions=data.get("pending_decisions"),
            budget_fired=[(str(m), float(t)) for m, t in data.get("budget_fired") or []] or None,
            budget_unpriced_reported=bool(data.get("budget_unpriced_reported", False)),
            budget_exceeded=data.get("budget_exceeded"),
            budget_known_cost_usd=data.get("budget_known_cost_usd"),
            budget_unpriced_calls=int(data.get("budget_unpriced_calls") or 0),
            image_view=(
                ImageViewState.from_dict(data["image_view"])
                if data.get("image_view") is not None
                else None
            ),
            pause_id=data.get("pause_id"),
            schema_version=data.get("schema_version") or 1,
            provider_kwargs={
                k: v
                for k, v in (data.get("provider_kwargs") or {}).items()
                if k in {"model", "max_tokens", "temperature", "thinking", "effort"}
            },
        )


# ------------------------------------------------------------------
# PauseState serialization helpers
# ------------------------------------------------------------------


def tool_call_parse_meta(tc: ToolCall) -> dict[str, Any]:
    """Persistable parse-failure fields of a tool call; empty when it parsed.

    Returns ``parse_error`` and ``raw_length`` only, never the raw text.
    """
    if tc.parse_error is None:
        return {}
    return {
        "parse_error": tc.parse_error,
        "raw_length": len(tc.raw_arguments) if tc.raw_arguments is not None else None,
    }


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    return {
        "name": tc.name,
        "params": tc.params,
        "id": tc.id,
        "provider_tool_call_id": tc.provider_tool_call_id,
        **tool_call_parse_meta(tc),
    }


def _tool_call_from_dict(d: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=d["name"],
        params=d.get("params", {}),
        id=d["id"],
        provider_tool_call_id=d.get("provider_tool_call_id"),
        parse_error=d.get("parse_error"),
    )


def _message_to_dict(m: Message) -> dict[str, Any]:
    import json as _json

    d: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.name is not None:
        d["name"] = m.name
    if m.tool_calls is not None:
        d["tool_calls"] = [_tool_call_to_dict(tc) for tc in m.tool_calls]
    if m.call_id is not None:
        d["call_id"] = m.call_id
    if m.meta:
        # Validate meta is JSON-safe; coerce non-serializable values
        safe_meta: dict[str, Any] = {}
        for k, v in m.meta.items():
            try:
                _json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                safe_meta[k] = str(v)
        d["meta"] = safe_meta
    # Origin envelope — only persist non-default values (kind for TOOL/SYSTEM is
    # re-derived from role on load, so serialize only when it carries new info).
    if m.kind not in ("chat", "tool_result", "system"):
        d["kind"] = m.kind
    if m.placement != "dynamic":
        d["placement"] = m.placement
    if m.source is not None:
        d["source"] = m.source
    # Reasoning summary + opaque replay blocks — persisted so a run paused
    # mid-tool-loop keeps its reasoning context across resume.
    if m.reasoning is not None:
        d["reasoning"] = m.reasoning
    if m.reasoning_blocks is not None:
        d["reasoning_blocks"] = m.reasoning_blocks
    if m.parts is not None:
        d["parts"] = [content_part_to_dict(p) for p in m.parts]
    return d


def _message_from_dict(d: dict[str, Any]) -> Message:
    tool_calls = None
    if "tool_calls" in d:
        tool_calls = [_tool_call_from_dict(tc) for tc in d["tool_calls"]]
    if d.get("parts") is not None:
        return Message.with_parts(
            Role(d["role"]),
            [content_part_from_dict(p) for p in d["parts"]],
            name=d.get("name"),
            call_id=d.get("call_id"),
            meta=d.get("meta", {}),
            kind=d.get("kind", "chat"),
            placement=d.get("placement", "dynamic"),
            source=d.get("source"),
            reasoning=d.get("reasoning"),
            reasoning_blocks=d.get("reasoning_blocks"),
        )
    return Message(
        role=Role(d["role"]),
        content=d["content"],
        name=d.get("name"),
        tool_calls=tool_calls,
        call_id=d.get("call_id"),
        meta=d.get("meta", {}),
        kind=d.get("kind", "chat"),
        placement=d.get("placement", "dynamic"),
        source=d.get("source"),
        reasoning=d.get("reasoning"),
        reasoning_blocks=d.get("reasoning_blocks"),
    )


def _action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, ToolCall):
        return {"type": "tool_call", **_tool_call_to_dict(action)}
    if isinstance(action, Finish):
        return {"type": "finish", "answer": action.answer, "meta": action.meta}
    if isinstance(action, Clarification):
        return {"type": "clarification", "question": action.question, "options": action.options}
    raise TypeError(f"Unknown action type: {type(action)}")


def _action_from_dict(d: dict[str, Any]) -> Action:
    t = d["type"]
    if t == "tool_call":
        return _tool_call_from_dict(d)
    if t == "finish":
        return Finish(answer=d["answer"], meta=d.get("meta", {}))
    if t == "clarification":
        return Clarification(question=d["question"], options=d.get("options", []))
    raise ValueError(f"Unknown action type: {t!r}")


def _step_to_dict(s: AgentStep) -> dict[str, Any]:
    import json as _json

    d: dict[str, Any] = {
        "reasoning": s.reasoning,
        "action": _action_to_dict(s.action),
        "raw_response": s.raw_response,
    }
    # Serialize meta — handle known non-JSON types explicitly
    safe_meta: dict[str, Any] = {}
    for k, v in s.meta.items():
        if k == "all_tool_calls":
            safe_meta[k] = [_tool_call_to_dict(tc) for tc in v]
        else:
            # Validate each value is JSON-serializable
            try:
                _json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                # Coerce non-serializable values to string representation
                safe_meta[k] = str(v)
    if safe_meta:
        d["meta"] = safe_meta
    return d


def _step_from_dict(d: dict[str, Any]) -> AgentStep:
    meta = d.get("meta", {})
    if "all_tool_calls" in meta:
        meta = dict(meta)
        meta["all_tool_calls"] = [_tool_call_from_dict(tc) for tc in meta["all_tool_calls"]]
    return AgentStep(
        reasoning=d.get("reasoning"),
        action=_action_from_dict(d["action"]),
        raw_response=d.get("raw_response"),
        meta=meta,
    )


def _usage_to_dict(u: UsageStats) -> dict[str, Any]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "cost_usd": u.cost_usd,
        "cache_read_input_tokens": u.cache_read_input_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
        "cost_source": u.cost_source,
        "reasoning_tokens": u.reasoning_tokens,
        "usage_reported": u.usage_reported,
        "cost_unknown": u.cost_unknown,
    }


def _usage_from_dict(d: dict[str, Any]) -> UsageStats:
    return UsageStats(
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        total_tokens=d.get("total_tokens", 0),
        cost_usd=d.get("cost_usd"),
        cache_read_input_tokens=d.get("cache_read_input_tokens"),
        cache_creation_input_tokens=d.get("cache_creation_input_tokens"),
        cost_source=d.get("cost_source"),
        reasoning_tokens=d.get("reasoning_tokens"),
        usage_reported=d.get("usage_reported", True),
        cost_unknown=d.get("cost_unknown", False),
    )


class StreamEventType(StrEnum):
    """Types of events emitted during streaming LLM responses.

    TEXT_DELTA: Incremental text token from the LLM.
    REASONING_DELTA: Incremental reasoning/thinking token (``text`` carries it).
        Only emitted when thinking is enabled; arrives before TEXT_DELTA.
    TOOL_USE_START: A tool call block has begun (name known, args pending).
    TOOL_USE_END: A tool call is fully assembled and ready to execute.
    DONE: Stream finished. ``raw`` carries the full ``LLMResponse``.
    """

    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"
    DONE = "done"


@dataclass(frozen=True)
class StreamEvent:
    """A single event from a streaming LLM response.

    Providers yield these during complete_stream(). The loop and transport
    layer consume them — strategies never touch provider-specific streaming APIs.
    """

    type: StreamEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    raw: Any = None
    # Provider-native answer phase on TEXT_DELTA ("commentary" | "final_answer"),
    # set only when the vendor labels it (OpenAI Responses message ``phase``).
    phase: str | None = None


# ------------------------------------------------------------------
# Run-level streaming types
# ------------------------------------------------------------------


class RunEventType(StrEnum):
    """Types of events emitted during agent.stream() and agent.resume_stream().

    These span the full multi-turn run, not just one LLM call.

    Lifecycle events (runner-owned):
        RUN_STARTED: First event for new runs. Carries run_id for correlation.
        RUN_RESUMED: First event for resumed runs. Carries run_id.
        RUN_COMPLETED: Terminal. Run finished (success or max_iterations).
        RUN_PAUSED: Terminal. Run paused for client tool or human input.
        RUN_ERROR: Terminal. Run failed. No exception is raised to the consumer.
        RUN_CANCELLED: Terminal. Run was cancelled cooperatively.

    LLM output events (translated from provider StreamEvent):
        TEXT_DELTA: Incremental text token from the LLM.
        REASONING_DELTA: Incremental reasoning/thinking token (in ``text``).
        TOOL_USE_START: A tool call block has begun (name known, args pending).
        TOOL_USE_END: A tool call is fully assembled.

    Execution events (loop-owned):
        TOOL_RESULT: A tool has been executed and its result is available.
    """

    # Lifecycle (runner-owned)
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    RUN_COMPLETED = "run_completed"
    RUN_PAUSED = "run_paused"
    RUN_ERROR = "run_error"
    RUN_CANCELLED = "run_cancelled"

    # LLM output (translated from provider StreamEvent)
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"

    # Execution (loop-owned)
    TOOL_RESULT = "tool_result"


# Terminal event types — after any of these, the stream ends.
_TERMINAL_RUN_EVENTS = frozenset(
    {
        RunEventType.RUN_COMPLETED,
        RunEventType.RUN_PAUSED,
        RunEventType.RUN_ERROR,
        RunEventType.RUN_CANCELLED,
    }
)


@dataclass(frozen=True)
class RunEvent:
    """A single event from agent.stream() or agent.resume_stream().

    Carries data relevant to its type:
        RUN_STARTED:     run_id (first event for new runs)
        RUN_RESUMED:     run_id (first event for resumed runs)
        TEXT_DELTA:      text, iteration, phase (when already known)
        REASONING_DELTA: text, iteration
        TOOL_USE_START:  tool_name, tool_call_id, iteration, phase="commentary"
        TOOL_USE_END:    tool_call, tool_name, tool_call_id, iteration, phase="commentary"
        TOOL_RESULT:     tool_call, tool_result, iteration, phase="commentary"
        RUN_COMPLETED:   run_result, phase="final_answer" on success;
                         ``stop_reason`` / ``truncated`` mirror run_result
        RUN_PAUSED:      run_result (pause state in run_result.meta)
        RUN_ERROR:       run_result, error

    ``iteration`` is the loop step (one LLM call) an LLM-output or tool event
    belongs to; it is None on lifecycle events. ``phase`` labels assistant
    text as ``"commentary"`` (narration in a step that ends in tool calls) or
    ``"final_answer"``. A text block usually streams before the tool call
    that makes it commentary, so TEXT_DELTA.phase is often None. Clients
    bucket TEXT_DELTA by ``iteration``, mark a bucket commentary when a tool
    event with that iteration arrives, and treat the last unmarked bucket as
    the answer once RUN_COMPLETED carries ``phase="final_answer"``
    (``run_result.answer`` stays authoritative).
    """

    type: RunEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_result: ToolResult | None = None
    run_result: RunResult | None = None
    run_id: str | None = None
    error: str | None = None
    iteration: int | None = None
    phase: str | None = None

    @property
    def stop_reason(self) -> StopReason | None:
        """The run's final stop reason on RUN_COMPLETED (``run_result.stop_reason``)."""
        if self.type != RunEventType.RUN_COMPLETED or self.run_result is None:
            return None
        return self.run_result.stop_reason

    @property
    def truncated(self) -> bool:
        """True on a RUN_COMPLETED whose final answer hit the output-token limit."""
        return self.stop_reason == "max_tokens"


class RunStream:
    """Async iterable of RunEvents with lifecycle management.

    Single-use: iterating twice raises RuntimeError.

    With an idempotency key, ``run_id`` is provisional before iteration and
    resolves to the existing run's ID during setup if the key matches.
    Cached outcomes yield one terminal event; ``result`` holds the cached
    summary, and no text or tool events are replayed.

    Cleanup: ``__aiter__`` returns an async generator wrapper whose
    ``finally`` block runs CAS-guarded cancellation if no terminal event
    was received. The cleanup fires reliably in these cases:

      - The generator is exhausted normally (all events consumed).
      - The consumer calls ``await stream.aclose()`` explicitly.
      - The consumer uses ``async with stream:`` — **recommended**
        for deterministic immediate cleanup on break or exception.

    After ``async for ...: break`` without ``async with``, cleanup
    depends on asyncio's async generator finalization hooks, which
    schedule cleanup when the wrapper is garbage-collected. This is
    **not immediate** and not guaranteed before the event loop ends.
    For production use, prefer ``async with``::

        async with agent.stream("task") as stream:
            async for event in stream:
                if done:
                    break  # __aexit__ guarantees immediate cleanup
    """

    def __init__(
        self,
        run_id: str,
        generator: AsyncGenerator[RunEvent, None],
        cleanup: Callable[[], Coroutine[Any, Any, RunResult | None]],
    ) -> None:
        self.run_id = run_id
        self._gen = generator
        self._cleanup = cleanup
        self._result: RunResult | None = None
        self._terminated = False
        self._started = False
        self._closed = False

    @property
    def result(self) -> RunResult | None:
        """Terminal result after a terminal event or explicit stream closure."""
        return self._result

    def __aiter__(self) -> AsyncGenerator[RunEvent, None]:
        if self._started:
            raise RuntimeError("RunStream is single-use; cannot iterate twice")
        self._started = True
        return self._wrap()

    async def _wrap(self) -> AsyncGenerator[RunEvent, None]:
        """Async generator wrapper — cleanup runs in the finally block.

        Fires reliably on exhaustion and explicit ``aclose()``.
        After ``break`` without ``async with``, depends on asyncio's
        async gen finalization hooks (non-deterministic timing).
        """
        try:
            async for event in self._gen:
                if event.type in _TERMINAL_RUN_EVENTS:
                    self._terminated = True
                    self._result = event.run_result
                yield event
        finally:
            await self._finalize()

    async def __aenter__(self) -> RunStream:
        """Enter async context — for deterministic immediate cleanup."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the stream. Idempotent — safe to call multiple times."""
        await self._finalize()

    async def _finalize(self) -> None:
        """Run cleanup exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._terminated:
                try:
                    self._result = await self._cleanup()
                except Exception:
                    logger.warning("RunStream cleanup failed", exc_info=True)
        finally:
            await self._gen.aclose()

    async def text(self) -> AsyncGenerator[str, None]:
        """Convenience filter — yields only text deltas as plain strings."""
        async for event in self:
            if event.type == RunEventType.TEXT_DELTA and event.text:
                yield event.text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreateRunResult:
    """Outcome of a create_run() call, including idempotency resolution."""

    run_id: str
    outcome: Literal["created", "existing_active", "existing_terminal"]
    status: RunStatus


class RunAlreadyActiveError(RuntimeError):
    """Raised when an idempotency key matches a run that is still executing.

    The caller should use ``agent.resume()`` or poll for completion
    instead of starting a new run.
    """

    def __init__(self, run_id: str, current_status: RunStatus) -> None:
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(
            f"Run {run_id} is already {current_status.value}. "
            f"Use agent.resume() or wait for completion."
        )


class IdempotencyConflictError(RuntimeError):
    """Raised when an idempotency key is reused with different request parameters.

    This indicates a bug in the caller — keys must not be reused across
    different requests. Generate a new key for each distinct request.
    ``run_id`` is None when the existing run must not be disclosed (for
    example, it belongs to another tenant).
    """

    def __init__(self, run_id: str | None, idempotency_key: str) -> None:
        self.run_id = run_id
        self.idempotency_key = idempotency_key
        where = f" for run {run_id}" if run_id is not None else ""
        super().__init__(
            f"Idempotency key '{idempotency_key}' already used{where} "
            f"with different input. Keys must not be reused across different requests."
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (type(self), (self.run_id, self.idempotency_key))


class StructuredOutputValidationError(RuntimeError):
    """Raised when the LLM response fails validation against the requested output_type.

    Carries the raw response text and the validation error for debugging.
    ``usage``, ``model`` and ``stop_reason`` describe the LLM call that
    produced the invalid response (``None`` when unknown), so its tokens and
    cost are not lost: the call was made and billed even though its output was
    rejected. ``stop_reason == "max_tokens"`` means the JSON was cut off at
    the output-token limit.
    """

    def __init__(
        self,
        raw_response: str,
        output_type_name: str,
        validation_error: str,
        *,
        usage: UsageStats | None = None,
        model: str | None = None,
        stop_reason: StopReason | None = None,
    ) -> None:
        self.raw_response = raw_response
        self.output_type_name = output_type_name
        self.validation_error = validation_error
        self.usage = usage
        self.model = model
        self.stop_reason = stop_reason
        # The full rejected response, set by ``structured_complete`` so the
        # loop can record the call with its evidence. Internal.
        self._response: LLMResponse | None = None
        super().__init__(
            f"Structured output validation failed for {output_type_name}. "
            f"Validation error: {validation_error}\n"
            f"Raw response: {raw_response[:500]}"
        )


def compute_idempotency_fingerprint(
    agent_name: str,
    user_input: str,
    output_type_name: str | None = None,
    history: list[Message] | None = None,
    context: list[Any] | None = None,
    replay: dict[str, Any] | None = None,
    skills: dict[str, Any] | None = None,
    user_input_parts: Sequence[ContentPart] | None = None,
) -> str:
    """Deterministic SHA-256 fingerprint for idempotency conflict detection.

    Includes agent_name + user_input + output_type + chat history + per-run
    context (request identity). Does NOT include provider kwargs (temperature,
    max_tokens) — those are execution tuning, not request identity.

    output_type changes the return shape contract, so two calls with
    different output_types must be treated as different requests.

    History is hashed as the canonical list of (role, content) pairs from
    the normalized Message objects — different construction paths producing
    equivalent normalized history hash identically. Context blocks are hashed
    by (kind, placement, source, content) so the same input with different
    context is treated as a different request. ``replay`` carries the
    ``runs`` / ``replay_last`` / ``replay_reasoning`` knobs when ``runs=`` is
    used; omitted otherwise so existing fingerprints are unchanged.

    Images are additive: ``user_input_parts`` (a multimodal input; ``user_input``
    is then its text projection) adds ``"user_input_parts"``, and a history
    message with parts adds ``"parts"``, each as :func:`parts_digest`.
    Text-only requests hash exactly as in v0.3.0.
    """
    data: dict[str, Any] = {"agent_name": agent_name, "user_input": user_input}
    if user_input_parts is not None:
        data["user_input_parts"] = parts_digest(user_input_parts)
    if output_type_name is not None:
        data["output_type"] = output_type_name
    if history:
        data["history"] = [_history_fingerprint_entry(m) for m in history]
    if context:
        data["context"] = [
            {
                "kind": b.kind,
                "placement": b.placement,
                "source": b.source,
                "content": b.content,
            }
            for b in context
        ]
    if replay:
        data["replay"] = replay
    if skills is not None:
        data["skills"] = skills
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _history_fingerprint_entry(m: Message) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.parts is not None:
        entry["parts"] = parts_digest(m.parts)
    return entry
