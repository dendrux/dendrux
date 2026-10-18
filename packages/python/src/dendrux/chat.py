"""Public chat-history surface for multi-turn agent applications.

Use ``ChatMessage`` to pass prior conversation turns into ``agent.run()``
and ``agent.stream()`` via ``history=``. The dev's app owns conversation
storage; Dendrux only reads it as input for the next turn.

This type is intentionally narrower than the internal ``dendrux.types.Message``:
it exposes only role, text content and (user turns only) images, hiding tool
calls, call IDs, and runtime metadata so devs cannot accidentally inject
malformed runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from dendrux.media._image import Image
from dendrux.types import ImagePart, ImageRef, Message, Role

if TYPE_CHECKING:
    from dendrux.media._image import ImageInput, UserContent
    from dendrux.types import ContentPart

_IMAGE_TYPES = (Image, ImageRef, ImagePart)


def split_user_content(content: Any, *, owner: str = "user_input") -> tuple[str, tuple[Any, ...]]:
    """Type-check ``UserContent``; return ``(text projection, items)``.

    ``items`` is empty for a plain string or a sequence without images (the
    text-only shape). Text items are joined by blank lines; empty strings are
    skipped. Raises ``TypeError`` for any other item type.
    """
    if isinstance(content, str):
        return content, ()
    if isinstance(content, (bytes, bytearray, Mapping, *_IMAGE_TYPES)) or not isinstance(
        content, Sequence
    ):
        raise TypeError(
            f"{owner} must be a str or a sequence of str / Image / ImageRef / ImagePart, "
            f"got {type(content).__name__}"
        )
    items = tuple(content)
    has_image = False
    for i, item in enumerate(items):
        if isinstance(item, _IMAGE_TYPES):
            has_image = True
        elif not isinstance(item, str):
            raise TypeError(
                f"{owner}[{i}] must be str, Image, ImageRef or ImagePart, got {type(item).__name__}"
            )
    text = "\n\n".join(item for item in items if isinstance(item, str) and item)
    return text, (
        tuple(item for item in items if not isinstance(item, str) or item) if has_image else ()
    )


class ChatRole(StrEnum):
    """Roles allowed in chat history passed via ``agent.run(history=...)``.

    Excludes ``system`` (set on the agent, not in chat history) and
    ``tool`` (an internal runtime concept, not part of external chat).
    """

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """One turn in a chat history passed to ``agent.run(history=...)``.

    Frozen so equivalent messages serialize byte-identically across turns
    — required for prompt-cache stability when the dev's app re-sends the
    same prior turns each request.

    ``parts`` (user turns only) holds the ordered text and image items of a
    multimodal turn; ``content`` is then its text projection. Build one with
    :meth:`user`. Images are ingested (validated and stored) when the run
    starts, and count as already sent.
    """

    role: ChatRole
    content: str
    parts: tuple[str | ImageInput, ...] | None = None

    def __post_init__(self) -> None:
        if self.parts is None:
            return
        if self.role != ChatRole.USER:
            raise ValueError("assistant chat messages cannot carry images")
        text, items = split_user_content(self.parts, owner="ChatMessage.parts")
        if not items:
            raise ValueError(
                "ChatMessage.parts must contain at least one image; use content= for text-only"
            )
        object.__setattr__(self, "parts", items)
        if self.content != text:
            raise ValueError(
                "ChatMessage.content must equal the text of parts joined by blank lines; "
                "use ChatMessage.user([...])"
            )

    @classmethod
    def user(cls, content: UserContent) -> ChatMessage:
        """Construct a user-role chat message from text, or text and images."""
        text, items = split_user_content(content, owner="ChatMessage.user()")
        if not items:
            return cls(role=ChatRole.USER, content=text)
        return cls(role=ChatRole.USER, content=text, parts=items)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        """Construct an assistant-role chat message."""
        return cls(role=ChatRole.ASSISTANT, content=content)


def validate_chat_history(history: list[ChatMessage] | None) -> None:
    """Validate chat history against the four boundary rules.

    Rules:
      1. First message (if any) must be USER.
      2. Last message (if any) must be ASSISTANT.
      3. No two consecutive messages may share the same role.
      4. Each message must have non-empty content.

    ``None`` and ``[]`` are both no-ops (treated as "no prior conversation").
    Raises ``ValueError`` with an actionable message on the first violation.
    """
    if not history:
        return

    if history[0].role != ChatRole.USER:
        raise ValueError("history must start with a user message, not an assistant message")

    if history[-1].role != ChatRole.ASSISTANT:
        raise ValueError("history ends with a user message; pass it as user_input instead")

    for i, msg in enumerate(history):
        if not msg.content and not msg.parts:
            raise ValueError(f"history message at index {i} has empty content")
        if i > 0 and msg.role == history[i - 1].role:
            raise ValueError(f"history contains consecutive {msg.role.value} messages at index {i}")


def history_has_images(history: list[ChatMessage] | None) -> bool:
    """Whether any history turn carries images (they need ingest before use)."""
    return bool(history) and any(msg.parts for msg in history or ())


def normalize_chat_history(
    history: list[ChatMessage] | None,
    parts_for: Mapping[int, tuple[ContentPart, ...]] | None = None,
) -> list[Message]:
    """Convert chat history to internal ``Message`` objects, deterministically.

    Validates first via ``validate_chat_history`` so the boundary is harder
    to misuse — call this directly in the runner and validation is implicit.

    Content is preserved verbatim: no trimming, no normalization, no metadata
    injection. Equivalent input always produces byte-identical ``Message``
    objects, which is required for prompt-cache stability across turns.

    ``parts_for`` maps the index of each image-bearing turn to its ingested
    canonical parts; such a turn without an entry raises ``ValueError``.
    """
    validate_chat_history(history)
    if not history:
        return []

    out: list[Message] = []
    for i, msg in enumerate(history):
        if msg.parts:
            parts = (parts_for or {}).get(i)
            if parts is None:
                raise ValueError(f"history message at index {i} has images that were not ingested")
            out.append(Message.with_parts(Role.USER, parts))
        else:
            out.append(Message(role=Role(msg.role.value), content=msg.content))
    return out
