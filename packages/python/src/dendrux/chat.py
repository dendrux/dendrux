"""Public chat-history surface for multi-turn agent applications.

Use ``ChatMessage`` to pass prior conversation turns into ``agent.run()``
and ``agent.stream()`` via ``history=``. The dev's app owns conversation
storage; Dendrux only reads it as input for the next turn.

This type is intentionally narrower than the internal ``dendrux.types.Message``:
it exposes only role and text content, hiding tool calls, call IDs, and
runtime metadata so devs cannot accidentally inject malformed runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dendrux.types import Message, Role


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
    """

    role: ChatRole
    content: str

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        """Construct a user-role chat message."""
        return cls(role=ChatRole.USER, content=content)

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
        if not msg.content:
            raise ValueError(f"history message at index {i} has empty content")
        if i > 0 and msg.role == history[i - 1].role:
            raise ValueError(f"history contains consecutive {msg.role.value} messages at index {i}")


def normalize_chat_history(history: list[ChatMessage] | None) -> list[Message]:
    """Convert chat history to internal ``Message`` objects, deterministically.

    Validates first via ``validate_chat_history`` so the boundary is harder
    to misuse — call this directly in the runner and validation is implicit.

    Content is preserved verbatim: no trimming, no normalization, no metadata
    injection. Equivalent input always produces byte-identical ``Message``
    objects, which is required for prompt-cache stability across turns.
    """
    validate_chat_history(history)
    if not history:
        return []

    return [Message(role=Role(msg.role.value), content=msg.content) for msg in history]
