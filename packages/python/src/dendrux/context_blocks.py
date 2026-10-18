"""Public per-run context surface for agent applications.

Use ``ContextBlock`` to inject per-run context — retrieved documents,
project instructions/memory, referenced files, surface state — into
``agent.run()`` / ``agent.stream()`` via ``context=``. Unlike ``history=``
(prior conversation, dev-owned), context blocks are supplemental material
for THIS run only: read-only, not persisted as forward state, re-supplied
each call.

``placement`` is a cache hint, the dev's promise about how the content
changes turn-to-turn:
  - ``"stable"``  → byte-identical every turn. Folded into the FIRST user
                    message so it sits in the frozen, cacheable prefix. If
                    the dev changes it, they simply lose the cache hit — their
                    call. Dendrux never reclassifies.
  - ``"dynamic"`` → per-turn / rebuilt content (default). Folded into the
                    CURRENT user message at the tail, where growth never
                    invalidates history's cached prefix. Ephemeral: NOT
                    persisted into history — the dev stores only raw turns.

Wire shape (context folds into message *content*, not separate messages,
so the list stays a clean user/assistant alternation on every provider):

    system
    [user]      "STABLE CONTEXT:\\n…\\n\\nUSER MESSAGE:\\n<first turn>"
    [assistant] …
     … history …
    [user]      "ADDITIONAL CONTEXT:\\n…\\n\\nUSER INPUT:\\n<current input>"

``kind`` / ``source`` / ``metadata`` are opaque: Dendrux carries them for
audit but never interprets them. The app owns its taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dendrux.skills._session import SkillSession

from dendrux.types import Message, Role, TextPart

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.types import ContentPart

STABLE_HEADER = "STABLE CONTEXT:"
DYNAMIC_HEADER = "ADDITIONAL CONTEXT:"
MESSAGE_HEADER = "USER MESSAGE:"
INPUT_HEADER = "USER INPUT:"


@dataclass(frozen=True)
class ContextBlock:
    """One labeled block of per-run context passed via ``context=``.

    Frozen so equivalent blocks render byte-identically across turns —
    required for prompt-cache stability when ``stable`` blocks are re-sent.
    """

    content: str
    kind: str = "context"
    placement: Literal["stable", "dynamic"] = "dynamic"
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_context(context: list[ContextBlock] | None) -> None:
    """Validate context blocks before any side effects.

    Rules:
      1. Each item must be a ``ContextBlock``.
      2. Each block must have non-empty content.
      3. ``placement`` must be ``"stable"`` or ``"dynamic"``.

    ``None`` and ``[]`` are both no-ops. Raises ``TypeError`` / ``ValueError``
    with an actionable message on the first violation.
    """
    if not context:
        return

    for i, block in enumerate(context):
        if not isinstance(block, ContextBlock):
            raise TypeError(f"context[{i}] must be a ContextBlock, got {type(block).__name__}")
        if not block.content:
            raise ValueError(f"context[{i}] has empty content")
        if block.placement not in ("stable", "dynamic"):
            raise ValueError(
                f"context[{i}] placement must be 'stable' or 'dynamic', got {block.placement!r}"
            )


def render_band(context: list[ContextBlock] | None, placement: str) -> str:
    """Render one placement band to text, preserving the caller's order.

    Returns ``""`` when the band is empty. Block contents are joined verbatim
    by blank lines; no header is added here (see ``fold_context``).
    """
    if not context:
        return ""
    return "\n\n".join(b.content for b in context if b.placement == placement)


def fold_context(
    history_messages: list[Message],
    context: list[ContextBlock] | None,
    user_input: str,
    *,
    session: SkillSession | None = None,
    user_parts: Sequence[ContentPart] | None = None,
) -> list[Message] | None:
    """Fold context blocks into user-message content per the locked format.

    Stable band → prepended to the first user message. Dynamic band →
    prepended to the current (final) user message alongside ``user_input``.
    Labels are added only when a band is non-empty, so the no-context path
    is byte-identical to a plain history + user_input assembly.

    ``user_parts`` (a multimodal input; ``user_input`` is then its text
    projection) builds the current message with :func:`wire_user_parts`.

    Returns ``None`` only when there is no history AND no context AND no
    image input — the single-turn path where the loop builds the
    ``user_input`` message itself.
    """
    activations = session.envelopes() if session else []
    validate_context(context)
    stable = render_band(context, "stable")
    dynamic = render_band(context, "dynamic")

    if history_messages:
        first = history_messages[0]
        # placement="stable" marks the frozen-prefix boundary for the
        # provider translator's cross-run cache breakpoint.
        first_folded = fold_first_message(first, stable) if stable else first
        if user_parts is not None:
            current = Message.with_parts(
                Role.USER,
                wire_user_parts(user_parts, activations, context, include_stable=False),
            )
        else:
            current = Message(
                role=Role.USER,
                content=wire_user_input(user_input, activations, context, include_stable=False),
            )
        return [first_folded, *history_messages[1:], current]

    if user_parts is not None:
        return [
            Message.with_parts(
                Role.USER,
                wire_user_parts(user_parts, activations, context),
                placement="stable" if stable else "dynamic",
            )
        ]

    if not stable and not dynamic and not activations:
        return None

    # Single-turn (no history): render the first-message stable wrapper around
    # the current-message content so that, once this turn becomes history[0]
    # next run, its stable head is byte-identical → cross-run cache hit.
    content = wire_user_input(user_input, activations, context)
    return [
        Message(
            role=Role.USER,
            content=content,
            placement="stable" if stable else "dynamic",
        )
    ]


def fold_first_message(first: Message, stable: str) -> Message:
    """Fold the stable band into the first history message (text or parts)."""
    if first.parts is None:
        return replace(first, content=_fold_first(stable, first.content), placement="stable")
    return Message.with_parts(
        first.role,
        fold_first_parts(stable, first.parts),
        meta=first.meta,
        kind=first.kind,
        placement="stable",
        source=first.source,
    )


def fold_first_parts(stable: str, parts: Sequence[ContentPart]) -> tuple[ContentPart, ...]:
    """``_fold_first`` for a parts message: the stable header precedes the first part."""
    if not stable:
        return tuple(parts)
    if parts and isinstance(parts[0], TextPart):
        return (TextPart(_fold_first(stable, parts[0].text)), *parts[1:])
    return (TextPart(_fold_first(stable, "")), *parts)


def _fold_first(stable: str, original: str) -> str:
    if not stable:
        return original
    return f"{STABLE_HEADER}\n{stable}\n\n{MESSAGE_HEADER}\n{original}"


def _fold_current(dynamic: str, user_input: str) -> str:
    if not dynamic:
        return user_input
    return f"{DYNAMIC_HEADER}\n{dynamic}\n\n{INPUT_HEADER}\n{user_input}"


def wire_user_input(
    raw_input: str,
    activations: list[str],
    context_blocks: list[ContextBlock] | None,
    *,
    include_stable: bool = True,
) -> str:
    """Canonical first-input framing; activation envelopes always precede context."""
    content = _fold_current(render_band(context_blocks, "dynamic"), raw_input)
    if include_stable:
        content = _fold_first(render_band(context_blocks, "stable"), content)
    return "\n\n".join([*activations, content])


def wire_user_parts(
    parts: Sequence[ContentPart],
    activations: list[str],
    context_blocks: list[ContextBlock] | None,
    *,
    include_stable: bool = True,
) -> tuple[ContentPart, ...]:
    """:func:`wire_user_input` for a multimodal input; caller part order is kept.

    A leading text part is replaced by its framed form; otherwise the framing
    (when non-empty) is prepended as its own text part, so framing text always
    precedes the caller's first part and stable-band bytes never depend on images.
    """
    if parts and isinstance(parts[0], TextPart):
        framed = wire_user_input(
            parts[0].text, activations, context_blocks, include_stable=include_stable
        )
        return (TextPart(framed), *parts[1:])
    framing = wire_user_input("", activations, context_blocks, include_stable=include_stable)
    return (TextPart(framing), *parts) if framing else tuple(parts)
