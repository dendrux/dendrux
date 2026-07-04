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
from typing import Any, Literal

from dendrux.types import Message, Role

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
) -> list[Message] | None:
    """Fold context blocks into user-message content per the locked format.

    Stable band → prepended to the first user message. Dynamic band →
    prepended to the current (final) user message alongside ``user_input``.
    Labels are added only when a band is non-empty, so the no-context path
    is byte-identical to a plain history + user_input assembly.

    Returns ``None`` only when there is no history AND no context — the
    single-turn path where the loop builds the ``user_input`` message itself.
    """
    validate_context(context)
    stable = render_band(context, "stable")
    dynamic = render_band(context, "dynamic")

    if history_messages:
        first = history_messages[0]
        if stable:
            # placement="stable" marks the frozen-prefix boundary for the
            # provider translator's cross-run cache breakpoint.
            first_folded = replace(
                first, content=_fold_first(stable, first.content), placement="stable"
            )
        else:
            first_folded = first
        current = Message(role=Role.USER, content=_fold_current(dynamic, user_input))
        return [first_folded, *history_messages[1:], current]

    if not stable and not dynamic:
        return None

    # Single-turn (no history): render the first-message stable wrapper around
    # the current-message content so that, once this turn becomes history[0]
    # next run, its stable head is byte-identical → cross-run cache hit.
    content = _fold_first(stable, _fold_current(dynamic, user_input))
    return [
        Message(
            role=Role.USER,
            content=content,
            placement="stable" if stable else "dynamic",
        )
    ]


def _fold_first(stable: str, original: str) -> str:
    if not stable:
        return original
    return f"{STABLE_HEADER}\n{stable}\n\n{MESSAGE_HEADER}\n{original}"


def _fold_current(dynamic: str, user_input: str) -> str:
    if not dynamic:
        return user_input
    return f"{DYNAMIC_HEADER}\n{dynamic}\n\n{INPUT_HEADER}\n{user_input}"
