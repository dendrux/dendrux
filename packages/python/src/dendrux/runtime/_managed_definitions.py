"""Check a root's registered subagent definitions before managed execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux.errors import SubagentDefinitionError, SubagentTreeTooLargeError
from dendrux.loops.react import ReActLoop
from dendrux.runtime import _subagents

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dendrux.agent import Agent


def _check_definitions(root: Agent, store: Any) -> None:
    """Every registered child has a provider and shares the root's store object."""
    for parent, child in _definition_edges(root):
        _require_react(parent, child)
        if child.provider is None:
            raise ValueError(f"Subagent '{child.name}' needs a provider.")
        own = getattr(child, "_state_store", None)
        if (own is not None and own is not store) or (
            own is None and getattr(child, "_database_url", None) is not None
        ):
            raise ValueError(f"Subagent '{child.name}' must use the root's store object or none.")


def require_streamable(root: Agent, entry: str) -> None:
    """Streaming a workflow needs every subagent to be streamable (no guardrails).

    The stream path never scans with guardrails, so a subagent that declares
    them is refused before anything runs, as the root itself is.
    """
    for _, child in _definition_edges(root):
        if child.guardrails or getattr(child, "_output_type", None) is not None:
            raise ValueError(
                f"{entry}() cannot stream subagent '{child.name}': it declares guardrails "
                "or output_type. Use run() for this workflow."
            )


def _definition_edges(root: Agent) -> Iterator[tuple[Agent, Agent]]:
    remaining = root._limits.max_subagent_definitions - 1
    if len(root.subagents) > remaining:
        _subagents.refuse(SubagentTreeTooLargeError(root.name, max_nodes=remaining + 1))
    stack = [(root, child) for child in root.subagents]
    while stack:
        parent, child = stack.pop()
        remaining -= 1
        yield parent, child
        if len(stack) + len(child.subagents) > remaining:
            _subagents.refuse(
                SubagentTreeTooLargeError(
                    root.name, max_nodes=root._limits.max_subagent_definitions
                )
            )
        stack.extend((child, grandchild) for grandchild in child.subagents)


def _require_react(parent: Agent, child: Agent) -> None:
    """Managed participants are durable ReAct runs only (owner 2026-11-12)."""
    loop = child._effective_loop(None)
    if loop is not None and not isinstance(loop, ReActLoop):
        _subagents.refuse(
            SubagentDefinitionError(
                parent.name, reason="loop_unsupported", subagent_name=child.name
            )
        )
