"""Which subagent definitions and loops a managed workflow supports, checked before effects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux.errors import (
    RunNotReclaimableError,
    SubagentDefinitionError,
    SubagentTreeTooLargeError,
)
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.runtime import _subagents

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dendrux.agent import Agent

logger = logging.getLogger(__name__)


def _check_definitions(root: Agent, store: Any) -> None:
    """Every registered child has a supported loop, a provider and the root's store object."""
    for parent, child in _definition_edges(root):
        require_bindable(parent, child)
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


def require_bindable(parent: Agent, child: Agent) -> None:
    """A child may be bound: a supported loop, and not guardrails with output_type.

    Checked at root entry and again at every bind, live or recovered (resume, submit,
    reclaim), before any write. SingleCall refuses the pair only inside its run, after
    the bind, so the parent's model could otherwise retry a call that can never work.
    """
    _require_supported_loop(parent, child)
    if child.guardrails and getattr(child, "_output_type", None) is not None:
        raise ValueError(f"Subagent '{child.name}' cannot combine guardrails with output_type.")


def _require_supported_loop(parent: Agent, child: Agent, *, paused: bool = False) -> None:
    """A managed child runs as ReAct or SingleCall; any other loop is refused.

    ``paused``: the child continues a paused run, which only ReAct can do.
    """
    loop = child._effective_loop(None)
    allowed = ReActLoop if paused else (ReActLoop, SingleCall)
    if loop is not None and not isinstance(loop, allowed):
        _subagents.refuse(
            SubagentDefinitionError(
                parent.name, reason="loop_unsupported", subagent_name=child.name
            )
        )


def require_resumable_leaf(found: Any) -> None:
    """A paused subagent is continued by ReAct only: a SingleCall agent never pauses.

    Refuses a definition that changed loop since the leaf paused, before its claim.
    """
    if found.routed and not isinstance(found.leaf_agent._effective_loop(None), ReActLoop):
        _require_supported_loop(found.waiting[-1].parent_agent, found.leaf_agent, paused=True)


def require_claimable(leaf_id: str, saved_loop: str | None) -> None:
    """A running SingleCall leaf is never claimed, on any store.

    It cannot continue from a checkpoint: ``sweep()`` fails it, then the root's reclaim
    adopts that failure. Refused before the store is asked, so nothing changes; for
    the caller ``loop_unsupported`` here means "run sweep(), then reclaim the root".
    """
    if saved_loop == "single_call":
        _codes.log_code(logger, _codes.LOOP_UNSUPPORTED, run_id=leaf_id)
        raise RunNotReclaimableError(leaf_id, "loop_unsupported")
