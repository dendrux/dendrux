"""Managed subagents (0.7): definition checks and the model-visible tool definition.

Pure functions over ``Agent`` definitions. Nothing here runs a child or touches a
store. ``check_subagent_tree`` is called when an ``Agent`` is built (early
validation) and is reusable at execution time, after MCP discovery and with the
run's effective depth ceiling.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, NoReturn

from dendrux import _codes
from dendrux._sentinel import _UnsetType
from dendrux.errors import (
    SubagentConfigurationError,
    SubagentCycleError,
    SubagentDefinitionError,
    SubagentDepthError,
    SubagentNameConflictError,
    SubagentTreeTooLargeError,
)
from dendrux.skills._session import NAMES as _SKILL_TOOL_NAMES
from dendrux.tool import get_tool_def, is_tool
from dendrux.types import Limits, ToolDef, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_MAX_DELEGATION_DEPTH",
    "MAX_DIRECT_SUBAGENTS_WARNING",
    "SUBAGENT_TASK_FIELD",
    "SubagentDefinitionError",
    "check_subagent_tree",
    "normalize_subagents",
    "refuse",
    "subagent_tool_def",
    "warn_fanout",
]

logger = logging.getLogger(__name__)

MAX_DIRECT_SUBAGENTS_WARNING = 10
"""More direct subagents than this logs ``subagent_fanout`` (a hint, never a refusal)."""

DEFAULT_MAX_DELEGATION_DEPTH = 10
"""The runner's default ceiling for a root run (``runtime/runner.py``); a test keeps them equal."""

SUBAGENT_TASK_FIELD = "task"
TASK_FIELD_DESCRIPTION = (
    "The complete task for this subagent. It starts fresh and sees only this text."
)

# Valid as a Python identifier (local @tool rule) and provider-safe ([a-zA-Z0-9_-]{1,64}).
_SUBAGENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_FROM_ROOT = _UnsetType()


_REFUSAL_CODES: dict[type[SubagentConfigurationError], _codes.Code] = {
    SubagentDefinitionError: _codes.SUBAGENT_INVALID,
    SubagentNameConflictError: _codes.SUBAGENT_NAME_CONFLICT,
    SubagentCycleError: _codes.SUBAGENT_CYCLE,
    SubagentDepthError: _codes.SUBAGENT_TOO_DEEP,
    SubagentTreeTooLargeError: _codes.SUBAGENT_TREE_TOO_LARGE,
}


def refuse(error: SubagentConfigurationError) -> NoReturn:
    """Log the error's stable code (names and numbers only), then raise it."""
    fields: dict[str, Any] = {}
    if isinstance(error, SubagentDepthError):
        fields = {"count": error.depth, "limit": error.max_depth}
    elif isinstance(error, SubagentTreeTooLargeError):
        fields = {"limit": error.max_nodes}
    _codes.log_code(logger, _REFUSAL_CODES[type(error)], **fields)
    raise error


def normalize_subagents(owner_name: str, subagents: Any) -> tuple[Any, ...]:
    """The ``subagents=`` value as a new tuple; refuses anything but a list or tuple."""
    if subagents is None:
        return ()
    if not isinstance(subagents, (list, tuple)):
        refuse(SubagentDefinitionError(owner_name, reason="not_a_list"))
    return tuple(subagents)


def warn_fanout(agent: Any) -> None:
    """Log ``subagent_fanout`` once when ``agent`` has many direct subagents."""
    count = len(agent.subagents)
    if count > MAX_DIRECT_SUBAGENTS_WARNING:
        _codes.log_code(
            logger, _codes.SUBAGENT_FANOUT, count=count, limit=MAX_DIRECT_SUBAGENTS_WARNING
        )


def check_subagent_tree(
    root: Any,
    *,
    agent_type: type[Any],
    extra_tool_names: Iterable[str] = (),
    max_depth: int | None | _UnsetType = _FROM_ROOT,
    max_nodes: int | None = None,
) -> None:
    """Refuse an invalid subagent tree below ``root`` with a typed error.

    Checks every agent's direct subagents: each is an ``agent_type`` with a
    valid name and a non-empty description, names are unique per parent and do
    not clash with that parent's tools (and its skill tools when it has skills;
    ``extra_tool_names`` adds names for the root, for example tools discovered
    later). Cycles are found on the current
    path by object identity, so one Agent may appear at several paths. Depth is
    checked against ``max_depth`` (default: the root's own setting, else 10;
    ``None`` is unbounded), tightened by each agent's own limit on its subtree.
    At most the root's definition cap (or ``max_nodes``) nodes are walked,
    root included: a list that
    does not fit the remaining budget is refused before any entry is inspected.
    """
    max_nodes = root._limits.max_subagent_definitions if max_nodes is None else max_nodes
    Limits(max_subagent_definitions=max_nodes)
    limit = _root_limit(root) if isinstance(max_depth, _UnsetType) else max_depth
    root_names = _tool_names(root) | frozenset(extra_tool_names)
    stack: list[tuple[Any, int, int | None, tuple[int, ...]]] = [(root, 0, limit, (id(root),))]
    walked = 1
    while stack:
        node, depth, node_limit, path = stack.pop()
        children = _children(node)
        if walked + len(children) > max_nodes:
            refuse(SubagentTreeTooLargeError(root.name, max_nodes=max_nodes))
        names = root_names if node is root else _tool_names(node)
        _check_siblings(node.name, children, agent_type, names)
        walked += len(children)
        for child in children:
            _check_edge(node, child, depth + 1, node_limit, path)
            child_limit = _tighten(node_limit, child.max_delegation_depth)
            stack.append((child, depth + 1, child_limit, (*path, id(child))))


def _check_edge(
    parent: Any, child: Any, child_depth: int, limit: int | None, path: tuple[int, ...]
) -> None:
    """Refuse ``child`` under ``parent`` if it closes a cycle or sits too deep."""
    if id(child) in path:
        refuse(SubagentCycleError(parent.name, subagent_name=child.name))
    if limit is not None and child_depth > limit:
        refuse(
            SubagentDepthError(
                parent.name, subagent_name=child.name, depth=child_depth, max_depth=limit
            )
        )


def subagent_tool_def(agent: Any) -> ToolDef:
    """The tool the parent's model sees for ``agent``: one required string, ``task``.

    ``target`` is the reserved ``ToolTarget.AGENT`` and ``parallel`` is False.
    Not wired into any loop: today an AGENT-target call would pause like a
    client tool instead of running.
    """
    return ToolDef(
        name=agent.name,
        description=agent.description,
        parameters={
            "type": "object",
            "properties": {
                SUBAGENT_TASK_FIELD: {"type": "string", "description": TASK_FIELD_DESCRIPTION}
            },
            "required": [SUBAGENT_TASK_FIELD],
            "additionalProperties": False,
        },
        target=ToolTarget.AGENT,
        parallel=False,
    )


def _children(agent: Any) -> Sequence[Any]:
    children = getattr(agent, "_subagents", ())
    return children if isinstance(children, (list, tuple)) else ()


def _root_limit(root: Any) -> int | None:
    own = root.max_delegation_depth
    return DEFAULT_MAX_DELEGATION_DEPTH if isinstance(own, _UnsetType) else own


def _tighten(limit: int | None, own: Any) -> int | None:
    """A child's own limit can only lower the ceiling on its subtree."""
    if not isinstance(own, int) or isinstance(own, bool):
        return limit
    return own if limit is None else min(limit, own)


def _tool_names(agent: Any) -> frozenset[str]:
    """Local tool names, plus the generated skill tool names when skills are configured."""
    names = frozenset(get_tool_def(fn).name for fn in agent.tools if is_tool(fn))
    if getattr(agent, "_skills_dir", None) or getattr(agent, "_explicit_skills", None):
        return names | _SKILL_TOOL_NAMES
    return names


def _check_siblings(
    owner: str, children: Sequence[Any], agent_type: type[Any], tool_names: frozenset[str]
) -> None:
    seen: set[str] = set()
    for index, child in enumerate(children):
        if not isinstance(child, agent_type):
            refuse(SubagentDefinitionError(owner, reason="not_agent", index=index))
        name = child.name
        if not isinstance(name, str) or _SUBAGENT_NAME.fullmatch(name) is None:
            refuse(SubagentDefinitionError(owner, reason="invalid_name", index=index))
        description = child.description
        if not isinstance(description, str) or not description.strip():
            refuse(
                SubagentDefinitionError(
                    owner, reason="missing_description", subagent_name=name, index=index
                )
            )
        if name in seen:
            refuse(
                SubagentNameConflictError(
                    owner, reason="duplicate_name", subagent_name=name, index=index
                )
            )
        if name in tool_names:
            refuse(
                SubagentNameConflictError(
                    owner, reason="tool_name_clash", subagent_name=name, index=index
                )
            )
        seen.add(name)
