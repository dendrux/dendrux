"""Pure policy inputs shared by live, approved and recovered tool calls."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from dendrux._lifecycle import DenialReason
from dendrux._meta_keys import NOT_EXECUTED_META_KEY
from dendrux.types import Role, ToolCallContext, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dendrux.types import BeforeToolHook, ImageRef, Message, ToolTarget


def history_call_counts(history: Iterable[Message]) -> dict[str, int]:
    """Count named tool observations unless explicitly marked unexecuted."""
    counts: dict[str, int] = {}
    for message in history:
        if (
            message.role == Role.TOOL
            and message.name is not None
            and not message.meta.get(NOT_EXECUTED_META_KEY)
        ):
            counts[message.name] = counts.get(message.name, 0) + 1
    return counts


def static_denial(name: str, call_id: str) -> tuple[ToolResult, dict[str, Any]]:
    """Build the model result and event data for a static policy denial."""
    message = f"Tool '{name}' is denied by policy."
    return (
        ToolResult(
            name=name,
            call_id=call_id,
            payload=json.dumps({"denied": message}),
            success=False,
            error=message,
        ),
        {"tool_name": name, "call_id": call_id, "reason": DenialReason.DENIED_BY_POLICY.value},
    )


def tool_call_context(
    *,
    run_id: str,
    call_id: str,
    tool_name: str,
    params: dict[str, Any],
    target: ToolTarget,
    source: Literal["local", "mcp", "skill"],
    meta: dict[str, Any],
    iteration: int,
    tenant_id: str | None,
    images: tuple[ImageRef, ...] = (),
) -> ToolCallContext:
    """Build a hook context whose mutable inputs are isolated from execution."""
    return ToolCallContext(
        run_id=run_id,
        call_id=call_id,
        tool_name=tool_name,
        params=copy.deepcopy(params),
        target=target,
        source=source,
        meta=copy.deepcopy(meta),
        iteration=iteration,
        tenant_id=tenant_id,
        images=images,
    )


class _DispatchPolicy(TypedDict):
    require_approval: frozenset[str]
    before_tool: BeforeToolHook | None
    approved_skills: bool


def dispatch_policy(
    *,
    name: str,
    phase: str,
    kind: str = "",
    require_approval: frozenset[str] = frozenset(),
    before_tool: BeforeToolHook | None = None,
) -> _DispatchPolicy:
    """Keep human approval authoritative while applying recovery approval gates."""
    if phase == "approved":
        return {"require_approval": frozenset(), "before_tool": None, "approved_skills": True}
    if kind == "approval":
        require_approval = require_approval | {name}
    return {
        "require_approval": require_approval,
        "before_tool": before_tool,
        "approved_skills": False,
    }
