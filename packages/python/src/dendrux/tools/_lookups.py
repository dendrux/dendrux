"""Tool lookup tables for execution dispatch.

ToolLookups and build_tool_lookups were originally in loops/react.py.
Extracted here so agent.py, runtime/runner.py, and loops can all
import from a neutral module without layering violations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux.tool import get_tool_def

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.types import ToolDef, ToolTarget


class ToolLookups(NamedTuple):
    """Pre-computed lookups for tool execution — built once per run."""

    fn: dict[str, Callable[..., Any]]
    target: dict[str, ToolTarget]
    timeout: dict[str, float]
    explicit_timeout: dict[str, bool]
    max_calls: dict[str, int | None]
    parallel: dict[str, bool]


def build_tool_lookups(
    tools: list[Callable[..., Any]],
    *,
    mcp_executors: dict[str, Callable[..., Any]] | None = None,
    mcp_tool_defs: list[ToolDef] | None = None,
) -> ToolLookups:
    """Build all tool lookups from local tools and optional MCP tools.

    Args:
        tools: List of @tool-decorated Python functions.
        mcp_executors: Optional dict mapping namespaced MCP tool names
            to their executor callables.
        mcp_tool_defs: Optional list of ToolDefs for MCP tools.
            Must be provided alongside mcp_executors.
    """
    fn: dict[str, Callable[..., Any]] = {}
    target: dict[str, ToolTarget] = {}
    timeout: dict[str, float] = {}
    explicit_timeout: dict[str, bool] = {}
    max_calls: dict[str, int | None] = {}
    parallel: dict[str, bool] = {}

    # Local tools
    for func in tools:
        td = get_tool_def(func)
        if td.name in fn:
            raise ValueError(
                f"Duplicate tool name '{td.name}'. "
                f"Each tool registered on an agent must have a unique name."
            )
        fn[td.name] = func
        target[td.name] = td.target
        timeout[td.name] = td.timeout_seconds
        explicit_timeout[td.name] = td.has_explicit_timeout
        max_calls[td.name] = td.max_calls_per_run
        parallel[td.name] = td.parallel

    # MCP tools (if any)
    has_executors = mcp_executors is not None
    has_defs = mcp_tool_defs is not None
    if has_executors != has_defs:
        raise ValueError("mcp_executors and mcp_tool_defs must both be provided or both be None.")
    if has_executors and has_defs:
        assert mcp_executors is not None and mcp_tool_defs is not None
        def_names = {td.name for td in mcp_tool_defs}
        exec_names = set(mcp_executors.keys())
        if def_names != exec_names:
            raise ValueError(
                f"MCP tool defs and executors are out of sync. "
                f"Defs only: {def_names - exec_names}, "
                f"Executors only: {exec_names - def_names}. "
                f"This is an internal error — discovery should produce both."
            )
        for td in mcp_tool_defs:
            if td.name in fn:
                raise ValueError(
                    f"MCP tool name '{td.name}' collides with an existing tool. "
                    f"Each tool must have a unique namespaced name."
                )
            fn[td.name] = mcp_executors[td.name]
            target[td.name] = td.target
            timeout[td.name] = td.timeout_seconds
            explicit_timeout[td.name] = td.has_explicit_timeout
            max_calls[td.name] = td.max_calls_per_run
            parallel[td.name] = td.parallel

    return ToolLookups(fn, target, timeout, explicit_timeout, max_calls, parallel)
