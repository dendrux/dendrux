"""Tool execution utilities.

Extracted from loops/react.py so that agent, loops, and runner
can all import from the same neutral module. ``tool_pool_stats()`` reports
the process-wide sync-tool pool, including timed-out calls still running.
"""

from dendrux.tools._limits import cap_tool_result, enforce_call_limits
from dendrux.tools._lookups import ToolLookups, build_tool_lookups
from dendrux.tools._pool import ToolPoolStats, tool_pool_stats

__all__ = [
    "ToolLookups",
    "ToolPoolStats",
    "build_tool_lookups",
    "cap_tool_result",
    "enforce_call_limits",
    "tool_pool_stats",
]
