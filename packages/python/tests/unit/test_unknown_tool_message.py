"""Unknown-tool result lists the tools the model was offered (#38)."""

from __future__ import annotations

import json
from typing import Any

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import _execute_tool
from dendrux.tools import build_tool_lookups
from dendrux.types import LLMResponse, RunStatus, ToolCall


@tool()
async def ping() -> str:
    """Ping."""
    return "pong"


# ---------------------------------------------------------------------------
# T1-c
# ---------------------------------------------------------------------------


def _tools(n: int) -> list[Any]:
    out = []
    for i in range(n):

        async def fn() -> str:
            return ""

        fn.__name__ = f"tool_{i:03d}"
        fn.__doc__ = "A tool."
        out.append(tool()(fn))
    return out


async def _unknown(lookups: Any) -> tuple[str, str]:
    result = await _execute_tool(ToolCall("bash", {}), lookups)
    assert result.success is False
    assert result.error is not None
    return result.error, json.loads(result.payload)["error"]


class TestUnknownTool:
    async def test_lists_available_tools_sorted(self) -> None:

        @tool()
        async def write_file() -> str:
            """W."""
            return ""

        @tool()
        async def read_file() -> str:
            """R."""
            return ""

        error, payload = await _unknown(build_tool_lookups([write_file, read_file]))
        expected = "Unknown tool: bash. Available tools: read_file, write_file"
        assert error == expected
        assert payload == expected

    async def test_no_tools(self) -> None:
        error, payload = await _unknown(build_tool_lookups([]))
        assert error == "Unknown tool: bash. No tools are available."
        assert payload == error

    async def test_caps_at_fifty(self) -> None:
        error, payload = await _unknown(build_tool_lookups(_tools(53)))
        listed = ", ".join(f"tool_{i:03d}" for i in range(50))
        assert error == f"Unknown tool: bash. Available tools: {listed}, … and 3 more"
        assert payload == error

    async def test_exactly_fifty_has_no_suffix(self) -> None:
        error, _ = await _unknown(build_tool_lookups(_tools(50)))
        assert error.endswith("tool_049")

    async def test_end_to_end_model_sees_the_list(self) -> None:
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("bash", {})]),
                LLMResponse(text="ok"),
            ]
        )
        result = await Agent(provider=llm, prompt="t", tools=[ping]).run("hi")
        assert result.status == RunStatus.SUCCESS
        tool_msg = llm.wire_messages[1][-1]
        assert "Unknown tool: bash. Available tools: ping" in tool_msg.content
