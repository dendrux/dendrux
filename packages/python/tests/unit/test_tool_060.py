"""Tool replay is always an explicit opt in."""

import inspect

from dendrux.tool import get_tool_def, tool


def test_tool_idempotent_defaults_and_override():
    assert "idempotent" in inspect.signature(tool).parameters, (
        "replay needs a declared safety promise"
    )

    @tool(idempotent=True)
    async def read() -> str:
        """Read a stable value."""
        return "value"

    @tool()
    async def write() -> str:
        """Write an effect."""
        return "done"

    assert get_tool_def(read).idempotent is True
    assert get_tool_def(write).idempotent is False
