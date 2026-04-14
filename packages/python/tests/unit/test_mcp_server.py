"""Unit tests for MCPServer and MCP-aware Agent validation.

Tests construction validation, name sanitization, provider-safe
name checks, result normalization, close idempotency, and
governance refactor for MCP-only agents.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dendrux.agent import Agent
from dendrux.tool import tool

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestMCPServerConstruction:
    """MCPServer(name, url=..., command=[...]) validation."""

    def test_stdio_transport(self) -> None:
        from dendrux.mcp import MCPServer

        server = MCPServer("filesystem", command=["npx", "-y", "server-fs", "/tmp"])
        assert server.name == "filesystem"
        assert server._command == ["npx", "-y", "server-fs", "/tmp"]
        assert server._url is None

    def test_http_transport(self) -> None:
        from dendrux.mcp import MCPServer

        server = MCPServer("linear", url="http://localhost:8080/mcp")
        assert server.name == "linear"
        assert server._url == "http://localhost:8080/mcp"
        assert server._command is None

    def test_no_transport_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="exactly one.*url.*command"):
            MCPServer("test")

    def test_both_transports_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="exactly one.*url.*command"):
            MCPServer("test", url="http://x", command=["npx"])

    def test_empty_command_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="non-empty"):
            MCPServer("test", command=[])

    def test_name_with_double_underscore_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="__"):
            MCPServer("my__server", command=["npx"])

    def test_name_with_dots_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="identifier"):
            MCPServer("my.server", command=["npx"])

    def test_empty_name_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="identifier"):
            MCPServer("", command=["npx"])

    def test_name_with_invalid_chars_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="identifier"):
            MCPServer("my-server!", command=["npx"])

    def test_valid_name_with_hyphens(self) -> None:
        """Hyphens are valid in source names — provider-safe."""
        from dendrux.mcp import MCPServer

        server = MCPServer("my-server", command=["npx"])
        assert server.name == "my-server"

    def test_command_as_string_raises(self) -> None:
        """Bare string command would split into single chars."""
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="non-empty list"):
            MCPServer("test", command="npx")  # type: ignore[arg-type]

    def test_command_with_non_string_elements_raises(self) -> None:
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="only strings"):
            MCPServer("test", command=["npx", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Provider-safe name validation
# ---------------------------------------------------------------------------


class TestProviderSafeNames:
    """Namespaced names must be safe for Anthropic/OpenAI tool schemas."""

    def test_valid_canonical_name(self) -> None:
        from dendrux.mcp._server import _validate_canonical_name

        # Should not raise
        _validate_canonical_name("filesystem__read_file", "filesystem", "read_file")

    def test_name_too_long_raises(self) -> None:
        from dendrux.mcp._server import _validate_canonical_name

        long_name = "a" * 65
        with pytest.raises(ValueError, match="provider-safe"):
            _validate_canonical_name(long_name, "src", "tool")

    def test_name_with_dots_raises(self) -> None:
        from dendrux.mcp._server import _validate_canonical_name

        with pytest.raises(ValueError, match="provider-safe"):
            _validate_canonical_name("fs__read.file", "fs", "read.file")

    def test_name_with_spaces_raises(self) -> None:
        from dendrux.mcp._server import _validate_canonical_name

        with pytest.raises(ValueError, match="provider-safe"):
            _validate_canonical_name("fs__read file", "fs", "read file")


class TestNameSanitization:
    """MCP tool names with non-alnum chars are sanitized."""

    def test_dots_replaced(self) -> None:
        from dendrux.mcp._server import _sanitize_tool_name

        assert _sanitize_tool_name("read.file") == "read_file"

    def test_slashes_replaced(self) -> None:
        from dendrux.mcp._server import _sanitize_tool_name

        assert _sanitize_tool_name("read/file") == "read_file"

    def test_hyphens_preserved(self) -> None:
        from dendrux.mcp._server import _sanitize_tool_name

        assert _sanitize_tool_name("read-file") == "read-file"

    def test_underscores_preserved(self) -> None:
        from dendrux.mcp._server import _sanitize_tool_name

        assert _sanitize_tool_name("read_file") == "read_file"

    def test_multiple_special_chars(self) -> None:
        from dendrux.mcp._server import _sanitize_tool_name

        assert _sanitize_tool_name("créer/ticket.now") == "cr_er_ticket_now"


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


class TestResultNormalization:
    """_normalize_mcp_result handles text, structuredContent, errors."""

    def _make_result(
        self,
        content: list[Any] | None = None,
        is_error: bool = False,
        structured_content: dict[str, Any] | None = None,
    ) -> MagicMock:
        """Create a mock CallToolResult."""
        result = MagicMock()
        result.isError = is_error
        result.structuredContent = structured_content
        if content is None:
            content = []
        result.content = content
        return result

    def _make_text_content(self, text: str) -> MagicMock:
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def _make_image_content(self) -> MagicMock:
        block = MagicMock()
        block.type = "image"
        return block

    def test_text_content(self) -> None:
        from dendrux.mcp._server import _normalize_mcp_result

        result = self._make_result(content=[self._make_text_content("hello world")])
        assert _normalize_mcp_result(result) == "hello world"

    def test_multiple_text_blocks(self) -> None:
        from dendrux.mcp._server import _normalize_mcp_result

        result = self._make_result(
            content=[self._make_text_content("line1"), self._make_text_content("line2")]
        )
        assert _normalize_mcp_result(result) == "line1\nline2"

    def test_empty_content(self) -> None:
        from dendrux.mcp._server import _normalize_mcp_result

        result = self._make_result(content=[])
        assert _normalize_mcp_result(result) == ""

    def test_unsupported_content_type(self) -> None:
        from dendrux.mcp._server import _normalize_mcp_result

        result = self._make_result(content=[self._make_image_content()])
        normalized = _normalize_mcp_result(result)
        assert "unsupported content type" in normalized
        assert "image" in normalized

    def test_structured_content_preferred(self) -> None:
        """structuredContent takes priority over text blocks."""
        from dendrux.mcp._server import _normalize_mcp_result

        result = self._make_result(
            content=[self._make_text_content("fallback")],
            structured_content={"key": "value", "count": 42},
        )
        normalized = _normalize_mcp_result(result)
        # Should return the dict, not a JSON string (loop handles serialization)
        assert isinstance(normalized, dict)
        assert normalized == {"key": "value", "count": 42}

    def test_structured_content_none_falls_back(self) -> None:
        from dendrux.mcp._server import _normalize_mcp_result

        result = self._make_result(
            content=[self._make_text_content("hello")],
            structured_content=None,
        )
        assert _normalize_mcp_result(result) == "hello"


# ---------------------------------------------------------------------------
# Close idempotency
# ---------------------------------------------------------------------------


class TestCloseIdempotency:
    """MCPServer.close() is safe to call multiple times."""

    @pytest.mark.asyncio
    async def test_close_without_connect(self) -> None:
        from dendrux.mcp import MCPServer

        server = MCPServer("test", command=["echo"])
        # Should not raise
        await server.close()
        await server.close()

    @pytest.mark.asyncio
    async def test_close_clears_session(self) -> None:
        from dendrux.mcp import MCPServer

        server = MCPServer("test", command=["echo"])
        # Simulate an exit stack
        server._exit_stack = AsyncMock()
        server._session = MagicMock()

        await server.close()

        assert server._session is None
        assert server._exit_stack is None

    @pytest.mark.asyncio
    async def test_discover_twice_raises(self) -> None:
        """Double _discover() without close() raises instead of leaking."""
        from dendrux.mcp import MCPServer

        server = MCPServer("test", command=["echo"])
        server._session = MagicMock()  # simulate already connected

        with pytest.raises(RuntimeError, match="already connected"):
            await server._discover()


# ---------------------------------------------------------------------------
# build_tool_lookups MCP validation
# ---------------------------------------------------------------------------


class TestBuildToolLookupsMCP:
    """build_tool_lookups rejects mismatched MCP args."""

    def test_executors_without_defs_raises(self) -> None:
        from dendrux.tools import build_tool_lookups

        with pytest.raises(ValueError, match="both be provided"):
            build_tool_lookups([], mcp_executors={"x": lambda: None}, mcp_tool_defs=None)

    def test_defs_without_executors_raises(self) -> None:
        from dendrux.tools import build_tool_lookups
        from dendrux.types import ToolDef

        td = ToolDef(name="x", description="", parameters={})
        with pytest.raises(ValueError, match="both be provided"):
            build_tool_lookups([], mcp_executors=None, mcp_tool_defs=[td])

    def test_def_without_matching_executor_raises(self) -> None:
        from dendrux.tools import build_tool_lookups
        from dendrux.types import ToolDef

        td = ToolDef(name="x__tool", description="", parameters={})
        with pytest.raises(ValueError, match="out of sync"):
            build_tool_lookups([], mcp_executors={}, mcp_tool_defs=[td])

    def test_valid_mcp_tools_merged(self) -> None:
        """Positive test — MCP tools actually appear in lookups."""
        from dendrux.tools import build_tool_lookups
        from dendrux.types import ToolDef, ToolTarget

        async def mock_executor(**params: Any) -> str:
            return "ok"

        td = ToolDef(
            name="fs__read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {}},
            target=ToolTarget.SERVER,
            meta={"source_name": "fs", "mcp_tool_name": "read_file"},
        )
        lookups = build_tool_lookups(
            [],
            mcp_executors={"fs__read_file": mock_executor},
            mcp_tool_defs=[td],
        )
        assert "fs__read_file" in lookups.fn
        assert lookups.fn["fs__read_file"] is mock_executor
        assert lookups.target["fs__read_file"] == ToolTarget.SERVER

    def test_extra_executor_raises(self) -> None:
        from dendrux.tools import build_tool_lookups
        from dendrux.types import ToolDef

        td = ToolDef(name="x__tool", description="", parameters={})
        with pytest.raises(ValueError, match="out of sync"):
            build_tool_lookups(
                [],
                mcp_executors={"x__tool": lambda: None, "x__extra": lambda: None},
                mcp_tool_defs=[td],
            )


# ---------------------------------------------------------------------------
# Agent governance with MCP tool_sources
# ---------------------------------------------------------------------------


@tool(target="server")
def _dummy_tool(x: int) -> int:
    """A dummy tool for tests."""
    return x


class TestAgentMCPGovernance:
    """Agent validation accepts MCP-only governance configs."""

    def test_deny_with_tool_sources_accepted(self) -> None:
        """deny= with no local tools but tool_sources should not raise."""
        from dendrux.mcp import MCPServer

        agent = Agent(
            prompt="test",
            tools=[],
            tool_sources=[MCPServer("fs", command=["echo"])],
            deny=["fs__delete_file"],
        )
        assert "fs__delete_file" in agent.deny

    def test_require_approval_with_tool_sources_accepted(self) -> None:
        from dendrux.mcp import MCPServer

        agent = Agent(
            prompt="test",
            tools=[],
            tool_sources=[MCPServer("fs", command=["echo"])],
            require_approval=["fs__write_file"],
        )
        assert "fs__write_file" in agent.require_approval

    def test_deny_no_tools_no_sources_still_raises(self) -> None:
        """Without tools or tool_sources, deny should still fail."""
        with pytest.raises(ValueError, match="no tools"):
            Agent(
                prompt="test",
                tools=[],
                deny=["something"],
            )

    def test_require_approval_no_tools_no_sources_still_raises(self) -> None:
        with pytest.raises(ValueError, match="no tools"):
            Agent(
                prompt="test",
                tools=[],
                require_approval=["something"],
            )

    def test_singlecall_rejects_tool_sources(self) -> None:
        from dendrux.loops.single import SingleCall
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="tool_source"):
            Agent(
                prompt="test",
                loop=SingleCall(),
                tool_sources=[MCPServer("fs", command=["echo"])],
            )

    def test_deny_name_check_skipped_with_tool_sources(self) -> None:
        """Name existence is deferred — unknown names don't raise at init."""
        from dendrux.mcp import MCPServer

        # This name doesn't exist in local tools, but tool_sources present
        # so name check is deferred to discovery time.
        agent = Agent(
            prompt="test",
            tools=[_dummy_tool],
            tool_sources=[MCPServer("fs", command=["echo"])],
            deny=["fs__nonexistent_tool"],
        )
        assert "fs__nonexistent_tool" in agent.deny

    def test_deny_name_check_without_sources_still_works(self) -> None:
        """Without tool_sources, name check fires at init as before."""
        with pytest.raises(ValueError, match="unknown tool"):
            Agent(
                prompt="test",
                tools=[_dummy_tool],
                deny=["nonexistent"],
            )

    def test_tool_sources_validates_instances(self) -> None:
        """Non-MCPServer entries in tool_sources raise at construction."""
        with pytest.raises(ValueError, match="not an MCPServer"):
            Agent(
                prompt="test",
                tool_sources=["bad"],  # type: ignore[list-item]
            )

    def test_require_approval_rejects_client_tool_with_sources(self) -> None:
        """require_approval on a local client tool fails at construction,
        even when tool_sources are present. No MCP discovery needed.

        This prevents the double-pause flow: client tool already pauses,
        approval on top would create a second pause.
        """
        from dendrux.mcp import MCPServer

        @tool(target="client")
        def client_tool(x: int) -> int:
            """A client-side tool."""
            return x

        with pytest.raises(ValueError, match="require_approval.*client_tool.*target=client"):
            Agent(
                prompt="test",
                tools=[client_tool],
                tool_sources=[MCPServer("fs", command=["echo"])],
                require_approval=["client_tool"],
            )

    def test_duplicate_source_names_rejected(self) -> None:
        """Two MCPServer entries with the same name are rejected."""
        from dendrux.mcp import MCPServer

        with pytest.raises(ValueError, match="duplicate.*fs"):
            Agent(
                prompt="test",
                tool_sources=[
                    MCPServer("fs", command=["echo"]),
                    MCPServer("fs", command=["echo", "2"]),
                ],
            )

    def test_client_tool_approval_caught_early_with_sources(self) -> None:
        """Client tool in require_approval fails at construction, not discovery."""
        from dendrux.mcp import MCPServer

        @tool(target="client")
        def my_client_tool(x: int) -> int:
            """Client tool."""
            return x

        with pytest.raises(ValueError, match="require_approval.*my_client_tool.*target=client"):
            Agent(
                prompt="test",
                tools=[my_client_tool],
                tool_sources=[MCPServer("fs", command=["echo"])],
                require_approval=["my_client_tool"],
            )

    @pytest.mark.asyncio
    async def test_close_clears_discovery_cache(self) -> None:
        """close() clears discovery caches so stale executors aren't reused."""
        from dendrux.mcp import MCPServer

        agent = Agent(
            prompt="test",
            tool_sources=[MCPServer("fs", command=["echo"])],
        )
        # Simulate discovery having run
        agent._discovered_tool_defs = []
        agent._mcp_executors = {}

        await agent.close()

        assert agent._discovered_tool_defs is None
        assert agent._mcp_executors is None
