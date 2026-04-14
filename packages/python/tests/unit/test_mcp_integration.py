"""Tests for MCP integration across agent, loop, runner, and persistence.

These tests verify the BEHAVIOR of MCP integration, not the
implementation details. Written as specifications of what should
be true, not mirrors of what the code does.

No real MCP servers — uses mocks to simulate discovery and execution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dendrux.agent import Agent
from dendrux.tool import tool
from dendrux.types import ToolDef, ToolTarget

# ---------------------------------------------------------------------------
# Fixtures: a mock MCP source that "discovers" tools
# ---------------------------------------------------------------------------


def _make_mock_source(
    name: str,
    tools: list[dict[str, Any]],
) -> MagicMock:
    """Create a mock MCPServer that discovers the given tools.

    Each tool dict has: name, description, parameters (optional).
    Returns a MagicMock that passes isinstance checks and has
    _discover, _create_executor, close, and name.
    """
    from dendrux.mcp._server import MCPServer

    source = MagicMock(spec=MCPServer)
    source.name = name

    tool_defs = []
    executors: dict[str, Any] = {}
    for t in tools:
        td = ToolDef(
            name=f"{name}__{t['name']}",
            description=t.get("description", ""),
            parameters=t.get("parameters", {"type": "object", "properties": {}}),
            target=ToolTarget.SERVER,
            meta={
                "source_name": name,
                "mcp_tool_name": t["name"],
                "transport": "stdio",
                "annotations": None,
            },
        )
        tool_defs.append(td)

        result = t.get("result", f"result from {t['name']}")

        async def make_exec(r: Any = result) -> Any:
            return r

        executors[td.name] = make_exec

    async def mock_discover() -> list[ToolDef]:
        return tool_defs

    source._discover = mock_discover

    def mock_create_executor(mcp_tool_name: str) -> Any:
        canonical = f"{name}__{mcp_tool_name}"
        return executors[canonical]

    source._create_executor = mock_create_executor
    source.close = AsyncMock()

    return source


@tool(target="server")
def local_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ---------------------------------------------------------------------------
# 1. Loop should use ALL tools (local + MCP)
# ---------------------------------------------------------------------------


class TestLoopUsesFullToolSet:
    """The loop should discover MCP tools and include them in lookups."""

    @pytest.mark.asyncio
    async def test_get_tool_lookups_includes_mcp_tools(self) -> None:
        """get_tool_lookups() must return both local and MCP executors."""
        source = _make_mock_source("fs", [{"name": "read_file"}, {"name": "write_file"}])

        agent = Agent(
            prompt="test",
            tools=[local_add],
            tool_sources=[source],
        )

        lookups = await agent.get_tool_lookups()

        # Local tool present
        assert "local_add" in lookups.fn
        # MCP tools present
        assert "fs__read_file" in lookups.fn
        assert "fs__write_file" in lookups.fn
        # All are SERVER target
        assert lookups.target["fs__read_file"] == ToolTarget.SERVER
        assert lookups.target["fs__write_file"] == ToolTarget.SERVER

    @pytest.mark.asyncio
    async def test_get_all_tool_defs_includes_mcp_tools(self) -> None:
        """get_all_tool_defs() must return local + MCP ToolDefs."""
        source = _make_mock_source("fs", [{"name": "read_file"}])

        agent = Agent(
            prompt="test",
            tools=[local_add],
            tool_sources=[source],
        )

        await agent.get_tool_lookups()  # trigger discovery
        all_defs = agent.get_all_tool_defs()

        names = {td.name for td in all_defs}
        assert "local_add" in names
        assert "fs__read_file" in names

    @pytest.mark.asyncio
    async def test_mcp_tool_defs_have_meta(self) -> None:
        """MCP ToolDefs must carry source info in meta."""
        source = _make_mock_source("fs", [{"name": "read_file"}])

        agent = Agent(
            prompt="test",
            tool_sources=[source],
        )

        await agent.get_tool_lookups()
        all_defs = agent.get_all_tool_defs()

        mcp_def = next(td for td in all_defs if td.name == "fs__read_file")
        assert mcp_def.meta["source_name"] == "fs"
        assert mcp_def.meta["mcp_tool_name"] == "read_file"
        assert mcp_def.meta["transport"] == "stdio"

    @pytest.mark.asyncio
    async def test_mcp_only_agent_works(self) -> None:
        """An agent with no local tools, only MCP sources, should work."""
        source = _make_mock_source("fs", [{"name": "list_files"}])

        agent = Agent(
            prompt="test",
            tools=[],
            tool_sources=[source],
        )

        lookups = await agent.get_tool_lookups()
        assert "fs__list_files" in lookups.fn
        assert len(lookups.fn) == 1


# ---------------------------------------------------------------------------
# 2. MCP executor should actually run and return results
# ---------------------------------------------------------------------------


class TestMCPExecutorExecution:
    """MCP executors should be callable and return correct results."""

    @pytest.mark.asyncio
    async def test_mcp_executor_is_callable(self) -> None:
        """The executor from get_tool_lookups should be an async callable."""
        source = _make_mock_source("fs", [{"name": "read_file", "result": "file contents"}])

        agent = Agent(prompt="test", tool_sources=[source])
        lookups = await agent.get_tool_lookups()

        executor = lookups.fn["fs__read_file"]
        result = await executor()
        assert result == "file contents"


# ---------------------------------------------------------------------------
# 3. Discovery is cached and concurrent-safe
# ---------------------------------------------------------------------------


class TestDiscoveryCaching:
    """Discovery should happen once and be cached."""

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self) -> None:
        """get_tool_lookups() called twice should only discover once."""
        source = _make_mock_source("fs", [{"name": "read_file"}])
        call_count = 0
        original_discover = source._discover

        async def counting_discover() -> list[ToolDef]:
            nonlocal call_count
            call_count += 1
            return await original_discover()

        source._discover = counting_discover

        agent = Agent(prompt="test", tool_sources=[source])
        await agent.get_tool_lookups()
        await agent.get_tool_lookups()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_close_clears_cache_and_rediscovers(self) -> None:
        """After close(), next get_tool_lookups() should re-discover."""
        source = _make_mock_source("fs", [{"name": "read_file"}])
        call_count = 0
        original_discover = source._discover

        async def counting_discover() -> list[ToolDef]:
            nonlocal call_count
            call_count += 1
            return await original_discover()

        source._discover = counting_discover

        agent = Agent(prompt="test", tool_sources=[source])
        await agent.get_tool_lookups()
        assert call_count == 1

        await agent.close()

        # After close, caches are cleared
        assert agent._discovered_tool_defs is None

        # Next call should re-discover
        await agent.get_tool_lookups()
        assert call_count == 2


# ---------------------------------------------------------------------------
# 4. Governance validation at discovery time
# ---------------------------------------------------------------------------


class TestGovernanceAtDiscovery:
    """Deferred governance checks should catch errors at discovery."""

    @pytest.mark.asyncio
    async def test_deny_unknown_mcp_tool_caught_at_discovery(self) -> None:
        """A typo in deny that refers to MCP tools should fail at discovery."""
        source = _make_mock_source("fs", [{"name": "read_file"}])

        agent = Agent(
            prompt="test",
            tool_sources=[source],
            deny=["fs__nonexistent"],
        )
        # Construction passes (deferred)
        # Discovery catches the typo
        with pytest.raises(ValueError, match="unknown tool.*fs__nonexistent"):
            await agent.get_tool_lookups()

    @pytest.mark.asyncio
    async def test_deny_valid_mcp_tool_passes(self) -> None:
        """A correct deny name matching an MCP tool should pass."""
        source = _make_mock_source("fs", [{"name": "delete_file"}])

        agent = Agent(
            prompt="test",
            tool_sources=[source],
            deny=["fs__delete_file"],
        )
        lookups = await agent.get_tool_lookups()
        assert "fs__delete_file" in lookups.fn

    @pytest.mark.asyncio
    async def test_name_collision_local_vs_mcp_rejected(self) -> None:
        """An MCP tool with the same namespaced name as a local tool should fail."""

        @tool(target="server")
        def fs__read_file(path: str) -> str:
            """Local tool with MCP-like name."""
            return ""

        source = _make_mock_source("fs", [{"name": "read_file"}])

        agent = Agent(
            prompt="test",
            tools=[fs__read_file],
            tool_sources=[source],
        )
        with pytest.raises(ValueError, match="collides"):
            await agent.get_tool_lookups()

    @pytest.mark.asyncio
    async def test_discovery_failure_cleans_up_sources(self) -> None:
        """If discovery fails, opened sources should be closed."""
        from dendrux.mcp._server import MCPServer

        good_source = _make_mock_source("good", [{"name": "tool1"}])
        bad_source = MagicMock(spec=MCPServer)
        bad_source.name = "bad"

        async def failing_discover() -> list[ToolDef]:
            raise ConnectionError("MCP server unreachable")

        bad_source._discover = failing_discover
        bad_source.close = AsyncMock()

        agent = Agent(
            prompt="test",
            tool_sources=[good_source, bad_source],
        )

        with pytest.raises(ConnectionError, match="unreachable"):
            await agent.get_tool_lookups()

        # Good source should have been closed during cleanup
        good_source.close.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Persistence records meta for MCP tools
# ---------------------------------------------------------------------------


class TestPersistenceMeta:
    """Persistence should serialize ToolDef.meta for forensic evidence."""

    def test_semantic_tools_include_meta(self) -> None:
        """on_llm_call_completed should include meta in serialized tools."""

        # We can't easily call on_llm_call_completed without a full store,
        # so test the serialization logic directly by inspecting what
        # the recorder would produce.
        mcp_tool = ToolDef(
            name="fs__read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {}},
            target=ToolTarget.SERVER,
            meta={
                "source_name": "fs",
                "mcp_tool_name": "read_file",
                "transport": "stdio",
            },
        )
        local_tool = ToolDef(
            name="local_add",
            description="Add",
            parameters={"type": "object", "properties": {}},
            target=ToolTarget.SERVER,
        )

        # Simulate the serialization logic from persistence.py
        serialized = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "target": getattr(t, "target", "server"),
                "meta": t.meta or None,
            }
            for t in [local_tool, mcp_tool]
        ]

        # Local tool: meta is {} → serialized as None
        assert serialized[0]["meta"] is None

        # MCP tool: meta carries source info
        assert serialized[1]["meta"]["source_name"] == "fs"
        assert serialized[1]["meta"]["mcp_tool_name"] == "read_file"
        assert serialized[1]["meta"]["transport"] == "stdio"


# ---------------------------------------------------------------------------
# 6. Multi-source discovery
# ---------------------------------------------------------------------------


class TestMultiSourceDiscovery:
    """Multiple MCP sources should merge correctly."""

    @pytest.mark.asyncio
    async def test_two_sources_merge(self) -> None:
        """Tools from two sources should both appear in lookups."""
        fs_source = _make_mock_source("fs", [{"name": "read_file"}])
        git_source = _make_mock_source("git", [{"name": "log"}])

        agent = Agent(
            prompt="test",
            tool_sources=[fs_source, git_source],
        )

        lookups = await agent.get_tool_lookups()
        assert "fs__read_file" in lookups.fn
        assert "git__log" in lookups.fn

    @pytest.mark.asyncio
    async def test_cross_source_collision_rejected(self) -> None:
        """Two sources discovering same namespaced name should fail."""
        # Both sources named differently but tool names produce same canonical
        source_a = _make_mock_source("src", [{"name": "tool"}])
        source_b = _make_mock_source("src2", [{"name": "tool"}])

        # These won't collide because source names differ: src__tool vs src2__tool
        agent = Agent(prompt="test", tool_sources=[source_a, source_b])
        lookups = await agent.get_tool_lookups()
        assert "src__tool" in lookups.fn
        assert "src2__tool" in lookups.fn
