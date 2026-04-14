"""Tests for agent.refresh() — universal cache invalidation.

Tests define what refresh() should do before implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dendrux.agent import Agent
from dendrux.skills import Skill
from dendrux.tool import tool

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "skills"


@tool(target="server")
def dummy_tool(x: int) -> int:
    """Dummy."""
    return x


class TestRefreshSkills:
    """refresh() clears skill caches so next run re-scans."""

    @pytest.mark.asyncio
    async def test_refresh_clears_skill_cache(self) -> None:
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])

        # Load skills
        agent._ensure_skills_loaded()
        assert agent._loaded_skills is not None
        assert len(agent._loaded_skills) == 1

        # Refresh
        await agent.refresh()

        # Caches cleared
        assert agent._loaded_skills is None
        assert agent._denied_skill_names is None

    @pytest.mark.asyncio
    async def test_refresh_picks_up_new_skills(self) -> None:
        """After refresh, new skills added to dir are discovered."""
        with tempfile.TemporaryDirectory() as d:
            skills_dir = Path(d)
            # Start with one skill
            skill_dir = skills_dir / "alpha"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: alpha\ndescription: First skill\n---\nAlpha body"
            )

            agent = Agent(
                prompt="Test.",
                tools=[dummy_tool],
                skills_dir=str(skills_dir),
            )

            # First load
            agent._ensure_skills_loaded()
            assert len(agent._loaded_skills) == 1

            # Add a second skill
            new_skill_dir = skills_dir / "beta"
            new_skill_dir.mkdir()
            (new_skill_dir / "SKILL.md").write_text(
                "---\nname: beta\ndescription: Second skill\n---\nBeta body"
            )

            # Before refresh — still cached with 1 skill
            agent._ensure_skills_loaded()
            assert len(agent._loaded_skills) == 1

            # Refresh and re-load
            await agent.refresh()
            agent._ensure_skills_loaded()
            assert len(agent._loaded_skills) == 2
            names = {s.name for s in agent._loaded_skills}
            assert "alpha" in names
            assert "beta" in names


class TestRefreshMCP:
    """refresh() clears MCP caches."""

    @pytest.mark.asyncio
    async def test_refresh_clears_mcp_cache(self) -> None:
        from dendrux.mcp import MCPServer

        source = MagicMock(spec=MCPServer)
        source.name = "fs"
        source.close = AsyncMock()

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            tool_sources=[source],
        )

        # Simulate discovery having run
        agent._discovered_tool_defs = []
        agent._mcp_executors = {}

        await agent.refresh()

        assert agent._discovered_tool_defs is None
        assert agent._mcp_executors is None
        source.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_clears_both_mcp_and_skills(self) -> None:
        """refresh() clears everything — one method, all caches."""
        from dendrux.mcp import MCPServer

        source = MagicMock(spec=MCPServer)
        source.name = "fs"
        source.close = AsyncMock()

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            tool_sources=[source],
            skills=[skill],
        )

        # Simulate both having loaded
        agent._discovered_tool_defs = []
        agent._mcp_executors = {}
        agent._ensure_skills_loaded()
        assert agent._loaded_skills is not None

        await agent.refresh()

        # Both cleared
        assert agent._discovered_tool_defs is None
        assert agent._mcp_executors is None
        assert agent._loaded_skills is None
        assert agent._denied_skill_names is None


class TestRefreshIdempotent:
    """refresh() is safe to call multiple times."""

    @pytest.mark.asyncio
    async def test_double_refresh(self) -> None:
        agent = Agent(prompt="Test.", tools=[dummy_tool])

        await agent.refresh()
        await agent.refresh()  # should not raise

    @pytest.mark.asyncio
    async def test_refresh_without_any_sources(self) -> None:
        """Agent with no MCP or skills — refresh is a no-op."""
        agent = Agent(prompt="Test.", tools=[dummy_tool])

        await agent.refresh()

        # get_system_prompt still works
        assert agent.get_system_prompt() == "Test."
