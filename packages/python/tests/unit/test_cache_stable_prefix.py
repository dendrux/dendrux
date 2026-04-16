"""Tests for cache-stable prefix sorting (PR 1).

Written before implementation — these tests specify the invariants
PR 1 must establish so provider prompt caches can hit across runs.

Three invariants:
  1. get_system_prompt() is byte-identical regardless of skill load order
  2. get_tool_defs() returns tools sorted by name
  3. get_all_tool_defs() — and the underlying _discovered_tool_defs
     storage — is sorted, regardless of MCP source or discovery order
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dendrux.agent import Agent
from dendrux.skills import Skill
from dendrux.tool import tool
from dendrux.types import ToolDef, ToolTarget

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "skills"


# ---------------------------------------------------------------------------
# MCP mock helper (duplicated from test_mcp_integration.py for isolation)
# ---------------------------------------------------------------------------


def _make_mock_source(name: str, tool_names: list[str]) -> MagicMock:
    """Build a mock MCPServer that discovers tools in the given order."""
    from dendrux.mcp._server import MCPServer

    source = MagicMock(spec=MCPServer)
    source.name = name

    tool_defs = [
        ToolDef(
            name=f"{name}__{tn}",
            description=f"{tn} tool",
            parameters={"type": "object", "properties": {}},
            target=ToolTarget.SERVER,
            meta={
                "source_name": name,
                "mcp_tool_name": tn,
                "transport": "stdio",
                "annotations": None,
            },
        )
        for tn in tool_names
    ]

    async def mock_discover() -> list[ToolDef]:
        return tool_defs

    source._discover = mock_discover
    source._create_executor = lambda mcp_tool_name: AsyncMock(return_value="ok")
    source.close = AsyncMock()
    return source


# ---------------------------------------------------------------------------
# Test tools used in get_tool_defs sort tests
# ---------------------------------------------------------------------------


@tool(target="server")
def zebra(x: int) -> int:
    """A z-tool."""
    return x


@tool(target="server")
def alpha(x: int) -> int:
    """An a-tool."""
    return x


@tool(target="server")
def mango(x: int) -> int:
    """An m-tool."""
    return x


# ---------------------------------------------------------------------------
# Invariant 1: system prompt byte-stability across skill load order
# ---------------------------------------------------------------------------


class TestSystemPromptByteStability:
    """get_system_prompt() output must not depend on the order skills
    were passed to Agent(skills=[...])."""

    def test_system_prompt_identical_across_skill_reversal(self) -> None:
        """Two agents with the same skills in reversed order produce
        byte-identical system prompts."""
        skill_a = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        skill_b = Skill.from_dir(FIXTURES_DIR / "report-gen")

        agent1 = Agent(prompt="Base prompt.", skills=[skill_a, skill_b])
        agent2 = Agent(prompt="Base prompt.", skills=[skill_b, skill_a])

        assert agent1.get_system_prompt() == agent2.get_system_prompt()

    def test_system_prompt_lists_skills_alphabetically(self) -> None:
        """Skill names appear in alphabetical order in the Available Skills
        section, so downstream cache prefixes match byte-for-byte."""
        skill_a = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        skill_b = Skill.from_dir(FIXTURES_DIR / "report-gen")

        # Pass in reverse alphabetical order
        agent = Agent(prompt="Base prompt.", skills=[skill_b, skill_a])
        prompt = agent.get_system_prompt()

        pdf_pos = prompt.find("pdf-processing")
        report_pos = prompt.find("report-gen")
        assert pdf_pos != -1 and report_pos != -1
        assert pdf_pos < report_pos, "pdf-processing must appear before report-gen (alphabetical)"

    def test_system_prompt_does_not_mutate_loaded_skills_order(self) -> None:
        """Sort must happen at serialization, not by mutating storage.
        self._loaded_skills preserves load order so anything reading it
        directly sees the dev-facing order."""
        skill_a = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        skill_b = Skill.from_dir(FIXTURES_DIR / "report-gen")

        agent = Agent(prompt="Base prompt.", skills=[skill_b, skill_a])
        agent.get_system_prompt()  # triggers lazy load + sort-at-serialize

        # Storage stays in load order
        assert agent._loaded_skills is not None
        assert [s.name for s in agent._loaded_skills] == [
            "report-gen",
            "pdf-processing",
        ]


# ---------------------------------------------------------------------------
# Invariant 2: local tool defs sorted by name
# ---------------------------------------------------------------------------


class TestToolDefsSortedByName:
    """get_tool_defs() must return local tools in alphabetical order so the
    provider tools[] payload is byte-stable across runs."""

    def test_local_tools_sorted_by_name(self) -> None:
        agent = Agent(prompt="p", tools=[zebra, alpha, mango])
        names = [td.name for td in agent.get_tool_defs()]
        assert names == ["alpha", "mango", "zebra"]

    def test_self_tools_preserves_caller_order(self) -> None:
        """Sort happens on the getter output, not in self.tools storage —
        dev-facing list reflects what they passed in."""
        agent = Agent(prompt="p", tools=[zebra, alpha, mango])
        assert agent.tools == [zebra, alpha, mango]

    def test_use_skill_sorts_alphabetically_not_last(self) -> None:
        """When skills are loaded, use_skill is a regular tool that slots
        into its alphabetical position (between tool names starting with
        letters before and after 'u'). It is not forcibly appended."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="p", tools=[zebra, alpha], skills=[skill])
        names = [td.name for td in agent.get_tool_defs()]
        # alphabetical: alpha, use_skill, zebra
        assert names == ["alpha", "use_skill", "zebra"]


# ---------------------------------------------------------------------------
# Invariant 3: get_all_tool_defs sorted, MCP storage sorted
# ---------------------------------------------------------------------------


class TestAllToolDefsWithMCPSortedEverywhere:
    """MCP tools must sort at the storage level so developers inspecting
    _discovered_tool_defs see alphabetical order. get_all_tool_defs()
    output is also sorted."""

    @pytest.mark.asyncio
    async def test_mcp_discovered_storage_is_sorted(self) -> None:
        """After _ensure_discovered(), _discovered_tool_defs is sorted
        by ToolDef.name regardless of MCP-yield order."""
        # Yield in reverse alphabetical order
        source = _make_mock_source("fs", ["write_file", "read_file", "list_dir"])
        agent = Agent(prompt="p", tool_sources=[source])

        await agent._ensure_discovered()

        assert agent._discovered_tool_defs is not None
        names = [td.name for td in agent._discovered_tool_defs]
        assert names == sorted(names)
        # Concretely: fs__list_dir, fs__read_file, fs__write_file
        assert names == ["fs__list_dir", "fs__read_file", "fs__write_file"]

    @pytest.mark.asyncio
    async def test_get_all_tool_defs_sorted_across_source_order(self) -> None:
        """Two agents with the same MCP sources passed in different order
        produce the same sorted get_all_tool_defs() output."""
        source_fs_1 = _make_mock_source("fs", ["read_file"])
        source_git_1 = _make_mock_source("git", ["log"])
        agent1 = Agent(
            prompt="p",
            tools=[alpha, zebra],
            tool_sources=[source_git_1, source_fs_1],
        )

        source_fs_2 = _make_mock_source("fs", ["read_file"])
        source_git_2 = _make_mock_source("git", ["log"])
        agent2 = Agent(
            prompt="p",
            tools=[zebra, alpha],
            tool_sources=[source_fs_2, source_git_2],
        )

        await agent1._ensure_discovered()
        await agent2._ensure_discovered()

        names1 = [td.name for td in agent1.get_all_tool_defs()]
        names2 = [td.name for td in agent2.get_all_tool_defs()]

        assert names1 == names2
        assert names1 == sorted(names1)
        assert names1 == ["alpha", "fs__read_file", "git__log", "zebra"]
