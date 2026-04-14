"""Tests for Agent skills integration.

Written before implementation — defines what the Agent should do
with skills_dir, skills, deny_skills, and get_system_prompt().
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.skills import Skill
from dendrux.tool import tool

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "skills"


@tool(target="server")
def dummy_tool(x: int) -> int:
    """A dummy tool for tests."""
    return x


def _make_clean_skills_dir() -> tuple[Path, Any]:
    """Create a temp dir with valid skills. Returns (path, cleanup_handle)."""
    d = tempfile.mkdtemp()
    shutil.copytree(FIXTURES_DIR / "pdf-processing", Path(d) / "pdf-processing")
    shutil.copytree(FIXTURES_DIR / "report-gen", Path(d) / "report-gen")
    return Path(d), d


# ---------------------------------------------------------------------------
# skills_dir loading
# ---------------------------------------------------------------------------


class TestSkillsDirParam:
    """Agent(skills_dir=...) scans and loads skills."""

    def test_skills_dir_accepted(self) -> None:
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base prompt.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
            )
            assert agent._skills_dir == skills_path
        finally:
            shutil.rmtree(cleanup)

    @pytest.mark.asyncio
    async def test_skills_loaded_lazily(self) -> None:
        """Skills are not loaded at __init__, only at first run/get_system_prompt."""
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base prompt.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
            )
            # Not loaded yet
            assert agent._loaded_skills is None

            # Trigger lazy loading
            agent._ensure_skills_loaded()

            # Now loaded
            assert agent._loaded_skills is not None
            assert len(agent._loaded_skills) == 2
        finally:
            shutil.rmtree(cleanup)

    @pytest.mark.asyncio
    async def test_skills_cached_after_first_load(self) -> None:
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base prompt.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
            )
            agent._ensure_skills_loaded()
            first_load = agent._loaded_skills

            agent._ensure_skills_loaded()
            assert agent._loaded_skills is first_load  # same object
        finally:
            shutil.rmtree(cleanup)


# ---------------------------------------------------------------------------
# Explicit skills param
# ---------------------------------------------------------------------------


class TestSkillsParam:
    """Agent(skills=[...]) accepts explicit skill list."""

    def test_explicit_skills(self) -> None:
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Base prompt.",
            tools=[dummy_tool],
            skills=[skill],
        )
        agent._ensure_skills_loaded()

        assert len(agent._loaded_skills) == 1
        assert agent._loaded_skills[0].name == "pdf-processing"

    def test_skills_and_skills_dir_together(self) -> None:
        """Both skills= and skills_dir= should merge."""
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            extra_skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
            # skills_dir has pdf-processing + report-gen
            # skills= has pdf-processing (duplicate)
            # Should either deduplicate or raise
            agent = Agent(
                prompt="Base prompt.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
                skills=[extra_skill],
            )
            with pytest.raises(ValueError, match="duplicate"):
                agent._ensure_skills_loaded()
        finally:
            shutil.rmtree(cleanup)

    def test_no_skills_is_fine(self) -> None:
        """Agent without any skills should work normally."""
        agent = Agent(
            prompt="Base prompt.",
            tools=[dummy_tool],
        )
        agent._ensure_skills_loaded()
        assert agent._loaded_skills == []


# ---------------------------------------------------------------------------
# deny_skills governance
# ---------------------------------------------------------------------------


class TestDenySkills:
    """deny_skills filters skills from the loaded set."""

    def test_deny_removes_skill(self) -> None:
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base prompt.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
                deny_skills=["pdf-processing"],
            )
            agent._ensure_skills_loaded()

            names = [s.name for s in agent._loaded_skills]
            assert "pdf-processing" not in names
            assert "report-gen" in names
            assert "pdf-processing" in agent._denied_skill_names
        finally:
            shutil.rmtree(cleanup)

    def test_deny_unknown_name_allowed(self) -> None:
        """Unknown deny names are accepted — future-proof for refresh."""
        agent = Agent(
            prompt="Base prompt.",
            tools=[dummy_tool],
            deny_skills=["nonexistent-skill"],
        )
        assert "nonexistent-skill" in agent._deny_skills

    def test_deny_all_skills(self) -> None:
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base prompt.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
                deny_skills=["pdf-processing", "report-gen"],
            )
            agent._ensure_skills_loaded()

            assert agent._loaded_skills == []
            assert len(agent._denied_skill_names) == 2
        finally:
            shutil.rmtree(cleanup)


# ---------------------------------------------------------------------------
# get_system_prompt()
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    """get_system_prompt() composes base prompt + skill bodies."""

    def test_no_skills_returns_base(self) -> None:
        agent = Agent(prompt="You are helpful.", tools=[dummy_tool])
        agent._ensure_skills_loaded()

        assert agent.get_system_prompt() == "You are helpful."

    def test_with_skills_has_names_and_descriptions(self) -> None:
        """Progressive disclosure: prompt has names + descriptions, NOT full bodies."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="You are a doc assistant.",
            tools=[dummy_tool],
            skills=[skill],
        )
        agent._ensure_skills_loaded()

        prompt = agent.get_system_prompt()
        assert prompt.startswith("You are a doc assistant.")
        # Name and description present
        assert "pdf-processing" in prompt
        assert "Extract text" in prompt  # from description
        # Full body NOT in system prompt (progressive disclosure)
        assert "Edge Cases" not in prompt

    def test_prompt_mentions_use_skill(self) -> None:
        """System prompt should tell the LLM how to activate a skill."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Base.",
            tools=[dummy_tool],
            skills=[skill],
        )
        prompt = agent.get_system_prompt()
        assert "use_skill" in prompt

    def test_multiple_skills_all_listed(self) -> None:
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
            )
            agent._ensure_skills_loaded()

            prompt = agent.get_system_prompt()
            assert "pdf-processing" in prompt
            assert "report-gen" in prompt
        finally:
            shutil.rmtree(cleanup)

    def test_denied_skills_not_in_prompt(self) -> None:
        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base.",
                tools=[dummy_tool],
                skills_dir=str(skills_path),
                deny_skills=["pdf-processing"],
            )
            agent._ensure_skills_loaded()

            prompt = agent.get_system_prompt()
            assert "pdf-processing" not in prompt
            assert "report-gen" in prompt
        finally:
            shutil.rmtree(cleanup)

    def test_prompt_property_unchanged(self) -> None:
        """agent.prompt should return the base prompt, not composed."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Base only.",
            tools=[dummy_tool],
            skills=[skill],
        )

        assert agent.prompt == "Base only."
        assert "pdf-processing" in agent.get_system_prompt()

    def test_get_system_prompt_auto_loads(self) -> None:
        """get_system_prompt() should work without explicit _ensure_skills_loaded()."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Auto load test.",
            tools=[dummy_tool],
            skills=[skill],
        )
        prompt = agent.get_system_prompt()
        assert "Auto load test." in prompt
        assert "pdf-processing" in prompt


# ---------------------------------------------------------------------------
# use_skill tool
# ---------------------------------------------------------------------------


class TestUseSkillTool:
    """use_skill meta-tool is auto-registered and returns skill body."""

    def test_use_skill_in_tool_defs(self) -> None:
        """Agent with skills should have use_skill in tool defs."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            skills=[skill],
        )
        agent._ensure_skills_loaded()

        tool_names = [td.name for td in agent.get_tool_defs()]
        assert "use_skill" in tool_names

    def test_no_skills_no_use_skill_tool(self) -> None:
        """Agent without skills should NOT have use_skill."""
        agent = Agent(prompt="Test.", tools=[dummy_tool])
        agent._ensure_skills_loaded()

        tool_names = [td.name for td in agent.get_tool_defs()]
        assert "use_skill" not in tool_names

    @pytest.mark.asyncio
    async def test_use_skill_returns_body(self) -> None:
        """Calling use_skill with a valid name returns the skill body."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            skills=[skill],
        )
        agent._ensure_skills_loaded()

        result = await agent._execute_use_skill(name="pdf-processing")
        assert "Instructions" in result
        assert "read_file" in result

    @pytest.mark.asyncio
    async def test_use_skill_unknown_name_returns_error(self) -> None:
        """Calling use_skill with unknown name returns error message."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            skills=[skill],
        )
        agent._ensure_skills_loaded()

        result = await agent._execute_use_skill(name="nonexistent")
        assert "not found" in result.lower() or "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_use_skill_denied_name_returns_error(self) -> None:
        """Calling use_skill with denied name returns error."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(
            prompt="Test.",
            tools=[dummy_tool],
            skills=[skill],
            deny_skills=["pdf-processing"],
        )
        agent._ensure_skills_loaded()

        result = await agent._execute_use_skill(name="pdf-processing")
        assert "not found" in result.lower() or "not available" in result.lower()


# ---------------------------------------------------------------------------
# SingleCall + skills
# ---------------------------------------------------------------------------


class TestSingleCallSkills:
    """SingleCall agents cannot have skills."""

    def test_singlecall_rejects_skills_dir(self) -> None:
        from dendrux.loops.single import SingleCall

        with pytest.raises(ValueError, match="skill"):
            Agent(
                prompt="Test.",
                loop=SingleCall(),
                skills_dir="./skills",
            )

    def test_singlecall_rejects_explicit_skills(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        with pytest.raises(ValueError, match="skill"):
            Agent(
                prompt="Test.",
                loop=SingleCall(),
                skills=[skill],
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSkillsValidation:
    """Agent validates skill-related config at construction."""

    def test_invalid_skills_dir_type(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            Agent(
                prompt="Test.",
                tools=[dummy_tool],
                skills_dir=123,  # type: ignore[arg-type]
            )

    def test_skills_entries_must_be_skill_instances(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            Agent(
                prompt="Test.",
                tools=[dummy_tool],
                skills=["not a skill"],  # type: ignore[list-item]
            )

    @pytest.mark.asyncio
    async def test_close_clears_skill_cache(self) -> None:
        """close() should clear skill caches like it clears MCP caches."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        agent = Agent(prompt="Test.", tools=[dummy_tool], skills=[skill])

        agent._ensure_skills_loaded()
        assert agent._loaded_skills is not None

        await agent.close()

        assert agent._loaded_skills is None
        assert agent._denied_skill_names is None

    def test_use_skill_name_reserved(self) -> None:
        """A local tool named 'use_skill' conflicts with the meta-tool."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        @tool(target="server")
        def use_skill(name: str) -> str:
            """Conflicts with meta-tool."""
            return name

        with pytest.raises(ValueError, match="reserved"):
            Agent(
                prompt="Test.",
                tools=[use_skill],
                skills=[skill],
            )
