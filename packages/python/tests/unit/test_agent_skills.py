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
# SingleCall + skills (inlined skill mode)
# ---------------------------------------------------------------------------


class TestSingleCallInlinedSkills:
    """SingleCall delivers skills inline into the system prompt.

    No use_skill tool is injected (one LLM call cannot do progressive
    disclosure). A UserWarning fires once at first prompt build with the
    rendered character count.
    """

    def test_singlecall_skills_dir_accepted(self) -> None:
        from dendrux.loops.single import SingleCall

        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Test.",
                loop=SingleCall(),
                skills_dir=str(skills_path),
            )
            assert agent._skills_dir == skills_path
        finally:
            shutil.rmtree(cleanup)

    def test_singlecall_explicit_skills_accepted(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(
            prompt="Test.",
            loop=SingleCall(),
            skills=[skill],
        )
        assert agent._explicit_skills == [skill]

    def test_inlined_prompt_contains_skill_body(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(
            prompt="Base prompt.",
            loop=SingleCall(),
            skills=[skill],
        )

        with pytest.warns(UserWarning, match="inlined skill mode"):
            prompt = agent.get_system_prompt()

        assert "Base prompt." in prompt
        assert "## Skills" in prompt
        assert "### Skill: pdf-processing" in prompt
        # Body text must be rendered verbatim, not just the description.
        assert skill.body in prompt
        # ReAct-mode catalogue text must NOT leak in.
        assert "use_skill" not in prompt
        assert "Available Skills" not in prompt

    def test_inlined_prompt_skills_alphabetically_sorted(self) -> None:
        from dendrux.loops.single import SingleCall

        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base.",
                loop=SingleCall(),
                skills_dir=str(skills_path),
            )
            with pytest.warns(UserWarning):
                prompt = agent.get_system_prompt()
            pdf_idx = prompt.index("### Skill: pdf-processing")
            report_idx = prompt.index("### Skill: report-gen")
            assert pdf_idx < report_idx
        finally:
            shutil.rmtree(cleanup)

    def test_inlined_prompt_respects_deny_skills(self) -> None:
        from dendrux.loops.single import SingleCall

        skills_path, cleanup = _make_clean_skills_dir()
        try:
            agent = Agent(
                prompt="Base.",
                loop=SingleCall(),
                skills_dir=str(skills_path),
                deny_skills=["report-gen"],
            )
            with pytest.warns(UserWarning):
                prompt = agent.get_system_prompt()
            assert "### Skill: pdf-processing" in prompt
            assert "### Skill: report-gen" not in prompt
        finally:
            shutil.rmtree(cleanup)

    def test_singlecall_skills_no_use_skill_tool(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(
            prompt="Test.",
            loop=SingleCall(),
            skills=[skill],
        )
        defs = agent.get_tool_defs()
        assert all(td.name != "use_skill" for td in defs)
        all_defs = agent.get_all_tool_defs()
        assert all(td.name != "use_skill" for td in all_defs)

    @pytest.mark.asyncio
    async def test_singlecall_skills_no_use_skill_in_lookups(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(
            prompt="Test.",
            loop=SingleCall(),
            skills=[skill],
        )
        lookups = await agent.get_tool_lookups()
        assert "use_skill" not in lookups.fn

    def test_warning_fires_once_per_agent(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(
            prompt="Base.",
            loop=SingleCall(),
            skills=[skill],
        )
        import warnings

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            agent.get_system_prompt()
            agent.get_system_prompt()
            agent.get_system_prompt()

        inline_warnings = [r for r in records if "inlined skill mode" in str(r.message)]
        assert len(inline_warnings) == 1

    def test_warning_includes_character_count(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(
            prompt="Base.",
            loop=SingleCall(),
            skills=[skill],
        )

        with pytest.warns(UserWarning, match=r"\d+ characters"):
            agent.get_system_prompt()

    def test_singlecall_tools_error_mentions_skills_supported(self) -> None:
        from dendrux.loops.single import SingleCall

        with pytest.raises(ValueError, match="Skills are still supported"):
            Agent(
                prompt="Test.",
                loop=SingleCall(),
                tools=[dummy_tool],
            )


class TestSkillDeliveryRespectsLoopOverride:
    """Skill delivery is driven by the *effective* loop passed to the
    getters, not ``Agent._loop``. Runtime ``loop=`` overrides via the
    runner therefore must produce a consistent prompt + tool set.
    """

    def test_react_agent_overridden_to_singlecall_inlines_bodies(self) -> None:
        """Agent built without an explicit loop (defaults to ReAct) but
        invoked with a SingleCall override must render bodies inline,
        not the catalogue + use_skill instructions."""
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", skills=[skill])  # no loop= → ReAct

        with pytest.warns(UserWarning, match="inlined skill mode"):
            prompt = agent.get_system_prompt(loop=SingleCall())

        assert "### Skill: pdf-processing" in prompt
        assert skill.body in prompt
        assert "use_skill" not in prompt
        assert "Available Skills" not in prompt

    def test_react_agent_overridden_to_singlecall_drops_use_skill_tool(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", skills=[skill])

        defs = agent.get_tool_defs(loop=SingleCall())
        assert all(td.name != "use_skill" for td in defs)

    @pytest.mark.asyncio
    async def test_react_override_drops_use_skill_in_lookups(self) -> None:
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", skills=[skill])

        lookups = await agent.get_tool_lookups(loop=SingleCall())
        assert "use_skill" not in lookups.fn

    def test_singlecall_agent_overridden_to_react_uses_catalogue(self) -> None:
        """Agent built with SingleCall but invoked with a ReAct override
        must produce the catalogue + use_skill view, not the inline one."""
        from dendrux.loops.react import ReActLoop
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", loop=SingleCall(), skills=[skill])

        prompt = agent.get_system_prompt(loop=ReActLoop())
        assert "Available Skills" in prompt
        assert "use_skill" in prompt
        # Body must NOT be inlined when delivery flips to catalogue.
        assert skill.body not in prompt
        assert "### Skill: pdf-processing" not in prompt

    def test_singlecall_agent_overridden_to_react_keeps_use_skill_tool(self) -> None:
        from dendrux.loops.react import ReActLoop
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", loop=SingleCall(), skills=[skill])

        defs = agent.get_tool_defs(loop=ReActLoop())
        assert any(td.name == "use_skill" for td in defs)

    @pytest.mark.asyncio
    async def test_singlecall_agent_overridden_to_react_keeps_use_skill_in_lookups(
        self,
    ) -> None:
        from dendrux.loops.react import ReActLoop
        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", loop=SingleCall(), skills=[skill])

        lookups = await agent.get_tool_lookups(loop=ReActLoop())
        assert "use_skill" in lookups.fn


class TestRunnerSkillCompatRelaxation:
    """``runner._validate_loop_skill_compat`` no longer rejects skills
    on a SingleCall override — delivery flips to inline instead."""

    def test_runner_validation_allows_singlecall_with_skills(self) -> None:
        from dendrux.loops.single import SingleCall
        from dendrux.runtime.runner import _validate_loop_skill_compat

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", skills=[skill])

        # No exception expected.
        _validate_loop_skill_compat(agent, SingleCall())

    def test_runner_validation_still_rejects_singlecall_with_tool_sources(self) -> None:
        from unittest.mock import AsyncMock

        from dendrux.loops.single import SingleCall
        from dendrux.runtime.runner import _validate_loop_skill_compat

        agent = Agent(prompt="Base.", tools=[dummy_tool])
        agent._tool_sources = [AsyncMock()]

        with pytest.raises(ValueError, match="(?i)tool_sources"):
            _validate_loop_skill_compat(agent, SingleCall())


class TestRefreshRearmsInlinedSkillWarning:
    """``refresh()`` clears the one-shot warning flag so the next
    prompt build re-warns about the (potentially new) rendered size.
    """

    def test_refresh_re_arms_warning(self) -> None:
        import warnings

        from dendrux.loops.single import SingleCall

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        agent = Agent(prompt="Base.", loop=SingleCall(), skills=[skill])

        with warnings.catch_warnings(record=True) as first_records:
            warnings.simplefilter("always")
            agent.get_system_prompt()
            agent.get_system_prompt()
        assert len([r for r in first_records if "inlined skill mode" in str(r.message)]) == 1

        # refresh() resets discovery + skill caches AND the warning flag.
        import asyncio

        asyncio.run(agent.refresh())

        with warnings.catch_warnings(record=True) as second_records:
            warnings.simplefilter("always")
            agent.get_system_prompt()
        assert len([r for r in second_records if "inlined skill mode" in str(r.message)]) == 1


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
