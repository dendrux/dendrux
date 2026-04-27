"""SingleCall + skills end-to-end (inlined skill mode).

The runner builds the system prompt via ``Agent.get_system_prompt()`` and
sends it to the provider on the first (and only) LLM call. This test
asserts the skill body is in that recorded prompt and confirms no
``use_skill`` tool fires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.skills import Skill
from dendrux.types import LLMResponse, RunStatus, UsageStats

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "skills"


def _resp(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
    )


@pytest.fixture
def db_store(engine):
    return SQLAlchemyStateStore(engine)


class TestSingleCallInlinedSkillsE2E:
    async def test_skill_body_lands_in_recorded_system_prompt(self, db_store) -> None:
        """The skill's full markdown body must appear in the first LLM call's
        system message — proving inline delivery actually reaches the model."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        llm = MockLLM([_resp("ack")])
        agent = Agent(
            provider=llm,
            prompt="You are a chat participant.",
            loop=SingleCall(),
            skills=[skill],
            state_store=db_store,
        )

        with pytest.warns(UserWarning, match="inlined skill mode"):
            result = await agent.run("hello")

        assert result.status == RunStatus.SUCCESS

        # MockLLM records every call; the first call's messages must have
        # the skill body inline (system prompt is the first message in
        # NativeToolCalling strategy).
        assert len(llm.call_history) == 1
        messages = llm.call_history[0]["messages"]
        system_text = messages[0].content
        assert "### Skill: pdf-processing" in system_text
        assert skill.body in system_text

    async def test_no_use_skill_tool_offered(self, db_store) -> None:
        """SingleCall must not advertise use_skill in the call's tool list."""
        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")
        llm = MockLLM([_resp("ack")])
        agent = Agent(
            provider=llm,
            prompt="P.",
            loop=SingleCall(),
            skills=[skill],
            state_store=db_store,
        )

        with pytest.warns(UserWarning):
            await agent.run("hi")

        tool_names = [td.name for td in (llm.call_history[0]["tools"] or [])]
        assert "use_skill" not in tool_names
