"""Tests for Agent definition class."""

import pytest

from dendrite.agent import Agent
from dendrite.tool import tool
from dendrite.types import ToolTarget


@tool(target="server")
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool(target="client")
async def read_range(sheet: str, range: str) -> list[list[str]]:
    """Read cell values from a range."""
    pass


class TestAgentSubclass:
    """Subclass style: class MyAgent(Agent)."""

    def test_basic_subclass(self):
        class MathAgent(Agent):
            model = "claude-sonnet-4-6"
            tools = [add]
            prompt = "You are a math assistant."

        agent = MathAgent()
        assert agent.name == "MathAgent"
        assert agent.model == "claude-sonnet-4-6"
        assert agent.prompt == "You are a math assistant."
        assert agent.tools == [add]
        assert agent.max_iterations == 10

    def test_subclass_custom_name(self):
        class MyAgent(Agent):
            name = "custom-agent"
            model = "claude-sonnet-4-6"
            prompt = "Hello."

        agent = MyAgent()
        assert agent.name == "custom-agent"

    def test_subclass_override_max_iterations(self):
        class AuditAgent(Agent):
            model = "claude-sonnet-4-6"
            prompt = "Audit."
            max_iterations = 15

        agent = AuditAgent()
        assert agent.max_iterations == 15

    def test_subclass_constructor_overrides_class_attrs(self):
        class MyAgent(Agent):
            model = "claude-sonnet-4-6"
            prompt = "Default prompt."
            max_iterations = 5

        agent = MyAgent(prompt="Override prompt.", max_iterations=20)
        assert agent.prompt == "Override prompt."
        assert agent.max_iterations == 20
        assert agent.model == "claude-sonnet-4-6"

    def test_subclass_no_tools_is_valid(self):
        class SimpleAgent(Agent):
            model = "claude-sonnet-4-6"
            prompt = "No tools needed."

        agent = SimpleAgent()
        assert agent.tools == []

    def test_two_instances_dont_share_tools(self):
        class AgentA(Agent):
            model = "claude-sonnet-4-6"
            prompt = "A."
            tools = [add]

        class AgentB(Agent):
            model = "claude-sonnet-4-6"
            prompt = "B."
            tools = [read_range]

        a = AgentA()
        b = AgentB()
        assert a.tools == [add]
        assert b.tools == [read_range]


class TestAgentConstructor:
    """Constructor style: Agent(model=..., tools=...)."""

    def test_basic_constructor(self):
        agent = Agent(
            model="claude-sonnet-4-6",
            tools=[add],
            prompt="You are a math assistant.",
        )
        assert agent.name == "Agent"
        assert agent.model == "claude-sonnet-4-6"
        assert agent.tools == [add]

    def test_constructor_with_name(self):
        agent = Agent(
            name="my-agent",
            model="claude-sonnet-4-6",
            prompt="Hello.",
        )
        assert agent.name == "my-agent"

    def test_constructor_with_max_iterations(self):
        agent = Agent(
            model="claude-sonnet-4-6",
            prompt="Hello.",
            max_iterations=25,
        )
        assert agent.max_iterations == 25


class TestAgentValidation:
    """Agent validates configuration at creation time."""

    def test_missing_model_raises(self):
        with pytest.raises(ValueError, match="requires a model"):
            Agent(prompt="Hello.")

    def test_missing_prompt_raises(self):
        with pytest.raises(ValueError, match="requires a prompt"):
            Agent(model="claude-sonnet-4-6")

    def test_non_tool_in_tools_raises(self):
        def plain_function():
            pass

        with pytest.raises(ValueError, match="non-tool"):
            Agent(
                model="claude-sonnet-4-6",
                prompt="Hello.",
                tools=[plain_function],
            )

    def test_max_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            Agent(
                model="claude-sonnet-4-6",
                prompt="Hello.",
                max_iterations=0,
            )

    def test_max_iterations_negative_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            Agent(
                model="claude-sonnet-4-6",
                prompt="Hello.",
                max_iterations=-1,
            )


class TestAgentToolDefs:
    """Agent.get_tool_defs() extracts ToolDefs from registered tools."""

    def test_get_tool_defs(self):
        agent = Agent(
            model="claude-sonnet-4-6",
            prompt="Hello.",
            tools=[add, read_range],
        )
        defs = agent.get_tool_defs()
        assert len(defs) == 2
        assert defs[0].name == "add"
        assert defs[0].target == ToolTarget.SERVER
        assert defs[1].name == "read_range"
        assert defs[1].target == ToolTarget.CLIENT

    def test_get_tool_defs_empty(self):
        agent = Agent(model="claude-sonnet-4-6", prompt="Hello.")
        assert agent.get_tool_defs() == []


class TestAgentRepr:
    """Agent has a useful repr."""

    def test_repr(self):
        agent = Agent(
            model="claude-sonnet-4-6",
            prompt="Hello.",
            tools=[add],
        )
        r = repr(agent)
        assert "Agent" in r
        assert "claude-sonnet-4-6" in r
        assert "tools=1" in r
        assert "max_iterations=10" in r
