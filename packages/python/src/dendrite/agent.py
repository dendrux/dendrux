"""Agent definition.

An Agent describes WHAT an agent is — model, tools, prompt, limits.
It does not run itself. The runner (runtime/runner.py) takes an Agent
definition and executes it through the loop.

Two creation styles:
    # Subclass — for reusable agents with custom logic
    class AuditAgent(Agent):
        model = "claude-sonnet-4-6"
        tools = [readRange, finishAudit]
        prompt = "You are a workbook auditor."
        max_iterations = 15

    # Constructor — for simple, config-driven agents
    agent = Agent(
        model="claude-sonnet-4-6",
        tools=[add, multiply],
        prompt="You are a math assistant.",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrite.tool import get_tool_def, is_tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrite.types import ToolDef

# Sentinel to detect "not provided" vs explicitly set to a value
_UNSET: Any = object()


class Agent:
    """Definition of a Dendrite agent.

    Describes the agent's identity, capabilities, and limits. The runtime
    reads this definition to configure the loop, strategy, and LLM provider.

    Attributes:
        name: Agent identifier. Auto-derived from class name if subclassed.
        model: LLM model identifier (e.g., "claude-sonnet-4-6").
        prompt: System prompt for the agent.
        tools: List of @tool-decorated functions this agent can use.
        max_iterations: Maximum ReAct loop iterations before stopping.
    """

    name: str = ""
    model: str = ""
    prompt: str = ""
    tools: list[Callable[..., Any]] = []
    max_iterations: int = 10

    def __init__(
        self,
        *,
        name: str = _UNSET,
        model: str = _UNSET,
        prompt: str = _UNSET,
        tools: list[Callable[..., Any]] = _UNSET,
        max_iterations: int = _UNSET,
    ) -> None:
        if name is not _UNSET:
            self.name = name
        elif not self.name:
            self.name = type(self).__name__

        if model is not _UNSET:
            self.model = model
        if prompt is not _UNSET:
            self.prompt = prompt
        if tools is not _UNSET:
            self.tools = list(tools)
        if max_iterations is not _UNSET:
            self.max_iterations = max_iterations

        self._validate()

    def _validate(self) -> None:
        """Validate agent configuration at creation time."""
        if not self.model:
            raise ValueError(
                f"Agent '{self.name}' requires a model. "
                f"Set model as a class attribute or pass it to the constructor."
            )

        if not self.prompt:
            raise ValueError(
                f"Agent '{self.name}' requires a prompt. "
                f"Set prompt as a class attribute or pass it to the constructor."
            )

        for fn in self.tools:
            if not is_tool(fn):
                raise ValueError(
                    f"Agent '{self.name}' has a non-tool in its tools list: "
                    f"'{getattr(fn, '__name__', fn)}'. Decorate it with @tool()."
                )

        if self.max_iterations < 1:
            raise ValueError(
                f"Agent '{self.name}' max_iterations must be >= 1, got {self.max_iterations}."
            )

    def get_tool_defs(self) -> list[ToolDef]:
        """Get ToolDef for each tool registered on this agent."""
        return [get_tool_def(fn) for fn in self.tools]

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, max_iterations={self.max_iterations})"
        )
