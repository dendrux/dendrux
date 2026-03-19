"""Agent registry — configuration for hosted agent deployments.

Maps agent names to HostedAgentConfig, which stores factories (not live
instances) to avoid shared-state concurrency bugs across concurrent runs.

Lives at the package root because it serves multiple consumers:
server, worker, CLI, and future integrations. Not tied to any
specific deployment layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.loops.base import Loop
    from dendrite.strategies.base import Strategy


@dataclass
class HostedAgentConfig:
    """Configuration for a hosted agent.

    Stores factories instead of live instances so each run gets fresh
    provider/strategy/loop objects — no shared state across concurrent runs.

    Args:
        agent: Agent definition (immutable, safe to share).
        provider_factory: Creates a fresh LLMProvider per run.
        strategy_factory: Creates a fresh Strategy per run. None = NativeToolCalling.
        loop_factory: Creates a fresh Loop per run. None = ReActLoop.
        redact: Optional redaction policy applied to all persisted content.
    """

    agent: Agent
    provider_factory: Callable[[], LLMProvider]
    strategy_factory: Callable[[], Strategy] | None = None
    loop_factory: Callable[[], Loop] | None = None
    redact: Callable[[str], str] | None = None


class AgentRegistry:
    """Registry of hosted agent configurations.

    Usage:
        registry = AgentRegistry()
        registry.register(HostedAgentConfig(
            agent=my_agent,
            provider_factory=lambda: AnthropicProvider(api_key=key, model="claude-sonnet-4-6"),
        ))

        config = registry.get("MyAgent")
        provider = config.provider_factory()  # fresh instance per run
    """

    def __init__(self) -> None:
        self._configs: dict[str, HostedAgentConfig] = {}

    def register(self, config: HostedAgentConfig) -> None:
        """Register an agent configuration by agent name."""
        name = config.agent.name
        if name in self._configs:
            raise ValueError(f"Agent '{name}' is already registered.")
        self._configs[name] = config

    def get(self, agent_name: str) -> HostedAgentConfig:
        """Look up a registered agent config by name. Raises KeyError if not found."""
        if agent_name not in self._configs:
            raise KeyError(
                f"Agent '{agent_name}' is not registered. Available: {list(self._configs.keys())}"
            )
        return self._configs[agent_name]

    def list_agents(self) -> list[str]:
        """Return names of all registered agents."""
        return list(self._configs.keys())

    def __len__(self) -> int:
        return len(self._configs)
