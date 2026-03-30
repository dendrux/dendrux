"""Tests for bridge() — paused-run interaction layer (G2)."""

from __future__ import annotations

import pytest

from dendrite.agent import Agent
from dendrite.llm.mock import MockLLM
from dendrite.tool import tool


@tool(target="server")
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool(target="client")
async def read_range(sheet: str, range: str) -> list[list[str]]:
    """Read cell values from a range."""
    pass


def _mock_provider(model: str = "claude-sonnet-4-6") -> MockLLM:
    return MockLLM([], model=model)


# ------------------------------------------------------------------
# bridge() factory validation
# ------------------------------------------------------------------


class TestBridgeFactory:
    def test_bridge_returns_fastapi_app(self):
        from dendrite.bridge import bridge

        agent = Agent(
            provider=_mock_provider(),
            prompt="Test.",
            tools=[add, read_range],
            database_url="sqlite+aiosqlite:///test.db",
        )
        app = bridge(agent, allow_insecure_dev_mode=True)

        from fastapi import FastAPI

        assert isinstance(app, FastAPI)

    def test_bridge_requires_auth_or_dev_mode(self):
        from dendrite.bridge import bridge

        agent = Agent(
            provider=_mock_provider(),
            prompt="Test.",
            database_url="sqlite+aiosqlite:///test.db",
        )
        with pytest.raises(ValueError, match="requires a secret"):
            bridge(agent)

    def test_bridge_requires_provider(self):
        from dendrite.bridge import bridge

        agent = Agent(
            prompt="Test.",
            database_url="sqlite+aiosqlite:///test.db",
        )
        with pytest.raises(ValueError, match="requires an agent with a provider"):
            bridge(agent, allow_insecure_dev_mode=True)

    def test_bridge_requires_persistence(self):
        from dendrite.bridge import bridge

        agent = Agent(
            provider=_mock_provider(),
            prompt="Test.",
        )
        with pytest.raises(ValueError, match="requires persistence"):
            bridge(agent, allow_insecure_dev_mode=True)

    def test_bridge_with_secret(self):
        from dendrite.bridge import bridge

        agent = Agent(
            provider=_mock_provider(),
            prompt="Test.",
            database_url="sqlite+aiosqlite:///test.db",
        )
        app = bridge(agent, secret="my-secret")

        from fastapi import FastAPI

        assert isinstance(app, FastAPI)
