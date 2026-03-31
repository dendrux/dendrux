"""Tests for Agent — definition, validation, and runtime facade (G1)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.tool import tool
from dendrux.types import ToolTarget


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
# Subclass style
# ------------------------------------------------------------------


class TestAgentSubclass:
    """Subclass style: class MyAgent(Agent)."""

    def test_basic_subclass(self):
        class MathAgent(Agent):
            tools = [add]
            prompt = "You are a math assistant."

        provider = _mock_provider()
        agent = MathAgent(provider=provider)
        assert agent.name == "MathAgent"
        assert agent.model == "claude-sonnet-4-6"
        assert agent.prompt == "You are a math assistant."
        assert agent.tools == [add]
        assert agent.max_iterations == 10

    def test_subclass_custom_name(self):
        class MyAgent(Agent):
            name = "custom-agent"
            prompt = "Hello."

        agent = MyAgent(provider=_mock_provider())
        assert agent.name == "custom-agent"

    def test_subclass_override_max_iterations(self):
        class AuditAgent(Agent):
            prompt = "Audit."
            max_iterations = 15

        agent = AuditAgent(provider=_mock_provider())
        assert agent.max_iterations == 15

    def test_subclass_constructor_overrides_class_attrs(self):
        class MyAgent(Agent):
            prompt = "Default prompt."
            max_iterations = 5

        agent = MyAgent(provider=_mock_provider(), prompt="Override prompt.", max_iterations=20)
        assert agent.prompt == "Override prompt."
        assert agent.max_iterations == 20

    def test_subclass_no_tools_is_valid(self):
        class SimpleAgent(Agent):
            prompt = "No tools needed."

        agent = SimpleAgent(provider=_mock_provider())
        assert agent.tools == []

    def test_two_instances_dont_share_tools(self):
        class AgentA(Agent):
            prompt = "A."
            tools = [add]

        class AgentB(Agent):
            prompt = "B."
            tools = [read_range]

        a = AgentA(provider=_mock_provider())
        b = AgentB(provider=_mock_provider())
        assert a.tools == [add]
        assert b.tools == [read_range]

    def test_subclass_instances_get_separate_tool_lists(self):
        """C1: Two instances of the same subclass must have independent tool lists."""

        class SharedAgent(Agent):
            prompt = "Shared."
            tools = [add]

        a1 = SharedAgent(provider=_mock_provider())
        a2 = SharedAgent(provider=_mock_provider())
        assert a1.tools is not a2.tools
        assert a1.tools == a2.tools  # same contents, different lists

    def test_mutating_one_instance_tools_doesnt_affect_another(self):
        """C1: Mutating one instance's tools must not cross-contaminate."""

        class MutableAgent(Agent):
            prompt = "Test."
            tools = [add]

        a1 = MutableAgent(provider=_mock_provider())
        a2 = MutableAgent(provider=_mock_provider())
        a1.tools.append(read_range)
        assert len(a1.tools) == 2
        assert len(a2.tools) == 1


# ------------------------------------------------------------------
# Constructor style
# ------------------------------------------------------------------


class TestAgentConstructor:
    """Constructor style: Agent(provider=..., tools=...)."""

    def test_basic_constructor(self):
        agent = Agent(
            provider=_mock_provider(),
            tools=[add],
            prompt="You are a math assistant.",
        )
        assert agent.name == "Agent"
        assert agent.model == "claude-sonnet-4-6"
        assert agent.tools == [add]

    def test_constructor_with_name(self):
        agent = Agent(
            name="my-agent",
            provider=_mock_provider(),
            prompt="Hello.",
        )
        assert agent.name == "my-agent"

    def test_constructor_with_max_iterations(self):
        agent = Agent(
            provider=_mock_provider(),
            prompt="Hello.",
            max_iterations=25,
        )
        assert agent.max_iterations == 25

    def test_constructor_without_provider_is_valid(self):
        """Agent without provider is valid — standalone run() provides its own."""
        agent = Agent(prompt="Hello.")
        assert agent.provider is None
        assert agent.model == ""


# ------------------------------------------------------------------
# Provider
# ------------------------------------------------------------------


class TestAgentProvider:
    def test_provider_stored(self):
        provider = _mock_provider()
        agent = Agent(provider=provider, prompt="Hello.")
        assert agent.provider is provider

    def test_model_reads_from_provider(self):
        agent = Agent(provider=_mock_provider("gpt-4"), prompt="Hello.")
        assert agent.model == "gpt-4"

    def test_model_empty_when_no_provider(self):
        agent = Agent(prompt="Hello.")
        assert agent.model == ""

    def test_class_level_model_warns(self):
        with pytest.warns(DeprecationWarning, match="Class-level 'model'"):

            class OldStyleAgent(Agent):
                model = "claude-sonnet-4-6"  # type: ignore[assignment]
                prompt = "Hello."

        # model attribute is removed, property takes over
        agent = OldStyleAgent(provider=_mock_provider("actual-model"))
        assert agent.model == "actual-model"

    def test_class_level_provider_blocked(self):
        with pytest.raises(ValueError, match="provider must be passed to __init__"):

            class BadAgent(Agent):
                provider = _mock_provider()
                prompt = "Hello."

            BadAgent()


# ------------------------------------------------------------------
# Persistence config
# ------------------------------------------------------------------


class TestAgentPersistence:
    def test_database_url_stored(self):
        agent = Agent(provider=_mock_provider(), prompt="Hello.", database_url="sqlite:///test.db")
        assert agent._database_url == "sqlite:///test.db"

    def test_state_store_stored(self):
        mock_store = AsyncMock()
        agent = Agent(provider=_mock_provider(), prompt="Hello.", state_store=mock_store)
        assert agent._state_store is mock_store

    def test_database_url_and_state_store_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                database_url="sqlite:///test.db",
                state_store=AsyncMock(),
            )

    def test_database_options_requires_database_url(self):
        with pytest.raises(ValueError, match="database_options requires database_url"):
            Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                database_options={"pool_size": 5},
            )

    def test_database_options_with_url_is_valid(self):
        agent = Agent(
            provider=_mock_provider(),
            prompt="Hello.",
            database_url="sqlite:///test.db",
            database_options={"pool_size": 5},
        )
        assert agent._database_options == {"pool_size": 5}

    def test_redact_stored(self):
        redact_fn = lambda text: text.replace("secret", "[REDACTED]")  # noqa: E731
        agent = Agent(provider=_mock_provider(), prompt="Hello.", redact=redact_fn)
        assert agent._redact is redact_fn


# ------------------------------------------------------------------
# Persistence: private engine (finding 2 + 3)
# ------------------------------------------------------------------


class TestAgentPrivateEngine:
    """database_url creates a private engine — no global singleton conflict."""

    async def test_two_agents_different_urls_coexist(self):
        """Two agents with different database_urls can both resolve stores."""
        with tempfile.TemporaryDirectory() as tmp:
            url_a = f"sqlite+aiosqlite:///{Path(tmp) / 'a.db'}"
            url_b = f"sqlite+aiosqlite:///{Path(tmp) / 'b.db'}"

            agent_a = Agent(provider=_mock_provider(), prompt="A.", database_url=url_a)
            agent_b = Agent(provider=_mock_provider(), prompt="B.", database_url=url_b)

            store_a = await agent_a._resolve_state_store()
            store_b = await agent_b._resolve_state_store()

            assert store_a is not None
            assert store_b is not None
            # Different engines — no singleton conflict
            assert agent_a._private_engine is not agent_b._private_engine

            await agent_a.close()
            await agent_b.close()

    async def test_database_options_reach_engine(self):
        """database_options are passed through to create_async_engine."""
        with tempfile.TemporaryDirectory() as tmp:
            url = f"sqlite+aiosqlite:///{Path(tmp) / 'opts.db'}"
            agent = Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                database_url=url,
                database_options={"pool_pre_ping": True},
            )

            store = await agent._resolve_state_store()
            assert store is not None
            assert agent._private_engine is not None
            # pool_pre_ping is stored on the engine's pool
            assert agent._private_engine.pool._pre_ping is True

            await agent.close()

    async def test_close_disposes_private_engine(self):
        """agent.close() disposes the agent-owned engine."""
        with tempfile.TemporaryDirectory() as tmp:
            url = f"sqlite+aiosqlite:///{Path(tmp) / 'close.db'}"
            agent = Agent(provider=_mock_provider(), prompt="Hello.", database_url=url)

            await agent._resolve_state_store()
            engine = agent._private_engine
            assert engine is not None

            await agent.close()
            # Private engine reference is cleared after close
            assert agent._private_engine is None

    async def test_close_clears_lazy_store_so_reuse_works(self):
        """After close(), re-resolving creates a fresh engine and store."""
        with tempfile.TemporaryDirectory() as tmp:
            url = f"sqlite+aiosqlite:///{Path(tmp) / 'reuse.db'}"
            agent = Agent(provider=_mock_provider(), prompt="Hello.", database_url=url)

            store_before = await agent._resolve_state_store()
            engine_before = agent._private_engine
            assert store_before is not None

            await agent.close()
            assert agent._private_engine is None

            # Re-resolve after close — must get a fresh engine, not stale store
            store_after = await agent._resolve_state_store()
            assert store_after is not None
            assert store_after is not store_before
            assert agent._private_engine is not engine_before

            await agent.close()

    async def test_env_var_fallback_uses_shared_engine(self):
        """DENDRUX_DATABASE_URL env var path uses the global get_engine(), not private."""
        with (
            patch("dendrux.agent.os.environ.get", return_value="sqlite+aiosqlite:///env.db"),
            patch("dendrux.db.session.get_engine", new_callable=AsyncMock) as mock_get,
        ):
            from sqlalchemy.ext.asyncio import create_async_engine

            # Create a real in-memory engine for the mock to return
            mock_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
            mock_get.return_value = mock_engine

            agent = Agent(provider=_mock_provider(), prompt="Hello.")
            store = await agent._resolve_state_store()

            assert store is not None
            # Used the global singleton, not a private engine
            mock_get.assert_awaited_once_with("sqlite+aiosqlite:///env.db")
            assert agent._private_engine is None

            await mock_engine.dispose()


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestAgentValidation:
    def test_missing_prompt_raises(self):
        with pytest.raises(ValueError, match="requires a prompt"):
            Agent(provider=_mock_provider())

    def test_non_tool_in_tools_raises(self):
        def plain_function():
            pass

        with pytest.raises(ValueError, match="non-tool"):
            Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                tools=[plain_function],
            )

    def test_max_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                max_iterations=0,
            )

    def test_max_iterations_negative_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                max_iterations=-1,
            )

    def test_max_iterations_exceeds_ceiling_raises(self):
        from dendrux.agent import MAX_ITERATIONS_CEILING

        with pytest.raises(ValueError, match="cannot exceed"):
            Agent(
                provider=_mock_provider(),
                prompt="Hello.",
                max_iterations=MAX_ITERATIONS_CEILING + 1,
            )

    def test_max_iterations_at_ceiling_is_valid(self):
        from dendrux.agent import MAX_ITERATIONS_CEILING

        agent = Agent(
            provider=_mock_provider(),
            prompt="Hello.",
            max_iterations=MAX_ITERATIONS_CEILING,
        )
        assert agent.max_iterations == MAX_ITERATIONS_CEILING


# ------------------------------------------------------------------
# Runtime: agent.run()
# ------------------------------------------------------------------


class TestAgentRun:
    async def test_run_without_provider_raises(self):
        agent = Agent(prompt="Hello.")
        with pytest.raises(ValueError, match="requires a provider"):
            await agent.run("test")


# ------------------------------------------------------------------
# Runtime: agent.resume()
# ------------------------------------------------------------------


class TestAgentResume:
    async def test_resume_without_provider_raises(self):
        agent = Agent(prompt="Hello.", database_url="sqlite:///test.db")
        with pytest.raises(ValueError, match="requires a provider"):
            await agent.resume("run-123", tool_results=[])

    async def test_resume_without_persistence_raises(self):
        agent = Agent(provider=_mock_provider(), prompt="Hello.")
        with pytest.raises(ValueError, match="requires persistence"):
            await agent.resume("run-123", tool_results=[])

    async def test_resume_with_both_args_raises(self):
        agent = Agent(
            provider=_mock_provider(),
            prompt="Hello.",
            database_url="sqlite:///test.db",
        )
        with pytest.raises(ValueError, match="Cannot provide both"):
            await agent.resume("run-123", tool_results=[], user_input="test")

    async def test_resume_with_no_args_raises(self):
        agent = Agent(
            provider=_mock_provider(),
            prompt="Hello.",
            database_url="sqlite:///test.db",
        )
        with pytest.raises(ValueError, match="requires either tool_results or user_input"):
            await agent.resume("run-123")


# ------------------------------------------------------------------
# Lifecycle: async with
# ------------------------------------------------------------------


class TestAgentLifecycle:
    async def test_close_closes_provider(self):
        provider = _mock_provider()
        provider.close = AsyncMock()  # type: ignore[method-assign]
        agent = Agent(provider=provider, prompt="Hello.")
        await agent.close()
        provider.close.assert_awaited_once()

    async def test_context_manager_closes_on_exit(self):
        provider = _mock_provider()
        provider.close = AsyncMock()  # type: ignore[method-assign]
        agent = Agent(provider=provider, prompt="Hello.")
        async with agent as a:
            assert a is agent
        provider.close.assert_awaited_once()

    async def test_close_without_provider_is_noop(self):
        agent = Agent(prompt="Hello.")
        await agent.close()  # should not raise


# ------------------------------------------------------------------
# Introspection
# ------------------------------------------------------------------


class TestAgentToolDefs:
    def test_get_tool_defs(self):
        agent = Agent(
            provider=_mock_provider(),
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
        agent = Agent(provider=_mock_provider(), prompt="Hello.")
        assert agent.get_tool_defs() == []


class TestAgentRepr:
    def test_repr(self):
        agent = Agent(
            provider=_mock_provider(),
            prompt="Hello.",
            tools=[add],
        )
        r = repr(agent)
        assert "Agent" in r
        assert "claude-sonnet-4-6" in r
        assert "tools=1" in r
        assert "max_iterations=10" in r
