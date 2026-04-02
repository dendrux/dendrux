"""Tests for delegation context — automatic parent-child run linking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.context import (
    DelegationContext,
    get_delegation_context,
    get_store_identity,
    reset_delegation_context,
    resolve_parent_link,
    set_delegation_context,
)
from dendrux.runtime.runner import run
from dendrux.types import LLMResponse, ToolCall

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ------------------------------------------------------------------
# Fake stores
# ------------------------------------------------------------------


class FakeStoreWithIdentity:
    """Fake store that exposes store_identity (like SQLAlchemyStateStore)."""

    def __init__(self, identity: str) -> None:
        self._identity = identity

    @property
    def store_identity(self) -> str:
        return self._identity


@dataclass
class RecordingStateStore:
    """Fake StateStore that records create_run calls and has a store_identity."""

    _identity: str = "sqlite:///test.db"
    created_runs: list[dict[str, Any]] = field(default_factory=list)
    finalized_runs: list[dict[str, Any]] = field(default_factory=list)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def store_identity(self) -> str:
        return self._identity

    async def create_run(self, run_id: str, agent_name: str, **kwargs: Any) -> None:
        self.created_runs.append(
            {"run_id": run_id, "agent_name": agent_name, **kwargs}
        )

    async def finalize_run(self, run_id: str, **kwargs: Any) -> bool:
        kwargs.pop("expected_current_status", None)
        self.finalized_runs.append({"run_id": run_id, **kwargs})
        return True

    async def save_trace(self, run_id: str, **kwargs: Any) -> None:
        pass

    async def save_tool_call(self, run_id: str, **kwargs: Any) -> None:
        pass

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        pass

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self._events.setdefault(run_id, []).append(kwargs)

    async def get_run_events(self, run_id: str) -> list[Any]:
        return []


# ------------------------------------------------------------------
# Unit tests: get_store_identity
# ------------------------------------------------------------------


class TestGetStoreIdentity:
    def test_none_store(self) -> None:
        assert get_store_identity(None) is None

    def test_store_with_property(self) -> None:
        store = FakeStoreWithIdentity("sqlite:///app.db")
        assert get_store_identity(store) == "sqlite:///app.db"

    def test_store_without_property_falls_back_to_id(self) -> None:
        store = object()
        assert get_store_identity(store) == str(id(store))

    def test_same_url_different_objects_match(self) -> None:
        """Two store objects with same URL have matching identity."""
        store_a = FakeStoreWithIdentity("sqlite:///shared.db")
        store_b = FakeStoreWithIdentity("sqlite:///shared.db")
        assert store_a is not store_b
        assert get_store_identity(store_a) == get_store_identity(store_b)

    def test_different_urls_do_not_match(self) -> None:
        store_a = FakeStoreWithIdentity("sqlite:///a.db")
        store_b = FakeStoreWithIdentity("sqlite:///b.db")
        assert get_store_identity(store_a) != get_store_identity(store_b)


# ------------------------------------------------------------------
# Unit tests: resolve_parent_link
# ------------------------------------------------------------------


class TestResolveParentLink:
    def test_root_run_no_parent(self) -> None:
        """Root run (no parent context) gets parent_run_id=None, level=0."""
        parent_run_id, level = resolve_parent_link(None, object())
        assert parent_run_id is None
        assert level == 0

    def test_same_store_both_persisted_links(self) -> None:
        """Both persisted + same store identity → link."""
        store = FakeStoreWithIdentity("sqlite:///app.db")
        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=True,
            store_identity="sqlite:///app.db",
        )
        parent_run_id, level = resolve_parent_link(parent_ctx, store)
        assert parent_run_id == "run_parent"
        assert level == 1

    def test_same_url_different_store_objects_links(self) -> None:
        """Two separate store objects with same DB URL → link (the critical fix)."""
        parent_store = FakeStoreWithIdentity("sqlite:///shared.db")
        child_store = FakeStoreWithIdentity("sqlite:///shared.db")
        assert parent_store is not child_store

        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=True,
            store_identity=get_store_identity(parent_store),
        )
        parent_run_id, level = resolve_parent_link(parent_ctx, child_store)
        assert parent_run_id == "run_parent"
        assert level == 1

    def test_ephemeral_parent_skips_link(self) -> None:
        """Parent not persisted → no link, level still increments."""
        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=False,
            store_identity=None,
        )
        parent_run_id, level = resolve_parent_link(parent_ctx, object())
        assert parent_run_id is None
        assert level == 1

    def test_different_store_skips_link(self) -> None:
        """Different store URLs → no link, level still increments."""
        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=True,
            store_identity="sqlite:///parent.db",
        )
        child_store = FakeStoreWithIdentity("sqlite:///child.db")
        parent_run_id, level = resolve_parent_link(parent_ctx, child_store)
        assert parent_run_id is None
        assert level == 1

    def test_ephemeral_child_no_link_needed(self) -> None:
        """Child has no store → no link (no DB row to write)."""
        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=True,
            store_identity="sqlite:///app.db",
        )
        parent_run_id, level = resolve_parent_link(parent_ctx, None)
        assert parent_run_id is None
        assert level == 1

    def test_nested_levels_increment(self) -> None:
        """Level increments through multiple nesting layers."""
        store = FakeStoreWithIdentity("sqlite:///app.db")
        ctx_level_2 = DelegationContext(
            run_id="run_grandparent",
            delegation_level=2,
            persisted=True,
            store_identity="sqlite:///app.db",
        )
        parent_run_id, level = resolve_parent_link(ctx_level_2, store)
        assert parent_run_id == "run_grandparent"
        assert level == 3


# ------------------------------------------------------------------
# Unit tests: contextvar set/get/reset
# ------------------------------------------------------------------


class TestContextVar:
    def test_default_is_none(self) -> None:
        assert get_delegation_context() is None

    def test_set_and_get(self) -> None:
        ctx = DelegationContext(run_id="run_1", delegation_level=0)
        token = set_delegation_context(ctx)
        try:
            assert get_delegation_context() is ctx
        finally:
            reset_delegation_context(token)

    def test_reset_restores_previous(self) -> None:
        ctx_outer = DelegationContext(run_id="outer", delegation_level=0)
        token_outer = set_delegation_context(ctx_outer)
        try:
            ctx_inner = DelegationContext(run_id="inner", delegation_level=1)
            token_inner = set_delegation_context(ctx_inner)
            assert get_delegation_context() is ctx_inner
            reset_delegation_context(token_inner)
            assert get_delegation_context() is ctx_outer
        finally:
            reset_delegation_context(token_outer)

    def test_no_leak_after_reset(self) -> None:
        ctx = DelegationContext(run_id="temp", delegation_level=0)
        token = set_delegation_context(ctx)
        reset_delegation_context(token)
        assert get_delegation_context() is None


# ------------------------------------------------------------------
# Warning dedup
# ------------------------------------------------------------------


class TestWarningDedup:
    def test_ephemeral_parent_warns_once(self) -> None:
        """Multiple calls with same parent should warn only once."""
        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=False,
            store_identity=None,
        )

        resolve_parent_link(parent_ctx, object())
        assert "ephemeral_parent" in parent_ctx.warned_mismatches

        resolve_parent_link(parent_ctx, object())
        assert len(parent_ctx.warned_mismatches) == 1

    def test_different_store_warns_once(self) -> None:
        parent_ctx = DelegationContext(
            run_id="run_parent",
            delegation_level=0,
            persisted=True,
            store_identity="sqlite:///parent.db",
        )
        child_store = FakeStoreWithIdentity("sqlite:///child.db")

        resolve_parent_link(parent_ctx, child_store)
        assert "different_store" in parent_ctx.warned_mismatches

        resolve_parent_link(parent_ctx, child_store)
        assert len(parent_ctx.warned_mismatches) == 1


# ------------------------------------------------------------------
# Integration: runner creates run with parent_run_id
# ------------------------------------------------------------------


class TestDelegationRunnerIntegration:
    async def test_root_run_has_no_parent(self) -> None:
        """Root agent.run() → parent_run_id=None, delegation_level=0."""
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store)

        created = store.created_runs[0]
        assert created.get("parent_run_id") is None
        assert created.get("delegation_level", 0) == 0

    async def test_nested_run_links_to_parent(self) -> None:
        """Tool that calls agent.run() → child gets parent_run_id."""
        store = RecordingStateStore()

        child_agent = Agent(
            prompt="You are a helper. Just say 'helped'.",
            tools=[add],
        )

        @tool()
        async def delegate(task: str) -> str:
            """Delegate to child agent."""
            result = await run(
                child_agent,
                provider=MockLLM([LLMResponse(text="helped")]),
                user_input=task,
                state_store=store,
            )
            return result.answer or ""

        parent_agent = Agent(prompt="Test.", tools=[delegate])

        parent_llm = MockLLM([
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(name="delegate", params={"task": "help me"})],
            ),
            LLMResponse(text="all done"),
        ])

        await run(
            parent_agent,
            provider=parent_llm,
            user_input="go",
            state_store=store,
        )

        assert len(store.created_runs) == 2

        parent_created = store.created_runs[0]
        child_created = store.created_runs[1]

        assert parent_created.get("parent_run_id") is None
        assert parent_created.get("delegation_level", 0) == 0

        assert child_created["parent_run_id"] == parent_created["run_id"]
        assert child_created["delegation_level"] == 1

    async def test_nested_run_links_with_separate_store_objects(self) -> None:
        """Parent and child use different store objects with same DB URL → link.

        This is the common developer path: each agent resolves its own store
        from the same database_url. The stores are different Python objects
        but point at the same database.
        """
        parent_store = RecordingStateStore(_identity="sqlite:///shared.db")
        child_store = RecordingStateStore(_identity="sqlite:///shared.db")
        assert parent_store is not child_store

        child_agent = Agent(
            prompt="Helper.",
            tools=[add],
        )

        @tool()
        async def delegate(task: str) -> str:
            """Delegate to child agent."""
            result = await run(
                child_agent,
                provider=MockLLM([LLMResponse(text="helped")]),
                user_input=task,
                state_store=child_store,
            )
            return result.answer or ""

        parent_agent = Agent(prompt="Test.", tools=[delegate])

        parent_llm = MockLLM([
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(name="delegate", params={"task": "help"})],
            ),
            LLMResponse(text="done"),
        ])

        await run(
            parent_agent,
            provider=parent_llm,
            user_input="go",
            state_store=parent_store,
        )

        # Parent on parent_store, child on child_store
        assert len(parent_store.created_runs) == 1
        assert len(child_store.created_runs) == 1

        parent_created = parent_store.created_runs[0]
        child_created = child_store.created_runs[0]

        # Child links to parent — same store_identity despite different objects
        assert child_created["parent_run_id"] == parent_created["run_id"]
        assert child_created["delegation_level"] == 1

    async def test_context_resets_after_run(self) -> None:
        """Delegation context does not leak after run completes."""
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store)

        assert get_delegation_context() is None

    async def test_ephemeral_parent_persistent_child_no_link(self) -> None:
        """Parent without store, child with store → no FK violation."""
        store = RecordingStateStore()

        child_agent = Agent(
            prompt="Helper.",
            tools=[add],
        )

        @tool()
        async def delegate(task: str) -> str:
            """Delegate to child agent."""
            result = await run(
                child_agent,
                provider=MockLLM([LLMResponse(text="helped")]),
                user_input=task,
                state_store=store,
            )
            return result.answer or ""

        parent_agent = Agent(prompt="Test.", tools=[delegate])

        parent_llm = MockLLM([
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(name="delegate", params={"task": "help"})],
            ),
            LLMResponse(text="done"),
        ])

        await run(
            parent_agent,
            provider=parent_llm,
            user_input="go",
        )

        assert len(store.created_runs) == 1
        child_created = store.created_runs[0]
        assert child_created.get("parent_run_id") is None
        assert child_created["delegation_level"] == 1
