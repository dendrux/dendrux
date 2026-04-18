"""Tests for delegation context — automatic parent-child run linking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.context import (
    DelegationContext,
    DelegationDepthExceededError,
    get_delegation_context,
    get_store_identity,
    reset_delegation_context,
    resolve_parent_link,
    set_delegation_context,
)
from dendrux.runtime.runner import run
from dendrux.types import LLMResponse, ToolCall
from tests._helpers.state_store_mocks import CancellationStubsMixin

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
class RecordingStateStore(CancellationStubsMixin):
    """Fake StateStore that records create_run calls and has a store_identity."""

    _identity: str = "sqlite:///test.db"
    created_runs: list[dict[str, Any]] = field(default_factory=list)
    finalized_runs: list[dict[str, Any]] = field(default_factory=list)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def store_identity(self) -> str:
        return self._identity

    async def create_run(self, run_id: str, agent_name: str, **kwargs: Any):
        self.created_runs.append({"run_id": run_id, "agent_name": agent_name, **kwargs})
        from dendrux.types import CreateRunResult, RunStatus

        return CreateRunResult(run_id=run_id, outcome="created", status=RunStatus.RUNNING)

    async def finalize_run(self, run_id: str, **kwargs: Any) -> bool:
        kwargs.pop("expected_current_status", None)
        self.finalized_runs.append({"run_id": run_id, **kwargs})
        return True

    async def save_trace(self, run_id: str, *args: Any, **kwargs: Any) -> None:
        pass

    async def save_tool_call(self, run_id: str, *args: Any, **kwargs: Any) -> None:
        pass

    async def save_usage(self, run_id: str, *args: Any, **kwargs: Any) -> None:
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

        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="delegate", params={"task": "help me"})],
                ),
                LLMResponse(text="all done"),
            ]
        )

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

        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="delegate", params={"task": "help"})],
                ),
                LLMResponse(text="done"),
            ]
        )

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

        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="delegate", params={"task": "help"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(
            parent_agent,
            provider=parent_llm,
            user_input="go",
        )

        assert len(store.created_runs) == 1
        child_created = store.created_runs[0]
        assert child_created.get("parent_run_id") is None
        assert child_created["delegation_level"] == 1


# ------------------------------------------------------------------
# max_delegation_depth tests
# ------------------------------------------------------------------


class TestMaxDelegationDepth:
    """Tests for the delegation depth safety guard."""

    def test_resolve_parent_link_within_limit(self) -> None:
        """Child at depth 1 with max_depth=2 is allowed."""
        parent = DelegationContext(
            run_id="parent",
            delegation_level=0,
            persisted=True,
            store_identity="sqlite:///test.db",
            max_delegation_depth=2,
        )
        child_store = FakeStoreWithIdentity("sqlite:///test.db")
        parent_run_id, level = resolve_parent_link(parent, child_store)
        assert parent_run_id == "parent"
        assert level == 1

    def test_resolve_parent_link_at_limit(self) -> None:
        """Child at depth exactly equal to max is allowed."""
        parent = DelegationContext(
            run_id="parent",
            delegation_level=1,
            persisted=True,
            store_identity="sqlite:///test.db",
            max_delegation_depth=2,
        )
        child_store = FakeStoreWithIdentity("sqlite:///test.db")
        parent_run_id, level = resolve_parent_link(parent, child_store)
        assert parent_run_id == "parent"
        assert level == 2

    def test_resolve_parent_link_exceeds_limit(self) -> None:
        """Child that would exceed max_depth raises DelegationDepthExceededError."""
        parent = DelegationContext(
            run_id="parent",
            delegation_level=2,
            persisted=True,
            store_identity="sqlite:///test.db",
            max_delegation_depth=2,
        )
        child_store = FakeStoreWithIdentity("sqlite:///test.db")
        with pytest.raises(DelegationDepthExceededError) as exc_info:
            resolve_parent_link(parent, child_store)
        assert exc_info.value.delegation_level == 3
        assert exc_info.value.max_depth == 2

    def test_unbounded_depth(self) -> None:
        """max_delegation_depth=None means no limit."""
        parent = DelegationContext(
            run_id="parent",
            delegation_level=100,
            persisted=True,
            store_identity="sqlite:///test.db",
            max_delegation_depth=None,
        )
        child_store = FakeStoreWithIdentity("sqlite:///test.db")
        parent_run_id, level = resolve_parent_link(parent, child_store)
        assert parent_run_id == "parent"
        assert level == 101

    def test_depth_zero_blocks_all_children(self) -> None:
        """max_delegation_depth=0 blocks any child run."""
        parent = DelegationContext(
            run_id="parent",
            delegation_level=0,
            persisted=True,
            store_identity="sqlite:///test.db",
            max_delegation_depth=0,
        )
        child_store = FakeStoreWithIdentity("sqlite:///test.db")
        with pytest.raises(DelegationDepthExceededError):
            resolve_parent_link(parent, child_store)

    async def test_depth_exceeded_blocks_child_creation(self) -> None:
        """DelegationDepthExceededError prevents child run from being created.

        The error is caught by the tool executor (returned as tool error to
        the LLM). The parent run completes; the child was never created.
        """
        child_store = RecordingStateStore()
        parent_store = RecordingStateStore()

        child_agent = Agent(prompt="Helper.", tools=[add])

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

        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="delegate", params={"task": "help"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        # max_delegation_depth=0 means no children allowed
        result = await run(
            parent_agent,
            provider=parent_llm,
            user_input="go",
            state_store=parent_store,
            max_delegation_depth=0,
        )

        # Parent completed (tool error was handled gracefully by ReAct loop)
        assert result.answer == "done"
        # Parent was created
        assert len(parent_store.created_runs) == 1
        # Child was NOT created — blocked by depth guard
        assert len(child_store.created_runs) == 0

    async def test_depth_limit_propagates_to_children(self) -> None:
        """max_delegation_depth propagates through context to nested runs."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_context(msg: str) -> str:
            """Capture current delegation context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        agent = Agent(prompt="Test.", tools=[capture_context])

        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="capture_context", params={"msg": "hi"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(
            agent,
            provider=llm,
            user_input="go",
            max_delegation_depth=5,
        )

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 5

    async def test_default_depth_is_10(self) -> None:
        """Default max_delegation_depth is 10 when not explicitly set."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_context(msg: str) -> str:
            """Capture current delegation context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        agent = Agent(prompt="Test.", tools=[capture_context])

        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="capture_context", params={"msg": "hi"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(agent, provider=llm, user_input="go")

        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 10

    async def test_explicit_none_means_unbounded(self) -> None:
        """max_delegation_depth=None explicitly sets unbounded."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_context(msg: str) -> str:
            """Capture current delegation context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        agent = Agent(prompt="Test.", tools=[capture_context])

        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="capture_context", params={"msg": "hi"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(agent, provider=llm, user_input="go", max_delegation_depth=None)

        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth is None

    async def test_depth_persisted_in_run_metadata(self) -> None:
        """max_delegation_depth is written to run metadata for resume."""
        store = RecordingStateStore()
        agent = Agent(prompt="Test.", tools=[add])
        llm = MockLLM([LLMResponse(text="done")])

        await run(
            agent,
            provider=llm,
            user_input="go",
            state_store=store,
            max_delegation_depth=7,
        )

        assert len(store.created_runs) == 1
        meta = store.created_runs[0].get("meta", {})
        assert meta["dendrux.max_delegation_depth"] == 7

    async def test_unbounded_depth_not_persisted(self) -> None:
        """max_delegation_depth=None means no key in metadata."""
        store = RecordingStateStore()
        agent = Agent(prompt="Test.", tools=[add])
        llm = MockLLM([LLMResponse(text="done")])

        await run(
            agent,
            provider=llm,
            user_input="go",
            state_store=store,
            max_delegation_depth=None,
        )

        meta = store.created_runs[0].get("meta", {})
        assert "dendrux.max_delegation_depth" not in meta

    async def test_negative_depth_raises_value_error(self) -> None:
        """Negative max_delegation_depth raises ValueError at the Agent level."""
        agent = Agent(prompt="Test.", tools=[add])

        with pytest.raises(ValueError, match="non-negative integer"):
            await agent.run("go", max_delegation_depth=-1)

    async def test_negative_depth_raises_on_runner_run(self) -> None:
        """Negative max_delegation_depth raises ValueError on dendrux.run() too."""
        agent = Agent(prompt="Test.", tools=[add])
        llm = MockLLM([LLMResponse(text="done")])

        with pytest.raises(ValueError, match="non-negative integer"):
            await run(agent, provider=llm, user_input="go", max_delegation_depth=-1)

    def test_negative_depth_raises_on_runner_run_stream(self) -> None:
        """Negative max_delegation_depth raises ValueError on run_stream() too."""
        from dendrux.runtime.runner import run_stream

        agent = Agent(prompt="Test.", tools=[add])
        llm = MockLLM([LLMResponse(text="done")])

        with pytest.raises(ValueError, match="non-negative integer"):
            run_stream(agent, provider=llm, user_input="go", max_delegation_depth=-1)

    async def test_child_cannot_loosen_parent_depth(self) -> None:
        """A child run that sets a higher depth gets clamped with a warning."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn a child that tries to loosen the depth limit."""
            child_agent = Agent(prompt="Child.", tools=[capture_inner])
            child_llm = MockLLM(
                [
                    LLMResponse(
                        text=None,
                        tool_calls=[ToolCall(name="capture_inner", params={"msg": "hi"})],
                    ),
                    LLMResponse(text="done"),
                ]
            )
            # Child tries to set depth=100, but parent is 3
            await run(
                child_agent,
                provider=child_llm,
                user_input="go",
                max_delegation_depth=100,
            )
            return "spawned"

        @tool()
        async def capture_inner(msg: str) -> str:
            """Capture context inside the child."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(
            parent_agent,
            provider=parent_llm,
            user_input="go",
            max_delegation_depth=3,
        )

        # Child's context should have depth clamped to 3, not 100
        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 3

    async def test_child_loosen_emits_warning(self, caplog) -> None:
        """Explicit child loosen emits a warning with requested and effective values."""

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn child with loosened depth."""
            child_agent = Agent(prompt="Child.", tools=[noop])
            child_llm = MockLLM([LLMResponse(text="done")])
            await run(
                child_agent,
                provider=child_llm,
                user_input="go",
                max_delegation_depth=100,
            )
            return "ok"

        @tool()
        async def noop(msg: str) -> str:
            """No-op."""
            return "ok"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        with caplog.at_level("WARNING", logger="dendrux.runtime.runner"):
            await run(
                parent_agent,
                provider=parent_llm,
                user_input="go",
                max_delegation_depth=3,
            )

        assert any("100 clamped to 3" in msg for msg in caplog.messages)

    async def test_child_can_tighten_parent_depth(self) -> None:
        """A child run that sets a lower depth is allowed (no warning)."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn a child that tightens the depth limit."""
            child_agent = Agent(prompt="Child.", tools=[capture_inner])
            child_llm = MockLLM(
                [
                    LLMResponse(
                        text=None,
                        tool_calls=[ToolCall(name="capture_inner", params={"msg": "hi"})],
                    ),
                    LLMResponse(text="done"),
                ]
            )
            # Child tightens from 10 → 2
            await run(
                child_agent,
                provider=child_llm,
                user_input="go",
                max_delegation_depth=2,
            )
            return "spawned"

        @tool()
        async def capture_inner(msg: str) -> str:
            """Capture context inside the child."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(parent_agent, provider=parent_llm, user_input="go", max_delegation_depth=10)

        # Child's context should have depth=2 (tighter than parent's 10)
        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 2

    async def test_child_unbounded_clamped_to_parent(self, caplog) -> None:
        """A child that sets None (unbounded) gets clamped with a warning."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn a child that tries unbounded depth."""
            child_agent = Agent(prompt="Child.", tools=[capture_inner])
            child_llm = MockLLM(
                [
                    LLMResponse(
                        text=None,
                        tool_calls=[ToolCall(name="capture_inner", params={"msg": "hi"})],
                    ),
                    LLMResponse(text="done"),
                ]
            )
            # Child tries unbounded, but parent has limit=5
            await run(
                child_agent,
                provider=child_llm,
                user_input="go",
                max_delegation_depth=None,
            )
            return "spawned"

        @tool()
        async def capture_inner(msg: str) -> str:
            """Capture context inside the child."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        with caplog.at_level("WARNING", logger="dendrux.runtime.runner"):
            await run(
                parent_agent,
                provider=parent_llm,
                user_input="go",
                max_delegation_depth=5,
            )

        # Child's context should be clamped to 5, not None
        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 5
        # Warning was emitted
        assert any("None" in msg and "clamped to 5" in msg for msg in caplog.messages)

    async def test_child_omitted_inherits_silently(self, caplog) -> None:
        """A child that omits max_delegation_depth inherits without warning."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn child without explicit depth."""
            child_agent = Agent(prompt="Child.", tools=[capture_inner])
            child_llm = MockLLM(
                [
                    LLMResponse(
                        text=None,
                        tool_calls=[ToolCall(name="capture_inner", params={"msg": "hi"})],
                    ),
                    LLMResponse(text="done"),
                ]
            )
            # No max_delegation_depth — should inherit silently
            await run(child_agent, provider=child_llm, user_input="go")
            return "spawned"

        @tool()
        async def capture_inner(msg: str) -> str:
            """Capture context inside the child."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        with caplog.at_level("WARNING", logger="dendrux.runtime.runner"):
            await run(
                parent_agent,
                provider=parent_llm,
                user_input="go",
                max_delegation_depth=3,
            )

        # Inherited parent's limit
        assert len(captured_contexts) == 1
        assert captured_contexts[0].max_delegation_depth == 3
        # No warning about clamping
        assert not any("clamped" in msg for msg in caplog.messages)

    # --- Agent-level default tests ---

    async def test_agent_default_used_on_root_run(self) -> None:
        """Agent's max_delegation_depth is used when agent.run() omits the kwarg."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_context(msg: str) -> str:
            """Capture current delegation context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="capture_context", params={"msg": "hi"})],
                ),
                LLMResponse(text="done"),
            ]
        )
        agent = Agent(
            prompt="Test.",
            tools=[capture_context],
            max_delegation_depth=7,
            provider=llm,
        )

        await agent.run("go")

        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 7

    async def test_run_kwarg_overrides_agent_default(self) -> None:
        """Explicit agent.run() kwarg takes precedence over agent default."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_context(msg: str) -> str:
            """Capture current delegation context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="capture_context", params={"msg": "hi"})],
                ),
                LLMResponse(text="done"),
            ]
        )
        agent = Agent(
            prompt="Test.",
            tools=[capture_context],
            max_delegation_depth=7,
            provider=llm,
        )

        await agent.run("go", max_delegation_depth=3)

        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 3

    async def test_child_agent_default_cannot_loosen_parent(self) -> None:
        """A child agent's default depth cannot loosen the parent's limit."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_inner(msg: str) -> str:
            """Capture context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn child with agent-level default higher than parent's limit."""
            child_llm = MockLLM(
                [
                    LLMResponse(
                        text=None,
                        tool_calls=[ToolCall(name="capture_inner", params={"msg": "hi"})],
                    ),
                    LLMResponse(text="done"),
                ]
            )
            # Child agent has default=50, but parent limit is 3
            child_agent = Agent(
                prompt="Child.",
                tools=[capture_inner],
                max_delegation_depth=50,
                provider=child_llm,
            )
            await child_agent.run("go")
            return "spawned"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(parent_agent, provider=parent_llm, user_input="go", max_delegation_depth=3)

        assert len(captured_contexts) == 1
        assert captured_contexts[0].max_delegation_depth == 3

    async def test_child_agent_default_can_tighten_parent(self) -> None:
        """A child agent's default depth that is lower than parent is allowed."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_inner(msg: str) -> str:
            """Capture context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        @tool()
        async def spawn_child(msg: str) -> str:
            """Spawn child with agent-level default lower than parent's limit."""
            child_llm = MockLLM(
                [
                    LLMResponse(
                        text=None,
                        tool_calls=[ToolCall(name="capture_inner", params={"msg": "hi"})],
                    ),
                    LLMResponse(text="done"),
                ]
            )
            child_agent = Agent(
                prompt="Child.",
                tools=[capture_inner],
                max_delegation_depth=2,
                provider=child_llm,
            )
            await child_agent.run("go")
            return "spawned"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_child])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_child", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        await run(parent_agent, provider=parent_llm, user_input="go", max_delegation_depth=10)

        assert len(captured_contexts) == 1
        assert captured_contexts[0].max_delegation_depth == 2

    def test_agent_constructor_validates_negative_depth(self) -> None:
        """Agent constructor rejects negative max_delegation_depth."""
        with pytest.raises(ValueError, match="non-negative integer"):
            Agent(prompt="Test.", max_delegation_depth=-1)

    def test_subclass_default_negative_depth_raises(self) -> None:
        """Subclass with class-level max_delegation_depth=-1 raises at construction."""
        with pytest.raises(ValueError, match="non-negative integer"):

            class BadAgent(Agent):
                prompt = "Bad."
                max_delegation_depth = -1

            BadAgent()

    async def test_fanout_clamp_warning_deduped(self, caplog) -> None:
        """Two sibling children both trying to loosen emit only one warning."""

        @tool()
        async def noop(msg: str) -> str:
            """No-op."""
            return "ok"

        @tool()
        async def spawn_two(msg: str) -> str:
            """Spawn two children that both try to loosen."""
            for _ in range(2):
                child_agent = Agent(prompt="Child.", tools=[noop])
                child_llm = MockLLM([LLMResponse(text="done")])
                await run(
                    child_agent,
                    provider=child_llm,
                    user_input="go",
                    max_delegation_depth=100,
                )
            return "spawned"

        parent_agent = Agent(prompt="Parent.", tools=[spawn_two])
        parent_llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="spawn_two", params={"msg": "go"})],
                ),
                LLMResponse(text="done"),
            ]
        )

        with caplog.at_level("WARNING", logger="dendrux.runtime.runner"):
            await run(
                parent_agent,
                provider=parent_llm,
                user_input="go",
                max_delegation_depth=3,
            )

        # Only one warning, not two
        clamp_msgs = [m for m in caplog.messages if "clamped" in m]
        assert len(clamp_msgs) == 1
        assert "100 clamped to 3" in clamp_msgs[0]

    async def test_root_run_no_kwarg_no_agent_default_gets_runner_default(self) -> None:
        """Root run with neither explicit kwarg nor agent default → runner picks 10."""
        captured_contexts: list[DelegationContext | None] = []

        @tool()
        async def capture_context(msg: str) -> str:
            """Capture current delegation context."""
            captured_contexts.append(get_delegation_context())
            return "ok"

        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="capture_context", params={"msg": "hi"})],
                ),
                LLMResponse(text="done"),
            ]
        )
        # No max_delegation_depth on agent or run call
        agent = Agent(prompt="Test.", tools=[capture_context], provider=llm)

        await agent.run("go")

        ctx = captured_contexts[0]
        assert ctx is not None
        assert ctx.max_delegation_depth == 10  # runner default
