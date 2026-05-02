"""Tests for Wave 1 — tool deny (governance v1)."""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool()
async def refund(order_id: int) -> str:
    """Refund an order."""
    return f"Refunded order {order_id}"


@tool()
async def delete_account(user_id: int) -> str:
    """Delete a user account."""
    return f"Deleted user {user_id}"


def _make_agent(**overrides) -> Agent:
    defaults = {
        "prompt": "You are a customer support agent.",
        "tools": [search, refund, delete_account],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


# ------------------------------------------------------------------
# Construction validation
# ------------------------------------------------------------------


class TestDenyValidation:
    def test_deny_unknown_tool_raises(self):
        """Typo in deny list raises ValueError at construction."""
        with pytest.raises(ValueError, match="unknown tool.*typo"):
            _make_agent(deny=["typo"])

    def test_deny_with_single_call_raises(self):
        """deny + SingleCall raises ValueError (SingleCall can't have tools)."""
        # SingleCall with tools already raises, so deny + SingleCall also raises.
        # The error comes from either "SingleCall agents must have zero tools"
        # or "deny is not supported with SingleCall" depending on validation order.
        with pytest.raises(ValueError):
            Agent(
                prompt="Test",
                tools=[search],
                loop=SingleCall(),
                deny=["search"],
            )

    def test_deny_without_tools_raises(self):
        """deny without any tools raises ValueError."""
        with pytest.raises(ValueError, match="no tools"):
            Agent(prompt="Test", tools=[], deny=["search"])

    def test_deny_none_is_valid(self):
        """deny=None (default) is valid."""
        agent = _make_agent()
        assert agent.deny == frozenset()

    def test_deny_empty_list_is_valid(self):
        """deny=[] is valid, same as None."""
        agent = _make_agent(deny=[])
        assert agent.deny == frozenset()

    def test_deny_valid_names(self):
        """Valid tool names in deny are accepted."""
        agent = _make_agent(deny=["delete_account"])
        assert agent.deny == frozenset({"delete_account"})

    def test_deny_multiple_valid_names(self):
        """Multiple valid tool names in deny are accepted."""
        agent = _make_agent(deny=["delete_account", "refund"])
        assert agent.deny == frozenset({"delete_account", "refund"})

    def test_deny_partial_unknown_raises(self):
        """Mix of valid and invalid names raises for the invalid ones."""
        with pytest.raises(ValueError, match="unknown tool.*nonexistent"):
            _make_agent(deny=["search", "nonexistent"])


# ------------------------------------------------------------------
# Runtime behavior
# ------------------------------------------------------------------


class TestDenyRuntime:
    async def test_denied_tool_not_executed(self):
        """Denied tool gets a synthesized error, never executes."""
        tc = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="I'll delete the account", tool_calls=[tc]),
                LLMResponse(text="Sorry, I can't delete accounts."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete user 42",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Sorry, I can't delete accounts."
        # LLM was called twice: first produced tool call, second finished
        assert llm.calls_made == 2

    async def test_denied_tool_error_message(self):
        """The model sees 'denied by policy' in the tool result."""
        tc = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Deleting...", tool_calls=[tc]),
                LLMResponse(text="Cannot do that."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete user 42",
        )

        # Check that the second LLM call saw the deny error in history
        second_call = llm.call_history[1]
        messages = second_call["messages"]
        # Find the tool result message
        tool_result_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_result_msgs) == 1
        assert "denied by policy" in tool_result_msgs[0].content

    async def test_allowed_tool_still_executes(self):
        """Non-denied tools execute normally."""
        tc = ToolCall(
            name="search",
            params={"query": "hello"},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Searching...", tool_calls=[tc]),
                LLMResponse(text="Found results."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search for hello",
        )

        assert result.status == RunStatus.SUCCESS
        # Check that search actually executed (result in history)
        second_call = llm.call_history[1]
        messages = second_call["messages"]
        tool_result_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_result_msgs) == 1
        assert "Results for: hello" in tool_result_msgs[0].content

    async def test_parallel_batch_only_denies_target(self):
        """In a parallel batch, only the denied tool is blocked."""
        tc_search = ToolCall(
            name="search",
            params={"query": "order 123"},
            provider_tool_call_id="t1",
        )
        tc_delete = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t2",
        )
        llm = MockLLM(
            [
                LLMResponse(
                    text="Let me search and delete",
                    tool_calls=[tc_search, tc_delete],
                ),
                LLMResponse(text="Search worked, delete was denied."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search and delete",
        )

        assert result.status == RunStatus.SUCCESS
        # Second LLM call should have two tool results:
        # one success (search) and one failure (delete_account)
        second_call = llm.call_history[1]
        messages = second_call["messages"]
        tool_result_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_result_msgs) == 2

        contents = [m.content for m in tool_result_msgs]
        has_search_result = any("Results for: order 123" in c for c in contents)
        has_deny_result = any("denied by policy" in c for c in contents)
        assert has_search_result
        assert has_deny_result

    async def test_multiple_denied_tools_in_batch(self):
        """Multiple denied tools in same batch all get errors."""
        tc_delete = ToolCall(
            name="delete_account",
            params={"user_id": 1},
            provider_tool_call_id="t1",
        )
        tc_refund = ToolCall(
            name="refund",
            params={"order_id": 99},
            provider_tool_call_id="t2",
        )
        llm = MockLLM(
            [
                LLMResponse(
                    text="Deleting and refunding",
                    tool_calls=[tc_delete, tc_refund],
                ),
                LLMResponse(text="Both denied."),
            ]
        )
        agent = _make_agent(deny=["delete_account", "refund"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete and refund",
        )

        assert result.status == RunStatus.SUCCESS
        second_call = llm.call_history[1]
        messages = second_call["messages"]
        tool_result_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_result_msgs) == 2
        assert all("denied by policy" in m.content for m in tool_result_msgs)

    async def test_no_deny_does_not_affect_execution(self):
        """Agent without deny= executes all tools normally."""
        tc = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Deleting...", tool_calls=[tc]),
                LLMResponse(text="Done."),
            ]
        )
        agent = _make_agent()  # no deny

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete user 42",
        )

        assert result.status == RunStatus.SUCCESS
        second_call = llm.call_history[1]
        messages = second_call["messages"]
        tool_result_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_result_msgs) == 1
        assert "Deleted user 42" in tool_result_msgs[0].content


# ------------------------------------------------------------------
# Governance event emission
# ------------------------------------------------------------------


class TestDenyGovernanceEvent:
    async def test_governance_event_emitted(self):
        """Denied tool emits a governance event via the recorder."""
        tc = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Deleting...", tool_calls=[tc]),
                LLMResponse(text="Cannot do that."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        # Track governance events via a simple recorder
        governance_events: list[dict] = []

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
                governance_events.append(
                    {
                        "event_type": event_type,
                        "iteration": iteration,
                        "data": data,
                        "correlation_id": correlation_id,
                    }
                )

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete user 42",
            recorder=SpyRecorder(),
        )

        assert len(governance_events) == 1
        evt = governance_events[0]
        assert evt["event_type"] == "policy.denied"
        assert evt["data"]["tool_name"] == "delete_account"
        assert evt["data"]["reason"] == "denied_by_policy"
        assert evt["correlation_id"] is not None

    async def test_denied_tool_does_not_emit_tool_completed(self):
        """Denied tool should NOT emit tool.completed — only policy.denied."""
        tc = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Deleting...", tool_calls=[tc]),
                LLMResponse(text="Cannot do that."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        tool_completed_calls: list[str] = []
        governance_calls: list[str] = []

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                tool_completed_calls.append(tool_call.name)

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
                governance_calls.append(event_type)

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete user 42",
            recorder=SpyRecorder(),
        )

        # Denied tool should NOT trigger on_tool_completed
        assert "delete_account" not in tool_completed_calls
        # But SHOULD trigger governance event
        assert "policy.denied" in governance_calls


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------


class TestDenyStreaming:
    async def test_denied_tool_in_stream(self):
        """Denied tool works correctly via run_stream()."""
        tc = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Deleting...", tool_calls=[tc]),
                LLMResponse(text="Cannot do that."),
            ]
        )
        agent = _make_agent(deny=["delete_account"])

        events = []
        async for event in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete user 42",
        ):
            events.append(event)

        # Should have a tool result event with the deny error
        tool_results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert len(tool_results) == 1
        assert tool_results[0].tool_result.success is False
        assert "denied by policy" in tool_results[0].tool_result.error

        # Should complete successfully
        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
