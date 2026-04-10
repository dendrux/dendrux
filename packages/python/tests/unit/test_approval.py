"""Tests for Wave 2 — require_approval (governance v1)."""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
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


class TestApprovalValidation:
    def test_require_approval_unknown_tool_raises(self):
        """Typo in require_approval list raises ValueError."""
        with pytest.raises(ValueError, match="unknown tool.*typo"):
            _make_agent(require_approval=["typo"])

    def test_require_approval_with_single_call_raises(self):
        """require_approval + SingleCall raises ValueError."""
        with pytest.raises(ValueError):
            Agent(
                prompt="Test",
                tools=[search],
                loop=SingleCall(),
                require_approval=["search"],
            )

    def test_require_approval_without_tools_raises(self):
        """require_approval without any tools raises ValueError."""
        with pytest.raises(ValueError, match="no tools"):
            Agent(prompt="Test", tools=[], require_approval=["search"])

    def test_require_approval_none_is_valid(self):
        """require_approval=None (default) is valid."""
        agent = _make_agent()
        assert agent.require_approval == frozenset()

    def test_require_approval_empty_list_is_valid(self):
        """require_approval=[] is valid, same as None."""
        agent = _make_agent(require_approval=[])
        assert agent.require_approval == frozenset()

    def test_require_approval_valid_names(self):
        """Valid tool names in require_approval are accepted."""
        agent = _make_agent(require_approval=["refund"])
        assert agent.require_approval == frozenset({"refund"})

    def test_require_approval_multiple_valid_names(self):
        """Multiple valid tool names in require_approval are accepted."""
        agent = _make_agent(require_approval=["refund", "delete_account"])
        assert agent.require_approval == frozenset({"refund", "delete_account"})

    def test_deny_and_require_approval_overlap_raises(self):
        """Tool in both deny and require_approval raises ValueError."""
        with pytest.raises(ValueError, match="both deny and require_approval"):
            _make_agent(deny=["refund"], require_approval=["refund"])

    def test_deny_and_require_approval_no_overlap_ok(self):
        """Non-overlapping deny and require_approval is fine."""
        agent = _make_agent(deny=["delete_account"], require_approval=["refund"])
        assert agent.deny == frozenset({"delete_account"})
        assert agent.require_approval == frozenset({"refund"})

    def test_require_approval_partial_unknown_raises(self):
        """Mix of valid and invalid names raises for the invalid ones."""
        with pytest.raises(ValueError, match="unknown tool.*nonexistent"):
            _make_agent(require_approval=["search", "nonexistent"])

    def test_require_approval_client_tool_raises(self):
        """Client tool in require_approval raises ValueError."""

        @tool(target="client")
        async def client_tool(data: str) -> str:
            """A client-side tool."""
            return data

        with pytest.raises(ValueError, match="only supports server tools"):
            Agent(
                prompt="Test",
                tools=[search, client_tool],
                require_approval=["client_tool"],
            )


# ------------------------------------------------------------------
# Runtime behavior
# ------------------------------------------------------------------


class TestApprovalRuntime:
    async def test_approval_pauses_run(self):
        """Tool needing approval causes WAITING_APPROVAL pause."""
        tc = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="I'll refund the order", tool_calls=[tc]),
            ]
        )
        agent = _make_agent(require_approval=["refund"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Refund order 123",
        )

        assert result.status == RunStatus.WAITING_APPROVAL
        pause = result.meta["pause_state"]
        assert len(pause.pending_tool_calls) == 1
        assert pause.pending_tool_calls[0].name == "refund"

    async def test_non_approval_tool_executes_normally(self):
        """Tools not in require_approval execute without pause."""
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
        agent = _make_agent(require_approval=["refund"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search for hello",
        )

        assert result.status == RunStatus.SUCCESS

    async def test_entire_batch_pauses_when_one_needs_approval(self):
        """If any tool in batch needs approval, entire batch pauses."""
        tc_search = ToolCall(
            name="search",
            params={"query": "order 123"},
            provider_tool_call_id="t1",
        )
        tc_refund = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t2",
        )
        llm = MockLLM(
            [
                LLMResponse(
                    text="Let me search and refund",
                    tool_calls=[tc_search, tc_refund],
                ),
            ]
        )
        agent = _make_agent(require_approval=["refund"])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search and refund",
        )

        assert result.status == RunStatus.WAITING_APPROVAL
        pause = result.meta["pause_state"]
        # Both tools are pending (entire batch pauses)
        assert len(pause.pending_tool_calls) == 2
        names = {tc.name for tc in pause.pending_tool_calls}
        assert names == {"search", "refund"}

    async def test_deny_fires_before_approval(self):
        """Denied tool gets error, remaining tool triggers approval pause."""
        tc_delete = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        tc_refund = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t2",
        )
        llm = MockLLM(
            [
                LLMResponse(
                    text="Deleting and refunding",
                    tool_calls=[tc_delete, tc_refund],
                ),
            ]
        )
        agent = _make_agent(
            deny=["delete_account"],
            require_approval=["refund"],
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete and refund",
        )

        assert result.status == RunStatus.WAITING_APPROVAL
        pause = result.meta["pause_state"]
        # Only refund is pending — delete_account was denied
        assert len(pause.pending_tool_calls) == 1
        assert pause.pending_tool_calls[0].name == "refund"

    async def test_no_approval_does_not_affect_execution(self):
        """Agent without require_approval executes all tools normally."""
        tc = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Refunding...", tool_calls=[tc]),
                LLMResponse(text="Done."),
            ]
        )
        agent = _make_agent()  # no require_approval

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Refund order 123",
        )

        assert result.status == RunStatus.SUCCESS
        second_call = llm.call_history[1]
        messages = second_call["messages"]
        tool_result_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_result_msgs) == 1
        assert "Refunded order 123" in tool_result_msgs[0].content


# ------------------------------------------------------------------
# Governance event emission
# ------------------------------------------------------------------


class TestApprovalGovernanceEvent:
    async def test_approval_requested_event_emitted(self):
        """Approval pause emits approval.requested governance event."""
        tc = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Refunding...", tool_calls=[tc]),
            ]
        )
        agent = _make_agent(require_approval=["refund"])

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
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
            user_input="Refund order 123",
            recorder=SpyRecorder(),
        )

        assert len(governance_events) == 1
        evt = governance_events[0]
        assert evt["event_type"] == "approval.requested"
        assert evt["data"]["tool_name"] == "refund"
        assert evt["data"]["reason"] == "requires_approval"
        assert evt["correlation_id"] is not None

    async def test_approval_requested_only_for_governed_tools(self):
        """In a mixed batch, only the governed tool emits approval.requested."""
        tc_search = ToolCall(
            name="search",
            params={"query": "test"},
            provider_tool_call_id="t1",
        )
        tc_refund = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t2",
        )
        llm = MockLLM(
            [
                LLMResponse(
                    text="Searching and refunding",
                    tool_calls=[tc_search, tc_refund],
                ),
            ]
        )
        agent = _make_agent(require_approval=["refund"])

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_events.append(
                    {
                        "event_type": event_type,
                        "data": data,
                    }
                )

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search and refund",
            recorder=SpyRecorder(),
        )

        # Only refund triggers approval.requested — search does not
        approval_events = [e for e in governance_events if e["event_type"] == "approval.requested"]
        assert len(approval_events) == 1
        assert approval_events[0]["data"]["tool_name"] == "refund"

    async def test_approval_does_not_emit_tool_completed(self):
        """Approval-pending tools do NOT emit tool.completed."""
        tc = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Refunding...", tool_calls=[tc]),
            ]
        )
        agent = _make_agent(require_approval=["refund"])

        tool_completed_calls: list[str] = []
        governance_calls: list[str] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                tool_completed_calls.append(tool_call.name)

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_calls.append(event_type)

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Refund order 123",
            recorder=SpyRecorder(),
        )

        # No tool execution — only governance event
        assert "refund" not in tool_completed_calls
        assert "approval.requested" in governance_calls


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------


class TestApprovalStreaming:
    async def test_approval_pause_in_stream(self):
        """Approval works correctly via run_stream()."""
        tc = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Refunding...", tool_calls=[tc]),
            ]
        )
        agent = _make_agent(require_approval=["refund"])

        events = []
        async for event in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Refund order 123",
        ):
            events.append(event)

        # Should pause, not complete
        paused = [e for e in events if e.type == RunEventType.RUN_PAUSED]
        assert len(paused) == 1
        assert paused[0].run_result.status == RunStatus.WAITING_APPROVAL

        # Should NOT have tool results (tools weren't executed)
        tool_results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert len(tool_results) == 0

    async def test_deny_plus_approval_in_stream(self):
        """Denied tool emits tool result, approval tool pauses in stream."""
        tc_delete = ToolCall(
            name="delete_account",
            params={"user_id": 42},
            provider_tool_call_id="t1",
        )
        tc_refund = ToolCall(
            name="refund",
            params={"order_id": 123},
            provider_tool_call_id="t2",
        )
        llm = MockLLM(
            [
                LLMResponse(
                    text="Deleting and refunding",
                    tool_calls=[tc_delete, tc_refund],
                ),
            ]
        )
        agent = _make_agent(
            deny=["delete_account"],
            require_approval=["refund"],
        )

        events = []
        async for event in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Delete and refund",
        ):
            events.append(event)

        # Denied tool emits a tool result (deny error)
        tool_results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert len(tool_results) == 1
        assert tool_results[0].tool_result.success is False
        assert "denied by policy" in tool_results[0].tool_result.error

        # Run pauses for approval
        paused = [e for e in events if e.type == RunEventType.RUN_PAUSED]
        assert len(paused) == 1
        assert paused[0].run_result.status == RunStatus.WAITING_APPROVAL
