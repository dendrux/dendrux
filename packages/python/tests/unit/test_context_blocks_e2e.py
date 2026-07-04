"""End-to-end tests for context blocks (agent.run / agent.stream).

Covers text-folding assembly (stable → first user message, dynamic → current
user message), the no-context path staying byte-identical, streaming, the
stable-placement tag reaching the provider, and idempotency hashing.
"""

from __future__ import annotations

from dendrux.agent import Agent
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    LLMResponse,
    Message,
    Role,
    RunEventType,
    RunStatus,
    UsageStats,
    compute_idempotency_fingerprint,
)


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=None, usage=UsageStats())


def _agent(llm: MockLLM) -> Agent:
    return Agent(provider=llm, prompt="You are helpful.", tools=[])


def _user_messages(llm: MockLLM) -> list[Message]:
    sent = llm.call_history[0]["messages"]
    return [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]


class TestFoldingReachesProvider:
    async def test_stable_folds_into_first_user_message(self) -> None:
        llm = MockLLM([_resp("Paris")])
        agent = _agent(llm)
        result = await agent.run(
            user_input="Capital of France?",
            history=[ChatMessage.user("Hello"), ChatMessage.assistant("Hi")],
            context=[ContextBlock("project spec", kind="spec", placement="stable")],
        )
        assert result.status == RunStatus.SUCCESS
        msgs = _user_messages(llm)
        assert msgs[0].content == "STABLE CONTEXT:\nproject spec\n\nUSER MESSAGE:\nHello"
        assert msgs[-1].content == "Capital of France?"

    async def test_dynamic_folds_into_current_message(self) -> None:
        llm = MockLLM([_resp("ok")])
        agent = _agent(llm)
        await agent.run(
            user_input="Summarize",
            history=[ChatMessage.user("Hello"), ChatMessage.assistant("Hi")],
            context=[ContextBlock("retrieved doc", kind="doc")],
        )
        msgs = _user_messages(llm)
        assert msgs[0].content == "Hello"  # first untouched
        assert msgs[-1].content == "ADDITIONAL CONTEXT:\nretrieved doc\n\nUSER INPUT:\nSummarize"

    async def test_stable_and_dynamic_together(self) -> None:
        llm = MockLLM([_resp("ok")])
        agent = _agent(llm)
        await agent.run(
            user_input="Q",
            history=[ChatMessage.user("H"), ChatMessage.assistant("A")],
            context=[
                ContextBlock("instructions", placement="stable"),
                ContextBlock("doc", placement="dynamic"),
            ],
        )
        msgs = _user_messages(llm)
        assert msgs[0].content == "STABLE CONTEXT:\ninstructions\n\nUSER MESSAGE:\nH"
        assert msgs[-1].content == "ADDITIONAL CONTEXT:\ndoc\n\nUSER INPUT:\nQ"

    async def test_no_history_single_turn(self) -> None:
        llm = MockLLM([_resp("ok")])
        agent = _agent(llm)
        await agent.run(
            user_input="Q",
            context=[ContextBlock("instructions", placement="stable")],
        )
        msgs = _user_messages(llm)
        assert msgs[0].content == "STABLE CONTEXT:\ninstructions\n\nUSER MESSAGE:\nQ"

    async def test_stable_message_is_placement_tagged(self) -> None:
        llm = MockLLM([_resp("ok")])
        agent = _agent(llm)
        await agent.run(
            user_input="Q",
            history=[ChatMessage.user("H"), ChatMessage.assistant("A")],
            context=[ContextBlock("spec", placement="stable")],
        )
        user_msgs = [m for m in llm.call_history[0]["messages"] if m.role == Role.USER]
        assert user_msgs[0].placement == "stable"  # cache-breakpoint signal
        assert user_msgs[-1].placement == "dynamic"


class TestNoContextUnchanged:
    async def test_no_context_byte_identical_to_plain(self) -> None:
        llm_ctx = MockLLM([_resp("a")])
        llm_plain = MockLLM([_resp("b")])
        hist = [ChatMessage.user("Hello"), ChatMessage.assistant("Hi")]
        await Agent(provider=llm_ctx, prompt="X", tools=[]).run(
            user_input="Q", history=hist, context=None
        )
        await Agent(provider=llm_plain, prompt="X", tools=[]).run(user_input="Q", history=hist)
        assert llm_ctx.call_history[0]["messages"] == llm_plain.call_history[0]["messages"]

    async def test_empty_context_is_noop(self) -> None:
        llm = MockLLM([_resp("ok")])
        result = await _agent(llm).run(user_input="hi", context=[])
        assert result.status == RunStatus.SUCCESS


class TestStreamAcceptsContext:
    async def test_stream_folds_context(self) -> None:
        llm = MockLLM([_resp("Berlin")])
        agent = Agent(provider=llm, prompt="Geo.", tools=[])
        events = []
        async for ev in agent.stream(
            user_input="Capital?",
            context=[ContextBlock("hint", placement="stable")],
        ):
            events.append(ev)
        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
        msgs = _user_messages(llm)
        assert msgs[-1].content == "STABLE CONTEXT:\nhint\n\nUSER MESSAGE:\nCapital?"


class TestCrossRunStablePrefix:
    async def test_stable_first_message_byte_stable_across_turns(self) -> None:
        # Turn 1
        llm1 = MockLLM([_resp("a1")])
        await Agent(provider=llm1, prompt="X", tools=[]).run(
            user_input="Q1", context=[ContextBlock("spec", placement="stable")]
        )
        # Turn 2 — Q1 now in history, same stable context re-supplied
        llm2 = MockLLM([_resp("a2")])
        await Agent(provider=llm2, prompt="X", tools=[]).run(
            user_input="Q2",
            history=[ChatMessage.user("Q1"), ChatMessage.assistant("a1")],
            context=[ContextBlock("spec", placement="stable")],
        )
        first_turn1 = [m for m in llm1.call_history[0]["messages"] if m.role == Role.USER][0]
        first_turn2 = [m for m in llm2.call_history[0]["messages"] if m.role == Role.USER][0]
        # The stable head is byte-identical across turns (cacheable prefix).
        assert (
            first_turn1.content
            == first_turn2.content
            == "STABLE CONTEXT:\nspec\n\nUSER MESSAGE:\nQ1"
        )


class TestIdempotencyWithContext:
    def _ctx(self, content: str) -> list[ContextBlock]:
        return [ContextBlock(content, kind="doc", placement="dynamic")]

    def test_same_context_same_hash(self) -> None:
        h1 = compute_idempotency_fingerprint("a", "in", context=self._ctx("x"))
        h2 = compute_idempotency_fingerprint("a", "in", context=self._ctx("x"))
        assert h1 == h2

    def test_different_context_different_hash(self) -> None:
        h1 = compute_idempotency_fingerprint("a", "in", context=self._ctx("x"))
        h2 = compute_idempotency_fingerprint("a", "in", context=self._ctx("y"))
        assert h1 != h2

    def test_context_present_changes_hash(self) -> None:
        h_none = compute_idempotency_fingerprint("a", "in")
        h_with = compute_idempotency_fingerprint("a", "in", context=self._ctx("x"))
        assert h_none != h_with

    def test_none_vs_empty_match(self) -> None:
        h_none = compute_idempotency_fingerprint("a", "in", context=None)
        h_empty = compute_idempotency_fingerprint("a", "in", context=[])
        assert h_none == h_empty

    def test_placement_change_changes_hash(self) -> None:
        stable = [ContextBlock("x", placement="stable")]
        dynamic = [ContextBlock("x", placement="dynamic")]
        h_s = compute_idempotency_fingerprint("a", "in", context=stable)
        h_d = compute_idempotency_fingerprint("a", "in", context=dynamic)
        assert h_s != h_d
