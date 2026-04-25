"""End-to-end tests for chat history support in agent.run / agent.stream.

Covers ReActLoop and SingleCall, both run() and run_stream(), validation,
idempotency hashing, and byte-stable provider input across turns.
"""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.chat import ChatMessage, ChatRole
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
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


def _agent(**overrides) -> Agent:
    defaults: dict = {
        "provider": MockLLM([_resp("ok")]),
        "prompt": "You are helpful.",
        "tools": [],
    }
    defaults.update(overrides)
    return Agent(**defaults)


# -----------------------------------------------------------------------
# ReActLoop + history — agent.run()
# -----------------------------------------------------------------------


class TestReActLoopRunWithHistory:
    async def test_history_seeds_provider_messages(self) -> None:
        llm = MockLLM([_resp("Paris")])
        agent = Agent(provider=llm, prompt="Geography.", tools=[])
        history = [
            ChatMessage.user("Hello"),
            ChatMessage.assistant("Hi! Ask me about geography."),
        ]

        result = await agent.run(user_input="Capital of France?", history=history)

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Paris"
        # Inspect messages sent to the provider
        sent = llm.call_history[0]["messages"]
        # Skip system; find user/assistant turns
        user_assistant = [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]
        assert len(user_assistant) == 3
        assert user_assistant[0].role == Role.USER
        assert user_assistant[0].content == "Hello"
        assert user_assistant[1].role == Role.ASSISTANT
        assert user_assistant[1].content == "Hi! Ask me about geography."
        assert user_assistant[2].role == Role.USER
        assert user_assistant[2].content == "Capital of France?"

    async def test_no_history_works_unchanged(self) -> None:
        llm = MockLLM([_resp("hello")])
        agent = _agent(provider=llm)
        result = await agent.run(user_input="hi")
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "hello"

    async def test_empty_history_treated_as_no_history(self) -> None:
        llm = MockLLM([_resp("ok")])
        agent = _agent(provider=llm)
        result = await agent.run(user_input="hi", history=[])
        assert result.status == RunStatus.SUCCESS

    async def test_invalid_history_raises_before_loop_runs(self) -> None:
        llm = MockLLM([_resp("never reached")])
        agent = _agent(provider=llm)
        with pytest.raises(ValueError, match="must start with a user message"):
            await agent.run(
                user_input="hi",
                history=[ChatMessage.assistant("oops")],
            )
        assert llm.calls_made == 0


# -----------------------------------------------------------------------
# ReActLoop + history — agent.stream()
# -----------------------------------------------------------------------


class TestReActLoopStreamWithHistory:
    async def test_history_seeds_provider_messages(self) -> None:
        llm = MockLLM([_resp("Berlin")])
        agent = Agent(provider=llm, prompt="Geography.", tools=[])
        history = [
            ChatMessage.user("Hi"),
            ChatMessage.assistant("Yes?"),
        ]

        events = []
        async for ev in agent.stream(user_input="Capital of Germany?", history=history):
            events.append(ev)

        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
        assert completed[0].run_result.answer == "Berlin"

        sent = llm.call_history[0]["messages"]
        user_assistant = [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]
        assert [m.role for m in user_assistant] == [Role.USER, Role.ASSISTANT, Role.USER]
        assert [m.content for m in user_assistant] == ["Hi", "Yes?", "Capital of Germany?"]

    async def test_invalid_history_raises_synchronously(self) -> None:
        llm = MockLLM([_resp("never")])
        agent = _agent(provider=llm)
        with pytest.raises(ValueError, match="pass it as user_input instead"):
            agent.stream(
                user_input="hi",
                history=[
                    ChatMessage.user("a"),
                    ChatMessage.assistant("b"),
                    ChatMessage.user("c"),
                ],
            )
        assert llm.calls_made == 0


# -----------------------------------------------------------------------
# SingleCall + history — both run() and run_stream()
# -----------------------------------------------------------------------


class TestSingleCallWithHistory:
    async def test_run_accepts_history(self) -> None:
        llm = MockLLM([_resp("complaint")])
        agent = Agent(
            provider=llm,
            prompt="Classify intent: greeting, question, complaint.",
            tools=[],
            loop=SingleCall(),
        )
        history = [
            ChatMessage.user("show me Q1 revenue"),
            ChatMessage.assistant("Q1 revenue was $1.2M"),
        ]

        result = await agent.run(
            user_input="that's not what I wanted",
            history=history,
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "complaint"
        sent = llm.call_history[0]["messages"]
        user_assistant = [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]
        assert [m.content for m in user_assistant] == [
            "show me Q1 revenue",
            "Q1 revenue was $1.2M",
            "that's not what I wanted",
        ]

    async def test_run_no_history_works_unchanged(self) -> None:
        llm = MockLLM([_resp("greeting")])
        agent = Agent(provider=llm, prompt="Classify.", tools=[], loop=SingleCall())
        result = await agent.run(user_input="hi")
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "greeting"

    async def test_stream_accepts_history(self) -> None:
        llm = MockLLM([_resp("Chess opening: Italian Game.")])
        agent = Agent(
            provider=llm,
            prompt="You are a chess teacher.",
            tools=[],
            loop=SingleCall(),
        )
        history = [
            ChatMessage.user("hello"),
            ChatMessage.assistant("ready to learn?"),
        ]

        events = []
        async for ev in agent.stream(user_input="what's a good opening?", history=history):
            events.append(ev)

        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
        assert completed[0].run_result.answer == "Chess opening: Italian Game."

        sent = llm.call_history[0]["messages"]
        user_assistant = [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]
        assert [m.content for m in user_assistant] == [
            "hello",
            "ready to learn?",
            "what's a good opening?",
        ]


class TestSingleCallStillRejectsGenuineResume:
    async def test_rejects_initial_steps(self) -> None:
        from dendrux.types import AgentStep, Finish

        llm = MockLLM([_resp("never")])
        agent = Agent(provider=llm, prompt="x", tools=[], loop=SingleCall())
        loop = SingleCall()
        from dendrux.strategies.native import NativeToolCalling

        with pytest.raises(RuntimeError, match="does not support resume"):
            await loop.run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="hi",
                initial_steps=[AgentStep(reasoning=None, action=Finish(answer="x"))],
            )

    async def test_rejects_iteration_offset(self) -> None:
        llm = MockLLM([_resp("never")])
        agent = Agent(provider=llm, prompt="x", tools=[], loop=SingleCall())
        loop = SingleCall()
        from dendrux.strategies.native import NativeToolCalling

        with pytest.raises(RuntimeError, match="does not support resume"):
            await loop.run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="hi",
                iteration_offset=3,
            )

    async def test_rejects_initial_usage(self) -> None:
        llm = MockLLM([_resp("never")])
        agent = Agent(provider=llm, prompt="x", tools=[], loop=SingleCall())
        loop = SingleCall()
        from dendrux.strategies.native import NativeToolCalling

        with pytest.raises(RuntimeError, match="does not support resume"):
            await loop.run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="hi",
                initial_usage=UsageStats(input_tokens=100),
            )


# -----------------------------------------------------------------------
# Idempotency hashing
# -----------------------------------------------------------------------


class TestIdempotencyFingerprintWithHistory:
    def test_same_history_same_hash(self) -> None:
        history = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b"),
        ]
        h1 = compute_idempotency_fingerprint("agent", "input", history=history)
        h2 = compute_idempotency_fingerprint("agent", "input", history=history)
        assert h1 == h2

    def test_different_history_different_hash(self) -> None:
        history_a = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b"),
        ]
        history_b = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="DIFFERENT"),
        ]
        h1 = compute_idempotency_fingerprint("agent", "input", history=history_a)
        h2 = compute_idempotency_fingerprint("agent", "input", history=history_b)
        assert h1 != h2

    def test_no_history_vs_empty_history_match(self) -> None:
        h_none = compute_idempotency_fingerprint("agent", "input", history=None)
        h_empty = compute_idempotency_fingerprint("agent", "input", history=[])
        assert h_none == h_empty

    def test_history_present_changes_hash_from_no_history(self) -> None:
        history = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b"),
        ]
        h_none = compute_idempotency_fingerprint("agent", "input", history=None)
        h_with = compute_idempotency_fingerprint("agent", "input", history=history)
        assert h_none != h_with

    def test_history_order_matters(self) -> None:
        h_ab = compute_idempotency_fingerprint(
            "agent",
            "input",
            history=[
                Message(role=Role.USER, content="a"),
                Message(role=Role.ASSISTANT, content="b"),
            ],
        )
        h_cd = compute_idempotency_fingerprint(
            "agent",
            "input",
            history=[
                Message(role=Role.USER, content="c"),
                Message(role=Role.ASSISTANT, content="d"),
            ],
        )
        assert h_ab != h_cd


# -----------------------------------------------------------------------
# Byte-stable provider input across turns
# -----------------------------------------------------------------------


class TestByteStableProviderInputAcrossTurns:
    async def test_same_history_same_provider_messages(self) -> None:
        """Two runs with identical inputs must send identical messages to provider."""
        llm1 = MockLLM([_resp("ok1")])
        llm2 = MockLLM([_resp("ok2")])
        history = [
            ChatMessage.user("Hello"),
            ChatMessage.assistant("Hi!"),
        ]
        agent1 = Agent(provider=llm1, prompt="X", tools=[])
        agent2 = Agent(provider=llm2, prompt="X", tools=[])

        await agent1.run(user_input="Q", history=history)
        await agent2.run(user_input="Q", history=history)

        msgs1 = llm1.call_history[0]["messages"]
        msgs2 = llm2.call_history[0]["messages"]
        assert msgs1 == msgs2

    async def test_chatmessage_construction_path_does_not_affect_bytes(self) -> None:
        """Different construction paths producing equivalent history → identical bytes."""
        llm1 = MockLLM([_resp("ok1")])
        llm2 = MockLLM([_resp("ok2")])
        via_factory = [
            ChatMessage.user("Hello"),
            ChatMessage.assistant("Hi!"),
        ]
        via_direct = [
            ChatMessage(role=ChatRole.USER, content="Hello"),
            ChatMessage(role=ChatRole.ASSISTANT, content="Hi!"),
        ]
        agent1 = Agent(provider=llm1, prompt="X", tools=[])
        agent2 = Agent(provider=llm2, prompt="X", tools=[])

        await agent1.run(user_input="Q", history=via_factory)
        await agent2.run(user_input="Q", history=via_direct)

        assert llm1.call_history[0]["messages"] == llm2.call_history[0]["messages"]

    async def test_growing_history_preserves_prior_prefix(self) -> None:
        """Turn N+1's prefix bytes match what was sent in turn N."""
        # Turn 1
        llm1 = MockLLM([_resp("answer1")])
        agent1 = Agent(provider=llm1, prompt="X", tools=[])
        await agent1.run(user_input="first")
        msgs_turn1 = llm1.call_history[0]["messages"]

        # Turn 2 — dev passes turn 1 as history
        llm2 = MockLLM([_resp("answer2")])
        agent2 = Agent(provider=llm2, prompt="X", tools=[])
        await agent2.run(
            user_input="second",
            history=[
                ChatMessage.user("first"),
                ChatMessage.assistant("answer1"),
            ],
        )
        msgs_turn2 = llm2.call_history[0]["messages"]

        # The first messages of turn 2 must match what turn 1 sent
        # (same system + same first user message).
        assert msgs_turn2[: len(msgs_turn1)] == msgs_turn1


# -----------------------------------------------------------------------
# Trace policy: seeded history NOT recorded; new user_input IS recorded
# -----------------------------------------------------------------------


class _RecordingNotifier:
    """Test helper — captures everything the loop notifies."""

    def __init__(self) -> None:
        self.messages: list[tuple[Message, int]] = []

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        self.messages.append((message, iteration))

    async def on_llm_call_completed(self, response, iteration: int, **kwargs) -> None:
        pass

    async def on_tool_completed(self, tool_call, tool_result, iteration: int) -> None:
        pass


class TestTracePolicyForChatbotHistory:
    async def test_react_loop_records_user_input_only_not_seeded(self) -> None:
        """Seeded prior turns must NOT be recorded; new user_input MUST be."""
        llm = MockLLM([_resp("Paris")])
        agent = Agent(provider=llm, prompt="Geo.", tools=[])
        notifier = _RecordingNotifier()

        await agent.run(
            user_input="Capital of France?",
            history=[
                ChatMessage.user("Hello"),
                ChatMessage.assistant("Hi! Ask me anything."),
            ],
            notifier=notifier,
        )

        # Notifier should see EXACTLY: new user_input + final assistant.
        # Seeded "Hello" and "Hi!" must NOT appear.
        assert [m.role for m, _ in notifier.messages] == [Role.USER, Role.ASSISTANT]
        assert [m.content for m, _ in notifier.messages] == [
            "Capital of France?",
            "Paris",
        ]

    async def test_single_call_records_user_input_only_not_seeded(self) -> None:
        """SingleCall chatbot path: same trace policy as ReActLoop."""
        llm = MockLLM([_resp("complaint")])
        agent = Agent(provider=llm, prompt="Classify.", tools=[], loop=SingleCall())
        notifier = _RecordingNotifier()

        await agent.run(
            user_input="that's wrong",
            history=[
                ChatMessage.user("show Q1 revenue"),
                ChatMessage.assistant("Q1 was $1.2M"),
            ],
            notifier=notifier,
        )

        assert [m.role for m, _ in notifier.messages] == [Role.USER, Role.ASSISTANT]
        assert [m.content for m, _ in notifier.messages] == [
            "that's wrong",
            "complaint",
        ]

    async def test_react_loop_no_history_unchanged_recording(self) -> None:
        """Non-chatbot fresh runs: existing recording behavior preserved."""
        llm = MockLLM([_resp("hi back")])
        agent = Agent(provider=llm, prompt="X", tools=[])
        notifier = _RecordingNotifier()

        await agent.run(user_input="hi", notifier=notifier)

        assert [m.role for m, _ in notifier.messages] == [Role.USER, Role.ASSISTANT]
        assert [m.content for m, _ in notifier.messages] == ["hi", "hi back"]

    async def test_single_call_no_history_unchanged_recording(self) -> None:
        """Non-chatbot SingleCall: existing recording behavior preserved."""
        llm = MockLLM([_resp("greeting")])
        agent = Agent(provider=llm, prompt="Classify.", tools=[], loop=SingleCall())
        notifier = _RecordingNotifier()

        await agent.run(user_input="hi", notifier=notifier)

        assert [m.role for m, _ in notifier.messages] == [Role.USER, Role.ASSISTANT]
        assert [m.content for m, _ in notifier.messages] == ["hi", "greeting"]


# -----------------------------------------------------------------------
# Guardrail interaction with seeded history
# -----------------------------------------------------------------------


class _RecordingGuardrail:
    """Test guardrail — records every text it scans, returns no findings."""

    action = "warn"

    def __init__(self) -> None:
        self.scanned: list[str] = []

    async def scan(self, text: str):
        self.scanned.append(text)
        return []


class TestGuardrailScansSeededHistory:
    async def test_incoming_guardrail_scans_seeded_history_content(self) -> None:
        """Incoming guardrails must see content from seeded history, not just user_input."""
        llm = MockLLM([_resp("ok")])
        guard = _RecordingGuardrail()
        agent = Agent(
            provider=llm,
            prompt="X",
            tools=[],
            loop=SingleCall(),
            guardrails=[guard],
        )

        await agent.run(
            user_input="latest question",
            history=[
                ChatMessage.user("PRIOR-USER-CONTENT"),
                ChatMessage.assistant("PRIOR-ASSISTANT-CONTENT"),
            ],
        )

        scanned_text = " | ".join(guard.scanned)
        assert "PRIOR-USER-CONTENT" in scanned_text
        assert "PRIOR-ASSISTANT-CONTENT" in scanned_text
        assert "latest question" in scanned_text
