"""The chatbot-threads recipe, run from the page against SQLite with a mock model.

``docs/recipes/chatbot-threads.mdx`` skips its snippets in the docs runner (they
need a real database URL and API key). This test takes the recipe's own
``chat_turn`` function out of the page and runs it with a mock-model agent, so
the recipe cannot drift back into a shape that fails on its first call.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux.chat import ChatMessage
from dendrux.errors import TenantScopeRequiredError
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall

RECIPE = Path(__file__).resolve().parents[4] / "docs" / "recipes" / "chatbot-threads.mdx"


def _recipe_chat_turn(agent: Agent) -> Any:
    """Return the page's ``chat_turn``, bound to ``agent`` as its global."""
    page = RECIPE.read_text(encoding="utf-8")
    match = re.search(r"^async def chat_turn\(.*?(?=^```)", page, re.S | re.M)
    assert match, "chat-threads recipe no longer defines chat_turn"
    namespace: dict[str, Any] = {
        "agent": agent,
        "ChatMessage": ChatMessage,
        "RunStatus": RunStatus,
    }
    exec(compile(match.group(0), str(RECIPE), "exec"), namespace)
    return namespace["chat_turn"]


@pytest.fixture
def url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'chat.sqlite'}"


def _agent(url: str, model: MockLLM, **kw: Any) -> Agent:
    return Agent(
        name="ChatBot",
        provider=model,
        database_url=url,
        prompt="You are a friendly assistant. Reply briefly.",
        **kw,
    )


async def test_a_turn_without_tenant_scope_is_refused(url: str) -> None:
    async with _agent(url, MockLLM([LLMResponse(text="Hello!")])) as agent:
        with pytest.raises(TenantScopeRequiredError):
            await agent.run(user_input="Hi", history=[])


async def test_turns_build_history_and_reads_stay_in_the_tenant(url: str) -> None:
    history: list[ChatMessage] = []
    model = MockLLM(
        [LLMResponse(text="Hello!"), LLMResponse(text="Fine, thanks."), LLMResponse(text="Bye!")]
    )
    async with _agent(url, model) as agent:
        chat_turn = _recipe_chat_turn(agent)
        assert await chat_turn("acme", history, "Hi") == "Hello!"
        assert await chat_turn("acme", history, "How are you?") == "Fine, thanks."
        # The page's tagging example: one more run in this thread.
        await agent.run(
            user_input="Bye",
            history=history,
            tenant_id="acme",
            metadata={"thread_id": "thread_abc", "user_id": "u_123"},
        )

    assert [(m.role.value, m.content) for m in history] == [
        ("user", "Hi"),
        ("assistant", "Hello!"),
        ("user", "How are you?"),
        ("assistant", "Fine, thanks."),
    ]
    second_request = [m.content for m in model.call_history[1]["messages"]]
    assert "Hi" in second_request and "Hello!" in second_request

    async with _agent(url, MockLLM([LLMResponse(text="Other tenant.")])) as other:
        await other.run(user_input="Hi", tenant_id="globex", metadata={"thread_id": "thread_abc"})

    store = RunStore.from_database_url(url)
    try:
        mine = await store.list_runs(metadata_filter={"thread_id": "thread_abc"}, tenant_id="acme")
        everyone = await store.list_runs(metadata_filter={"thread_id": "thread_abc"})
        assert len(mine) == 1
        assert len(everyone) == 2
        assert await store.get_events(mine[0].run_id, limit=100, tenant_id="acme")
        assert not await store.get_events(mine[0].run_id, limit=100, tenant_id="globex")
    finally:
        await store.close()


async def test_a_paused_turn_stays_out_of_history(url: str) -> None:
    @tool()
    async def refund(order_id: str) -> str:
        """Refund an order."""
        return "refunded"

    history: list[ChatMessage] = [ChatMessage.user("Hi"), ChatMessage.assistant("Hello!")]
    model = MockLLM([LLMResponse(tool_calls=[ToolCall("refund", {"order_id": "A1"})])])
    async with _agent(url, model, tools=[refund], require_approval=["refund"]) as agent:
        chat_turn = _recipe_chat_turn(agent)
        assert await chat_turn("acme", history, "Refund order A1") is None
    assert len(history) == 2


async def test_a_failed_turn_stays_out_of_history(url: str) -> None:
    from dendrux.errors import ProviderResponseError

    history: list[ChatMessage] = [ChatMessage.user("Hi"), ChatMessage.assistant("Hello!")]
    async with _agent(url, MockLLM([LLMResponse(text=None)])) as agent:
        chat_turn = _recipe_chat_turn(agent)
        with pytest.raises(ProviderResponseError):
            await chat_turn("acme", history, "Still there?")
    assert len(history) == 2
