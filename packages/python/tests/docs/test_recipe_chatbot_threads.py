"""The chatbot-threads recipe, run as written against SQLite with a mock model.

``docs/recipes/chatbot-threads.mdx`` skips its snippets in the docs runner (they
need a real database URL and API key). This test runs the same ``chat_turn``
body and the same tenant-scoped ``RunStore`` reads, so the recipe cannot drift
back into a shape that fails on its first call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.errors import TenantScopeRequiredError
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus

if TYPE_CHECKING:
    from pathlib import Path


async def chat_turn(
    agent: Agent, tenant: str, history: list[ChatMessage], user_text: str
) -> str | None:
    """The recipe's ``chat_turn``; only ``agent`` is passed in instead of global."""
    result = await agent.run(
        user_input=user_text,
        history=history,
        tenant_id=tenant,
        metadata={"thread_id": "thread_abc", "user_id": "u_123"},
    )
    if result.status != RunStatus.SUCCESS or not result.answer:
        return None
    history.append(ChatMessage.user(user_text))
    history.append(ChatMessage.assistant(result.answer))
    return result.answer


def _agent(url: str, responses: list[LLMResponse]) -> Agent:
    return Agent(
        name="ChatBot",
        provider=MockLLM(responses),
        database_url=url,
        prompt="You are a friendly assistant. Reply briefly.",
    )


@pytest.fixture
def url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'chat.sqlite'}"


async def test_a_turn_without_tenant_scope_is_refused(url: str) -> None:
    async with _agent(url, [LLMResponse(text="Hello!")]) as agent:
        with pytest.raises(TenantScopeRequiredError):
            await agent.run(user_input="Hi", history=[])


async def test_turns_build_history_and_reads_stay_in_the_tenant(url: str) -> None:
    history: list[ChatMessage] = []
    model = MockLLM([LLMResponse(text="Hello!"), LLMResponse(text="Fine, thanks.")])
    async with Agent(
        name="ChatBot",
        provider=model,
        database_url=url,
        prompt="You are a friendly assistant. Reply briefly.",
    ) as agent:
        assert await chat_turn(agent, "acme", history, "Hi") == "Hello!"
        assert await chat_turn(agent, "acme", history, "How are you?") == "Fine, thanks."

    assert [(m.role.value, m.content) for m in history] == [
        ("user", "Hi"),
        ("assistant", "Hello!"),
        ("user", "How are you?"),
        ("assistant", "Fine, thanks."),
    ]
    second_request = [m.content for m in model.call_history[1]["messages"]]
    assert "Hi" in second_request and "Hello!" in second_request

    async with _agent(url, [LLMResponse(text="Other tenant.")]) as other:
        await chat_turn(other, "globex", [], "Hi")

    async with RunStore.from_database_url(url) as store:
        mine = await store.list_runs(metadata_filter={"thread_id": "thread_abc"}, tenant_id="acme")
        everyone = await store.list_runs(metadata_filter={"thread_id": "thread_abc"})
        assert len(mine) == 2
        assert len(everyone) == 3
        run = mine[0]
        assert await store.get_events(run.run_id, limit=100, tenant_id="acme")
        assert not await store.get_events(run.run_id, limit=100, tenant_id="globex")


async def test_a_failed_turn_stays_out_of_history(url: str) -> None:
    from dendrux.errors import ProviderResponseError

    history: list[ChatMessage] = [ChatMessage.user("Hi"), ChatMessage.assistant("Hello!")]
    async with _agent(url, [LLMResponse(text=None)]) as agent:
        with pytest.raises(ProviderResponseError):
            await chat_turn(agent, "acme", history, "Still there?")
    assert len(history) == 2
