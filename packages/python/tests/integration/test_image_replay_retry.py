"""I-05 / §5.14: replay and retry of image runs on SQLite + Postgres.

State initialisation for seeded runs (history images count as sent, carried
rejections / descriptions / guardrail verdicts, single-source drop view) and
the approval -> client pause -> resume carry of ``image_view`` (N-2). These need
the persistence (B), provider (C) and loop (D1) tracks; the coordinator removes
the xfail marks at integration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import UnsupportedModalityError
from dendrux.llm.mock import MockLLM
from dendrux.media import Image, ImageDropPolicy, ImagePolicy, ToolOutput
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, Role, RunStatus, ToolCall, ToolResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "images"
PNG = (FIXTURES / "zebra42.png").read_bytes()
PNG2 = (FIXTURES / "kiwi77.png").read_bytes()
SHA = hashlib.sha256(PNG).hexdigest()

pytestmark = pytest.mark.xfail(strict=False, reason="needs B/C/D1 merged")


@tool()
async def screenshot(n: int) -> ToolOutput:
    """Server screenshot."""
    return ToolOutput(f"shot {n}", images=[Image.from_bytes(PNG if n % 2 else PNG2)])


@tool(target="client")
async def client_shot() -> str:
    """Client screenshot."""
    return "never"


def _mock(responses: list[LLMResponse], support: str = "yes") -> MockLLM:
    return MockLLM(responses, image_support=support)  # type: ignore[call-arg]


def _agent(store: Any, provider: MockLLM, **kw: Any) -> Agent:
    return Agent(
        name="img_agent",
        prompt="p",
        provider=provider,
        tools=kw.pop("tools", [screenshot, client_shot]),
        state_store=store,
        **kw,
    )


def _sent(provider: MockLLM, call: int = 0) -> list[Any]:
    return [m for m in provider.call_history[call]["messages"] if m.role != Role.SYSTEM]


async def test_replay_into_text_only_model_degrades_history(engine: Any) -> None:
    store = SQLAlchemyStateStore(engine)
    source = await _agent(store, _mock([LLMResponse(text="one")])).run(
        ["see", Image.from_bytes(PNG), Image.from_bytes(PNG2)]
    )
    text_only = _mock([LLMResponse(text="two")], support="no")
    result = await _agent(store, text_only).run("and?", runs=[source.run_id], replay_last=1)
    assert result.status == RunStatus.SUCCESS
    first = _sent(text_only)[0]
    assert not first.images
    assert "not shown: this model does not accept images" in first.content
    with pytest.raises(UnsupportedModalityError):
        await _agent(store, _mock([LLMResponse(text="x")], support="no")).run(
            ["new", Image.from_bytes(PNG)], runs=[source.run_id]
        )


async def test_retry_never_resends_rejected_tool_image(engine: Any) -> None:
    store = SQLAlchemyStateStore(engine)
    provider = _mock(
        [
            LLMResponse(text=None, tool_calls=[ToolCall(name="screenshot", params={"n": 1})]),
            LLMResponse(text="done"),
        ]
    )
    source = await _agent(store, provider).run("shoot")
    await store.save_run_event(
        source.run_id,
        event_type="image.rejected",
        sequence_index=999,
        iteration_index=2,
        data={"image_id": "img_" + SHA[:12], "sha256": SHA, "tenant_scope": "-"},
    )
    retry_provider = _mock([LLMResponse(text="again")])
    await _agent(store, retry_provider).retry(source.run_id)
    tool_msg = next(m for m in _sent(retry_provider) if m.role == Role.TOOL)
    assert not tool_msg.images
    assert "was rejected by the model provider" in tool_msg.content


async def test_single_source_full_replay_keeps_drop_view(engine: Any) -> None:
    store = SQLAlchemyStateStore(engine)
    calls = [
        LLMResponse(text=None, tool_calls=[ToolCall(name="screenshot", params={"n": i})])
        for i in range(5)
    ]
    policy = ImagePolicy(drop=ImageDropPolicy(keep_recent=1, interval=2))
    provider = _mock([*calls, LLMResponse(text="done")])
    source = await _agent(store, provider, images=policy).run("shoot five")
    last_request = _sent(provider, -1)
    follow = _mock([LLMResponse(text="next")])
    await _agent(store, follow, images=policy).run("more", runs=[source.run_id], replay_last=1)
    replayed = _sent(follow)
    # N-8: the source's sticky drop placeholders carry over slot for slot (the
    # follow-up may drop more: the source's last tool batch is no longer the
    # current turn).
    dropped = [i for i, m in enumerate(last_request) if "omitted from context" in m.content]
    assert dropped
    for i in dropped:
        assert replayed[i].content == last_request[i].content


async def test_approval_client_pause_resume_carries_view(engine: Any) -> None:
    store = SQLAlchemyStateStore(engine)
    policy = ImagePolicy(drop=ImageDropPolicy(keep_recent=1, interval=1))
    provider = _mock(
        [
            LLMResponse(text=None, tool_calls=[ToolCall(name="screenshot", params={"n": 1})]),
            LLMResponse(text=None, tool_calls=[ToolCall(name="screenshot", params={"n": 2})]),
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(name="screenshot", params={"n": 3}),
                    ToolCall(name="client_shot", params={}),
                ],
            ),
        ]
    )
    agent = _agent(store, provider, images=policy, require_approval=["screenshot"])
    paused = await agent.run("go")
    while paused.status == RunStatus.WAITING_APPROVAL:
        before = await store.get_pause_state(paused.run_id)
        paused = await agent.submit_approval(paused.run_id, approved=True)
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    client_pause = await store.get_pause_state(paused.run_id)
    assert client_pause["image_view"] == before["image_view"]
    call_id = client_pause["pending_tool_calls"][0]["id"]

    fresh_provider = _mock([LLMResponse(text="done")])
    fresh = _agent(SQLAlchemyStateStore(engine), fresh_provider, images=policy)
    result = await fresh.submit_tool_results(
        paused.run_id,
        [
            ToolResult(
                name="client_shot",
                call_id=call_id,
                payload='"ok"',
                images=(Image.from_bytes(PNG2),),
            )
        ],
    )
    assert result.status == RunStatus.SUCCESS
    assert json.dumps(client_pause["image_view"], sort_keys=True)
