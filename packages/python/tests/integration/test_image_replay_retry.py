"""I-05 / §5.14: replay and retry of image runs on SQLite + Postgres.

State initialisation for seeded runs (history images count as sent, carried
rejections / descriptions / guardrail verdicts, single-source drop view) and
the approval -> client pause -> resume carry of ``image_view`` (N-2). These need
the persistence (B), provider (C) and loop (D1) tracks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.chat import ChatMessage
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
        iteration_index=2,
        data={"image_id": "img_" + SHA[:12], "sha256": SHA, "tenant_scope": "-"},
    )
    retry_provider = _mock([LLMResponse(text="again")])
    await _agent(store, retry_provider).retry(source.run_id)
    tool_msg = next(m for m in _sent(retry_provider) if m.role == Role.TOOL)
    assert not tool_msg.images
    assert "was rejected by the model provider" in tool_msg.content


def _visible_tools(msgs: list[Any]) -> list[str]:
    return [m.content.split("\n")[0] for m in msgs if m.role == Role.TOOL and m.images]


def _view(msgs: list[Any]) -> list[tuple[str, str, list[str]]]:
    return [(m.role.value, m.content, [p.ref.sha256 for p in m.images]) for m in msgs]


async def _drop_source_then_follow(engine: Any, history: list[ChatMessage]) -> tuple[Any, Any]:
    store = SQLAlchemyStateStore(engine)
    policy = ImagePolicy(drop=ImageDropPolicy(keep_recent=2, interval=2))
    calls = [
        LLMResponse(text=None, tool_calls=[ToolCall(name="screenshot", params={"n": i})])
        for i in (1, 2, 3, 4, 5)
    ]
    provider = _mock([*calls, LLMResponse(text="done")])
    source = await _agent(store, provider, images=policy).run("go", history=history)
    follow = _mock([LLMResponse(text="next")])
    await _agent(store, follow, images=policy).run("more", runs=[source.run_id], replay_last=1)
    return _sent(provider, -1), _sent(follow)


async def test_single_source_full_replay_keeps_drop_view(engine: Any) -> None:
    # N-8: the follow-up's first request shows exactly the images the source's
    # last request showed, with the same sticky placeholders...
    src, fol = await _drop_source_then_follow(
        engine, [ChatMessage.user("old"), ChatMessage.assistant("ok")]
    )
    assert _visible_tools(fol) == _visible_tools(src)
    dropped = [m.content for m in src if "omitted from context" in m.content]
    assert dropped and [m.content for m in fol if "omitted from context" in m.content] == dropped
    # ...also when the source had its own seeded image (P1-1: slots are shifted).
    src_h, fol_h = await _drop_source_then_follow(
        engine, [ChatMessage.user(["old", Image.from_bytes(PNG)]), ChatMessage.assistant("ok")]
    )
    assert _visible_tools(src_h) == _visible_tools(src)
    assert _visible_tools(fol_h) == _visible_tools(src_h)


async def _approval_scenario(engine: Any, approval: bool) -> tuple[list[Any], dict[str, Any]]:
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
    extra = {"require_approval": ["screenshot"]} if approval else {}
    agent = _agent(store, provider, images=policy, **extra)
    paused = await agent.run("go")
    while paused.status == RunStatus.WAITING_APPROVAL:
        paused = await agent.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    client_pause = await store.get_pause_state(paused.run_id)
    call_id = client_pause["pending_tool_calls"][0]["id"]
    # Resume from a fresh store object and Agent ("new process").
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
    return _sent(fresh_provider), client_pause["image_view"]


async def test_approval_client_pause_resume_carries_view(engine: Any) -> None:
    """N-2: approval -> client pause -> fresh-process resume == the run without approvals."""
    with_approval, view_a = await _approval_scenario(engine, approval=True)
    without, view_b = await _approval_scenario(engine, approval=False)
    assert view_a == view_b
    assert _view(with_approval) == _view(without)


async def test_explicit_skill_retry_after_purge_shows_placeholder(engine: Any) -> None:
    """P2-1: the purged input image of an explicit-skills run degrades on retry."""
    from dendrux import Skill

    skill = Skill(name="demo", description="Demo", body="skill body")
    store = SQLAlchemyStateStore(engine)
    source = await _agent(store, _mock([LLMResponse(text="one")]), skills=[skill]).run(
        ["see", Image.from_bytes(PNG)], activate_skills=["demo"], tenant_id="acme"
    )
    assert await _agent(store, _mock([]), skills=[skill]).blobs.purge(SHA, tenant_id="acme")
    retry_provider = _mock([LLMResponse(text="again")])
    result = await _agent(store, retry_provider, skills=[skill]).retry(
        source.run_id, tenant_id="acme"
    )
    assert result.status == RunStatus.SUCCESS
    first_user = next(m for m in _sent(retry_provider) if m.role == Role.USER)
    assert "was deleted" in first_user.content and not first_user.images
