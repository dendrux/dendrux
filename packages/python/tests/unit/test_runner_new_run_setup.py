"""Characterization of new-run creation inputs before setup is consolidated."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from dendrux import Agent
from dendrux.chat import ChatMessage, normalize_chat_history
from dendrux.context_blocks import ContextBlock
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _run_images, context, runner
from dendrux.runtime.context import DelegationContext, get_store_identity
from dendrux.runtime.replay import PARTS_KEY, parts_to_dicts
from dendrux.runtime.state import RunRecord, TraceRecord
from dendrux.types import ImagePart, ImageRef, TextPart
from tests.unit.test_runner_persistence import RecordingStateStore


class Answer(BaseModel):
    value: str


class FalseyModelMeta(type(BaseModel)):
    def __bool__(cls):
        return False


class FalseyAnswer(BaseModel, metaclass=FalseyModelMeta):
    value: str


class CaptureCreateError(Exception):
    pass


@pytest.mark.parametrize("case", ["plain", "rich", "unbounded", "falsey"])
async def test_entry_creation_bytes(monkeypatch, case):
    store = RecordingStateStore()
    agent = Agent(name="setup", prompt="prompt")
    provider = MockLLM([], image_support="yes")
    rich = case in ("rich", "falsey")
    tenant = "tenant" if rich else None
    depth = None if case == "unbounded" else 10
    parts = (TextPart("hello"), ImagePart(ImageRef("a" * 64, "image/png", 10, 1, 1)))
    source = RunRecord(
        id="source",
        agent_name="setup",
        status="success",
        tenant_id=tenant,
        input_data={"input": "hello", **({PARTS_KEY: parts_to_dicts(parts)} if rich else {})},
    )
    store.get_run = AsyncMock(return_value=source)
    store.get_traces = AsyncMock(return_value=[TraceRecord("trace", "user", "hello", 0)])
    monkeypatch.setattr(runner, "_runtime_seeds", AsyncMock(return_value=({}, None)))
    monkeypatch.setattr(_run_images, "_image_runtime", lambda *args: None)
    history = [ChatMessage.user("before"), ChatMessage.assistant("reply")] if rich else None
    monkeypatch.setattr(
        _run_images,
        "_ingest_entry_images",
        AsyncMock(return_value=runner._EntryImages(None, parts, normalize_chat_history(history))),
    )
    if rich:
        parent = DelegationContext(
            "parent",
            delegation_level=2,
            persisted=True,
            store_identity=get_store_identity(store),
            tenant_id=tenant,
            max_delegation_depth=10,
        )
        monkeypatch.setattr(context, "get_delegation_context", lambda: parent)
    captures = {}

    async def capture(store, loop, run_id, name, **kwargs):
        captures[entry] = {"agent_name": name, **kwargs}
        raise CaptureCreateError("capture")

    monkeypatch.setattr(runner, "_create_runtime_run", capture)
    metadata = {"thread": "thread"} if rich else None
    for entry in ("run", "stream", "retry"):
        common = dict(
            agent=agent, provider=provider, state_store=store, tenant_id=tenant, metadata=metadata
        )
        if entry == "retry":
            agent.max_delegation_depth = depth
            with pytest.raises(CaptureCreateError):
                await runner.retry("source", **common)
            continue
        options = dict(
            user_input=["hello", parts[1].ref] if rich else "hello",
            history=history,
            context=[ContextBlock(content="context")] if rich else None,
            max_delegation_depth=depth,
            idempotency_key="key" if rich else None,
            output_type=(FalseyAnswer if case == "falsey" else Answer) if rich else None,
        )
        if entry == "run":
            with pytest.raises(CaptureCreateError):
                await runner.run(**common, **options)
        else:
            async with runner.run_stream(**common, **options) as stream:
                events = [event async for event in stream]
            assert len(events) == 1 and events[0].error == "capture"
    wire = json.dumps(captures, sort_keys=True, indent=2) + "\n"
    expected = Path(__file__).with_name("fixtures") / f"new_run_{case}.json"
    assert wire == expected.read_text()
    assert metadata == ({"thread": "thread"} if rich else None)


def test_pure_metadata_keeps_skills_replay_and_caller_metadata():
    from unittest.mock import Mock

    from dendrux.loops.react import ReActLoop
    from dendrux.runtime.replay import ReplayRequest
    from dendrux.types import compute_idempotency_fingerprint

    agent = Agent(name="pure", prompt="prompt")
    session = Mock(configured=True)
    session.manifest.return_value = {"explicit": ["skill"]}
    session.fingerprint.return_value = {"skill": "version"}
    replay = ReplayRequest(("source",), "all", False)
    metadata = {"thread": {"id": "thread"}}
    result, fingerprint = runner._new_run_metadata(
        agent,
        ReActLoop(),
        session,
        metadata,
        None,
        user_input="hello",
        idempotency_key="key",
        replay=replay,
    )
    assert result["dendrux.skills"] == {"explicit": ["skill"]}
    assert result["thread"] == {"id": "thread"}
    assert metadata == {"thread": {"id": "thread"}}
    assert "dendrux.max_delegation_depth" not in result
    assert fingerprint == compute_idempotency_fingerprint(
        "pure", "hello", replay=replay.fingerprint(), skills={"skill": "version"}
    )
    session.fingerprint.reset_mock()
    _, fingerprint = runner._new_run_metadata(
        agent, ReActLoop(), session, None, 10, user_input="hello", retry_of="source"
    )
    assert fingerprint is None
    session.fingerprint.assert_not_called()
