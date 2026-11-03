"""Resume and reclaim deliberately retain distinct decode sources and failures."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux import Agent
from dendrux.errors import PersistenceError, RunNotReclaimableError
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _reclaim, replay, runner
from dendrux.skills import _session
from dendrux.types import (
    ImageViewState,
    InputValidationError,
    Message,
    PauseState,
    Role,
    UsageStats,
)


def decode_case():
    agent = Agent(name="decode", prompt="p", provider=MockLLM([]), strict_tenancy=False)
    pause = PauseState(
        "decode",
        [],
        [],
        [],
        0,
        0,
        UsageStats(),
        provider_kwargs={"model": "blob-model"},
        schema_version=2,
    )
    state = dict(
        usage={"input_tokens": 7},
        iteration=3,
        trace_order_offset=12,
        provider_kwargs={"model": "checkpoint-model"},
        skills={},
    )
    store = SimpleNamespace(
        get_run=AsyncMock(return_value=SimpleNamespace(storage_format=2)),
        get_loop_state=AsyncMock(return_value=state),
        get_pause_state=AsyncMock(return_value=pause.to_dict()),
        get_trace=AsyncMock(),
        get_pii_mapping=AsyncMock(return_value={}),
    )
    rebuilt = replay.RebuiltHistory(
        [Message(Role.SYSTEM, "system"), Message(Role.USER, "user")], [], []
    )
    return agent, pause, state, store, rebuilt


@pytest.mark.parametrize(
    "branch", ["wrong_format", "missing_record", "empty_checkpoint", "missing_ref"]
)
async def test_resume_unsupported_format_logging(monkeypatch, caplog, branch):
    agent, pause, state, store, rebuilt = decode_case()
    monkeypatch.setattr(replay, "rebuild_history", AsyncMock(return_value=rebuilt))
    if branch == "wrong_format":
        store.get_run.return_value.storage_format = 1
    elif branch == "missing_record":
        store.get_run.return_value = None
    elif branch == "empty_checkpoint":
        store.get_loop_state.return_value = {}
    else:
        state["image_view_ref"] = "missing"
        store.get_trace.return_value = None
    with pytest.raises(PersistenceError) as caught:
        await runner._load_pause_state(
            store, "01J00000000000000000000000", pause.to_dict(), agent, agent.provider
        )
    assert caught.value.code == "pause_format_unsupported"
    codes = [getattr(r, "code", None) for r in caplog.records]
    assert codes.count("pause_format_unsupported") == (0 if branch == "missing_ref" else 1)


@pytest.mark.parametrize("entry", ["resume", "reclaim"])
async def test_history_limit_vocabulary(monkeypatch, entry):
    agent, pause, state, store, _ = decode_case()
    original = RunNotReclaimableError("01J00000000000000000000000", "history_too_large")
    monkeypatch.setattr(replay, "rebuild_history", AsyncMock(side_effect=original))
    if entry == "resume":
        with pytest.raises(InputValidationError) as caught:
            await runner._load_pause_state(
                store, "01J00000000000000000000000", pause.to_dict(), agent, agent.provider
            )
        assert caught.value.field == "history"
        assert caught.value.__cause__ is original
        assert "history_too_large" in str(caught.value)
    else:
        with pytest.raises(RunNotReclaimableError) as caught:
            await _reclaim.rebuild_pause(agent, store, "01J00000000000000000000000", state)
        assert caught.value is original


@pytest.mark.parametrize("entry", ["resume", "reclaim"])
@pytest.mark.parametrize("session_present", [False, True])
async def test_checkpoint_fields_refs_and_reasoning_source(monkeypatch, entry, session_present):
    agent, pause, state, store, rebuilt = decode_case()
    rebuild = AsyncMock(return_value=rebuilt)
    monkeypatch.setattr(replay, "rebuild_history", rebuild)
    view = ImageViewState(epoch=4, sent_through=7, warned=["prior"])
    snapshot = {"pins": {"skill": "digest"}}
    state.update(
        image_view_ref="image", skills={"session_ref": "session"}, clarification_ref="clarification"
    )
    payloads = {
        "image": view.to_dict(),
        "session": snapshot,
        "clarification": {"pause_message": "clarify"},
    }
    store.get_trace.side_effect = lambda run, ref: SimpleNamespace(
        content=json.dumps(payloads[ref])
    )
    session = Mock() if session_present else None
    monkeypatch.setattr(runner, "current_skill_session", lambda agent: session)
    monkeypatch.setattr(_session, "current", lambda agent: session)
    if entry == "resume":
        reads = runner._PauseReads(store.get_run.return_value)
        actual = await runner._load_pause_state(
            store, "01J00000000000000000000000", pause.to_dict(), agent, agent.provider, reads=reads
        )
        assert reads.history_records is rebuilt.records and reads.checkpoint is state
    else:
        actual = await _reclaim.rebuild_pause(agent, store, "01J00000000000000000000000", state)
    assert rebuild.await_args.kwargs["reasoning_for"] == (
        "blob-model" if entry == "resume" else "checkpoint-model"
    )
    assert actual.history == [rebuilt.messages[1]]
    assert actual.steps is rebuilt.steps
    assert actual.iteration == 3 and actual.trace_order_offset == 12
    assert actual.usage.input_tokens == 7
    assert actual.provider_kwargs == {"model": "checkpoint-model"}
    assert actual.image_view == view and actual.pause_message == "clarify"
    assert [call.args for call in store.get_trace.await_args_list] == [
        ("01J00000000000000000000000", "image"),
        ("01J00000000000000000000000", "session"),
        ("01J00000000000000000000000", "clarification"),
    ]
    if session is not None:
        session.restore_snapshot.assert_called_once_with(snapshot)


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
@pytest.mark.parametrize("problem", ["missing", "agent", "pause_id"])
async def test_decode_error_text_and_read_order(monkeypatch, entry, problem):
    agent, pause, _, store, _ = decode_case()
    pause.schema_version = 1
    pause.agent_name = "other" if problem == "agent" else agent.name
    store.get_pause_state.return_value = None if problem == "missing" else pause.to_dict()
    calls = []
    for name in ("get_run", "get_pause_state", "get_pii_mapping"):
        method = getattr(store, name)
        value = method.return_value

        async def read(*args, _name=name, _value=value):
            calls.append(_name)
            return _value

        method.side_effect = read
    monkeypatch.setattr(runner, "continuation", AsyncMock(return_value=Mock()))

    def require(*args, **kwargs):
        calls.append("require_pause_id")
        if problem == "pause_id":
            raise ValueError("pause sentinel")

    monkeypatch.setattr(runner, "require_pause_id", require)
    with pytest.raises(ValueError) as caught:
        if entry == "resume":
            await runner._read_resume_state(
                store, agent, agent.provider, "01J00000000000000000000000", False, None, None, None
            )
        else:
            await runner._stream_resume_state(
                agent,
                agent.provider,
                store,
                "01J00000000000000000000000",
                None,
                None,
                False,
                Mock(),
            )
    if problem == "missing":
        assert (
            str(caught.value)
            == "Run '01J00000000000000000000000' has no pause state — cannot resume."
        )
    elif problem == "pause_id":
        assert str(caught.value) == "pause sentinel"
    else:
        expected = (
            "Agent name mismatch: run '01J00000000000000000000000' was paused by "
            "'other', but resume called with 'decode'."
        )
        assert str(caught.value) == expected + (
            " This can cause state corruption." if entry == "resume" else ""
        )
    expected_calls = ["get_run", "get_pause_state"] if entry == "resume" else ["get_pause_state"]
    if problem != "missing":
        expected_calls += ["require_pause_id"]
    if problem == "agent":
        expected_calls += ["get_pii_mapping"] if entry == "resume" else ["get_run"]
    assert calls == expected_calls


@pytest.mark.parametrize("kind", ["image_view", "skills_session", "clarification"])
async def test_reclaim_missing_ref_vocabulary(monkeypatch, kind):
    agent, _, state, store, rebuilt = decode_case()
    monkeypatch.setattr(replay, "rebuild_history", AsyncMock(return_value=rebuilt))
    if kind == "skills_session":
        state["skills"] = {"session_ref": "missing"}
    else:
        state[f"{kind}_ref"] = "missing"
    store.get_trace.return_value = None
    with pytest.raises(PersistenceError) as caught:
        await _reclaim.rebuild_pause(agent, store, "01J00000000000000000000000", state)
    assert caught.value.code == "recovery_state_missing"


@pytest.mark.parametrize("entry", ["resume", "reclaim"])
async def test_missing_checkpoint_counters_keep_path_defaults(monkeypatch, entry):
    agent, pause, _, store, rebuilt = decode_case()
    state = {"skills": {}}
    store.get_loop_state.return_value = state
    monkeypatch.setattr(replay, "rebuild_history", AsyncMock(return_value=rebuilt))
    if entry == "resume":
        with pytest.raises(KeyError) as caught:
            await runner._load_pause_state(
                store, "01J00000000000000000000000", pause.to_dict(), agent, agent.provider
            )
        assert caught.value.args == ("usage",)
    else:
        actual = await _reclaim.rebuild_pause(agent, store, "01J00000000000000000000000", state)
        assert actual.iteration == actual.trace_order_offset == 0
        assert actual.usage == UsageStats() and actual.provider_kwargs == {}
