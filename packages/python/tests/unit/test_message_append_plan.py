"""Message decisions characterized through the recorder without a database."""

import asyncio
from copy import deepcopy

import pytest

from dendrux._meta_keys import MESSAGE_META_KEY, STEP_META_KEY
from dendrux.loops._budget import BudgetTracker
from dendrux.runtime import _cancel_guard
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import StoreCapabilities
from dendrux.types import (
    AgentStep,
    Clarification,
    Finish,
    ImageViewState,
    Message,
    Role,
    ToolCall,
    UsageStats,
    _step_to_dict,
    generate_ulid,
)
from tests._helpers.gates import wait_for_gate
from tests._helpers.lost_cancel import LostCancel

RUN_ID = generate_ulid()


class RecordingStore:
    storage_format = 2
    capabilities = StoreCapabilities(checkpointing=True, atomic_completion=True)

    def __init__(self):
        self.calls = []
        self.recorder = None
        self.fail = None

    async def checkpoint_step(self, run_id, **kwargs):
        self.calls.append(("checkpoint_step", deepcopy(kwargs), self.state()))
        if self.fail == "checkpoint_step":
            raise ValueError("checkpoint rejected")

    async def save_trace(self, run_id, role, content, **kwargs):
        self.calls.append(("save_trace", dict(role=role, content=content, **kwargs), self.state()))
        if self.fail == "save_trace":
            raise ValueError("state rejected")

    async def get_run_control(self, run_id):
        return None

    def state(self):
        r = self.recorder
        return deepcopy(
            (r._order_index, r._loop_state, r._durable_state, r._known_history, r._final_trace)
        )


def live_data(step=None):
    return dict(
        steps=[] if step is None else [step],
        tool_targets={
            "server": "server",
            "client": "client",
            "gated": "client",
            "denied": "server",
        },
        approval_tools={"gated", "denied"},
        denied_tools={"denied"},
        budget=BudgetTracker(
            None,
            run_id=generate_ulid(),
            usage=UsageStats(),
            recorder=None,
            notifier=None,
            warnings=[],
        ),
        usage=UsageStats(),
        tool_counts={},
        declared_tools=set(),
        pii_mapping=None,
        pii_counters={},
        provider_kwargs={},
        image_view=ImageViewState(),
        has_images=False,
    )


def recorder(*, atomic=True, live=None):
    store = RecordingStore()
    store.capabilities = StoreCapabilities(checkpointing=True, atomic_completion=atomic)
    r = PersistenceRecorder(store, RUN_ID, initial_order_index=10)
    store.recorder = r
    r._loop_state = {"pending": [], "phase": "model_next"}
    r._durable_state = deepcopy(r._loop_state)
    if live is not None:
        r._loop_source = lambda: live
    return r, store


def pending(call_id, order):
    return dict(
        call_id=call_id,
        tool="server",
        phase="dispatchable",
        decision_id=None,
        order_index=order,
        target="server",
    )


ROWS = [
    "consumed",
    "completed",
    "seed",
    "reserved_left",
    "reserved_last",
    "calls",
    "no_source",
    "step",
    "clarification",
    "atomic",
    "non_atomic",
    "user",
    "tool",
    "system",
]


@pytest.mark.parametrize("case", ROWS)
async def test_message_decision_table(case):
    step = AgentStep(
        "reason", Clarification("question") if case == "clarification" else Finish("answer")
    )
    live = live_data(step if case in {"step", "clarification", "calls"} else None)
    r, store = recorder(atomic=case != "non_atomic", live=None if case == "no_source" else live)
    message = Message(Role.ASSISTANT, "answer", meta={"custom": "kept"})
    expected_pending, expected_phase, expected_order, trace_order = [], "model_next", 11, 10
    if case in {"consumed", "seed", "user"}:
        message = Message(Role.USER, "input")
    if case in {"completed", "reserved_left", "reserved_last", "tool"}:
        message = Message(Role.TOOL, "result", call_id="result", name="server")
    if case == "system":
        message = Message(Role.SYSTEM, "instruction")
    if case == "consumed":
        r._consumed_user_count = 1
        r._seed_input_pending = True
        r._completed_observations.add(None)
    elif case == "completed":
        r._completed_observations.add("result")
        r._seed_input_pending = True
    elif case == "seed":
        r._seed_input_pending = True
    elif case.startswith("reserved"):
        r._loop_state["pending"] = [pending("result", 3)]
        if case == "reserved_left":
            expected_pending = [pending("left", 4)]
            r._loop_state["pending"] += expected_pending
            expected_phase = "dispatch"
        expected_order, trace_order = 10, 3
    elif case in {"calls", "no_source"}:
        calls = [
            ToolCall(name, id=name, parse_error="bad" if name == "parse" else None)
            for name in ["parse", "missing", "denied", "gated", "client", "server"]
        ]
        message = Message(Role.ASSISTANT, "calls", tool_calls=calls)
        r._loop_state["pending"] = [pending("old", 1)]
        phases = (
            ["invalid"] * 6
            if case == "no_source"
            else ["invalid", "invalid", "invalid", "gated", "client", "dispatchable"]
        )
        expected_pending = [
            dict(
                call_id=c.id,
                tool=c.name,
                phase=p,
                decision_id=None,
                order_index=11 + i,
                target=live["tool_targets"].get(c.name, "server") if case == "calls" else "server",
            )
            for i, (c, p) in enumerate(zip(calls, phases, strict=True))
        ]
        if case == "calls":
            expected_pending[4], expected_pending[5] = expected_pending[5], expected_pending[4]
            expected_pending[4]["order_index"] = 15
            expected_pending[5]["order_index"] = 16
        expected_order, expected_phase = 17, "dispatch"
    elif case == "clarification":
        expected_order, expected_phase = 12, "awaiting_input"
    elif case == "non_atomic":
        expected_phase = "finalizing"
    before = deepcopy(r._loop_state)
    await r.on_message_appended(RUN_ID, message, 7)
    if case in {"consumed", "completed", "seed"}:
        assert not store.calls
        assert r._order_index == 10 and r._loop_state == before
        assert r._consumed_user_count == 0
        assert r._seed_input_pending == (case != "seed")
        assert r._completed_observations == ({None} if case == "consumed" else set())
        return
    assert r._order_index == expected_order
    assert r._loop_state["pending"] == expected_pending
    assert r._loop_state["phase"] == expected_phase
    assert r._loop_state["trace_order_offset"] == expected_order
    assert [call[0] for call in store.calls] == (
        ["save_trace", "checkpoint_step"] if case == "clarification" else ["checkpoint_step"]
    )
    checkpoint = store.calls[-1][1]
    deferred = case in {"atomic", "step"}
    trace = r._final_trace if deferred else checkpoint["trace"]
    assert trace.order_index == trace_order
    assert trace.content == message.content and trace.role == message.role.value
    assert trace.meta[MESSAGE_META_KEY]["role"] == message.role.value
    assert "content" not in trace.meta[MESSAGE_META_KEY]
    assert trace.meta.get("custom") == message.meta.get("custom")
    assert (STEP_META_KEY in trace.meta) == (case in {"calls", "step", "clarification"})
    if STEP_META_KEY in trace.meta:
        assert trace.meta[STEP_META_KEY] == _step_to_dict(step)
    assert (checkpoint["trace"] is None) == deferred
    assert len(r._known_history) == (0 if deferred else 1)
    assert r._durable_state == checkpoint["loop_state"] == r._loop_state
    at_write = store.calls[-1][2]
    assert at_write[0] == expected_order and at_write[1] == checkpoint["loop_state"]
    assert at_write[2] != r._durable_state and at_write[3] == []
    assert (at_write[4] is not None) == deferred
    if case == "clarification":
        state = store.calls[0]
        assert state[1]["order_index"] == 11
        assert state[2][0] == 12 and state[2][1]["phase"] == "model_next"


@pytest.mark.parametrize("failure", ["checkpoint_step", "save_trace"])
async def test_failed_write_keeps_prewrite_mutations(failure):
    step = AgentStep(None, Clarification("question"))
    r, store = recorder(live=live_data(step))
    store.fail = failure
    durable = deepcopy(r._durable_state)
    with pytest.raises(ValueError):
        await r.on_message_appended(RUN_ID, Message(Role.ASSISTANT, "question"), 1)
    assert r._order_index == 12
    assert r._loop_state["phase"] == ("model_next" if failure == "save_trace" else "awaiting_input")
    assert r._durable_state == durable and not r._known_history


async def test_snapshot_effect_precedes_message_allocation():
    r, store = recorder()

    async def snapshot(iteration):
        await r._state_ref("image_view", {"v": 1})

    r._snapshot = snapshot
    await r.on_message_appended(RUN_ID, Message(Role.USER, "input"), 1)
    assert store.calls[0][1]["order_index"] == 10
    assert store.calls[1][1]["trace"].order_index == 11
    assert r._order_index == 12


@pytest.mark.parametrize("hook", ["model", "tool"])
async def test_append_lost_cancel_is_delivered_at_next_dispatch(monkeypatch, hook):
    monkeypatch.setattr(_cancel_guard, "_ENABLED", True)
    loss = LostCancel()
    store = RecordingStore()
    loss.install(monkeypatch, store, "on_message_appended", "checkpoint_step")

    async def run():
        r = PersistenceRecorder(store, RUN_ID)
        store.recorder = r
        r._loop_state = {
            "pending": [dict(pending("client", 1), target="client")],
            "phase": "dispatch",
        }
        await r.on_message_appended(RUN_ID, Message(Role.USER, "input"), 1)
        assert loss.dropped and len(store.calls) == 1
        assert asyncio.current_task().cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            if hook == "model":
                await r.on_llm_call_started(RUN_ID, 2)
            else:
                await r.on_tool_started(RUN_ID, ToolCall("client", id="client"), 2)

    task = asyncio.create_task(run())
    await wait_for_gate(loss.entered, task=task)
    task.cancel()
    await task


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("scope", [None, "tenant"])
async def test_image_scope_and_legacy_metadata(fmt, scope):
    from dendrux._meta_keys import PARTS_META_KEY
    from dendrux.errors import ImagePersistenceUnsupportedError
    from dendrux.types import ImagePart, ImageRef, TextPart

    r, store = recorder()
    r._format2 = fmt == 2
    r._tenant_scope = scope
    ref = ImageRef("a" * 64, "image/png", 10, 1, 1)
    message = Message.with_parts(
        Role.TOOL,
        [TextPart("result"), ImagePart(ref)],
        name="server",
        call_id="result",
        meta={"custom": "kept"},
    )
    if fmt == 1 and scope is None:
        with pytest.raises(ImagePersistenceUnsupportedError):
            await r.on_message_appended(RUN_ID, message, 7)
        assert store.calls == [] and r._order_index == 10
        return
    await r.on_message_appended(RUN_ID, message, 7)
    saved = store.calls[0][1]
    if fmt == 2:
        trace = saved["trace"]
        assert bool(trace.blob_refs) == bool(scope)
        assert trace.blob_refs_scope == scope
        assert trace.meta[PARTS_META_KEY] == [{"type": "text", "text": "result"}, ref.to_dict()]
    else:
        assert saved["meta"]["iteration"] == 7
        assert saved["meta"]["call_id"] == "result" and saved["meta"]["tool_name"] == "server"
        assert saved["blob_refs_scope"] == scope
        assert store.calls[0][2][0] == 10
        assert r._order_index == 11


@pytest.mark.parametrize("existing_parts", [False, True])
async def test_legacy_calls_reasoning_parts_and_retry_identity(existing_parts):
    from dendrux._meta_keys import PARTS_META_KEY
    from dendrux.types import ImagePart, ImageRef, TextPart

    r, store = recorder()
    r._format2 = False
    store.capabilities = StoreCapabilities(retry_ids=True)
    wire = [{"type": "text", "text": "projected"}]
    ref = ImageRef("a" * 64, "image/png", 10, 1, 1)
    r._tenant_scope = "tenant"
    message = Message.with_parts(
        Role.USER,
        [TextPart("raw"), ImagePart(ref)],
        reasoning_blocks=[{"type": "thinking", "thinking": "reason"}],
        meta={PARTS_META_KEY: wire} if existing_parts else {},
    )
    await r.on_message_appended(RUN_ID, message, 3)
    saved = store.calls[0][1]
    assert saved["trace_id"]
    assert saved["meta"]["iteration"] == 3
    assert saved["meta"]["reasoning_blocks"] == message.reasoning_blocks
    assert saved["meta"][PARTS_META_KEY] == (
        wire if existing_parts else [{"type": "text", "text": "raw"}, ref.to_dict()]
    )

    assistant = Message(Role.ASSISTANT, "calls", tool_calls=[ToolCall("server", id="call")])
    await r.on_message_appended(RUN_ID, assistant, 4)
    assert store.calls[1][1]["meta"]["tool_calls"][0]["id"] == "call"


async def test_loop_source_is_read_after_snapshot_effect_and_before_checkpoint():
    r, store = recorder()
    events = []
    live = live_data()

    def source():
        events.append("source")
        return live

    async def snapshot(iteration):
        events.append("snapshot")
        await r._state_ref("image_view", {"v": 1})
        live["steps"] = [AgentStep(None, Clarification("after snapshot"))]
        events.append("snapshot_done")

    r._snapshot = snapshot
    r._loop_source = source
    await r.on_message_appended(RUN_ID, Message(Role.ASSISTANT, "question"), 1)
    assert events[:2] == ["snapshot", "snapshot_done"]
    assert 2 <= events.count("source") <= 3
    assert store.calls[1][1]["order_index"] == 12
    assert '"question":"after snapshot"' in store.calls[1][1]["content"]
    assert store.calls[2][1]["trace"].order_index == 11


def test_observations_follow_durable_slots_after_reconstruction():
    from dendrux.runtime.state import TraceRecord

    r, _ = recorder()
    for call_id, order in (("a", 10), ("b", 11), ("c", 12)):
        r._remember(
            TraceRecord(
                id=call_id,
                role="tool",
                content=call_id,
                order_index=order,
                meta={"call_id": call_id},
            )
        )
    assistant = Message(Role.ASSISTANT, "calls")
    history = [
        assistant,
        *[Message(Role.TOOL, call, call_id=call, name="tool") for call in ("b", "c", "a")],
    ]
    r.order_observations(history)
    assert history[0] is assistant
    assert [m.call_id for m in history[1:]] == ["a", "b", "c"]
