"""Strict, bounded fan-out persistence and byte-identical single-child storage."""

import copy
import json

import pytest

from dendrux import types
from dendrux.errors import ManagedStateError, PersistenceError
from dendrux.types import ManagedWait, PauseState, ToolCall, UsageStats

PAUSE_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
CHILD_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
SENTINEL = "SENTINEL-brief-secret"
# Captured from the unchanged feat/0.8-t1 baseline (2f9513ca).
V3_GOLDEN = (
    b'{"agent_name":"lead","pending_tool_calls":[{"name":"worker","params":{},"id":"c0",'
    b'"provider_tool_call_id":"p0"}],"pending_targets":{"c0":"agent"},"iteration":2,'
    b'"trace_order_offset":5,"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,'
    b'"cost_usd":null,"cache_read_input_tokens":null,"cache_creation_input_tokens":null,'
    b'"cost_source":null,"reasoning_tokens":null,"usage_reported":true,"cost_unknown":false},'
    b'"pause_reason":"managed_wait","pause_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","schema_version":3,'
    b'"provider_kwargs":{"model":"m","custom_flag":true},"image_view_ref":null,'
    b'"loop_state_ref":true,"managed_wait":{"v":1,"tool_call_id":"c0",'
    b'"child_run_id":"01ARZ3NDEKTSV4RRFFQ69G5FAW"}}'
)


def encoded(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


def v4_data(n=2):
    data = json.loads(V3_GOLDEN)
    data.pop("managed_wait")
    data["schema_version"] = 4
    data["pending_tool_calls"] = [
        {"name": "worker", "params": {}, "id": f"c{i}", "provider_tool_call_id": None}
        for i in range(n)
    ]
    data["pending_targets"] = {f"c{i}": "agent" for i in range(n)}
    data["managed_fanout"] = {
        "v": 1,
        "n": n,
        "bindings": [{"tool_call_id": f"c{i}", "child_run_id": f"{i:026d}"} for i in range(n)],
    }
    return data


def test_v3_bytes_match_prechange_golden():
    pause = PauseState(
        agent_name="lead",
        pending_tool_calls=[
            ToolCall(name="worker", params={"task": SENTINEL}, id="c0", provider_tool_call_id="p0")
        ],
        history=[],
        steps=[],
        iteration=2,
        trace_order_offset=5,
        usage=UsageStats(input_tokens=10, output_tokens=4, total_tokens=14),
        pending_targets={"c0": "agent"},
        pause_reason="managed_wait",
        pause_id=PAUSE_ID,
        schema_version=3,
        provider_kwargs={"model": "m", "custom_flag": True},
        loop_state_ref=True,
        managed_wait=ManagedWait("c0", CHILD_ID),
    )
    assert encoded(pause.to_dict()) == V3_GOLDEN
    assert encoded(PauseState.from_managed_dict(json.loads(V3_GOLDEN)).to_dict()) == V3_GOLDEN


@pytest.mark.parametrize("n", [2, 4, 30, 32])
def test_v4_roundtrip_preserves_order_and_references(n):
    data = v4_data(n)
    pause = PauseState.from_managed_dict(data)
    assert pause.managed_wait is None
    assert pause.managed_fanout.to_dict() == data["managed_fanout"]
    assert pause.history == [] and pause.steps == [] and pause.loop_state_ref
    assert pause.to_dict() == data
    assert len(encoded(data)) <= 32 * 1024
    with pytest.raises(PersistenceError):
        PauseState.from_dict(data)


def test_v4_encoder_strips_briefs_and_binding_wins_over_checkpoint_version():
    fanout_type = getattr(types, "ManagedFanout", None)
    assert fanout_type is not None
    pause = PauseState.from_managed_dict(json.loads(V3_GOLDEN))
    pause.managed_wait = None
    bindings = tuple(ManagedWait(f"c{i}", f"{i:026d}") for i in range(2))
    pause.managed_fanout = fanout_type(bindings)
    pause.pending_tool_calls = [
        ToolCall(name="worker", params={"task": SENTINEL}, id=b.tool_call_id) for b in bindings
    ]
    pause.pending_targets = {b.tool_call_id: "agent" for b in bindings}
    pause.schema_version = 2
    data = pause.to_dict()
    assert data["schema_version"] == 4
    assert "managed_wait" not in data
    assert SENTINEL not in encoded(data).decode()
    assert PauseState.from_managed_dict(data).managed_fanout == pause.managed_fanout


def mutations():
    return {
        "version": lambda d: d.update(schema_version=3),
        "version_float": lambda d: d.update(schema_version=4.0),
        "reason": lambda d: d.update(pause_reason="budget"),
        "missing_envelope": lambda d: d.pop("managed_fanout"),
        "envelope_type": lambda d: d.update(managed_fanout=[]),
        "envelope_extra": lambda d: d["managed_fanout"].update(extra=1),
        "envelope_version": lambda d: d["managed_fanout"].update(v=True),
        "envelope_version_float": lambda d: d["managed_fanout"].update(v=1.0),
        "n_bool": lambda d: d["managed_fanout"].update(n=True),
        "n_float": lambda d: d["managed_fanout"].update(n=2.0),
        "n_mismatch": lambda d: d["managed_fanout"].update(n=3),
        "bindings_type": lambda d: d["managed_fanout"].update(bindings={}),
        "bindings_empty": lambda d: d["managed_fanout"].update(bindings=[]),
        "binding_type": lambda d: d["managed_fanout"]["bindings"].__setitem__(0, None),
        "binding_extra": lambda d: d["managed_fanout"]["bindings"][0].update(v=1),
        "duplicate_child": lambda d: d["managed_fanout"]["bindings"][1].update(
            child_run_id="0" * 26
        ),
        "child_invalid": lambda d: d["managed_fanout"]["bindings"][0].update(child_run_id="x"),
        "call_invalid": lambda d: d["managed_fanout"]["bindings"][0].update(tool_call_id=[]),
        "binding_order": lambda d: d["managed_fanout"]["bindings"].reverse(),
        "duplicate_call": lambda d: d["managed_fanout"]["bindings"][1].update(tool_call_id="c0"),
        "calls_empty": lambda d: d.update(pending_tool_calls=[]),
        "calls_type": lambda d: d.update(pending_tool_calls={}),
        "call_type": lambda d: d["pending_tool_calls"].__setitem__(0, None),
        "call_name": lambda d: d["pending_tool_calls"][0].update(name=3),
        "call_params": lambda d: d["pending_tool_calls"][0].update(params={"task": SENTINEL}),
        "provider_id": lambda d: d["pending_tool_calls"][0].update(provider_tool_call_id=3),
        "target": lambda d: d["pending_targets"].update(c0="client"),
        "extra_target": lambda d: d["pending_targets"].update(other="agent"),
        "v3_binding": lambda d: d.update(managed_wait={"v": 1}),
        "history": lambda d: d.update(history=[]),
        "steps": lambda d: d.update(steps=[]),
        "image_view": lambda d: d.update(image_view=None),
        "gates": lambda d: d.update(approval_gates={}),
        "decisions": lambda d: d.update(pending_decisions={}),
        "clarification": lambda d: d.update(clarification={}),
        "refs": lambda d: d.update(loop_state_ref=False),
        "pause_id": lambda d: d.update(pause_id="x"),
        "agent": lambda d: d.update(agent_name=None),
        "iteration": lambda d: d.update(iteration=True),
        "offset": lambda d: d.update(trace_order_offset=-1),
        "usage": lambda d: d.update(usage={"cost_usd": float("nan")}),
        "kwargs": lambda d: d.update(provider_kwargs=[]),
        "image_ref": lambda d: d.update(image_view_ref=3),
        "budget": lambda d: d.update(budget_known_cost_usd=-1),
        "pause_message": lambda d: d.update(pause_message=3),
        "oversize": lambda d: d.update(pause_message=SENTINEL * 4096),
        "non_json": lambda d: d.update(unknown=object()),
        "non_utf8": lambda d: d.update(pause_message="\ud800"),
    }


@pytest.mark.parametrize("case", sorted(mutations()))
def test_v4_decode_fails_closed_safely(case, caplog):
    data = v4_data()
    mutations()[case](data)
    with pytest.raises(ManagedStateError) as exc:
        PauseState.from_managed_dict(data, run_id=PAUSE_ID)
    assert exc.value.reason == "wait_invalid"
    assert exc.value.run_id == PAUSE_ID
    assert SENTINEL not in str(exc.value) + repr(exc.value.__dict__) + caplog.text


@pytest.mark.parametrize("n", [0, 1, 33])
def test_v4_refuses_outside_binding_count_bound(n):
    with pytest.raises(ManagedStateError) as exc:
        PauseState.from_managed_dict(v4_data(n))
    assert exc.value.reason == "wait_invalid"


@pytest.mark.parametrize(
    "field,value",
    [
        ("pending_targets", {}),
        ("pause_id", None),
        ("approval_gates", {}),
        ("pending_decisions", {}),
        ("clarification", {}),
        ("loop_state_ref", False),
        ("pause_reason", "requested"),
        ("pause_message", SENTINEL * 4096),
        ("iteration", -1),
        ("budget_known_cost_usd", float("inf")),
    ],
)
def test_v4_encoder_applies_decoder_validation(field, value):
    pause = PauseState.from_managed_dict(v4_data())
    setattr(pause, field, value)
    with pytest.raises(ValueError):
        pause.to_dict()


def test_v4_size_boundary_is_utf8_bytes_and_encoder_agrees():
    data = v4_data()
    data["pause_message"] = ""
    padding = 32768 - len(encoded(data))
    data["pause_message"] = "é" * (padding // 2) + "x" * (padding % 2)
    assert len(encoded(data)) == 32768
    pause = PauseState.from_managed_dict(data)
    assert pause.to_dict() == data
    assert len(encoded(pause.to_dict())) == 32768
    data["pause_message"] += "x"
    with pytest.raises(ManagedStateError) as exc:
        PauseState.from_managed_dict(data)
    assert exc.value.reason == "wait_invalid"
    pause.pause_message += "x"
    with pytest.raises(ValueError):
        pause.to_dict()


def test_v4_decode_owns_its_bindings():
    data = v4_data()
    expected = copy.deepcopy(data["managed_fanout"])
    pause = PauseState.from_managed_dict(data)
    data["managed_fanout"]["bindings"][0]["child_run_id"] = CHILD_ID
    assert pause.managed_fanout.to_dict() == expected
    with pytest.raises(AttributeError):
        pause.managed_fanout.bindings = ()


@pytest.mark.parametrize(
    "case",
    [
        "single",
        "too_many",
        "duplicate_call",
        "duplicate_child",
        "list",
        "item",
        "version_bool",
        "version_float",
    ],
)
def test_fanout_record_is_bounded_and_immutable_at_construction(case):
    fanout_type = getattr(types, "ManagedFanout", None)
    assert fanout_type is not None
    bindings = (ManagedWait("c0", "0" * 26), ManagedWait("c1", "1" * 26))
    version = 1
    if case == "single":
        bindings = bindings[:1]
    elif case == "too_many":
        bindings = tuple(ManagedWait(f"c{i}", f"{i:026d}") for i in range(33))
    elif case == "duplicate_call":
        bindings = (bindings[0], ManagedWait("c0", "1" * 26))
    elif case == "duplicate_child":
        bindings = (bindings[0], ManagedWait("c1", "0" * 26))
    elif case == "list":
        bindings = list(bindings)
    elif case == "item":
        bindings = (None, None)
    elif case == "version_bool":
        version = True
    elif case == "version_float":
        version = 1.0
    with pytest.raises(ValueError):
        fanout_type(bindings, version=version)


def test_v4_encoder_refuses_missing_or_conflicting_bindings():
    pause = PauseState.from_managed_dict(v4_data())
    pause.managed_wait = ManagedWait("c0", CHILD_ID)
    with pytest.raises(ValueError):
        pause.to_dict()
    pause.managed_wait = None
    pause.managed_fanout = None
    with pytest.raises(ValueError):
        pause.to_dict()


@pytest.mark.parametrize("field", ["approval_gates", "pending_decisions", "clarification"])
def test_v4_none_fields_allow_explicit_none(field):
    data = v4_data()
    data[field] = None
    pause = PauseState.from_managed_dict(data)
    assert getattr(pause, field) is None


@pytest.mark.parametrize("value", [None, [], "x", 4])
def test_managed_decode_refuses_non_object(value):
    with pytest.raises(ManagedStateError) as exc:
        PauseState.from_managed_dict(value)
    assert exc.value.reason == "wait_invalid"


def test_v4_checkpoint_record_preserves_binding_order_and_generation():
    pause = PauseState.from_managed_dict(v4_data())
    checkpoint = pause.managed_fanout.to_checkpoint(pause_id=PAUSE_ID)
    assert checkpoint == {
        "v": 1,
        "pause_id": PAUSE_ID,
        "bindings": {"c0": "0" * 26, "c1": "0" * 25 + "1"},
    }
    assert list(checkpoint["bindings"]) == ["c0", "c1"]
    checkpoint["bindings"]["c0"] = CHILD_ID
    assert pause.managed_fanout.bindings[0].child_run_id == "0" * 26


@pytest.mark.parametrize("pause_id", [None, "x", 4, True])
def test_v4_checkpoint_refuses_invalid_generation(pause_id):
    pause = PauseState.from_managed_dict(v4_data())
    with pytest.raises(ValueError):
        pause.managed_fanout.to_checkpoint(pause_id=pause_id)
