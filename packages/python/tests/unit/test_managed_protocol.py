"""Managed-subagent protocol (0.7, T2a): the pause v3 managed-wait codec.

A parent waiting on a managed child is saved as pause ``schema_version=3`` with
``pause_reason="managed_wait"``: storage format 2 references plus one outgoing
binding (the pending call and its child run). Old decoders must refuse it; the
strict managed decoder must accept only a complete, bounded wait.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dendrux.errors import ManagedStateError, PersistenceError
from dendrux.types import (
    ManagedWait,
    Message,
    PauseState,
    Role,
    ToolCall,
    UsageStats,
    generate_ulid,
)

BASELINE = "8b5b0b87a9f906c27ceb394fd96364b460ff817a"
SENTINEL = "SENTINEL-brief-text-7f3a"


def managed_wait_pause(
    *, call_id: str | None = None, child_run_id: str | None = None, **overrides
) -> PauseState:
    call_id = call_id or generate_ulid()
    child_run_id = child_run_id or generate_ulid()
    fields = dict(
        agent_name="lead",
        pending_tool_calls=[
            ToolCall(
                name="reviewer", params={"task": SENTINEL}, id=call_id, provider_tool_call_id="p1"
            )
        ],
        history=[],
        steps=[],
        iteration=2,
        trace_order_offset=5,
        usage=UsageStats(input_tokens=10, output_tokens=4, total_tokens=14),
        pending_targets={call_id: "agent"},
        pause_reason="managed_wait",
        pause_id=generate_ulid(),
        schema_version=3,
        provider_kwargs={"model": "m", "custom_flag": True},
        image_view_ref=None,
        loop_state_ref=True,
        managed_wait=ManagedWait(tool_call_id=call_id, child_run_id=child_run_id),
    )
    fields.update(overrides)
    return PauseState(**fields)


def _baseline_types():
    """``dendrux.types`` as shipped at the spec baseline, or None without git history."""
    root = Path(__file__).resolve().parents[4]
    try:
        source = subprocess.run(
            ["git", "-C", str(root), "show", f"{BASELINE}:packages/python/src/dendrux/types.py"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    name = "dendrux._baseline_types_for_test"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, "baseline_types.py", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def test_v3_managed_wait_roundtrip_and_old_decoder_refuses():
    pause = managed_wait_pause()
    stored = pause.to_dict()

    # The saved wait is format-2 references plus the binding, never history.
    assert stored["schema_version"] == 3
    assert stored["pause_reason"] == "managed_wait"
    assert stored["managed_wait"] == {
        "v": 1,
        "tool_call_id": pause.managed_wait.tool_call_id,
        "child_run_id": pause.managed_wait.child_run_id,
    }
    assert "history" not in stored and "steps" not in stored
    assert stored["loop_state_ref"] is True
    # The brief lives on the tool intent row; the wait keeps a reference only.
    assert stored["pending_tool_calls"][0]["params"] == {}
    assert SENTINEL not in json.dumps(stored)

    # The generic (old-format) decoder refuses v3, as main's decoder does.
    with pytest.raises(PersistenceError) as refused:
        PauseState.from_dict(stored)
    assert refused.value.code == "pause_format_unsupported"
    baseline = _baseline_types()
    if baseline is not None:
        with pytest.raises(Exception) as old:
            baseline.PauseState.from_dict(stored)
        assert getattr(old.value, "code", None) == "pause_format_unsupported"

    # The strict managed decoder preserves the binding and compact references.
    decoded = PauseState.from_managed_dict(stored)
    assert decoded.schema_version == 3
    assert decoded.managed_wait == pause.managed_wait
    assert decoded.pause_id == pause.pause_id
    assert [tc.id for tc in decoded.pending_tool_calls] == [pause.managed_wait.tool_call_id]
    assert decoded.pending_targets == {pause.managed_wait.tool_call_id: "agent"}
    assert decoded.provider_kwargs == {"model": "m", "custom_flag": True}
    assert decoded.history == [] and decoded.loop_state_ref is True
    assert decoded.to_dict() == stored

    # The recorder stamps schema_version=2 on pauses it checkpoints: the binding wins.
    decoded.schema_version = 2
    again = decoded.to_dict()
    assert again["schema_version"] == 3 and again["managed_wait"] == stored["managed_wait"]
    decoded.schema_version = 1
    assert decoded.to_dict() == stored


def test_v1_and_v2_round_trips_are_unchanged():
    call = ToolCall(name="lookup", params={"q": "x"}, id="c1")
    v1 = PauseState(
        agent_name="a",
        pending_tool_calls=[call],
        history=[Message(role=Role.USER, content="hi")],
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
        pending_targets={"c1": "client"},
        provider_kwargs={"model": "m", "custom": 1},
    )
    d1 = v1.to_dict()
    assert d1["schema_version"] == 1 and "history" in d1 and "managed_wait" not in d1
    assert d1["provider_kwargs"] == {"model": "m"}
    back1 = PauseState.from_dict(d1)
    assert back1.to_dict() == d1 and back1.managed_wait is None

    v2 = PauseState(
        agent_name="a",
        pending_tool_calls=[call],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
        pending_targets={"c1": "client"},
        schema_version=2,
        pause_id=generate_ulid(),
        provider_kwargs={"model": "m", "custom": 1},
        loop_state_ref=True,
    )
    d2 = v2.to_dict()
    assert d2["schema_version"] == 2 and d2["provider_kwargs"] == {"model": "m", "custom": 1}
    assert d2["pending_tool_calls"][0]["params"] == {"q": "x"}
    assert "managed_wait" not in d2
    back2 = PauseState.from_dict(d2)
    assert back2.to_dict() == d2 and back2.managed_wait is None
    # The strict decoder accepts only managed waits.
    with pytest.raises(ManagedStateError):
        PauseState.from_managed_dict(d2)


def test_v3_without_binding_is_never_serialized():
    pause = managed_wait_pause()
    pause.managed_wait = None
    with pytest.raises(ValueError):
        pause.to_dict()


@pytest.mark.parametrize(
    "overrides",
    [
        {"pause_reason": "requested"},
        {"pause_id": None},
        {"pending_targets": {}},
        {"approval_gates": {"x": {"source": "agent", "reason": None}}},
        {"pending_decisions": {"x": {"approved": True}}},
        {"clarification": {"question": "q", "options": []}},
        {"loop_state_ref": False},
    ],
    ids=["reason", "pause_id", "targets", "gates", "decisions", "clarification", "refs"],
)
def test_encoder_refuses_an_incomplete_managed_wait(overrides):
    with pytest.raises(ValueError):
        managed_wait_pause(**overrides).to_dict()


def test_encoder_refuses_a_second_pending_call_or_mismatched_binding():
    pause = managed_wait_pause()
    pause.pending_tool_calls.append(ToolCall(name="other", params={}, id="c2"))
    pause.pending_targets["c2"] = "server"
    with pytest.raises(ValueError):
        pause.to_dict()
    other = managed_wait_pause(managed_wait=ManagedWait("not-the-call", generate_ulid()))
    with pytest.raises(ValueError):
        other.to_dict()


def _mutations():
    def drop(key):
        return lambda d: d.pop(key)

    def set_(key, value):
        return lambda d: d.__setitem__(key, value)

    def wait(key, value):
        return lambda d: d["managed_wait"].__setitem__(key, value)

    def call(key, value):
        return lambda d: d["pending_tool_calls"][0].__setitem__(key, value)

    return {
        "no_binding": drop("managed_wait"),
        "binding_not_object": set_("managed_wait", ["x"]),
        "binding_version": wait("v", 2),
        "binding_child_not_ulid": wait("child_run_id", "not-a-ulid"),
        "binding_call_missing": wait("tool_call_id", None),
        "binding_extra_field": wait("extra", 1),
        "wrong_version": set_("schema_version", 2),
        "wrong_reason": set_("pause_reason", "budget"),
        "pause_id_invalid": set_("pause_id", "x"),
        "two_calls": lambda d: d["pending_tool_calls"].append(dict(d["pending_tool_calls"][0])),
        "no_calls": set_("pending_tool_calls", []),
        "call_mismatch": call("id", "other"),
        "params_copied": call("params", {"task": SENTINEL}),
        "target_not_agent": lambda d: d["pending_targets"].update(
            {d["managed_wait"]["tool_call_id"]: "client"}
        ),
        "extra_target": lambda d: d["pending_targets"].update({"other": "server"}),
        "history_copied": set_("history", [{"role": "user", "content": SENTINEL}]),
        "steps_copied": set_("steps", []),
        "refs_missing": set_("loop_state_ref", False),
        "gates": set_("approval_gates", {}),
        "decisions": set_("pending_decisions", {}),
        "clarification": set_("clarification", {"question": SENTINEL}),
        "agent_name_missing": drop("agent_name"),
        "oversize": set_("pause_message", SENTINEL * 4096),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_strict_decoder_refuses_incomplete_or_unbounded_waits(name, caplog):
    stored = managed_wait_pause().to_dict()
    _mutations()[name](stored)
    with pytest.raises(ManagedStateError) as refused:
        PauseState.from_managed_dict(stored)
    error = refused.value
    assert isinstance(error, PersistenceError)
    assert error.code == "managed_state_invalid"
    # Safe diagnostics: stable codes only, never stored text.
    assert SENTINEL not in str(error) and SENTINEL not in repr(error.__dict__)
    assert SENTINEL not in caplog.text
    assert any(getattr(r, "code", None) == "managed_state_invalid" for r in caplog.records)


def test_strict_decoder_refuses_non_objects():
    for value in (None, [], "x", 3):
        with pytest.raises(ManagedStateError):
            PauseState.from_managed_dict(value)  # type: ignore[arg-type]


def test_managed_wait_binding_is_immutable_and_validated():
    binding = ManagedWait(tool_call_id="c1", child_run_id=generate_ulid())
    with pytest.raises(AttributeError):
        binding.child_run_id = generate_ulid()  # type: ignore[misc]
    with pytest.raises(ValueError):
        ManagedWait(tool_call_id="c1", child_run_id="not-a-ulid")
    with pytest.raises(ValueError):
        ManagedWait(tool_call_id="", child_run_id=generate_ulid())


# ---------------------------------------------------------------------------
# Store protocol and the managed capability (T2a)
# ---------------------------------------------------------------------------


def _bind_request(**overrides):
    from dendrux.runtime.state import BindManagedChildRequest, EventWrite, LoopStatePatch

    root = overrides.pop("root_run_id", generate_ulid())
    call_id = overrides.pop("tool_call_id", generate_ulid())
    child = overrides.pop("child_run_id", generate_ulid())
    fields = dict(
        tenant_id="t1",
        root_run_id=root,
        parent_run_id=root,
        parent_claim_epoch=0,
        tool_call_id=call_id,
        row_id=generate_ulid(),
        provider_tool_call_id="p1",
        tool_name="reviewer",
        params={"task": "review"},
        iteration_index=1,
        child_run_id=child,
        child_agent_name="reviewer",
        child_path=("reviewer",),
        child_input_data={"input": "review"},
        child_loop_state={"v": 1, "loop": "react"},
        parent_wait=managed_wait_pause(call_id=call_id, child_run_id=child),
        parent_patch=LoopStatePatch(),
        intent_event=EventWrite(
            generate_ulid(), "tool.started", 1, {"tool": "reviewer"}, None, None
        ),
    )
    fields.update(overrides)
    return BindManagedChildRequest(**fields)


def test_bind_request_and_result_are_typed_and_immutable():
    from dataclasses import FrozenInstanceError

    from dendrux.runtime.state import BindManagedChildResult, ClaimResult

    request = _bind_request(child_path=["reviewer"], child_seed_traces=[])
    assert request.child_path == ("reviewer",) and request.child_seed_traces == ()
    with pytest.raises(FrozenInstanceError):
        request.child_run_id = generate_ulid()  # type: ignore[misc]
    result = BindManagedChildResult(
        outcome="created",
        child_run_id=request.child_run_id,
        parent_wait_id=request.parent_wait.pause_id,
        child_claim=ClaimResult(0, 2, {}, None),
    )
    with pytest.raises(FrozenInstanceError):
        result.child_claim = None  # type: ignore[misc]
    with pytest.raises(ValueError):
        BindManagedChildResult(
            outcome="existing",
            child_run_id=request.child_run_id,
            parent_wait_id=None,
            child_claim=ClaimResult(0, 2, {}, None),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"child_run_id": "not-a-ulid"},
        {"parent_run_id": "x"},
        {"root_run_id": ""},
        {"parent_claim_epoch": -1},
        {"parent_claim_epoch": True},
        {"tenant_id": ""},
        {"child_path": ()},
        {"child_path": ("",)},
        {"row_id": "x"},
        {"max_delegation_depth": 0},
    ],
    ids=lambda o: next(iter(o)),
)
def test_bind_request_refuses_invalid_identity(overrides):
    with pytest.raises(ValueError):
        _bind_request(**overrides)


def test_bind_request_requires_the_wait_to_carry_this_binding():
    other = managed_wait_pause()
    with pytest.raises(ValueError):
        _bind_request(parent_wait=other)
    unbound = managed_wait_pause()
    unbound.managed_wait = None
    with pytest.raises(ValueError):
        _bind_request(parent_wait=unbound)


@pytest.mark.parametrize("key", ["managed", "managed_wait", "submission", "pending"])
def test_bind_request_refuses_a_patch_of_reserved_checkpoint_keys(key):
    from dendrux.runtime.state import LoopStatePatch

    with pytest.raises(ValueError):
        _bind_request(parent_patch=LoopStatePatch(set_fields={key: None}))


def test_managed_header_round_trip_and_refusals():
    from dendrux.runtime.state import ManagedHeader

    root_id, child_id, call_id = generate_ulid(), generate_ulid(), generate_ulid()
    root = ManagedHeader(root_run_id=root_id, path=())
    assert ManagedHeader.from_dict(root.to_dict(), run_id=root_id) == root
    child = ManagedHeader(root_id, ("reviewer",), parent_run_id=root_id, tool_call_id=call_id)
    assert ManagedHeader.from_dict(child.to_dict(), run_id=child_id) == child
    bad = [
        None,
        {},
        {**root.to_dict(), "v": 2},
        root.to_dict(),  # a root header read on a different run
        {**child.to_dict(), "invocation": None},
        {**child.to_dict(), "path": "reviewer"},
        {**root.to_dict(), "invocation": {"parent_run_id": root_id, "tool_call_id": call_id}},
    ]
    for value in bad:
        with pytest.raises(ManagedStateError) as refused:
            ManagedHeader.from_dict(value, run_id=child_id)
        assert refused.value.reason == "header_invalid"


def test_capability_is_conservative():
    from dendrux.runtime.state import StoreCapabilities

    assert StoreCapabilities().managed_subagents is False


class _DeclaredWithoutBind:
    from dendrux.runtime.state import StoreCapabilities as _Caps

    capabilities = _Caps(fenced_writes=False, managed_subagents=True, deployment="single_process")


def test_misdeclared_managed_capability_is_refused():
    from dendrux.errors import StoreCapabilityError
    from dendrux.runtime._capabilities import validate_capabilities

    with pytest.raises(StoreCapabilityError) as refused:
        validate_capabilities(_DeclaredWithoutBind())
    assert refused.value.capability == "managed_subagents"
    assert refused.value.method == "bind_managed_child"


async def test_prerequisites_are_required_for_the_managed_capability():
    """Narrowing a prerequisite keeps the store valid but turns managed use off."""
    from dendrux.errors import StoreCapabilityError
    from dendrux.runtime._capabilities import require_managed_subagents, validate_capabilities
    from dendrux.runtime.state import StoreCapabilities

    class Partial:
        capabilities = StoreCapabilities(managed_subagents=True, deployment="single_process")

        async def bind_managed_child(self, request):
            raise AssertionError("never called")

        async def get_managed_run_count(self, root_run_id, *, limit, tenant_id):
            raise AssertionError("never called")

        async def get_managed_cancel_children_page(self, parents, *, after, limit, tenant_id):
            raise AssertionError("never called")

        async def is_managed_root_stopped(self, run_id):
            raise AssertionError("never called")

        async def get_managed_run_control(self, run_id):
            raise AssertionError("never called")

        async def get_run_control(self, run_id):
            raise AssertionError("never called")

        async def mark_cancel_requested(self, parents, *, tenant_id):
            raise AssertionError("never called")

        async def get_children_page(self, parents, *, after, limit, tenant_id):
            raise AssertionError("never called")

    store = Partial()
    validate_capabilities(store)  # an existing narrowing must not break construction
    with pytest.raises(StoreCapabilityError) as refused:
        require_managed_subagents(store)
    assert refused.value.capability == "managed_subagents"
    assert set(refused.value.missing) == {
        "fenced_writes",
        "retry_ids",
        "checkpointing",
        "atomic_completion",
        "recovery",
    }


async def test_store_without_capability_refuses_before_any_managed_write():
    from dendrux.errors import StoreCapabilityError
    from dendrux.runtime._capabilities import require_managed_subagents

    calls = []

    class Legacy:
        async def bind_managed_child(self, request):
            calls.append(request)

        async def create_run(self, *args, **kwargs):
            calls.append("create_run")

    with pytest.raises(StoreCapabilityError) as refused:
        require_managed_subagents(Legacy())
    assert refused.value.capability == "managed_subagents"
    assert calls == []


def test_managed_store_protocol_matches_the_sql_store():
    import inspect

    from dendrux.runtime.state import ManagedStateStore, StateStore
    from dendrux.runtime.stores.sql.base import SQLStateStore

    expected = inspect.signature(ManagedStateStore.bind_managed_child)
    actual = inspect.signature(SQLStateStore.bind_managed_child)
    assert list(expected.parameters) == list(actual.parameters) == ["self", "request"]
    # A separate, capability-gated protocol: the base StateStore is unchanged.
    assert not hasattr(StateStore, "bind_managed_child")


async def test_narrowing_recovery_on_a_sql_store_keeps_it_valid_but_unmanaged():
    from dataclasses import replace

    from dendrux.errors import StoreCapabilityError
    from dendrux.runtime._capabilities import require_managed_subagents, validate_capabilities
    from dendrux.runtime.stores._factory import store_for_engine
    from tests._helpers.v030_oracle import create_sqlite_engine

    engine = await create_sqlite_engine()
    try:
        store = store_for_engine(engine)
        store.capabilities = replace(store.capabilities, recovery=False)
        validate_capabilities(store)
        with pytest.raises(StoreCapabilityError) as refused:
            require_managed_subagents(store)
        assert refused.value.missing == ("recovery",)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Fix round 1, R4: strict continuation fields of a v3 wait
# ---------------------------------------------------------------------------


def _continuation_mutations():
    def set_(key, value):
        return lambda d: d.__setitem__(key, value)

    def usage(key, value):
        return lambda d: d["usage"].__setitem__(key, value)

    return {
        "iteration_text": set_("iteration", SENTINEL),
        "iteration_negative": set_("iteration", -1),
        "iteration_bool": set_("iteration", True),
        "iteration_float": set_("iteration", 1.5),
        "offset_negative": set_("trace_order_offset", -42),
        "offset_text": set_("trace_order_offset", SENTINEL),
        "schema_version_float": set_("schema_version", 3.0),
        "binding_version_bool": lambda d: d["managed_wait"].__setitem__("v", True),
        "binding_version_float": lambda d: d["managed_wait"].__setitem__("v", 1.0),
        "usage_not_object": set_("usage", [SENTINEL]),
        "usage_tokens_text": usage("input_tokens", SENTINEL),
        "usage_tokens_negative": usage("output_tokens", -5),
        "usage_tokens_bool": usage("total_tokens", True),
        "usage_cost_text": usage("cost_usd", SENTINEL),
        "usage_cost_negative": usage("cost_usd", -1.0),
        "usage_optional_text": usage("reasoning_tokens", SENTINEL),
        "usage_flag_text": usage("usage_reported", SENTINEL),
        "usage_source_number": usage("cost_source", 7),
        "image_ref_number": set_("image_view_ref", 7),
        "provider_kwargs_list": set_("provider_kwargs", [SENTINEL]),
        "pause_message_number": set_("pause_message", 7),
        "unpriced_calls_negative": set_("budget_unpriced_calls", -1),
        "unpriced_calls_bool": set_("budget_unpriced_calls", True),
        "known_cost_text": set_("budget_known_cost_usd", SENTINEL),
        "budget_fired_text": set_("budget_fired", SENTINEL),
        "budget_fired_bad_item": set_("budget_fired", [[SENTINEL]]),
        "unpriced_reported_text": set_("budget_unpriced_reported", SENTINEL),
        "budget_exceeded_text": set_("budget_exceeded", SENTINEL),
        "provider_call_id_number": lambda d: d["pending_tool_calls"][0].__setitem__(
            "provider_tool_call_id", 7
        ),
        "targets_not_object": set_("pending_targets", [SENTINEL]),
    }


@pytest.mark.parametrize("name", sorted(_continuation_mutations()))
def test_strict_decoder_refuses_malformed_continuation_fields(name, caplog):
    stored = managed_wait_pause(
        budget_fired=[("cost_usd", 0.5)], budget_known_cost_usd=0.25, budget_unpriced_calls=1
    ).to_dict()
    _continuation_mutations()[name](stored)
    with pytest.raises(ManagedStateError) as refused:
        PauseState.from_managed_dict(stored)
    assert refused.value.reason == "wait_invalid"
    assert SENTINEL not in str(refused.value) and SENTINEL not in caplog.text


def test_well_formed_optional_continuation_fields_round_trip():
    pause = managed_wait_pause(
        budget_fired=[("cost_usd", 0.5)],
        budget_known_cost_usd=0.25,
        budget_unpriced_calls=2,
        budget_unpriced_reported=True,
        budget_exceeded=[{"metric": "cost_usd", "used": 1.0, "max": 0.5}],
        pause_message="note",
        image_view_ref="ref-1",
        usage=UsageStats(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            cost_usd=0.01,
            reasoning_tokens=0,
            cost_source="table",
        ),
    )
    stored = pause.to_dict()
    assert PauseState.from_managed_dict(stored).to_dict() == stored


@pytest.mark.parametrize(
    "overrides",
    [{"agent_name": ""}, {"iteration": -1}, {"trace_order_offset": -3}, {"iteration": True}],
    ids=["agent_name", "iteration", "offset", "iteration_bool"],
)
def test_encoder_shares_the_strict_continuation_rules(overrides):
    with pytest.raises(ValueError):
        managed_wait_pause(**overrides).to_dict()


def test_binding_version_must_be_an_exact_int():
    for version in (True, 1.0):
        with pytest.raises(ValueError):
            ManagedWait(tool_call_id="c1", child_run_id=generate_ulid(), version=version)


def test_legacy_decoder_is_not_tightened():
    v2 = managed_wait_pause().to_dict()
    v2.pop("managed_wait")
    v2.update(schema_version=2, pause_reason=None, iteration="1", trace_order_offset=-1)
    # Legacy semantics unchanged: v1/v2 values are read as stored.
    back = PauseState.from_dict(v2)
    assert back.iteration == "1" and back.trace_order_offset == -1


# ---------------------------------------------------------------------------
# Fix round 2, V3: money validation is a total predicate
# ---------------------------------------------------------------------------

HUGE = 10**400


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["usage"].__setitem__("cost_usd", HUGE),
        lambda d: d.__setitem__("budget_known_cost_usd", HUGE),
        lambda d: d.__setitem__("budget_fired", [["cost_usd", HUGE]]),
        lambda d: d["usage"].__setitem__("cost_usd", -HUGE),
    ],
    ids=["cost_usd", "known_cost", "fired_threshold", "negative_cost"],
)
def test_v3_huge_integers_are_refused_not_raised(mutate):
    stored = managed_wait_pause().to_dict()
    mutate(stored)
    with pytest.raises(ManagedStateError) as refused:
        PauseState.from_managed_dict(stored)
    assert refused.value.reason == "wait_invalid"


@pytest.mark.parametrize(
    "overrides",
    [
        {"usage": UsageStats(cost_usd=HUGE)},
        {"budget_known_cost_usd": HUGE},
        {"budget_fired": [("cost_usd", HUGE)]},
    ],
    ids=["cost_usd", "known_cost", "fired_threshold"],
)
def test_v3_encoder_refuses_huge_integers_with_value_error(overrides):
    with pytest.raises(ValueError):
        managed_wait_pause(**overrides).to_dict()


def test_v3_large_finite_money_is_still_valid():
    pause = managed_wait_pause(budget_known_cost_usd=10**300, budget_fired=[("cost_usd", 1)])
    stored = pause.to_dict()
    assert PauseState.from_managed_dict(stored).budget_known_cost_usd == 10**300
