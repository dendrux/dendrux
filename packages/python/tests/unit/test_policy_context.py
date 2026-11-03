"""Database-free coverage of the shared pure policy builders."""

import json
from unittest.mock import Mock

import pytest

from dendrux._meta_keys import NOT_EXECUTED_META_KEY
from dendrux.loops._policy_context import (
    dispatch_policy,
    history_call_counts,
    static_denial,
    tool_call_context,
)
from dendrux.types import ImageRef, Message, Role, ToolCallContext, ToolResult, ToolTarget


def test_history_counts_skip_only_explicit_nonexecution():
    history = [
        Message(Role.USER, "input"),
        Message(Role.TOOL, "ok", name="work", call_id="1"),
        Message(Role.TOOL, "denied", name="work", call_id="2", meta={"is_error": True}),
        Message(Role.TOOL, "skip", name="work", call_id="3", meta={NOT_EXECUTED_META_KEY: True}),
        Message(Role.TOOL, "ok", name="other", call_id="4", meta={NOT_EXECUTED_META_KEY: False}),
        Message(Role.TOOL, "empty", name="legacy", call_id="5"),
        Message(Role.TOOL, "unnamed", name="legacy", call_id="6"),
    ]
    object.__setattr__(history[-2], "name", "")
    object.__setattr__(history[-1], "name", None)
    assert history_call_counts(iter(history)) == {"work": 2, "other": 1, "": 1}
    assert history_call_counts([]) == {}


@pytest.mark.parametrize("name", ["work", "quoted'\"工具"])
def test_static_denial_exact_wire_payload_and_event(name):
    result, event = static_denial(name, "call")
    message = f"Tool '{name}' is denied by policy."
    assert result == ToolResult(
        name=name,
        call_id="call",
        payload=json.dumps({"denied": message}),
        success=False,
        error=message,
    )
    assert event == {"tool_name": name, "call_id": "call", "reason": "denied_by_policy"}
    event["reason"] = "changed"
    assert static_denial(name, "call")[1]["reason"] == "denied_by_policy"


@pytest.mark.parametrize("source", ["local", "mcp", "skill"])
@pytest.mark.parametrize("target", [ToolTarget.SERVER, ToolTarget.CLIENT])
@pytest.mark.parametrize("tenant", [None, "tenant"])
def test_context_copies_nested_inputs_and_preserves_every_field(source, target, tenant):
    params = {"nested": {"values": [1]}}
    meta = {"nested": {"values": [2]}}
    images = (ImageRef(sha256="a" * 64, media_type="image/png", byte_length=1, width=1, height=1),)
    context = tool_call_context(
        run_id="run",
        call_id="call",
        tool_name="work",
        params=params,
        target=target,
        source=source,
        meta=meta,
        iteration=7,
        tenant_id=tenant,
        images=images,
    )
    assert context == ToolCallContext(
        run_id="run",
        call_id="call",
        tool_name="work",
        params=params,
        target=target,
        source=source,
        meta=meta,
        iteration=7,
        tenant_id=tenant,
        images=images,
    )
    assert context.images is images
    context.params["nested"]["values"].append(3)
    context.meta["nested"]["values"].append(4)
    assert params == {"nested": {"values": [1]}}
    assert meta == {"nested": {"values": [2]}}


@pytest.mark.parametrize("phase", ["approved", "dispatchable", "gated"])
@pytest.mark.parametrize("kind", ["dispatch", "approval", "replay"])
def test_dispatch_policy_keeps_approval_precedence(phase, kind):
    hook = Mock()
    requires = frozenset({"other"})
    result = dispatch_policy(
        name="work", phase=phase, kind=kind, require_approval=requires, before_tool=hook
    )
    assert result == {
        "require_approval": frozenset()
        if phase == "approved"
        else frozenset({"other", "work"})
        if kind == "approval"
        else requires,
        "before_tool": None if phase == "approved" else hook,
        "approved_skills": phase == "approved",
    }
    assert requires == frozenset({"other"})
    hook.assert_not_called()


def test_approved_dispatch_defaults():
    assert dispatch_policy(name="work", phase="approved") == {
        "require_approval": frozenset(),
        "before_tool": None,
        "approved_skills": True,
    }
