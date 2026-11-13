"""Root-inclusive structural caps for managed definitions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.agent import Agent
from dendrux.errors import SubagentTreeTooLargeError
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _managed
from dendrux.runtime._subagents import check_subagent_tree
from dendrux.runtime.state import RunRecord
from dendrux.types import InputValidationError, Limits, generate_ulid


def node(name, *, children=(), limits=None):
    return Agent(
        name=name,
        prompt="Work.",
        description="Specialist.",
        provider=MockLLM([]),
        subagents=list(children),
        limits=limits,
    )


def test_public_defaults():
    limits = Limits()
    assert getattr(limits, "max_managed_runs", None) == 128
    assert getattr(limits, "max_subagent_definitions", None) == 256


@pytest.mark.parametrize("field", ["max_managed_runs", "max_subagent_definitions"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, float("inf"), "SENTINEL", None])
def test_invalid_cap_is_typed_and_safe(field, value):
    try:
        Limits(**{field: value})
    except TypeError:
        pytest.fail("public cap field is missing")
    except InputValidationError as error:
        assert error.code == "invalid_limit"
        assert "SENTINEL" not in str(error)
    else:
        pytest.fail("invalid cap accepted")


def test_root_inclusive_hostile_definition_cap():
    root = node("root", children=[node("a"), node("b")])
    object.__setattr__(root._limits, "max_subagent_definitions", 2)
    with pytest.raises(SubagentTreeTooLargeError) as error:
        check_subagent_tree(root, agent_type=Agent)
    assert error.value.max_nodes == 2


def test_reused_paths_count_twice():
    shared = node("shared")
    a, b = node("a", children=[shared]), node("b", children=[shared])
    root = node("root", children=[a, b])
    object.__setattr__(root._limits, "max_subagent_definitions", 4)
    with pytest.raises(SubagentTreeTooLargeError):
        check_subagent_tree(root, agent_type=Agent)


def test_25_nodes_pass_default():
    root = node("root", children=[node(f"n{i}") for i in range(24)])
    assert len(root.subagents) == 24


def test_override_above_old_constant():
    root = node(
        "root",
        children=[node(f"n{i}") for i in range(300)],
        limits=Limits(max_subagent_definitions=301),
    )
    assert len(root.subagents) == 300


@pytest.mark.parametrize("check", ["stream", "provider", "store"])
def test_large_override_preflight_visits_every_child(check):
    children = [node(f"n{i}") for i in range(300)]
    if check == "stream":
        children[0]._guardrails = [PII()]
    elif check == "provider":
        children[0]._provider = None
    else:
        children[0]._state_store = object()
    root = node("root", children=children, limits=Limits(max_subagent_definitions=301))
    with pytest.raises(ValueError):
        if check == "stream":
            _managed.require_streamable(root, "stream")
        else:
            _managed._check_definitions(root, object())


@pytest.mark.parametrize("check", ["stream", "provider"])
@pytest.mark.parametrize("location", ["root", "nested"])
def test_mutated_tree_preflight_refuses_at_root_inclusive_cap(check, location):
    root = node("root", children=[node("one")], limits=Limits(max_subagent_definitions=2))
    if location == "root":
        root._subagents = (node("one"), node("two"))
    else:
        root.subagents[0]._subagents = (node("two"),)
    with pytest.raises(SubagentTreeTooLargeError):
        if check == "stream":
            _managed.require_streamable(root, "stream")
        else:
            _managed._check_definitions(root, None)


async def test_format_one_walk_does_not_read_a_checkpoint():
    record = RunRecord(id=generate_ulid(), agent_name="root", status="running", storage_format=1)
    store = SimpleNamespace(
        get_run=AsyncMock(return_value=record),
        get_loop_state=AsyncMock(side_effect=NotImplementedError),
    )
    waiting, leaf = await _managed.walk(node("root"), store, record.id)
    assert not waiting
    assert leaf is record
    store.get_loop_state.assert_not_awaited()


async def test_older_checkpoint_without_cap_keeps_original_default():
    record = RunRecord(id=generate_ulid(), agent_name="root", status="running", storage_format=2)
    store = SimpleNamespace(get_loop_state=AsyncMock(return_value={}))
    assert await _managed._saved_walk_limit(store, record) == 128
