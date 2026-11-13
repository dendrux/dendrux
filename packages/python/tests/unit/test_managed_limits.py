"""Root-inclusive structural caps for managed definitions."""

import pytest

from dendrux.agent import Agent
from dendrux.errors import SubagentTreeTooLargeError
from dendrux.llm.mock import MockLLM
from dendrux.runtime._subagents import check_subagent_tree
from dendrux.types import InputValidationError, Limits


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
