"""0.7 managed subagents, T1: registration, construction checks and the tool definition.

Pure unit tests: no database, no provider calls, nothing is executed.
"""

from __future__ import annotations

import copy
import logging
import pickle

import pytest

from dendrux.agent import Agent
from dendrux.errors import (
    SubagentConfigurationError,
    SubagentCycleError,
    SubagentDefinitionError,
    SubagentDepthError,
    SubagentNameConflictError,
    SubagentTreeTooLargeError,
)
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.runtime import _subagents
from dendrux.runtime._subagents import (
    MAX_DIRECT_SUBAGENTS_WARNING,
    MAX_SUBAGENT_TREE_NODES,
    SUBAGENT_TASK_FIELD,
    check_subagent_tree,
    subagent_tool_def,
)
from dendrux.tool import tool
from dendrux.types import ToolDef, ToolTarget

SECRET_DESCRIPTION = "SENTINEL-DESCRIPTION-TEXT-7f3a"


@tool()
async def lookup(query: str) -> str:
    """Look something up."""
    return query


class _Tripwire:
    """A subagent entry that counts every inspection of its definition."""

    inspected = 0

    def __getattr__(self, name: str) -> object:
        type(self).inspected += 1
        raise AssertionError(f"entry inspected: {name}")


@pytest.fixture(autouse=True)
def _reset_tripwire() -> None:
    _Tripwire.inspected = 0


def _provider() -> MockLLM:
    return MockLLM([])


def _child(name: str = "reviewer", description: str = "Reviews a draft.", **kwargs) -> Agent:
    return Agent(
        provider=_provider(),
        name=name,
        description=description,
        prompt="You review drafts.",
        **kwargs,
    )


def _parent(subagents, **kwargs) -> Agent:
    kwargs.setdefault("name", "lead")
    return Agent(provider=_provider(), prompt="You lead.", subagents=subagents, **kwargs)


def _chain(levels: int) -> Agent:
    """A chain of ``levels`` agents, each with one subagent; the top one is returned."""
    node = _child(name="n0")
    for i in range(1, levels):
        node = _child(name=f"n{i}", subagents=[node], max_delegation_depth=None)
    return node


# ------------------------------------------------------------------
# 1. description
# ------------------------------------------------------------------


class TestDescription:
    def test_defaults_to_empty_string(self) -> None:
        assert Agent(provider=_provider(), prompt="p").description == ""

    def test_constructor_kwarg(self) -> None:
        agent = Agent(provider=_provider(), prompt="p", description="Finds facts.")
        assert agent.description == "Finds facts."

    def test_class_attribute(self) -> None:
        class Researcher(Agent):
            prompt = "p"
            description = "Finds facts."

        assert Researcher(provider=_provider()).description == "Finds facts."

    def test_kwarg_overrides_class_attribute(self) -> None:
        class Researcher(Agent):
            prompt = "p"
            description = "Finds facts."

        assert Researcher(provider=_provider(), description="Other.").description == "Other."

    @pytest.mark.parametrize("bad", [123, None, b"bytes", ["x"]])
    def test_must_be_a_string(self, bad: object) -> None:
        with pytest.raises(ValueError, match="description must be a string"):
            Agent(provider=_provider(), prompt="p", description=bad)  # type: ignore[arg-type]

    def test_class_attribute_must_be_a_string(self) -> None:
        class Broken(Agent):
            prompt = "p"
            description = 5  # type: ignore[assignment]

        with pytest.raises(ValueError, match="description must be a string"):
            Broken(provider=_provider())


# ------------------------------------------------------------------
# 2. subagents= registration
# ------------------------------------------------------------------


class TestRegistration:
    def test_defaults_to_empty_tuple(self) -> None:
        assert Agent(provider=_provider(), prompt="p").subagents == ()

    def test_none_and_empty_list_mean_no_subagents(self) -> None:
        assert _parent(None).subagents == ()
        assert _parent([]).subagents == ()

    def test_stored_in_order_as_a_tuple(self) -> None:
        a, b = _child("alpha"), _child("beta")
        parent = _parent([a, b])
        assert parent.subagents == (a, b)
        assert parent.subagents[0] is a

    def test_caller_list_is_copied(self) -> None:
        a = _child("alpha")
        given = [a]
        parent = _parent(given)
        given.append(_child("beta"))
        assert parent.subagents == (a,)

    def test_tuple_accepted(self) -> None:
        a = _child("alpha")
        assert _parent((a,)).subagents == (a,)

    def test_children_are_not_mutated(self) -> None:
        child = _child("alpha", subagents=[_child("leaf")])
        containers = (list, dict, set)
        before = {
            key: (id(value), copy.copy(value) if isinstance(value, containers) else value)
            for key, value in vars(child).items()
        }
        _parent([child])
        _parent([child], name="second_lead")
        after = vars(child)
        assert after.keys() == before.keys()
        for key, (identity, snapshot) in before.items():
            assert id(after[key]) == identity, key
            assert after[key] == snapshot, key

    @pytest.mark.parametrize("bad", ["reviewer", 42, {"a": 1}])
    def test_must_be_a_list(self, bad: object) -> None:
        with pytest.raises(SubagentDefinitionError) as info:
            _parent(bad)
        assert info.value.reason == "not_a_list"

    def test_same_template_under_two_parents(self) -> None:
        shared = _child("reviewer")
        lead_a = _child("lead_a", subagents=[shared])
        lead_b = _child("lead_b", subagents=[shared])
        root = _parent([lead_a, lead_b])
        assert root.subagents[0].subagents[0] is root.subagents[1].subagents[0]


# ------------------------------------------------------------------
# 3. construction checks
# ------------------------------------------------------------------


class TestDefinitionChecks:
    @pytest.mark.parametrize("bad", ["reviewer", None, object(), lookup])
    def test_entry_must_be_an_agent(self, bad: object) -> None:
        with pytest.raises(SubagentDefinitionError) as info:
            _parent([_child("ok"), bad])
        err = info.value
        assert err.reason == "not_agent"
        assert err.index == 1
        assert err.agent_name == "lead"
        assert isinstance(err, SubagentConfigurationError)
        assert isinstance(err, ValueError)

    @pytest.mark.parametrize(
        "bad",
        ["", "has space", "has-dash", "1abc", "x" * 65, "naïve", "tab\tname", "dot.name"],
    )
    def test_invalid_names(self, bad: str) -> None:
        with pytest.raises(SubagentDefinitionError) as info:
            _parent([_child(name=bad)])
        assert info.value.reason == "invalid_name"
        assert info.value.index == 0
        assert info.value.subagent_name is None
        if bad:
            assert bad not in str(info.value)

    @pytest.mark.parametrize("good", ["a", "_x", "x" * 64, "Reviewer_2", "code_reviewer"])
    def test_valid_names(self, good: str) -> None:
        assert _parent([_child(name=good)]).subagents[0].name == good

    def test_class_derived_name_is_accepted(self) -> None:
        class Critic(Agent):
            prompt = "p"
            description = "Criticises."

        assert _parent([Critic(provider=_provider())]).subagents[0].name == "Critic"

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
    def test_description_required(self, bad: str) -> None:
        with pytest.raises(SubagentDefinitionError) as info:
            _parent([_child(description=bad)])
        assert info.value.reason == "missing_description"
        assert info.value.subagent_name == "reviewer"

    def test_single_call_parent_is_refused(self) -> None:
        with pytest.raises(SubagentDefinitionError) as info:
            _parent([_child()], loop=SingleCall())
        assert info.value.reason == "single_call_parent"

    def test_single_call_child_is_allowed(self) -> None:
        child = _child(loop=SingleCall())
        assert _parent([child]).subagents == (child,)


class TestNameConflicts:
    def test_duplicate_names_refused(self) -> None:
        with pytest.raises(SubagentNameConflictError) as info:
            _parent([_child("reviewer"), _child("reviewer", description=SECRET_DESCRIPTION)])
        err = info.value
        assert err.reason == "duplicate_name"
        assert err.subagent_name == "reviewer"
        assert SECRET_DESCRIPTION not in str(err)

    def test_same_instance_twice_refused(self) -> None:
        child = _child("reviewer")
        with pytest.raises(SubagentNameConflictError) as info:
            _parent([child, child])
        assert info.value.reason == "duplicate_name"

    def test_clash_with_parent_tool(self) -> None:
        with pytest.raises(SubagentNameConflictError) as info:
            _parent([_child("lookup")], tools=[lookup])
        assert info.value.reason == "tool_name_clash"
        assert info.value.subagent_name == "lookup"

    @pytest.mark.parametrize("generated", ["use_skill", "read_skill_resource", "run_skill_script"])
    def test_skill_free_parent_accepts_generated_skill_names(self, generated: str) -> None:
        assert _parent([_child(generated)]).subagents[0].name == generated

    @pytest.mark.parametrize("generated", ["use_skill", "read_skill_resource", "run_skill_script"])
    def test_parent_with_skills_refuses_generated_skill_names(
        self, generated: str, tmp_path
    ) -> None:
        with pytest.raises(SubagentNameConflictError) as info:
            _parent([_child(generated)], skills_dir=tmp_path)
        assert info.value.reason == "tool_name_clash"
        assert info.value.subagent_name == generated

    def test_names_are_unique_per_parent_not_globally(self) -> None:
        nested_helper = _child("helper", description="Nested helper.")
        lead = _child("lead_a", subagents=[nested_helper])
        root = _parent([_child("helper"), lead])
        assert [a.name for a in root.subagents] == ["helper", "lead_a"]

    def test_child_may_share_parent_name(self) -> None:
        assert _parent([_child("lead")], name="lead").subagents[0].name == "lead"

    def test_extra_tool_names_checked(self) -> None:
        # T2 re-runs the check after MCP discovery with the discovered names.
        parent = _parent([_child("reviewer")])
        with pytest.raises(SubagentNameConflictError) as info:
            check_subagent_tree(parent, agent_type=Agent, extra_tool_names={"reviewer"})
        assert info.value.reason == "tool_name_clash"


class TestCycles:
    def test_cycle_through_mutation_is_refused(self) -> None:
        b = _child("b")
        a = _child("a", subagents=[b])
        b._subagents = (a,)  # only reachable by mutating a built Agent
        with pytest.raises(SubagentCycleError) as info:
            _parent([a])
        assert info.value.subagent_name == "a"
        assert info.value.agent_name == "b"

    def test_self_cycle_is_refused(self) -> None:
        x = _child("x")
        x._subagents = (x,)
        with pytest.raises(SubagentCycleError):
            _parent([x])

    def test_root_reached_again_is_a_cycle(self) -> None:
        a = _child("a")
        root = _parent([a], description="Leads.")
        a._subagents = (root,)
        with pytest.raises(SubagentCycleError):
            check_subagent_tree(root, agent_type=Agent)

    def test_reuse_at_distinct_paths_is_not_a_cycle(self) -> None:
        leaf = _child("leaf")
        mid_a = _child("mid_a", subagents=[leaf])
        mid_b = _child("mid_b", subagents=[leaf])
        top = _child("top", subagents=[mid_a, mid_b])
        root = _parent([top, mid_a])
        assert len(root.subagents) == 2


class TestDepth:
    def test_default_depth_ceiling_is_ten(self) -> None:
        assert _parent([_chain(10)]).subagents  # deepest node at depth 10
        with pytest.raises(SubagentDepthError) as info:
            _parent([_chain(11)])
        err = info.value
        assert err.depth == 11
        assert err.max_depth == 10

    def test_explicit_root_limit(self) -> None:
        with pytest.raises(SubagentDepthError) as info:
            _parent([_child("a", subagents=[_child("b")])], max_delegation_depth=1)
        assert info.value.depth == 2
        assert info.value.max_depth == 1
        assert info.value.subagent_name == "b"

    def test_zero_limit_refuses_any_subagent(self) -> None:
        with pytest.raises(SubagentDepthError):
            _parent([_child()], max_delegation_depth=0)

    def test_none_is_unbounded(self) -> None:
        assert _parent([_chain(12)], max_delegation_depth=None).subagents

    def test_child_limit_tightens_its_subtree(self) -> None:
        # The child passes its own check (grandchild-of-child at relative depth 2),
        # but under the root that node sits at depth 3, above the child's ceiling of 2.
        inner = _child("g", subagents=[_child("gg")])
        child = _child("c", subagents=[inner], max_delegation_depth=2)
        with pytest.raises(SubagentDepthError) as info:
            _parent([child], max_delegation_depth=None)
        assert info.value.subagent_name == "gg"
        assert info.value.depth == 3
        assert info.value.max_depth == 2

    def test_child_cannot_loosen_parent_limit(self) -> None:
        child = _child("c", subagents=[_child("g")], max_delegation_depth=None)
        with pytest.raises(SubagentDepthError):
            _parent([child], max_delegation_depth=1)

    def test_effective_run_depth_can_be_passed(self) -> None:
        # T2 re-checks with the run's effective depth (run kwarg / inherited ceiling).
        root = _parent([_child("a", subagents=[_child("b")])])
        check_subagent_tree(root, agent_type=Agent, max_depth=2)
        with pytest.raises(SubagentDepthError):
            check_subagent_tree(root, agent_type=Agent, max_depth=1)

    def test_runner_default_matches(self) -> None:
        from dendrux._sentinel import _UnsetType
        from dendrux.runtime.runner import _resolve_max_delegation_depth

        assert _resolve_max_delegation_depth(_UnsetType(), None) == (
            _subagents.DEFAULT_MAX_DELEGATION_DEPTH
        )


class TestTreeSize:
    def test_default_bound(self) -> None:
        assert MAX_SUBAGENT_TREE_NODES == 256

    def test_hostile_reused_template_tree_is_refused_quickly(self) -> None:
        # Each level reaches the level below twice, so paths double per level while
        # objects grow by two. Thirty levels would be ~2**31 paths; the walk stops at
        # the bound instead (level 7: 381 paths).
        node = _child("n0")
        level = 0
        with pytest.raises(SubagentTreeTooLargeError) as info:
            for level in range(1, 31):  # noqa: B007
                mid = _child(f"m{level}", subagents=[node], max_delegation_depth=None)
                node = _child(f"n{level}", subagents=[node, mid], max_delegation_depth=None)
        assert info.value.max_nodes == MAX_SUBAGENT_TREE_NODES
        assert level == 7

    def test_ten_times_a_realistic_tree_passes(self) -> None:
        # Realistic: a lead with ~3 specialists, each with ~3 helpers (~12 nodes).
        # Ten times that: 10 leads x (1 + 10 helpers) = 110 nodes, under the bound.
        leads = [
            _child(f"lead{i}", subagents=[_child(f"h{j}") for j in range(10)]) for i in range(10)
        ]
        root = _parent(leads)
        assert len(root.subagents) == 10

    def test_wide_direct_list_is_refused_before_any_entry_is_inspected(self) -> None:
        tripwires = [_Tripwire() for _ in range(1_000)]
        with pytest.raises(SubagentTreeTooLargeError) as info:
            _parent(tripwires)
        assert info.value.max_nodes == MAX_SUBAGENT_TREE_NODES
        assert _Tripwire.inspected == 0

    def test_wide_nested_list_is_refused_before_any_entry_is_inspected(self) -> None:
        child = _child("lead_a")
        root = _parent([child], description="Leads.")
        child._subagents = tuple(_Tripwire() for _ in range(1_000))
        with pytest.raises(SubagentTreeTooLargeError):
            check_subagent_tree(root, agent_type=object)
        assert _Tripwire.inspected == 0

    def test_list_that_fits_the_remaining_budget_is_checked(self) -> None:
        # Budget 3: root's two children use 2, the nested list of 2 does not fit.
        inner = _child("inner", subagents=[_child("x"), _child("y")])
        root = _parent([inner, _child("other")], description="Leads.")
        check_subagent_tree(root, agent_type=Agent, max_nodes=4)
        with pytest.raises(SubagentTreeTooLargeError):
            check_subagent_tree(root, agent_type=Agent, max_nodes=3)

    def test_bound_is_overridable_on_the_check(self) -> None:
        root = _parent([_child("a"), _child("b")])
        check_subagent_tree(root, agent_type=Agent, max_nodes=2)
        with pytest.raises(SubagentTreeTooLargeError):
            check_subagent_tree(root, agent_type=Agent, max_nodes=1)


class TestFanoutWarning:
    def test_more_than_ten_direct_subagents_logs_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        children = [_child(f"s{i}") for i in range(11)]
        with caplog.at_level(logging.WARNING):
            parent = _parent(children)
        assert len(parent.subagents) == 11
        records = [r for r in caplog.records if getattr(r, "code", None) == "subagent_fanout"]
        assert len(records) == 1
        assert records[0].count == 11
        assert records[0].limit == MAX_DIRECT_SUBAGENTS_WARNING

    def test_ten_direct_subagents_do_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        children = [_child(f"s{i}") for i in range(10)]
        with caplog.at_level(logging.WARNING):
            _parent(children)
        assert not [r for r in caplog.records if getattr(r, "code", None) == "subagent_fanout"]


class TestErrorContract:
    @pytest.mark.parametrize(
        ("make", "cls"),
        [
            (lambda: _parent([42]), SubagentDefinitionError),
            (lambda: _parent([_child("a"), _child("a")]), SubagentNameConflictError),
            (lambda: _parent([_child()], max_delegation_depth=0), SubagentDepthError),
        ],
    )
    def test_refusal_logs_its_stable_code(self, make, cls, caplog) -> None:
        with caplog.at_level(logging.WARNING), pytest.raises(cls) as info:
            make()
        codes = [getattr(r, "code", None) for r in caplog.records]
        assert info.value.code in codes

    def test_codes_are_distinct_and_registered(self) -> None:
        from dendrux._codes import REGISTRY

        classes = [
            SubagentDefinitionError,
            SubagentNameConflictError,
            SubagentCycleError,
            SubagentDepthError,
            SubagentTreeTooLargeError,
        ]
        codes = [cls.code for cls in classes]
        assert len(set(codes)) == len(codes)
        assert {cls: code.name for cls, code in _subagents._REFUSAL_CODES.items()} == {
            cls: cls.code for cls in classes
        }
        assert set(codes) | {"subagent_fanout"} <= REGISTRY.keys()

    @pytest.mark.parametrize(
        "exc",
        [
            SubagentDefinitionError("lead", reason="not_agent", index=2),
            SubagentNameConflictError("lead", reason="duplicate_name", subagent_name="r"),
            SubagentCycleError("lead", subagent_name="r"),
            SubagentDepthError("lead", subagent_name="r", depth=3, max_depth=2),
            SubagentTreeTooLargeError("lead", max_nodes=256),
        ],
    )
    def test_pickle_and_copy_roundtrip(self, exc: SubagentConfigurationError) -> None:
        for clone in (pickle.loads(pickle.dumps(exc)), copy.copy(exc), copy.deepcopy(exc)):
            assert type(clone) is type(exc)
            assert str(clone) == str(exc)
            for name in type(exc)._init_fields:
                assert getattr(clone, name) == getattr(exc, name)


# ------------------------------------------------------------------
# 4. the model-visible tool definition
# ------------------------------------------------------------------


class TestToolDefinition:
    def test_shape(self) -> None:
        td = subagent_tool_def(_child("reviewer", description="Reviews a draft."))
        assert isinstance(td, ToolDef)
        assert td.name == "reviewer"
        assert td.description == "Reviews a draft."
        assert td.parameters == {
            "type": "object",
            "properties": {
                SUBAGENT_TASK_FIELD: {
                    "type": "string",
                    "description": _subagents.TASK_FIELD_DESCRIPTION,
                }
            },
            "required": [SUBAGENT_TASK_FIELD],
            "additionalProperties": False,
        }
        assert SUBAGENT_TASK_FIELD == "task"
        assert td.target is ToolTarget.AGENT
        assert td.parallel is False

    def test_schema_is_valid_and_validates_arguments(self) -> None:
        from dendrux.tools._validate import schema_problem, validate_tool_params

        td = subagent_tool_def(_child())
        assert schema_problem(td.parameters) is None
        assert validate_tool_params(td, {"task": "Check the totals."}) == []
        assert validate_tool_params(td, {}) != []
        assert validate_tool_params(td, {"task": 5}) != []
        assert validate_tool_params(td, {"task": "x", "extra": 1}) != []

    def test_each_call_returns_a_fresh_schema(self) -> None:
        child = _child()
        first, second = subagent_tool_def(child), subagent_tool_def(child)
        assert first == second
        first.parameters["properties"]["task"]["description"] = "changed"
        assert subagent_tool_def(child).parameters == second.parameters

    def test_provider_payload(self) -> None:
        pytest.importorskip("anthropic")
        from dendrux.llm.anthropic import AnthropicProvider

        td = subagent_tool_def(_child())
        payload = AnthropicProvider._legacy_tool(td)
        assert payload == {
            "name": "reviewer",
            "description": "Reviews a draft.",
            "input_schema": td.parameters,
        }

    def test_does_not_touch_the_agent(self) -> None:
        child = _child()
        before = dict(vars(child))
        subagent_tool_def(child)
        assert vars(child) == before

    def test_not_wired_into_the_parent_catalog_yet(self) -> None:
        parent = _parent([_child("reviewer")], tools=[lookup])
        assert [td.name for td in parent.get_tool_defs()] == ["lookup"]
        assert [td.name for td in parent.get_all_tool_defs()] == ["lookup"]
