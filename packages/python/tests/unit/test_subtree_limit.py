"""Signature compatibility and public subtree refusal without database I/O."""

import functools
import inspect
import pickle
import warnings

import pytest

from dendrux import errors
from dendrux.runtime.state import UNSCOPED, RunRecord, SubtreeNodeRecord, SubtreeRecord, UsageRollup
from dendrux.store import RunStore


def tree(descendants=0):
    return SubtreeRecord(
        nodes=[
            SubtreeNodeRecord(RunRecord(str(i), "agent", "running"), i, UsageRollup())
            for i in range(descendants + 1)
        ],
        usage=UsageRollup(),
    )


@pytest.mark.parametrize(
    "shape",
    ["named", "positional_or_keyword", "wrapped", "kwargs", "positional_only", "uninspectable"],
)
@pytest.mark.parametrize("max_nodes", [None, 3])
async def test_only_named_keyword_is_forwarded(shape, max_nodes):
    received = []

    async def named(run_id, *, tenant_id, max_nodes=None):
        received.append({"tenant_id": tenant_id, "max_nodes": max_nodes})
        return tree()

    async def flexible(run_id, **kwargs):
        received.append(kwargs)
        return tree()

    method = flexible
    supported = shape in {"named", "positional_or_keyword", "wrapped"}
    if shape == "named":
        method = named
    elif shape == "positional_or_keyword":

        async def method(run_id, max_nodes=None, *, tenant_id):
            received.append({"tenant_id": tenant_id, "max_nodes": max_nodes})
            return tree()
    elif shape == "wrapped":
        method = functools.wraps(named)(flexible)
    elif shape == "positional_only":
        method.__signature__ = inspect.Signature(
            [
                inspect.Parameter("run_id", inspect.Parameter.POSITIONAL_ONLY),
                inspect.Parameter("max_nodes", inspect.Parameter.POSITIONAL_ONLY, default=None),
                inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
            ]
        )
    elif shape == "uninspectable":
        method.__signature__ = "uninspectable"
    state = type("Custom", (), {})()
    state.get_subtree = method
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        result = await RunStore(state).get_subtree(
            "00000000000000000000000700", tenant_id="mine", max_nodes=max_nodes
        )
    assert result.root.run.run_id == "0"
    assert bool(caught) == (max_nodes is None)
    if supported:
        assert received == [{"tenant_id": "mine", "max_nodes": max_nodes}]
    else:
        assert received == [{"tenant_id": "mine"}]


@pytest.mark.parametrize("descendants,refuses", [(0, False), (1, False), (2, True), (3, True)])
async def test_custom_store_limit_after_loading(descendants, refuses, caplog):
    loaded = []

    class Legacy:
        async def get_subtree(self, run_id, *, tenant_id):
            loaded.append((run_id, tenant_id))
            return tree(descendants)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        if refuses:
            with pytest.raises(errors.SubtreeTooLargeError) as caught:
                await RunStore(Legacy()).get_subtree(
                    "00000000000000000000000700", tenant_id=UNSCOPED, max_nodes=2
                )
            assert isinstance(caught.value, ValueError)
            assert (caught.value.run_id, caught.value.max_nodes) == (
                "00000000000000000000000700",
                2,
            )
            restored = pickle.loads(pickle.dumps(caught.value))
            assert (restored.run_id, restored.max_nodes) == ("00000000000000000000000700", 2)
            records = [r for r in caplog.records if getattr(r, "code", None) == "subtree_limit"]
            assert (
                len(records) == 1
                and records[0].run_id == "00000000000000000000000700"
                and records[0].limit == 2
            )
        else:
            assert (
                len(
                    (
                        await RunStore(Legacy()).get_subtree(
                            "00000000000000000000000700", max_nodes=2
                        )
                    ).nodes
                )
                == descendants + 1
            )
    assert len(loaded) == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10", 2**63, 2**64])
async def test_invalid_max_nodes_before_io(value, caplog):
    class Legacy:
        async def get_subtree(self, *args, **kwargs):
            pytest.fail("read before validation")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            await RunStore(Legacy()).get_subtree("00000000000000000000000700", max_nodes=value)
    assert not caught
    records = [r for r in caplog.records if getattr(r, "code", None) == "read_argument_invalid"]
    assert len(records) == 1 and records[0].argument == "max_nodes"


@pytest.mark.parametrize("reason", ["subtree_limit", "unpriced_model"])
async def test_translate_only_subtree_budget_errors(reason):
    original = errors.BudgetConfigurationError(reason=reason)

    class Bounded:
        async def get_subtree(self, run_id, *, tenant_id, max_nodes):
            raise original

    expected = (
        errors.SubtreeTooLargeError
        if reason == "subtree_limit"
        else errors.BudgetConfigurationError
    )
    with pytest.raises(expected) as caught:
        await RunStore(Bounded()).get_subtree("00000000000000000000000700", max_nodes=3)
    if reason == "subtree_limit":
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original


def test_subtree_error_is_exported_publicly():
    assert "SubtreeTooLargeError" in errors.__all__
