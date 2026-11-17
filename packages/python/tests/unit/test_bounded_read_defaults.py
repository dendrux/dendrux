"""Bounded RunStore reads (since 0.7.0): default pages, refused oversize pages, no warnings."""

import inspect
import logging
import warnings
from types import SimpleNamespace

import pytest

from dendrux.runtime.state import StoreCapabilities, SubtreeRecord, UsageRollup
from dendrux.store import RunStore, _InitOnFirstUse
from tests._helpers.warnings import warnings_of

READS = ["get_events", "get_traces", "get_llm_calls", "get_tool_invocations"]
STORE_METHOD = {
    "get_events": "get_run_events",
    "get_traces": "get_traces",
    "get_llm_calls": "get_llm_interactions",
    "get_tool_invocations": "get_tool_calls",
}


class ReadAttemptError(Exception):
    pass


class ExplodingStore:
    """Every read is a refused I/O attempt, so a test can prove nothing was read."""

    async def get_run(self, *args, **kwargs):
        raise ReadAttemptError("tenant/format lookup")

    async def get_run_events(self, *args, **kwargs):
        raise ReadAttemptError("events")

    get_traces = get_run_events
    get_llm_interactions = get_run_events
    get_tool_calls = get_run_events
    get_subtree = get_run_events


async def _noop():
    return None


class RecordingStore:
    """A format-1 custom store that records the keyword arguments each read receives."""

    def __init__(self):
        self.seen: dict[str, dict] = {}

    async def get_run(self, run_id, *, tenant_id=None):
        return SimpleNamespace(id=run_id, tenant_id=tenant_id, storage_format=1)

    def __getattr__(self, name):
        if name not in STORE_METHOD.values():
            raise AttributeError(name)

        async def read(run_id, **kwargs):
            self.seen[name] = kwargs
            return []

        return read


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("kwargs", [{}, {"limit": None}])
@pytest.mark.parametrize("path", ["plain", "tenant", "lazy"])
async def test_omitted_limit_reads_one_bounded_page_without_warning(method, kwargs, path):
    underlying = RecordingStore()
    reader = RunStore(_InitOnFirstUse(underlying, _noop, _noop) if path == "lazy" else underlying)
    if path == "tenant":
        kwargs = {**kwargs, "tenant_id": "tenant"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert await getattr(reader, method)("root", **kwargs) == []
    assert underlying.seen[STORE_METHOD[method]]["limit"] == 1000
    assert not warnings_of(caught, FutureWarning)


@pytest.mark.parametrize("method", READS)
def test_signature_shows_the_bounded_default(method):
    assert inspect.signature(getattr(RunStore, method)).parameters["limit"].default == 1000


def test_subtree_signature_shows_the_bounded_default():
    assert inspect.signature(RunStore.get_subtree).parameters["max_nodes"].default == 10_000


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("limit", [1001, 2000, 2**63 - 1])
@pytest.mark.parametrize("path", ["plain", "tenant", "checkpoint", "lazy"])
async def test_limit_above_page_raises_before_any_io(method, limit, path, caplog):
    state = ExplodingStore()
    if path == "checkpoint":
        state.capabilities = StoreCapabilities(checkpointing=True, deployment="multi_server")
    if path == "lazy":
        state = _InitOnFirstUse(state, state.get_run, state.get_run)
    kwargs = {"limit": limit, **({"tenant_id": "tenant"} if path == "tenant" else {})}
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=r"limit must be at most 1000") as raised:
            await getattr(RunStore(state), method)("root", **kwargs)
    assert not isinstance(raised.value, ReadAttemptError)
    assert str(limit) not in str(raised.value)
    assert f"RunStore.{method}()" in str(raised.value)
    assert not warnings_of(caught, FutureWarning)
    records = [r for r in caplog.records if getattr(r, "code", None) == "read_argument_invalid"]
    assert [r.argument for r in records] == ["limit"]


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("limit", [-1, 0, 1, 1000])
async def test_explicit_page_passes_unchanged_without_warning(method, limit):
    state = RecordingStore()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await getattr(RunStore(state), method)("root", limit=limit)
    assert state.seen[STORE_METHOD[method]]["limit"] == limit
    assert not warnings_of(caught, FutureWarning)


@pytest.mark.parametrize("method", READS + ["get_subtree"])
@pytest.mark.parametrize("kwargs", [{"run_id": ""}, {"tenant_id": False}, {"tenant_id": ""}])
async def test_invalid_scope_validates_before_io(method, kwargs):
    kwargs = {"run_id": "root", **kwargs}
    with pytest.raises(ValueError):
        await getattr(RunStore(ExplodingStore()), method)(**kwargs)


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("value", [True, "100", 2**64])
async def test_invalid_limit_does_not_read(method, value):
    with pytest.raises(ValueError):
        await getattr(RunStore(ExplodingStore()), method)("root", limit=value)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("get_events", {"after_sequence_index": True}),
        ("get_events", {"event_types": [False]}),
        ("get_traces", {"offset": False}),
        ("get_llm_calls", {"iteration": False}),
        ("get_tool_invocations", {"before": "yesterday"}),
    ],
)
async def test_invalid_filters_do_not_read(method, kwargs):
    with pytest.raises(ValueError):
        await getattr(RunStore(ExplodingStore()), method)("root", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, 1000), ({"limit": None}, 1000), ({"limit": 1000}, 1000), ({"limit": 7}, 7)]
    + [({"limit": 0}, 1), ({"limit": -3}, 1)],
)
async def test_format_two_traces_use_the_bounded_page(monkeypatch, kwargs, expected):
    from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
    from dendrux.runtime import _public_traces

    class CheckpointStore(ExplodingStore):
        capabilities = StoreCapabilities(checkpointing=True, deployment="multi_server")

        async def get_run(self, run_id, *, tenant_id):
            assert tenant_id == "tenant"
            return SimpleNamespace(storage_format=STORAGE_FORMAT_CHECKPOINTED, retry_of_run_id=None)

    seen = []

    async def records(state, run_id, **kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(_public_traces, "public_traces", records)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        reader = RunStore(CheckpointStore())
        assert await reader.get_traces("root", tenant_id="tenant", offset=23, **kwargs) == []
    assert seen == [{"tenant_id": "tenant", "limit": expected, "offset": 23, "retry": False}]


@pytest.mark.parametrize("kwargs", [{}, {"max_nodes": None}])
async def test_omitted_max_nodes_bounds_the_subtree_without_warning(kwargs):
    received = []

    class Named:
        async def get_subtree(self, run_id, *, tenant_id, max_nodes=None):
            received.append(max_nodes)
            return SubtreeRecord(nodes=[], usage=UsageRollup())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await RunStore(Named()).get_subtree("00000000000000000000000700", **kwargs)
    assert received == [10_000]
    assert not warnings_of(caught, FutureWarning)


def test_shared_frame_walk_keeps_capability_warning_at_caller():
    from dendrux.runtime._capabilities import validate_capabilities

    class LegacyDeclaration:
        capabilities = StoreCapabilities()

    store = LegacyDeclaration()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        line = inspect.currentframe().f_lineno + 1
        validate_capabilities(store)
        assert validate_capabilities(store) is store.capabilities
    matching = warnings_of(caught, DeprecationWarning, "StateStore capabilities")
    assert len(matching) == 1
    assert matching[0].filename == __file__ and matching[0].lineno == line
