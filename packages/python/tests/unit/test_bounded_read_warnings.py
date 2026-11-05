"""Migration warnings precede every read and attribute to the awaiting caller."""

import asyncio
import inspect
import warnings
from types import SimpleNamespace

import pytest

from dendrux.runtime.state import StoreCapabilities
from dendrux.store import RunStore, _InitOnFirstUse

READS = ["get_events", "get_traces", "get_llm_calls", "get_tool_invocations"]


class ReadAttemptError(Exception):
    pass


class ExplodingStore:
    before_io = None

    def check_warning(self):
        if self.before_io is not None:
            self.before_io()

    async def get_run(self, *args, **kwargs):
        self.check_warning()
        raise ReadAttemptError("tenant/format lookup")

    async def get_run_events(self, *args, **kwargs):
        self.check_warning()
        raise ReadAttemptError("events")

    get_traces = get_run_events
    get_llm_interactions = get_run_events
    get_tool_calls = get_run_events
    get_subtree = get_run_events


@pytest.mark.parametrize(
    "method,limit",
    [(method, limit) for method in READS for limit in (None, 1001)] + [("get_subtree", None)],
)
@pytest.mark.parametrize("path", ["plain", "tenant", "checkpoint", "lazy"])
async def test_warning_before_any_io(method, limit, path):
    state = ExplodingStore()
    underlying = state
    if path == "checkpoint":
        state.capabilities = StoreCapabilities(checkpointing=True, deployment="multi_server")
    if path == "lazy":
        state = _InitOnFirstUse(state, state.get_run, state.get_run)
    reader = RunStore(state)
    kwargs = {"max_nodes" if method == "get_subtree" else "limit": limit}
    if path == "tenant":
        kwargs["tenant_id"] = "tenant"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)

        def before_io():
            assert len(caught) == 1, "the first store call preceded its warning"

        underlying.before_io = before_io
        with pytest.raises(ReadAttemptError):
            line = inspect.currentframe().f_lineno + 1
            await getattr(reader, method)("root", **kwargs)
    assert len(caught) == 1
    warning = caught[0]
    assert warning.category is FutureWarning
    assert warning.filename == __file__
    assert warning.lineno == line
    message = str(warning.message)
    assert f"RunStore.{method}" in message
    assert "0.7" in message
    if method == "get_subtree":
        assert "10,000" in message and "max_nodes=" in message
    else:
        assert ("after_sequence_index=" if method == "get_events" else "offset=") in message
        assert "reads every row" in message if limit is None else "clamped to 1,000 now" in message


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("limit", [-1, 0, 1, 1000])
async def test_explicit_page_does_not_warn(method, limit):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        with pytest.raises(ReadAttemptError):
            await getattr(RunStore(ExplodingStore()), method)("root", limit=limit)
    assert not caught


@pytest.mark.parametrize("method", READS + ["get_subtree"])
@pytest.mark.parametrize("kwargs", [{"run_id": ""}, {"tenant_id": False}, {"tenant_id": ""}])
async def test_invalid_scope_validates_before_warning_or_io(method, kwargs):
    kwargs = {"run_id": "root", **kwargs}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        with pytest.raises(ValueError):
            await getattr(RunStore(ExplodingStore()), method)(**kwargs)
    assert not caught


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("value", [True, "100", 2**64])
async def test_invalid_limit_does_not_warn_or_read(method, value):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        with pytest.raises(ValueError):
            await getattr(RunStore(ExplodingStore()), method)("root", limit=value)
    assert not caught


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
async def test_invalid_filters_do_not_warn_or_read(method, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        with pytest.raises(ValueError):
            await getattr(RunStore(ExplodingStore()), method)("root", **kwargs)
    assert not caught


async def test_format_two_clamp_is_at_facade_only(monkeypatch):
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
    with pytest.warns(FutureWarning):
        assert (
            await RunStore(CheckpointStore()).get_traces(
                "root", tenant_id="tenant", limit=2000, offset=23
            )
            == []
        )
    assert seen == [
        {
            "tenant_id": "tenant",
            "limit": 1000,
            "offset": 23,
            "retry": False,
            "before_order_index": None,
        }
    ]


async def test_warning_uses_stdlib_deduplication():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", FutureWarning)
        for _ in range(2):
            with pytest.raises(ReadAttemptError):
                await RunStore(ExplodingStore()).get_events("root")
    assert len(caught) == 1


def test_asyncio_run_attribution_is_asyncio():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        with pytest.raises(ReadAttemptError):
            asyncio.run(RunStore(ExplodingStore()).get_events("root"))
    assert len(caught) == 1
    assert "asyncio" in caught[0].filename


async def test_scheduled_task_attribution_is_asyncio():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        with pytest.raises(ReadAttemptError):
            await asyncio.create_task(RunStore(ExplodingStore()).get_events("root"))
    assert len(caught) == 1
    assert "asyncio" in caught[0].filename


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
    assert len(caught) == 1
    assert caught[0].filename == __file__ and caught[0].lineno == line
