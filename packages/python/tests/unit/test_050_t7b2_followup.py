"""Public pending type identity survives internal implementation moves."""

import importlib
import pickle

import pytest

from dendrux.store import PendingPause, PendingToolCall


@pytest.mark.parametrize("kind", [PendingPause, PendingToolCall])
def test_pending_types_report_and_pickle_the_public_module(kind):
    assert kind.__module__ == "dendrux.store", "public pending types must name their public module"
    assert getattr(importlib.import_module(kind.__module__), kind.__name__) is kind
    item = (
        kind(run_id="r", status="paused", kind="paused", pause_id="p")
        if kind is PendingPause
        else kind(id="c", name="lookup", target="client", params={"query": "x"})
    )
    assert pickle.loads(pickle.dumps(item)) == item
