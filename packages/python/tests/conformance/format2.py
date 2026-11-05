"""Format-2 records plus direct assertions against the immutable format-1 oracle."""

from __future__ import annotations

from itertools import groupby

from dendrux.store import RunStore
from tests.conformance import expected_060
from tests.conformance import harness as h


async def view(recording: dict, store) -> dict:
    """Record public traces as well as the harness's physical persistence evidence."""
    actual = h.golden_view(recording)
    public = []
    reader = RunStore(store)
    for run in recording["runs"]:
        rows = await reader.get_traces(
            run["run_id"], tenant_id=run["record"]["tenant_id"], limit=1000
        )
        public.append(
            [
                {k: h.raw(getattr(row, k)) for k in ("role", "content", "order_index", "meta")}
                for row in rows
            ]
        )
        _assert_tool_lifecycle(run["events"])
    actual["public_traces"] = h._norm(public)
    for run in actual["runs"]:
        run["events"] = _event_order(run["events"])
    return actual


def _assert_tool_lifecycle(events: list[dict]) -> None:
    started = set()
    for event in events:
        if event["event_type"] == "tool.started":
            assert event["correlation_id"] not in started
            started.add(event["correlation_id"])
        elif event["event_type"] == "tool.completed" and event["correlation_id"] in started:
            started.remove(event["correlation_id"])


def _event_order(events: list[list]) -> list[list]:
    result = []
    for key, group in groupby(
        events, key=lambda e: (e[0] in ("tool.started", "tool.completed"), e[1])
    ):
        block = list(group)
        result.extend(sorted(block, reverse=True) if key[0] else block)
    return result


def assert_shared_contract(cell_id: str, actual: dict) -> None:
    """Agreed fields compare to format 1 directly, never to a freshly recorded baseline."""
    legacy = expected_060.golden(cell_id)
    assert h._norm(actual["steps"]) == h._norm(legacy["steps"])
    assert actual["provider_calls"] == legacy["provider_calls"]
    assert len(actual["runs"]) == len(legacy["runs"])
    for current, previous in zip(actual["runs"], legacy["runs"], strict=True):
        for key in ("record", "tool_calls", "notes", "paused_with_state"):
            assert h._norm(current[key]) == h._norm(previous[key]), (cell_id, key)
        events = [
            [kind, 0 if kind in ("run.completed", "run.error") else iteration]
            for kind, iteration in current["events"]
            if kind != "tool.started"
        ]
        assert events == previous["events"], (cell_id, events, previous["events"])
    corpus = expected_060.corpus(cell_id)
    expected_traces = []
    for run in corpus["runs"]:
        expected_traces.append(
            [
                {k: row[k] for k in ("role", "content", "order_index", "meta")}
                for row in corpus["rows"]["react_traces"]
                if row["agent_run_id"] == run["run_id"]
            ]
        )
    if cell_id.startswith("pii_restore__"):
        for traces, run in zip(expected_traces, corpus["rows"]["agent_runs"], strict=True):
            mapping = run.get("pii_mapping") or {}
            for trace in traces:
                for placeholder, original in mapping.items():
                    if isinstance(original, str):
                        trace["content"] = trace["content"].replace(original, placeholder)
    assert actual["public_traces"] == h._norm(expected_traces), (
        cell_id,
        h.diff(h._norm(expected_traces), actual["public_traces"]),
    )
