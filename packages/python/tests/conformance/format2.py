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
    declared = {e["correlation_id"] for e in events if e["event_type"] == "tool.started"}
    active = set()
    completed = set()
    for event in events:
        call = event["correlation_id"]
        if event["event_type"] == "tool.started":
            assert call not in active and call not in completed
            active.add(call)
        elif event["event_type"] == "tool.completed" and call in declared:
            assert call in active
            active.remove(call)
            completed.add(call)
    assert not active


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
    if cell_id != "guardrail_block__run":
        _blocked_trace_expected(cell_id, expected_traces)
    assert actual["public_traces"] == h._norm(expected_traces), (
        cell_id,
        h.diff(h._norm(expected_traces), actual["public_traces"]),
    )


def _blocked_trace_expected(cell_id: str, traces: list) -> None:
    positions = {
        "guardrail_block__run": (0, "Mail jane.doe@example.com the report"),
        "edge_indexed_block__run": (2, '"person@example.com"'),
    }
    if cell_id not in positions:
        return
    index, original = positions[cell_id]
    assert len(traces) == 1
    trace = traces[0][index]
    assert trace["content"] == original
    trace["content"] = "[blocked: 1 finding]"


def _blocked_continuation_expected(cell_id: str, expected: dict) -> None:
    """A blocked submission now stops right after its claim, before resumed work.

    Only stored events and rows change: no resume/tool lifecycle events, no tool
    row and no tool trace. The result, error text and run record stay as before.
    """
    if cell_id != "edge_indexed_block__run":
        return
    for run in expected.get("runs", []):
        if "events" in run:
            assert [name for name, _ in run["events"][4:]] == [
                "run.resumed",
                "tools.declared",
                "tool.completed",
                "guardrail.blocked",
                "run.error",
            ]
            run["events"][4:] = [["guardrail.blocked", 0], ["run.error", 0]]
        if "trace_roles" in run:
            assert run["trace_roles"][-1] == "tool"
            run["trace_roles"].pop()
        assert [call["tool_name"] for call in run["tool_calls"]] == ["read_range"]
        run["tool_calls"] = []
    traces = expected["public_traces"][0]
    assert traces[-1]["role"] == "tool"
    assert traces[-1]["content"] in ('"person@example.com"', "[blocked: 1 finding]")
    traces.pop()
