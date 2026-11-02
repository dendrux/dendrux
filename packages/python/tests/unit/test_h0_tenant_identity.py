"""Public contracts select tenant scope or create identity; they cannot reassign it."""

import inspect

from dendrux import Agent
from dendrux.runtime.state import StateStore
from dendrux.store import RunStore


def test_public_run_apis_expose_no_tenant_reassignment_slot():
    # Enumerate reviewed tenant-bearing methods so a new setter, even one named
    # generically (e.g. update_run), requires an explicit contract review.
    scoped = {
        Agent: {
            "cancel_run",
            "get_result",
            "reclaim",
            "request_pause",
            "resume",
            "resume_stream",
            "submit_approval",
            "submit_input",
            "submit_note",
            "submit_resume",
            "submit_tool_results",
        },
        RunStore: {
            "count_runs",
            "follow",
            "get_event_view",
            "get_events",
            "get_llm_calls",
            "get_notes",
            "get_pause_page",
            "get_pauses",
            "get_pending",
            "get_pii_mapping",
            "get_run",
            "get_subtree",
            "get_tool_invocations",
            "get_traces",
            "list_runs",
            "read_image",
            "stream_events",
        },
        StateStore: {
            "add_note",
            "conversation_traces",
            "count_runs",
            "get_children_page",
            "get_notes",
            "get_replayed_tool_calls",
            "get_subtree",
            "get_trace",
            "get_traces_after",
            "list_runs",
            "mark_cancel_requested",
            "reclaim_run",
        },
    }
    creates = {Agent: {"run", "stream", "retry"}, RunStore: set(), StateStore: {"create_run"}}
    for owner in (Agent, RunStore, StateStore):
        found = set()
        for name, method in inspect.getmembers(owner, inspect.isfunction):
            if name.startswith("_"):
                continue
            params = inspect.signature(method).parameters
            tenant_slots = {key for key in params if "tenant" in key}
            if not tenant_slots:
                continue
            assert tenant_slots == {"tenant_id"}, (owner.__name__, name, tenant_slots)
            found.add(name)
            if name not in creates[owner] and (owner, name) != (Agent, "get_result"):
                assert "TenantScope" in str(params["tenant_id"].annotation), (owner.__name__, name)
        assert found == scoped[owner] | creates[owner], (owner.__name__, found)
