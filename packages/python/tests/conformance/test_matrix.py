"""The scenario matrix itself: every public entry path and every required scenario is covered."""

from __future__ import annotations

from tests.conformance.scenarios import CELLS

ENTRY_PATHS = {
    "run",
    "stream",
    "resume",
    "resume_stream",
    "submit_tool_results",
    "submit_input",
    "submit_approval",
    "cancel_run",
    "retry",
    "sweep",
    "submit_resume",
    "submit_note",
}
SCENARIOS = {
    "plain_answer",
    "server_tool",
    "parallel_tools",
    "client_tool",
    "human_input",
    "approval",
    "budget_unpriced",
    "guardrail_block",
    "delegation",
    "images",
    "pii_restore",
    "replay",
    "compaction_summary",
    "context_blocks",
    "seeded_history",
    "malformed_tool_args",
    "malformed_tool_json",
    "cancellation",
    "max_iterations",
    "retry",
    "sweep",
    "governance",
    "run_control",
}


def test_every_public_entry_path_is_exercised() -> None:
    declared = set().union(*(set(c.paths) for c in CELLS))
    assert declared >= ENTRY_PATHS, sorted(ENTRY_PATHS - declared)


def test_every_required_scenario_has_cells() -> None:
    present = {c.scenario for c in CELLS}
    assert present >= SCENARIOS, sorted(SCENARIOS - present)


def test_each_continuation_path_runs_after_both_run_and_stream_starts() -> None:
    cells = {c.id for c in CELLS}
    for start in ("run", "stream"):
        for cont in ("resume", "resume_stream", "submit_tool_results"):
            assert f"client_tool__{start}_{cont}" in cells


async def test_declared_paths_are_what_the_recording_shows() -> None:
    """A cell's ``paths`` must match the entry paths its recording actually used."""
    from tests._fixtures.corpus import load_cell

    for c in CELLS:
        recorded = load_cell(c.id).entry_paths
        assert set(c.paths) == recorded, (c.id, sorted(c.paths), sorted(recorded))
