"""Loader for the 0.5.0 recorded corpus (``tests/_fixtures/corpus_050``).

The corpus is the oracle for the 0.6 history rebuild: for each recorded scenario flow it holds
the exact message list a scripted provider received on every call (captured at the provider
boundary), the persisted rows after the flow (agent runs with pause data and PII mapping,
traces, tool calls, run events, blobs) and, for every pause, the pause data and traces as
they were when the run paused. Ids and timestamps are the real recorded values; compare
through ``tests.conformance.harness.normalised_corpus`` (ULIDs numbered by first appearance,
timestamps masked).

Recorded once from the 0.5.0 sources by ``python -m tests.conformance.regen`` (refused on any
other tree). ``load_corpus()`` returns every recorded flow; ``manifest.json`` indexes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_050"
MANIFEST = CORPUS_DIR / "manifest.json"


@dataclass(frozen=True)
class CorpusCall:
    """One provider call: what the loop handed the (scripted) provider."""

    index: int
    provider: str
    provider_call: int
    step: int  # index into CorpusRun.steps of the entry-path call that caused it
    run_id: str | None
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    options: dict[str, Any]


@dataclass(frozen=True)
class CorpusStep:
    """One public entry-path call of the flow."""

    path: str
    label: str
    run_id: str | None
    status: str | None
    raised: list[str] | None  # exception class chain, never the message
    at_pause: dict[str, Any] | None  # {"run_id", "pause_state", "traces"} when it paused


@dataclass(frozen=True)
class CorpusRun:
    """One recorded scenario flow."""

    id: str
    scenario: str
    dendrux_version: str
    agents: list[dict[str, Any]]  # {"name", "on_governance_change" ("warn"|"raise"), ...}
    steps: list[CorpusStep]
    calls: list[CorpusCall]
    runs: list[dict[str, Any]]  # {"run_id", "parent_run_id", "status", "pause_state"}
    rows: dict[str, list[dict[str, Any]]]  # table name -> rows

    @property
    def entry_paths(self) -> set[str]:
        return {s.path for s in self.steps}

    def calls_for(self, run_id: str) -> list[CorpusCall]:
        return [c for c in self.calls if c.run_id == run_id]

    def traces(self, run_id: str) -> list[dict[str, Any]]:
        rows = [r for r in self.rows["react_traces"] if r["agent_run_id"] == run_id]
        return sorted(rows, key=lambda r: r["order_index"])


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def load_cell(cell_id: str) -> CorpusRun:
    data = json.loads((CORPUS_DIR / f"{cell_id}.json").read_text(encoding="utf-8"))
    return CorpusRun(
        id=data["id"],
        scenario=data["scenario"],
        dendrux_version=data["dendrux_version"],
        agents=data["agents"],
        steps=[CorpusStep(**s) for s in data["steps"]],
        calls=[CorpusCall(**c) for c in data["calls"]],
        runs=data["runs"],
        rows=data["rows"],
    )


def load_corpus() -> list[CorpusRun]:
    """Every recorded flow, in manifest order."""
    return [load_cell(entry["id"]) for entry in load_manifest()["cells"]]
