"""The recorded 0.5.0 corpus (``tests/_fixtures/corpus_050``) and its loader.

Checks the corpus is complete and untouched (manifest hashes) and the structural facts 0.6
builds on: in 0.5 the persisted traces explain the provider messages of every unseeded run,
and fail to explain exactly the seeded kinds (``history=``, context blocks, replay, retry,
PII placeholders), which 0.6 closes with ``seed`` traces.
A fresh run of every flow is compared with the corpus in ``test_conformance.py``, on the same
run as the golden (so the oracle cannot drift from the code that is supposed to match it).
"""

from __future__ import annotations

import hashlib

from tests._fixtures import corpus as corpus_mod
from tests._fixtures.corpus import load_corpus, load_manifest
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS

CELL_IDS = [c.id for c in CELLS]

#: Scenarios whose provider messages carry material the 0.5 traces do not hold.
SEEDED_SCENARIOS = {
    "pii_restore",  # placeholders on the wire, raw text in the trace
    "seeded_history",
    "context_blocks",
    "compaction_summary",
    "replay",
    "retry",
}

#: What the corpus must cover (the 0.6 rebuild oracle's required inputs).
REQUIRED_COVERAGE = {
    "seeded history=": {"seeded_history__run", "seeded_history__client_tool_resume"},
    "context blocks": {"context_blocks__run", "context_blocks__no_history"},
    "replay-injected turns": {"replay__run_with_runs", "replay__stream_with_runs"},
    "images (blob refs)": {"images__run", "images__client_result_resume"},
    "PII placeholders": {"pii_restore__run", "pii_restore__client_tool_resume"},
    "unpriced budgets": {
        "budget_unpriced__run_resume",
        "budget_unpriced__stream_resume_stream_repauses",
    },
    "malformed calls": {"malformed_tool_args__run", "malformed_tool_json__run"},
    "compaction (context-block recipe)": {"compaction_summary__run"},
}
REQUIRED_CONTINUATIONS = {
    "resume",
    "resume_stream",
    "submit_tool_results",
    "submit_input",
    "submit_approval",
    "retry",
}


def test_manifest_indexes_every_file_and_no_other() -> None:
    manifest = load_manifest()
    assert manifest["dendrux_version"] == h.RECORDED_VERSION
    listed = {e["file"] for e in manifest["cells"]}
    present = {p.name for p in corpus_mod.CORPUS_DIR.glob("*.json")} - {"manifest.json"}
    assert listed == present
    assert [e["id"] for e in manifest["cells"]] == CELL_IDS
    for entry in manifest["cells"]:
        body = (corpus_mod.CORPUS_DIR / entry["file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"], f"{entry['id']} was modified"


def test_loader_returns_every_recorded_flow_with_its_counts() -> None:
    manifest = {e["id"]: e for e in load_manifest()["cells"]}
    runs = load_corpus()
    assert [r.id for r in runs] == CELL_IDS
    for run in runs:
        entry = manifest[run.id]
        assert len(run.calls) == entry["provider_calls"]
        assert len(run.steps) == entry["steps"]
        assert len(run.runs) == entry["runs"]
        assert run.dendrux_version == h.RECORDED_VERSION
        assert [c.index for c in run.calls] == list(range(len(run.calls)))


def test_corpus_covers_the_required_inputs() -> None:
    ids = set(CELL_IDS)
    for what, cells in REQUIRED_COVERAGE.items():
        assert cells <= ids, f"corpus is missing {what}: {sorted(cells - ids)}"
    seen = set().union(*(r.entry_paths for r in load_corpus()))
    assert seen >= REQUIRED_CONTINUATIONS, sorted(REQUIRED_CONTINUATIONS - seen)


def test_corpus_has_image_blob_refs_pii_placeholders_and_unpriced_runs() -> None:
    by_id = {r.id: r for r in load_corpus()}
    image_call = by_id["images__run"].calls[0]
    parts = [p for m in image_call.messages for p in (m["parts"] or []) if p["type"] == "image"]
    assert parts and parts[0]["ref"], "image parts are recorded as blob refs"
    assert by_id["images__run"].rows["dendrux_blobs"], "the blob rows are recorded"
    pii = by_id["pii_restore__run"].calls[0]
    assert any("<<EMAIL_ADDRESS_1>>" in m["content"] for m in pii.messages)
    assert not any("jane.doe@example.com" in m["content"] for m in pii.messages)
    unpriced = by_id["budget_unpriced__run_resume"]
    assert unpriced.runs[0]["status"] == "success"
    assert any(s.status == "paused" and s.at_pause for s in unpriced.steps)


def test_every_pause_records_its_pause_data_and_traces() -> None:
    for run in load_corpus():
        for step in run.steps:
            if step.status in h.PAUSE_STATUSES:
                assert step.at_pause is not None, f"{run.id}: {step.path} paused with no snapshot"
                assert step.at_pause["pause_state"], f"{run.id}: empty pause data"
                assert step.at_pause["traces"], f"{run.id}: no traces at pause"


def test_traces_explain_provider_messages_exactly_for_unseeded_runs() -> None:
    """0.5: for runs without seed material the last call's messages are the traces' prefix."""
    explained, unexplained = set(), set()
    for run in load_corpus():
        for r in run.runs:
            calls = run.calls_for(r["run_id"])
            if not calls:
                continue
            sent = [(m["role"], m["content"]) for m in calls[-1].messages if m["role"] != "system"]
            traces = [(t["role"], t["content"]) for t in run.traces(r["run_id"])]
            (explained if sent == traces[: len(sent)] else unexplained).add(run.scenario)
    assert unexplained <= SEEDED_SCENARIOS, sorted(unexplained - SEEDED_SCENARIOS)
    assert unexplained | explained >= SEEDED_SCENARIOS
    assert {"plain_answer", "client_tool", "approval", "human_input"} <= explained


def test_regeneration_is_refused_off_a_pristine_050_tree(monkeypatch) -> None:
    monkeypatch.setattr(h, "package_version", lambda: "0.6.0")
    assert "0.6.0" in (h.regen_refusal() or "")
    monkeypatch.setattr(h, "package_version", lambda: h.RECORDED_VERSION)
    monkeypatch.setattr(h, "RECORDED_TAG", "v0.0.0-missing-tag")
    assert h.regen_refusal() is not None
