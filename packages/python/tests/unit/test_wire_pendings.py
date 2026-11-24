"""The v1 browser DTO lists every pause as ``pendings`` (additive; ``pending`` is the first)."""

import json

from dendrux.store import PendingPause, PendingToolCall
from dendrux.types import RunResult, RunStatus

ROOT = "00000000000000000000000root"
SENTINEL = "RAW_PII_SENTINEL@example.test"


def _leaf(pause_id, name, path, origin, *, target="client", approval=None, params=None):
    return PendingPause(
        run_id=ROOT,
        status="paused",
        kind="paused",
        pause_id=pause_id,
        tool_calls=[
            PendingToolCall(
                id=f"c-{pause_id}",
                name="send",
                target=target,
                params=params if params is not None else {"to": "<<EMAIL_1>>"},
                approval=approval,
            )
        ],
        agent_name=name,
        agent_path=path,
        origin_run_id=origin,
    )


def _fanout(**kwargs):
    leaves = (
        _leaf("p1", "a", ["root", "a"], "run-a", **kwargs),
        _leaf("p2", "b", ["root", "b"], "run-b", **kwargs),
    )
    return RunResult(
        run_id=ROOT,
        status=RunStatus.PAUSED,
        pending=leaves[0],
        pendings=leaves,
        pause_id="p1",
    )


def test_fanout_pendings_keep_order_and_origin_fields():
    wire = _fanout().to_dict()
    assert [p["pause_id"] for p in wire["pendings"]] == ["p1", "p2"]
    assert [p["agent_name"] for p in wire["pendings"]] == ["a", "b"]
    assert [p["agent_path"] for p in wire["pendings"]] == [["root", "a"], ["root", "b"]]
    assert [p["origin_run_id"] for p in wire["pendings"]] == ["run-a", "run-b"]
    assert all(p["run_id"] == ROOT for p in wire["pendings"])
    assert wire["pending"] == wire["pendings"][0]
    assert json.loads(json.dumps(wire)) == wire


def test_non_paused_result_has_empty_pendings():
    wire = RunResult(run_id=ROOT, status=RunStatus.SUCCESS, answer="ok").to_dict()
    assert wire["pendings"] == []
    assert wire["pending"] is None


def test_old_single_pause_result_gives_one_item_list():
    pending = _leaf("p1", None, None, None)
    result = RunResult(run_id=ROOT, status=RunStatus.PAUSED, pending=pending)
    wire = result.to_dict()
    assert len(wire["pendings"]) == 1
    assert wire["pendings"][0] == wire["pending"]


def test_every_entry_hides_server_and_gated_arguments():
    for kwargs in ({"target": "server"}, {"approval": {"source": "static"}}):
        wire = _fanout(params={"secret": SENTINEL}, **kwargs).to_dict()
        assert SENTINEL not in json.dumps(wire)
        for entry in wire["pendings"]:
            call = entry["pending_tool_calls"][0]
            assert call["params"] is None and call["args_hidden"] is True


def test_every_entry_uses_pii_placeholders():
    result = _fanout(params={"to": SENTINEL})
    result.meta["pii_mapping"] = {"<<EMAIL_1>>": SENTINEL}
    wire = result.to_dict()
    assert SENTINEL not in json.dumps(wire["pendings"])
    assert SENTINEL not in json.dumps(wire["pending"])
