"""Every accept and refusal branch of ``wait_bindings`` (the §4.1 stored-wait shape)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dendrux.runtime._managed_fanout_store import wait_bindings
from dendrux.runtime.state import MANAGED_BINDING_KEY

PAUSE_ID = "01HZX3K9Q8F6M2N4P5R7S9T0VW"


def _wait() -> dict[str, Any]:
    return {
        "schema_version": 4,
        "pause_reason": "managed_wait",
        "pause_id": PAUSE_ID,
        "managed_fanout": {
            "v": 1,
            "n": 2,
            "bindings": [
                {"tool_call_id": "c1", "child_run_id": "r1"},
                {"tool_call_id": "c2", "child_run_id": "r2"},
            ],
        },
        "pending_tool_calls": [{"id": "c1"}, {"id": "c2"}],
    }


def test_accepts_valid_wait_in_call_order() -> None:
    assert wait_bindings(_wait()) == (("c1", "r1"), ("c2", "r2"))


def _mutate(fn: Any) -> object:
    data = copy.deepcopy(_wait())
    fn(data)
    return data


def _set(key: str, value: object) -> Any:
    return lambda d: d.__setitem__(key, value)


def _env(key: str, value: object) -> Any:
    return lambda d: d["managed_fanout"].__setitem__(key, value)


REFUSALS = {
    "not_a_dict": "nope",
    "missing_version": _mutate(lambda d: d.pop("schema_version")),
    "bool_version": _mutate(_set("schema_version", True)),
    "wrong_version": _mutate(_set("schema_version", 3)),
    "wrong_reason": _mutate(_set("pause_reason", "other")),
    "has_binding_key": _mutate(_set(MANAGED_BINDING_KEY, {})),
    "pause_id_not_str": _mutate(_set("pause_id", 7)),
    "pause_id_not_ulid": _mutate(_set("pause_id", "not-a-ulid")),
    "envelope_missing": _mutate(lambda d: d.pop("managed_fanout")),
    "envelope_not_dict": _mutate(_set("managed_fanout", [])),
    "envelope_extra_key": _mutate(_env("extra", 1)),
    "envelope_v_bool": _mutate(_env("v", True)),
    "envelope_v_wrong": _mutate(_env("v", 2)),
    "bindings_not_list": _mutate(_env("bindings", "x")),
    "n_mismatch": _mutate(_env("n", 3)),
    "binding_not_dict": _mutate(_env("bindings", ["a", "b"])),
    "binding_bad_keys": _mutate(_env("bindings", [{"tool_call_id": "c1"}, {"x": 1}])),
    "calls_not_list": _mutate(_set("pending_tool_calls", None)),
    "calls_wrong_order": _mutate(_set("pending_tool_calls", [{"id": "c2"}, {"id": "c1"}])),
    "call_not_dict": _mutate(_set("pending_tool_calls", ["c1", "c2"])),
}


@pytest.mark.parametrize("data", REFUSALS.values(), ids=REFUSALS.keys())
def test_refuses_malformed_wait(data: object) -> None:
    with pytest.raises(ValueError, match="Not a fan-out wait"):
        wait_bindings(data)
