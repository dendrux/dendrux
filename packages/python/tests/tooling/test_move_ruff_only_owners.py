"""Proposed Ruff-owner transfer tests when no other gate has an owner key."""

from __future__ import annotations

import copy

import pytest

from tools.move_keys import GATES, RUFF_ACTION, apply_key_map

OLD = "sql.base"
NEW = "sql.scan"
RULE = "PLC0415"
OLD_KEY = f"{OLD}:{RULE}"
NEW_KEY = f"{NEW}:{RULE}"


def _inputs():
    mapping = {gate: {} for gate in GATES}
    mapping["ruff"][OLD_KEY] = {"baseline": 253, "action": RUFF_ACTION, "new_modules": [NEW]}
    return mapping, {OLD_KEY: 253}, {OLD_KEY: 245, NEW_KEY: 8}, {OLD, NEW}


def test_explicit_ruff_only_owner_conserves_the_exact_rule_cap():
    mapping, baseline, scores, present = _inputs()
    baseline["unrelated:BLE001"] = 4
    assert apply_key_map("ruff", baseline, scores, mapping, present) == {
        OLD_KEY: 245,
        NEW_KEY: 8,
        "unrelated:BLE001": 4,
    }
    assert baseline == {OLD_KEY: 253, "unrelated:BLE001": 4}


@pytest.mark.parametrize(
    "owners",
    [NEW, [], [NEW, NEW], [OLD], ["unknown"], [None], [[NEW]], ["invalid-name"]],
)
def test_explicit_owners_require_unique_present_dotted_module_names(owners):
    mapping, baseline, scores, present = _inputs()
    mapping["ruff"][OLD_KEY]["new_modules"] = owners
    present.add("invalid-name")
    with pytest.raises(ValueError):
        apply_key_map("ruff", baseline, scores, mapping, present)


@pytest.mark.parametrize(
    "case",
    ["grown", "missing", "negative", "float", "existing-cap", "wrong-baseline", "unknown-field"],
)
def test_explicit_ruff_owner_keeps_existing_strict_checks(case):
    mapping, baseline, scores, present = _inputs()
    if case == "grown":
        scores[NEW_KEY] = 9
    elif case == "missing":
        del scores[NEW_KEY]
    elif case == "negative":
        scores[NEW_KEY] = -1
    elif case == "float":
        scores[NEW_KEY] = 8.0
    elif case == "existing-cap":
        baseline[NEW_KEY] = 8
    elif case == "wrong-baseline":
        mapping["ruff"][OLD_KEY]["baseline"] = 254
    else:
        mapping["ruff"][OLD_KEY]["allow_growth"] = True
    with pytest.raises(ValueError):
        apply_key_map("ruff", baseline, scores, mapping, present)


def test_existing_inferred_owners_still_work_without_the_new_field():
    mapping, baseline, scores, present = _inputs()
    del mapping["ruff"][OLD_KEY]["new_modules"]
    mapping["size"][OLD + ":f"] = {"baseline": 90, "new": {"mixin": NEW + ":f"}}
    unchanged = copy.deepcopy(mapping)
    assert apply_key_map("ruff", baseline, scores, mapping, present) == {OLD_KEY: 245, NEW_KEY: 8}
    assert mapping == unchanged
