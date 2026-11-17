"""Allowances follow exact lexical owners and cannot be grown or duplicated."""

from __future__ import annotations

import pytest

from tools.move_keys import apply_key_map


@pytest.mark.parametrize(
    "gate,score", [("size", 90), ("cognitive", 19), ("vulture", 1), ("semgrep", 1)]
)
def test_exact_one_to_one_conserved_transfer(gate: str, score: int) -> None:
    old, new = (
        ("a:f", "b:Mixin.f")
        if gate not in {"semgrep", "vulture"}
        else ("rule|a|f", "rule|b|Mixin.f")
        if gate == "semgrep"
        else ("a:f:method:f", "b:Mixin.f:method:f")
    )
    mapping = {gate: {old: {"baseline": score, "new": {"mixin": new}}}}
    assert apply_key_map(gate, {old: score, "unrelated": 4}, {new: score}, mapping, {new}) == {
        new: score,
        "unrelated": 4,
    }
    for scores, present, base in [
        ({new: score + 1}, {new}, {old: score}),
        ({new: score}, {old, new}, {old: score}),
        ({new: score}, {new}, {old: score, new: score}),
        ({new: score}, {new}, {old: score + 1}),
    ]:
        with pytest.raises(ValueError):
            apply_key_map(gate, base, scores, mapping, present)


def test_new_files_never_receive_size_allowances_and_small_functions_drop_debt() -> None:
    mapping = {"size": {"a:f": {"baseline": 90, "new": {"mixin": "b:f"}}}}
    assert apply_key_map("size", {"a:f": 90}, {"b:f": 80}, mapping, {"b:f"}) == {}
    with pytest.raises(ValueError):
        apply_key_map(
            "size",
            {"a": 1001},
            {"b": 1001},
            {"size": {"a": {"baseline": 1001, "new": {"mixin": "b"}}}},
            {"b"},
        )


def test_two_sources_cannot_claim_one_destination() -> None:
    rows = {x: {"baseline": 90, "new": {"mixin": "b:f"}} for x in ["a:f", "a:g"]}
    with pytest.raises(ValueError):
        apply_key_map("size", {"a:f": 90, "a:g": 90}, {"b:f": 90}, {"size": rows}, {"b:f"})


def test_crap_transfer_preserves_caps_and_uses_fresh_lexical_digest() -> None:
    row = {"complexity": 8, "crap": 10.0, "coverage": 0.9, "digest": "old"}
    fresh = dict(row, digest="new", crap=9.0)
    mapping = {"crap": {"src/a.py::f": {"baseline": row, "new": {"mixin": "src/b.py::Mixin.f"}}}}
    assert apply_key_map(
        "crap", {"src/a.py::f": row}, {"src/b.py::Mixin.f": fresh}, mapping, {"src/b.py::Mixin.f"}
    ) == {"src/b.py::Mixin.f": fresh}
    for key in ["crap", "complexity"]:
        bad = dict(fresh)
        bad[key] = row[key] + 1
        with pytest.raises(ValueError):
            apply_key_map(
                "crap",
                {"src/a.py::f": row},
                {"src/b.py::Mixin.f": bad},
                mapping,
                {"src/b.py::Mixin.f"},
            )


def test_ruff_conserves_per_rule_across_residual_and_created_owners() -> None:
    mapping = {
        "size": {"a:f": {"baseline": 90, "new": {"mixin": "b:f"}}},
        "ruff": {
            "a:PLR0912": {
                "baseline": 3,
                ("action"): (
                    "conserve per rule by re-running Ruff on before and after sources;"
                    " no per-function key"
                ),
            }
        },
    }
    assert apply_key_map(
        "ruff", {"a:PLR0912": 3}, {"a:PLR0912": 1, "b:PLR0912": 2}, mapping, {"a", "b"}
    ) == {"a:PLR0912": 1, "b:PLR0912": 2}
    with pytest.raises(ValueError):
        apply_key_map(
            "ruff", {"a:PLR0912": 3}, {"a:PLR0912": 2, "b:PLR0912": 2}, mapping, {"a", "b"}
        )


def test_empty_mapping_does_not_create_vulture_or_semgrep_allowances() -> None:
    for gate in ["vulture", "semgrep"]:
        assert apply_key_map(gate, {}, {"new": 10}, {gate: {}}, {"new"}) == {}


def test_semgrep_exemptions_follow_exact_function_owners() -> None:
    from tools.move_keys import exemption_transfers

    mapping = {
        "crap": {
            "src/dendrux/runtime/stores/sql/base.py::SQLStateStore._write_tx": {
                "baseline": {},
                "new": {
                    "mixin": "src/dendrux/runtime/stores/sql/_sql_writes.py::_SqlWrites._write_tx"
                },
            }
        }
    }
    old = "dendrux.runtime.stores.sql.base:SQLStateStore._write_tx"
    assert exemption_transfers(mapping, {old}) == {
        old: "dendrux.runtime.stores.sql._sql_writes:_SqlWrites._write_tx"
    }
    with pytest.raises(ValueError, match="exact"):
        exemption_transfers(mapping, {"dendrux.runtime.runner:unknown"})


def test_ruff_includes_small_owners_from_the_crap_map() -> None:
    mapping = {
        "crap": {"src/a.py::f": {"baseline": {}, "new": {"mixin": "src/b.py::f"}}},
        "ruff": {
            "a:PLC0415": {
                "baseline": 1,
                ("action"): (
                    "conserve per rule by re-running Ruff on before and after sources;"
                    " no per-function key"
                ),
            }
        },
    }
    assert apply_key_map(
        "ruff", {"a:PLC0415": 1}, {"a:PLC0415": 0, "b:PLC0415": 1}, mapping, {"b"}
    ) == {"b:PLC0415": 1}


def test_non_finite_crap_allowance_is_refused() -> None:
    row = {"complexity": 1, "crap": 1.0, "coverage": 1.0, "digest": "old"}
    mapping = {"crap": {"src/a.py::f": {"baseline": row, "new": {"mixin": "src/b.py::f"}}}}
    with pytest.raises(ValueError):
        apply_key_map(
            "crap",
            {"src/a.py::f": row},
            {"src/b.py::f": dict(row, crap=float("nan"))},
            mapping,
            {"src/b.py::f"},
        )


def test_key_proposal_cli_prints_without_writing_a_baseline(tmp_path, monkeypatch, capsys) -> None:
    import json
    import sys

    from tools.move_keys import main

    data = {
        "baseline": {"a:f": 90},
        "scores": {"b:f": 90},
        "present": ["b:f"],
        "map": {"size": {"a:f": {"baseline": 90, "new": {"mixin": "b:f"}}}},
    }
    args = ["move_keys.py", "size"]
    for name, value in data.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        args.extend(["--" + name, str(path)])
    original = (tmp_path / "baseline.json").read_text()
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    assert json.loads(capsys.readouterr().out) == {"b:f": 90}
    assert (tmp_path / "baseline.json").read_text() == original
