"""tools/vulture_ratchet.py: dead-code gate (vulture, min confidence 80, with a whitelist)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tools import vulture_ratchet as vr

if TYPE_CHECKING:
    from pathlib import Path

OUTPUT = """\
src/dendrux/skills/_fs.py:86: unused variable 'exc_type' (100% confidence)
src/dendrux/skills/_fs.py:88: unused variable 'tb' (100% confidence)
src/dendrux/skills/_fs.py:120: unused variable 'tb' (100% confidence)
src/dendrux/runtime/submit.py:164: unused variable 'json_payload' (100% confidence)
"""


def _in(function: str):
    return lambda path, line: function


def test_findings_are_keyed_by_function_without_line_numbers_and_counted() -> None:
    assert vr.findings_from_output(OUTPUT, lambda path, line: f"f{line}") == {
        "dendrux.runtime.submit:f164:variable:json_payload": 1,
        "dendrux.skills._fs:f120:variable:tb": 1,
        "dendrux.skills._fs:f86:variable:exc_type": 1,
        "dendrux.skills._fs:f88:variable:tb": 1,
    }


def test_every_diagnostic_kind_is_consumed() -> None:
    text = (
        "src/dendrux/m.py:5: unreachable code after 'return' (100% confidence)\n"
        "src/dendrux/m.py:9: unsatisfiable 'if' condition (100% confidence)\n"
        "src/dendrux/m.py:12: unsatisfiable 'ternary' condition (100% confidence)\n"
        "src/dendrux/m.py:14: unsatisfiable 'while' condition (100% confidence)\n"
        "src/dendrux/m.py:20: unused import 'os' (90% confidence)\n"
        "src/dendrux/m.py:21: unused function 'dead' (100% confidence)\n"
    )
    assert vr.findings_from_output(text, _in("f")) == {
        "dendrux.m:f:function:dead": 1,
        "dendrux.m:f:import:os": 1,
        "dendrux.m:f:unreachable_code:return": 1,
        "dendrux.m:f:unsatisfiable_condition:if": 1,
        "dendrux.m:f:unsatisfiable_condition:ternary": 1,
        "dendrux.m:f:unsatisfiable_condition:while": 1,
    }


def test_output_the_gate_does_not_understand_fails_loudly() -> None:
    with pytest.raises(vr.UnrecognisedOutputError, match="a brand new kind of finding"):
        vr.findings_from_output("src/dendrux/m.py:3: a brand new kind of finding (100% confidence)")
    with pytest.raises(vr.UnrecognisedOutputError):
        vr.findings_from_output("something odd with no location")


def test_moving_an_unused_name_to_another_function_is_a_new_finding() -> None:
    old = vr.findings_from_output(
        "src/dendrux/m.py:3: unused variable 'x' (100% confidence)", _in("one")
    )
    new = vr.findings_from_output(
        "src/dendrux/m.py:9: unused variable 'x' (100% confidence)", _in("two")
    )
    assert vr.compare(new, old).new == ["dendrux.m:two:variable:x"]


def test_a_new_finding_fails_and_a_known_one_passes() -> None:
    known = vr.findings_from_output(OUTPUT, _in("f"))
    assert not vr.compare(known, known).failed
    more = {**known, "dendrux.x:f:function:dead": 1}
    assert vr.compare(more, known).new == ["dendrux.x:f:function:dead"]


def test_a_second_unused_name_in_a_known_module_fails() -> None:
    known = {"dendrux.skills._fs:f:variable:tb": 2}
    more = {"dendrux.skills._fs:f:variable:tb": 3}
    assert vr.compare(more, known).worse == ["dendrux.skills._fs:f:variable:tb"]


def test_end_to_end_with_a_whitelist(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "m.py").write_text("def used(unused_arg):\n    return 2\n")
    whitelist = tmp_path / "whitelist.py"
    whitelist.write_text("# intentionally empty\n")
    baseline = tmp_path / "base.json"
    base = ["--src", str(src), "--baseline", str(baseline), "--whitelist", str(whitelist)]
    assert vr.main(["stamp", *base]) == 0
    assert vr.main(["check", *base]) == 0
    (src / "m.py").write_text("def used(unused_arg, other):\n    return 2\n")
    assert vr.main(["check", *base]) == 1
    # a whitelist entry marks the name as used on purpose
    whitelist.write_text("other\n")
    assert vr.main(["check", *base]) == 0


def test_end_to_end_unreachable_code_and_unsatisfiable_conditions_fail(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    mod = src / "m.py"
    mod.write_text("def f():\n    return 1\n")
    whitelist = tmp_path / "whitelist.py"
    whitelist.write_text("")
    baseline = tmp_path / "base.json"
    base = ["--src", str(src), "--baseline", str(baseline), "--whitelist", str(whitelist)]
    assert vr.main(["stamp", *base]) == 0
    assert vr.main(["check", *base]) == 0
    mod.write_text("def f():\n    return 1\n    print('never')\n")
    assert vr.main(["check", *base]) == 1
    mod.write_text("def f(x):\n    if False:\n        return 2\n    return x\n")
    assert vr.main(["check", *base]) == 1
    keys = vr.collect_findings(src, whitelist)
    assert any(":f:unsatisfiable_condition:if" in key for key in keys)
