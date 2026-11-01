"""tools/vulture_ratchet.py: dead-code gate (vulture, min confidence 80, with a whitelist)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools import vulture_ratchet as vr

if TYPE_CHECKING:
    from pathlib import Path

OUTPUT = """\
src/dendrux/skills/_fs.py:86: unused variable 'exc_type' (100% confidence)
src/dendrux/skills/_fs.py:88: unused variable 'tb' (100% confidence)
src/dendrux/skills/_fs.py:120: unused variable 'tb' (100% confidence)
src/dendrux/runtime/submit.py:164: unused variable 'json_payload' (100% confidence)
"""


def test_findings_are_keyed_without_line_numbers_and_counted() -> None:
    assert vr.findings_from_output(OUTPUT) == {
        "dendrux.runtime.submit:variable:json_payload": 1,
        "dendrux.skills._fs:variable:exc_type": 1,
        "dendrux.skills._fs:variable:tb": 2,
    }


def test_a_new_finding_fails_and_a_known_one_passes() -> None:
    known = vr.findings_from_output(OUTPUT)
    assert not vr.compare(known, known).failed
    more = {**known, "dendrux.x:function:dead": 1}
    assert vr.compare(more, known).new == ["dendrux.x:function:dead"]


def test_a_second_unused_name_in_a_known_module_fails() -> None:
    known = vr.findings_from_output(OUTPUT)
    more = {**known, "dendrux.skills._fs:variable:tb": 3}
    assert vr.compare(more, known).worse == ["dendrux.skills._fs:variable:tb"]


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
