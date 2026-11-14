"""Review regressions: imports are executable evidence, never an open waiver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.tooling.test_move_manifest import manifest, proof
from tools.move_check import check_move
from tools.move_keys import RUFF_ACTION, apply_key_map


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize("location", ["old", "new"])
@pytest.mark.parametrize("statement", ["from evil import *", "import evil_side_effect"])
def test_unexplained_imports_cannot_rebind_or_execute(
    manifest_mode: bool, location: str, statement: str
) -> None:
    old = "import os\ndef f(): return os.getcwd()\n"
    after = {"old": "from new import f\n", "new": old}
    after[location] = statement + "\n" + after[location]
    report = check_move(
        old, after, move_manifest=manifest(old, after, {"f": "new"}) if manifest_mode else None
    )
    assert not report.ok
    assert any(
        ("star-import" if "*" in statement else "extra-import") in p and statement in p
        for p in report.problems
    ), report.problems


def test_stray_import_next_to_an_allowed_hoist_is_refused() -> None:
    old = "def f():\n import math\n return math.pi\n"
    after = {
        "old": "from new import f\n",
        "new": "import math\nimport sys\ndef f():\n return math.pi\n",
    }
    report = proof(
        old,
        after,
        {"f": "new"},
        hoists=[{"root": "f", "module": "new", "statement": "import math", "comment": ""}],
    )
    assert not report.ok
    assert any("extra-import" in p and "import sys" in p for p in report.problems), report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_type_checking_does_not_make_a_star_import_explainable(manifest_mode: bool) -> None:
    old = "from typing import TYPE_CHECKING\ndef f(): return 1\n"
    after = {
        "old": "from new import f\n",
        "new": (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            " from evil import *\ndef f(): return 1\n"
        ),
    }
    report = check_move(
        old, after, move_manifest=manifest(old, after, {"f": "new"}) if manifest_mode else None
    )
    assert not report.ok
    assert any("star-import" in p for p in report.problems), report.problems


@pytest.mark.parametrize(
    "old,seams",
    [
        ("def f(): return 1\n", {"zz": "dep.zz"}),
        ("from dep import hook\ndef f(): return hook()\n", {"hook": "dep.hook"}),
    ],
)
def test_unused_seam_requires_an_original_import_and_a_lexical_consumer(
    old: str, seams: dict
) -> None:
    report = proof(old, {"old": old}, {"f": "old"}, seams=seams)
    assert not report.ok
    assert any("unused seam" in p for p in report.problems), report.problems


def test_metadata_pin_must_follow_its_class() -> None:
    old = "class Record: pass\n"
    after = {"old": "from new import Record\n", "new": 'Record.__module__ = "old"\n' + old}
    report = proof(
        old,
        after,
        {"Record": "new"},
        metadata=[{"root": "Record", "module": "new", "attribute": "__module__", "value": "old"}],
    )
    assert not report.ok
    assert any("metadata" in p and ("before" in p or "follow" in p) for p in report.problems), (
        report.problems
    )


@pytest.mark.parametrize(
    "field,value", [("root", "ghost"), ("module", "missing"), ("name", "unowned")]
)
def test_unknown_lexical_references_name_the_bad_entry(field: str, value: str) -> None:
    old = "def f(): return 1\n"
    after = {"old": "from new import f\n", "new": old}
    entry = {"root": "f", "module": "new", "name": "f", "target": "new.f", "count": 1}
    entry[field] = value
    report = proof(old, after, {"f": "new"}, lexical=[entry])
    assert not report.ok
    assert any("lexical" in p and value in p for p in report.problems), report.problems


def test_detect_job_pins_the_python_interpreter() -> None:
    path = Path(__file__).resolve().parents[4] / ".github/workflows/ci.yml"
    steps = yaml.safe_load(path.read_text())["jobs"]["move-manifests"]["steps"]
    setup = next((s for s in steps if s.get("uses", "").startswith("actions/setup-python@")), None)
    assert setup is not None
    assert setup["with"]["python-version"] == "3.11"


@pytest.mark.parametrize("missing", ["a:PLC0415", "b:PLC0415"])
def test_missing_claimed_ruff_score_is_not_zero(missing: str) -> None:
    mapping = {
        "size": {"a:f": {"baseline": 90, "new": {"mixin": "b:f"}}},
        "ruff": {"a:PLC0415": {"baseline": 2, "action": RUFF_ACTION}},
    }
    scores = {"a:PLC0415": 1, "b:PLC0415": 1}
    del scores[missing]
    with pytest.raises(ValueError, match="missing.*" + missing):
        apply_key_map("ruff", {"a:PLC0415": 2}, scores, mapping, {"a", "b"})


@pytest.mark.parametrize("gate", ["cli", "ci"])
def test_supported_gates_verify_provider_snapshots_before_comparison(
    tmp_path: Path, monkeypatch, gate: str
) -> None:
    from tests.tooling.test_move_ci import git, repository
    from tools import move_check, move_ci
    from tools._move_manifest import digest

    root, data = repository(tmp_path)
    source = "class Parent:\n def read(self): return 1\n"
    path = root / "src/dependency.py"
    path.parent.mkdir()
    path.write_text(source)
    # Keep the proof base's old module intact while adding real provider evidence.
    (root / "old.py").write_text("def f(): return 1\n")
    git(root, "add", "old.py", "src/dependency.py")
    git(root, "commit", "-qm", "Fixture provider base")
    data["base"] = git(root, "rev-parse", "HEAD")
    (root / "old.py").write_text("from new import f\n")
    data["providers"] = {
        "dependency.Parent": {
            "path": "src/dependency.py",
            "sha256": digest(source),
            "class": source.replace("return 1", "return 2"),
        }
    }
    (root / "tools/moves/fixture.json").write_text(json.dumps(data))
    monkeypatch.chdir(root)

    def forbidden_comparison(*args, **kwargs):
        pytest.fail("unverified snapshot reached check_move")

    if gate == "cli":
        monkeypatch.setattr(move_check, "check_move", forbidden_comparison)
        result = move_check.main(
            [
                "--base",
                data["base"],
                "--old",
                "old.py",
                "--new",
                "new.py",
                "--move-manifest",
                "tools/moves/fixture.json",
            ]
        )
    else:
        monkeypatch.setattr(move_ci, "check_move", forbidden_comparison)
        result = move_ci.run_manifests(Path("tools/moves"))
    assert result == 1


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_original_bindings_and_type_only_imports_still_pass(manifest_mode: bool) -> None:
    old = (
        "import os\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
        " from annotation_only import Shape\ndef f(): return os.getcwd()\n"
    )
    after = {
        "old": "from new import f\n",
        "new": (
            "import os\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            " from annotation_only import Shape\ndef f(): return os.getcwd()\n"
        ),
    }
    report = check_move(
        old, after, move_manifest=manifest(old, after, {"f": "new"}) if manifest_mode else None
    )
    assert report.ok, report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_original_dotted_import_must_recreate_the_same_binding(manifest_mode: bool) -> None:
    old = "import os.path\ndef f(): return os.path.sep\n"
    after = {
        "old": "from new import f\n",
        "new": old.replace("import os.path", "import os.path as os"),
    }
    report = check_move(
        old, after, move_manifest=manifest(old, after, {"f": "new"}) if manifest_mode else None
    )
    assert not report.ok
    assert any("extra-import" in p and "import os.path as os" in p for p in report.problems), (
        report.problems
    )


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_owner_self_import_is_not_a_facade_reexport(manifest_mode: bool) -> None:
    old = "def f(): return 1\n"
    after = {"old": "from new import f\n", "new": "from new import f\n" + old}
    report = check_move(
        old, after, move_manifest=manifest(old, after, {"f": "new"}) if manifest_mode else None
    )
    assert not report.ok
    assert any("extra-import" in p and "from new import f" in p for p in report.problems), (
        report.problems
    )
