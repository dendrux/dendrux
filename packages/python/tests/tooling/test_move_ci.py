"""The CI entry point consumes git-tracked, hash-pinned cumulative evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.tooling.test_move_manifest import manifest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/move_ci.py"


def run(cwd: Path, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, text=True, capture_output=True, check=False
    )


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    ).stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "pkg"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    (root / "old.py").write_text("def f(): return 1\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Fixture original")
    base = git(root, "rev-parse", "HEAD")
    old = (root / "old.py").read_text()
    after = {"old.py": "from new import f\n", "new.py": old}
    for name, source in after.items():
        (root / name).write_text(source)
    evidence = manifest(old, after, {"f": "new.py"}, base=base)
    evidence["old"] = "old.py"
    (root / "tools/moves").mkdir(parents=True)
    (root / "tools/moves/fixture.json").write_text(json.dumps(evidence))
    expected = {"old": {"f": "new"}}
    for kind in ["emitted", "expanded"]:
        (root / f"tools/moves/fixture-{kind}.json").write_text(json.dumps(expected))
    git(root, "add", ".")
    git(root, "commit", "-qm", "Fixture move")
    return root, evidence


def test_absent_manifest_is_an_error(tmp_path: Path) -> None:
    assert run(tmp_path).returncode == 1


def test_valid_tracked_manifest_runs_from_actual_base(tmp_path: Path) -> None:
    root, _ = repository(tmp_path)
    done = run(root)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    "damage", ["source", "hash", "base", "missing-map", "wrong-map", "untracked", "missing-old"]
)
def test_ci_refuses_stale_or_incomplete_evidence(tmp_path: Path, damage: str) -> None:
    root, evidence = repository(tmp_path)
    if damage == "source":
        (root / "new.py").write_text("def f(): return 2\n")
    elif damage == "hash":
        evidence["source_sha256"] = "0" * 64
    elif damage == "base":
        evidence["base"] = "HEAD"
    elif damage == "missing-map":
        (root / "tools/moves/fixture-expanded.json").unlink()
    elif damage == "wrong-map":
        (root / "tools/moves/fixture-emitted.json").write_text("{}")
    elif damage == "untracked":
        git(root, "rm", "--cached", "new.py")
    elif damage == "missing-old":
        (root / "old.py").unlink()
    (root / "tools/moves/fixture.json").write_text(json.dumps(evidence))
    assert run(root).returncode == 1


def test_discovery_reports_no_work_without_a_manifest(tmp_path: Path) -> None:
    done = run(tmp_path, "--discover")
    assert done.returncode == 0 and done.stdout.strip() == "false"


def test_move_job_uses_full_history_and_is_conditional() -> None:
    workflow = Path(__file__).resolve().parents[4] / ".github/workflows/ci.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    job = jobs["move-proof"]
    assert "move-manifests" in job["needs"]
    assert "true" in job["if"]
    checkout = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["fetch-depth"] == 0
    assert any("tools/move_ci.py" in s.get("run", "") for s in job["steps"])


def test_ci_entry_point_directly_checks_the_same_evidence(tmp_path: Path, monkeypatch) -> None:
    from tools.move_ci import run_manifests

    root, _ = repository(tmp_path)
    monkeypatch.chdir(root)
    assert run_manifests(Path("tools/moves")) == 0
    (root / "new.py").write_text("def f(): return 2\n")
    assert run_manifests(Path("tools/moves")) == 1


def test_runner_pin_and_actual_sql_receipt_are_required(tmp_path: Path, monkeypatch) -> None:
    from tools import move_ci

    monkeypatch.setattr(move_ci, "_git", lambda *args: "")
    with pytest.raises(ValueError, match="frozen"):
        move_ci._check_base({"base": "a" * 40, "old": "src/dendrux/runtime/runner.py"}, tmp_path)
    sql = {
        "base": "a" * 40,
        "old": "src/dendrux/runtime/stores/sql/base.py",
        "source_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="missing"):
        move_ci._check_base(sql, tmp_path)
    receipt = tmp_path / "sql-base.json"
    receipt.write_text(json.dumps({"base": "a" * 40, "source_sha256": "b" * 64}))
    move_ci._check_base(sql, tmp_path)
    receipt.write_text(json.dumps({"base": "a" * 40, "source_sha256": "c" * 64}))
    with pytest.raises(ValueError, match="receipt"):
        move_ci._check_base(sql, tmp_path)


def test_discovery_and_main_fail_closed_directly(tmp_path: Path, monkeypatch, capsys) -> None:
    from tools.move_ci import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["move_ci.py", "--discover"])
    assert main() == 0
    assert capsys.readouterr().out.strip() == "false"
    monkeypatch.setattr(sys, "argv", ["move_ci.py"])
    assert main() == 1


def test_move_check_cli_accepts_only_the_manifest_base(tmp_path: Path, monkeypatch) -> None:
    from tools.move_check import main

    root, evidence = repository(tmp_path)
    monkeypatch.chdir(root)
    args = [
        "--base",
        evidence["base"],
        "--old",
        "old.py",
        "--new",
        "new.py",
        "--move-manifest",
        "tools/moves/fixture.json",
        "--emit-map",
        "result.json",
    ]
    assert main(args) == 0
    assert json.loads((root / "result.json").read_text()) == {"old": {"f": "new"}}
    args[1] = "HEAD"
    assert main(args) == 1


def _receipt(root: Path, commit: str | None = None, manifests: list[str] | None = None) -> None:
    body = {
        "final_split_commit": commit or git(root, "rev-parse", "HEAD"),
        "manifests": ["fixture.json"] if manifests is None else manifests,
    }
    (root / "tools/moves/retirement.json").write_text(json.dumps(body))
    git(root, "add", ".")
    git(root, "commit", "-qm", "Retire")


def _importlinter(root: Path) -> None:
    (root / ".importlinter").write_text(
        "[importlinter:contract:split-no-facade-backedge]\nsource_modules = new\n"
    )
    (root / "src").mkdir(exist_ok=True)


def test_retirement_receipt_turns_discovery_off(tmp_path: Path) -> None:
    root, _ = repository(tmp_path)
    assert run(root, "--discover").stdout.strip() == "true"
    _receipt(root)
    done = run(root, "--discover")
    assert done.returncode == 0 and done.stdout.strip() == "false"


@pytest.mark.parametrize(
    "receipt",
    [
        "not json",
        '{"final_split_commit": "%s", "manifests": ["fixture.json"], "x": 1}',
        '{"final_split_commit": "z", "manifests": ["fixture.json"]}',
        '{"final_split_commit": "%s", "manifests": []}',
        '{"final_split_commit": "%s", "manifests": ["other.json"]}',
        '{"final_split_commit": "%s", "manifests": ["fixture.json"]}',
    ],
)
def test_bad_receipt_fails_closed(tmp_path: Path, receipt: str) -> None:
    root, _ = repository(tmp_path)
    head = git(root, "rev-parse", "HEAD")
    text = receipt % head if "%s" in receipt else receipt
    if receipt.endswith('["fixture.json"]}') and "x" not in receipt and "z" not in receipt:
        text = receipt % ("f" * 40)  # not an ancestor of HEAD
    (root / "tools/moves/retirement.json").write_text(text)
    assert run(root, "--discover").returncode == 1


def test_manifest_added_after_receipt_reactivates_discovery(tmp_path: Path) -> None:
    root, _ = repository(tmp_path)
    _receipt(root)
    (root / "tools/moves/pr99.json").write_text("{}")
    done = run(root, "--discover")
    assert done.returncode == 0 and done.stdout.strip() == "true"


def test_receipt_without_covered_manifest_fails_closed(tmp_path: Path) -> None:
    root, _ = repository(tmp_path)
    _receipt(root)
    (root / "tools/moves/fixture.json").unlink()
    assert run(root, "--discover").returncode == 1


def test_split_sources_without_manifest_or_receipt_still_fail(tmp_path: Path) -> None:
    _importlinter(tmp_path)
    (tmp_path / "src/new.py").write_text("x = 1\n")
    assert run(tmp_path, "--discover").returncode == 1
