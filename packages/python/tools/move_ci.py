"""Run tracked cumulative move evidence from immutable source bases."""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._move_manifest import verify_provider_sources
from tools._ratchet import module_name
from tools.move_check import _git_show, _patch_map, check_move

RUNNER_BASE = "c2e3169700c14341f85599b23066a43886234123"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError("move evidence must be tracked with an available immutable base")
    return result.stdout.strip()


def _tracked(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing move input: {path}")
    _git("ls-files", "--error-unmatch", str(path))


def _manifests(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.glob("*.json")
        if not p.name.endswith(("-emitted.json", "-expanded.json")) and p.name != "sql-base.json"
    )


def _created_sources() -> set[str]:
    config = configparser.ConfigParser()
    config.read(".importlinter")
    section = "importlinter:contract:split-no-facade-backedge"
    names = config.get(section, "source_modules", fallback="").split()
    return {
        f"src/{name.replace('.', '/')}.py"
        for name in names
        if Path(f"src/{name.replace('.', '/')}.py").is_file()
    }


def _check_base(manifest: dict[str, Any], directory: Path) -> None:
    base, old = manifest["base"], manifest["old"]
    if len(base) != 40 or any(c not in "0123456789abcdef" for c in base):
        raise ValueError("base must be the full immutable SHA")
    _git("merge-base", "--is-ancestor", base, "HEAD")
    if old == "src/dendrux/runtime/runner.py" and base != RUNNER_BASE:
        raise ValueError("runner proof base is not the frozen source pin")
    if old == "src/dendrux/runtime/stores/sql/base.py":
        receipt_path = directory / "sql-base.json"
        _tracked(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if (
            set(receipt) != {"base", "source_sha256"}
            or receipt["base"] != base
            or receipt["source_sha256"] != manifest["source_sha256"]
        ):
            raise ValueError("SQL requires the reviewed actual post-prerequisite base receipt")
        _git("merge-base", "--is-ancestor", RUNNER_BASE, base)


def _run(path: Path) -> tuple[str, set[str]]:
    _tracked(path)
    manifest = json.loads(path.read_text())
    _check_base(manifest, path.parent)
    old = manifest["old"]
    after = {}
    for key in manifest["sources"]:
        candidate = Path(key)
        if candidate.is_absolute() or ".." in candidate.parts or not key.endswith(".py"):
            raise ValueError("source paths must be repository-relative Python files")
        _tracked(candidate)
        after[key] = candidate.read_text()
    if old not in after:
        raise ValueError("residual facade is a required cumulative destination")
    verify_provider_sources(
        manifest,
        lambda path: _git_show(manifest["base"], path),
        lambda path: Path(path).read_text(),
    )
    for entry in manifest["providers"].values():
        _tracked(Path(entry["path"]))
    before = _git_show(manifest["base"], old)
    report = check_move(
        before,
        after,
        old_key=old,
        modules={key: module_name(key) for key in after},
        move_manifest=manifest,
    )
    if not report.ok:
        raise ValueError("\n".join(report.problems))
    emitted = _patch_map(Path(old), report.moved)
    expanded = {key: dict(value) for key, value in emitted.items()}
    for name, target in manifest["seams"].items():
        expanded.setdefault(module_name(old), {})[name] = target.rsplit(".", 1)[0]
    for suffix, expected in [("emitted", emitted), ("expanded", expanded)]:
        evidence_path = path.with_name(f"{path.stem}-{suffix}.json")
        _tracked(evidence_path)
        if json.loads(evidence_path.read_text()) != expected:
            raise ValueError(f"{suffix} map differs from proven/expanded ownership")
    print(f"move-proof: {path}: {len(report.moved)} moved definitions verified")
    return old, set(after)


def run_manifests(directory: Path) -> int:
    """Check every active family manifest; an absent manifest is an error."""
    try:
        paths = _manifests(directory)
        if not paths:
            raise ValueError("absent move manifest")
        roots = set()
        destinations = set()
        for path in paths:
            old, sources = _run(path)
            if old in roots:
                raise ValueError("keep one active cumulative manifest per original module")
            roots.add(old)
            destinations.update(sources)
        if _created_sources() - destinations:
            raise ValueError("cumulative manifest omits created split destinations")
    except (OSError, ValueError, KeyError, TypeError, SystemExit) as exc:
        print(f"move-proof: refused: {exc}")
        return 1
    return 0


def main() -> int:
    """Discover active work or verify all tracked manifests from the package root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("tools/moves"))
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    if args.discover:
        active = bool(_manifests(args.directory))
        if not active and _created_sources():
            parser.exit(1, "move-proof: split sources exist without a manifest\n")
        print("true" if active else "false")
        return 0
    return run_manifests(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
