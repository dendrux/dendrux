"""Semgrep gate: the custom rules in ``.semgrep/dendrux.yml`` over ``src/dendrux``.

``check`` fails on a new match, or more matches in a function that already had some.
Matches are keyed ``rule|module|enclosing function`` (no line numbers) in
``tools/semgrep_baseline.json``. ``test`` runs the rules against their fixture
(``.semgrep/dendrux.py``). A rule limits itself to part of the tree with
``metadata.include`` / ``metadata.exclude`` (path prefixes under ``src/dendrux``); they are
applied here rather than as Semgrep ``paths:`` so one fixture can exercise every rule.

Semgrep is a dev/CI tool, not a dependency of the package. Its pinned version is in
``tools/semgrep-requirements.txt``. It pins ``mcp`` and OpenTelemetry versions that clash
with the ``mcp`` and ``all`` extras, so install it in its own environment and point to it
with ``SEMGREP=/path/to/semgrep``.

    python tools/semgrep_ratchet.py check
    python tools/semgrep_ratchet.py test
    python tools/semgrep_ratchet.py stamp        # after improving; refuses to add matches
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

HERE = Path(__file__).resolve().parent
RULES_DIR = HERE.parent / ".semgrep"
RULES = RULES_DIR / "dendrux.yml"
BASELINE = HERE / "semgrep_baseline.json"
INSTALL_HINT = (
    "semgrep is not installed. Install the pinned version in its own environment "
    "(it conflicts with the mcp extra): python -m venv ~/.cache/dendrux-semgrep-venv && "
    "~/.cache/dendrux-semgrep-venv/bin/pip install -r tools/semgrep-requirements.txt "
    "(that is: pip install semgrep==<pinned>), then run with "
    "SEMGREP=~/.cache/dendrux-semgrep-venv/bin/semgrep. CI installs it in the quality job."
)


class SemgrepMissingError(Exception):
    pass


def find_semgrep() -> str:
    explicit = os.environ.get("SEMGREP")
    if explicit:
        path = os.path.expanduser(explicit)
        if Path(path).is_file():
            return path
        raise SemgrepMissingError(f"SEMGREP={explicit} does not exist. {INSTALL_HINT}")
    found = shutil.which("semgrep")
    if found:
        return found
    raise SemgrepMissingError(INSTALL_HINT)


def function_spans(source: str) -> list[tuple[int, int, str]]:
    """(first line, last line, qualified name) for every function, outermost first."""
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qual = f"{prefix}{child.name}"
                spans.append((child.lineno, child.end_lineno or child.lineno, qual))
                walk(child, f"{qual}.<locals>.")
            else:
                walk(child, prefix)

    walk(ast.parse(source), "")
    return spans


def enclosing(spans: list[tuple[int, int, str]], line: int) -> str:
    best = "<module>"
    best_size = None
    for first, last, qual in spans:
        if first <= line <= last and (best_size is None or last - first < best_size):
            best, best_size = qual, last - first
    return best


def _package_relative(path: str) -> str:
    parts = Path(path).parts
    anchor = max(i for i, part in enumerate(parts) if part == "dendrux")
    return "/".join(parts[anchor + 1 :])


def in_scope(path: str, metadata: Mapping[str, Any]) -> bool:
    rel = _package_relative(path)
    include = metadata.get("include") or []
    exclude = metadata.get("exclude") or []
    if include and not any(rel.startswith(prefix) for prefix in include):
        return False
    return not any(rel.startswith(prefix) for prefix in exclude)


def keys_from_results(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    spans_cache: dict[str, list[tuple[int, int, str]]] = {}
    for item in results:
        path = item["path"]
        metadata = item.get("extra", {}).get("metadata", {})
        if not in_scope(path, metadata):
            continue
        if path not in spans_cache:
            spans_cache[path] = function_spans(Path(path).read_text(encoding="utf-8"))
        rule = item["check_id"].rsplit(".", 1)[-1]
        func = enclosing(spans_cache[path], item["start"]["line"])
        counts[f"{rule}|{r.module_name(path)}|{func}"] += 1
    return dict(sorted(counts.items()))


def compare(current: Mapping[str, int], baseline: Mapping[str, int]) -> r.CountComparison:
    return r.compare_counts(current, baseline)


def collect_keys(src: Path = r.SRC, rules: Path = RULES) -> dict[str, int]:
    exe = find_semgrep()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "semgrep.json"
        cmd = [exe, "--config", str(rules), "--json", "--quiet", "--metrics=off"]
        cmd += ["--disable-version-check", "--no-git-ignore", "--output", str(out), str(src)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in (0, 1) or not out.is_file():
            raise RuntimeError(f"semgrep failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
        data = json.loads(out.read_text())
    if data.get("errors"):
        raise RuntimeError(f"semgrep reported errors: {data['errors'][:2]}")
    return keys_from_results(data["results"])


def run_fixture_tests() -> int:
    exe = find_semgrep()
    # Hidden directories are skipped by `semgrep --test`, so run from inside .semgrep.
    proc = subprocess.run(
        [exe, "--test", "--metrics=off", "--disable-version-check", "."],
        cwd=RULES_DIR,
        capture_output=True,
        text=True,
    )
    print((proc.stdout + proc.stderr).strip()[-2000:])
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    r.add_common_args(parser, ("check", "stamp", "test"))
    parser.add_argument("--src", type=Path, default=r.SRC)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    try:
        if args.command == "test":
            return run_fixture_tests()
        keys = collect_keys(args.src.resolve())
    except SemgrepMissingError as exc:
        print(f"semgrep: {exc}", file=sys.stderr)
        return 2
    old = r.baseline_or_none(args.baseline)
    return r.finish(
        args.command,
        label="semgrep",
        comparison=None if old is None else compare(keys, old),
        new_baseline=keys,
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=lambda key: f"{key.split('|')[0]} at {key.split('|', 1)[1]}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
