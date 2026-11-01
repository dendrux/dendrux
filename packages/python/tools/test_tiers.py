"""Test tiers: areas, the slow and flaky lists, changed-file linting and CI shards.

    python tools/test_tiers.py areas                      # the area names
    python tools/test_tiers.py workers                    # local worker count (at most 6)
    python tools/test_tiers.py check-area NAME            # exit 2 for an unknown area
    python tools/test_tiers.py ruff-changed               # ruff on the .py files changed vs HEAD
    python tools/test_tiers.py check-lists                # the slow/flaky lists name real tests
    python tools/test_tiers.py slow-list LOG [--threshold S] [--write]
    python tools/test_tiers.py weights LOG [--write]      # per-file seconds, for shard balance
    python tools/test_tiers.py shard --count N --index I  # the test files of shard I (1-based)
    python tools/test_tiers.py shard-proof --count N      # the shards are a disjoint cover
    python tools/test_tiers.py dbname URL                 # the database name of a URL

LOG is the output of a pytest run with ``--durations=0``. See tools/QUALITY.md, "Test tiers".
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tests._helpers import tiers  # noqa: E402

DEFAULT_SLOW_THRESHOLD = 2.0
UNKNOWN_FILE_SECONDS = 1.0
WEIGHTS = tiers.LISTS_DIR / "file_weights.json"
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s+(?:setup|call|teardown)\s+(\S.*?)\s*$")
_SLOW_HEADER = (
    "# Tests slower than {threshold:g} s (setup + call + teardown, one parallel run).\n"
    "# Regenerate: python tools/test_tiers.py slow-list <durations log> --write\n"
    "# The `slow` marker only moves a test out of the fast loop (make test-fast, make test-area);\n"
    "# make ci, CI and make test-merge still run it.\n"
)


def test_files() -> list[str]:
    """Every file pytest collects under ``tests/``, as paths relative to the package root."""
    found = {
        p.relative_to(PACKAGE_ROOT).as_posix()
        for pattern in ("test_*.py", "*_test.py")
        for p in (PACKAGE_ROOT / "tests").rglob(pattern)
        if "__pycache__" not in p.parts
    }
    return sorted(found)


def parse_durations(text: str) -> dict[str, float]:
    """Seconds per node id, summed over the setup, call and teardown phases."""
    totals: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        match = _DURATION.match(line)
        if match:
            totals[match.group(2)] += float(match.group(1))
    return dict(totals)


def slow_entries(durations: dict[str, float], threshold: float) -> list[str]:
    return sorted(nodeid for nodeid, seconds in durations.items() if seconds >= threshold)


def file_weights(durations: dict[str, float]) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    for nodeid, seconds in durations.items():
        weights[nodeid.split("::", 1)[0]] += seconds
    return {path: round(weights[path], 2) for path in sorted(weights)}


def load_weights() -> dict[str, float]:
    if not WEIGHTS.exists():
        return {}
    data: dict[str, float] = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return data


def make_shards(files: list[str], count: int, weights: dict[str, float]) -> list[list[str]]:
    """Longest-processing-time-first split of ``files`` into ``count`` shards.

    Deterministic: files sort by (weight desc, name). A file with no recorded weight counts as
    ``UNKNOWN_FILE_SECONDS``: the weights file leaves out files whose tests all run in under
    5 ms, and a new test file lands somewhere sensible without a regenerate.
    """
    default = UNKNOWN_FILE_SECONDS
    ordered = sorted(files, key=lambda f: (-weights.get(f, default), f))
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for name in ordered:
        target = min(range(count), key=lambda i: (loads[i], i))
        shards[target].append(name)
        loads[target] += weights.get(name, default)
    return [sorted(s) for s in shards]


def _function_names(path: Path) -> set[str]:
    """``name`` and ``Class::name`` for every test function defined in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.update(
                f"{node.name}::{sub.name}"
                for sub in node.body
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return names


def check_lists() -> list[str]:
    """Problems in the committed lists: a missing test, or a flaky entry with no reason."""
    problems = []
    cache: dict[str, set[str]] = {}
    for name in ("slow.txt", "flaky.txt"):
        for entry in tiers.load_list(name):
            path_part, _, rest = entry.nodeid.partition("::")
            test_name = rest.partition("[")[0]
            path = PACKAGE_ROOT / path_part
            if not path.is_file():
                problems.append(f"{name}: {entry.nodeid}: no such file")
                continue
            names = cache.setdefault(path_part, _function_names(path))
            if test_name not in names:
                problems.append(f"{name}: {entry.nodeid}: no such test in {path_part}")
            if name == "flaky.txt" and not entry.note:
                problems.append(f"{name}: {entry.nodeid}: a reason is required (`id | reason`)")
    return problems


def changed_python_files() -> list[str]:
    """Python files changed against HEAD (staged, unstaged, untracked), under this package."""
    commands = (
        ["git", "diff", "--name-only", "--relative", "--diff-filter=d", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    names: set[str] = set()
    for command in commands:
        out = subprocess.run(
            command, cwd=PACKAGE_ROOT, capture_output=True, text=True, check=False
        ).stdout
        names.update(line for line in out.splitlines() if line.endswith(".py"))
    return sorted(n for n in names if (PACKAGE_ROOT / n).is_file())


def _run_ruff(files: list[str]) -> int:
    code = 0
    for args in (["check"], ["format", "--check"]):
        code |= subprocess.run(
            [sys.executable, "-m", "ruff", *args, *files], cwd=PACKAGE_ROOT, check=False
        ).returncode
    return code


def _cmd_slow_list(args: argparse.Namespace) -> int:
    durations = parse_durations(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if not durations:
        print("no durations found: run pytest with --durations=0", file=sys.stderr)
        return 2
    entries = slow_entries(durations, args.threshold)
    body = _SLOW_HEADER.format(threshold=args.threshold) + "".join(f"{e}\n" for e in entries)
    if args.write:
        (tiers.LISTS_DIR / "slow.txt").write_text(body, encoding="utf-8")
        print(f"wrote {len(entries)} slow tests (>= {args.threshold:g} s)")
    else:
        sys.stdout.write(body)
    return 0


def _cmd_weights(args: argparse.Namespace) -> int:
    durations = parse_durations(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if not durations:
        print("no durations found: run pytest with --durations=0", file=sys.stderr)
        return 2
    text = json.dumps(file_weights(durations), indent=1, sort_keys=True) + "\n"
    if args.write:
        WEIGHTS.write_text(text, encoding="utf-8")
        print(f"wrote weights for {len(json.loads(text))} files")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_shard(args: argparse.Namespace) -> int:
    if not 1 <= args.index <= args.count:
        print("--index must be between 1 and --count", file=sys.stderr)
        return 2
    shards = make_shards(test_files(), args.count, load_weights())
    print("\n".join(shards[args.index - 1]))
    return 0


def _cmd_shard_proof(args: argparse.Namespace) -> int:
    files = test_files()
    shards = make_shards(files, args.count, load_weights())
    seen = [f for shard in shards for f in shard]
    if sorted(seen) != files or len(set(seen)) != len(seen):
        print("shards are not a disjoint cover of the test files", file=sys.stderr)
        return 1
    weights = load_weights()
    for number, shard in enumerate(shards, start=1):
        seconds = sum(weights.get(f, 0.0) for f in shard)
        print(f"shard {number}: {len(shard)} files, ~{seconds:.0f} s recorded")
    print(f"ok: {len(files)} files, each in exactly one of {args.count} shards")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("areas")
    sub.add_parser("workers")
    sub.add_parser("check-area").add_argument("name")
    sub.add_parser("ruff-changed")
    sub.add_parser("check-lists")
    for name in ("slow-list", "weights"):
        p = sub.add_parser(name)
        p.add_argument("log")
        p.add_argument("--write", action="store_true")
        if name == "slow-list":
            p.add_argument("--threshold", type=float, default=DEFAULT_SLOW_THRESHOLD)
    for name in ("shard", "shard-proof"):
        p = sub.add_parser(name)
        p.add_argument("--count", type=int, required=True)
        if name == "shard":
            p.add_argument("--index", type=int, required=True)
    sub.add_parser("dbname").add_argument("url")
    args = parser.parse_args(argv)

    if args.command == "areas":
        print(" ".join(tiers.AREAS))
    elif args.command == "workers":
        print(tiers.default_workers())
    elif args.command == "check-area":
        if args.name not in tiers.AREAS:
            print(f"unknown AREA {args.name!r}; one of: {' '.join(tiers.AREAS)}", file=sys.stderr)
            return 2
    elif args.command == "ruff-changed":
        files = changed_python_files()
        if not files:
            print("ruff: no changed Python files")
            return 0
        print(f"ruff: {len(files)} changed file(s)")
        return _run_ruff(files)
    elif args.command == "check-lists":
        problems = check_lists()
        print("\n".join(problems) if problems else "tier lists ok")
        return 1 if problems else 0
    elif args.command == "slow-list":
        return _cmd_slow_list(args)
    elif args.command == "weights":
        return _cmd_weights(args)
    elif args.command == "shard":
        return _cmd_shard(args)
    elif args.command == "shard-proof":
        return _cmd_shard_proof(args)
    elif args.command == "dbname":
        from sqlalchemy.engine import make_url

        print(make_url(args.url).database or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
