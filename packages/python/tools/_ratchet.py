"""Helpers shared by the quality-gate ratchets (see tools/QUALITY.md).

A ratchet records today's problems in a committed baseline and fails only on a new
problem or an existing one getting worse. Baselines are small, sorted JSON keyed by
``module:qualified_name`` so unrelated edits never churn them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC = PACKAGE_ROOT / "src" / "dendrux"
EXCLUDED_DIRS = frozenset({"migrations", "__pycache__"})


class BaselineMissingError(Exception):
    pass


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineMissingError(f"baseline not found: {path} (run 'make quality-stamp')")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def write_baseline(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(data), indent=1, sort_keys=True) + "\n")


@dataclass
class CountComparison:
    new: list[str] = field(default_factory=list)
    worse: list[str] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.new or self.worse)


def compare_counts(
    current: Mapping[str, int], baseline: Mapping[str, int], limit: int | None = None
) -> CountComparison:
    """New keys fail (when above ``limit``, if given); known keys may not grow."""
    result = CountComparison()
    for key, value in sorted(current.items()):
        old = baseline.get(key)
        if old is None:
            if limit is None or value > limit:
                result.new.append(key)
        elif value > old:
            result.worse.append(key)
        elif value < old:
            result.improved.append(key)
    result.improved.extend(sorted(k for k in baseline if k not in current))
    result.improved.sort()
    return result


def module_name(path: str | Path) -> str:
    """``.../src/dendrux/runtime/runner.py`` -> ``dendrux.runtime.runner`` (``__init__`` drops).

    Works on the path text alone (the last ``src/`` component anchors it), so it needs no
    knowledge of where the tree lives.
    """
    parts = list(Path(path).with_suffix("").parts)
    anchor = max(i for i, part in enumerate(parts) if part == "src")
    parts = parts[anchor + 1 :]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def iter_sources(src: Path = SRC) -> list[Path]:
    return sorted(p for p in src.rglob("*.py") if not EXCLUDED_DIRS & set(p.relative_to(src).parts))


def add_common_args(parser: Any, commands: tuple[str, ...] = ("check", "stamp")) -> None:
    parser.add_argument("command", choices=list(commands))
    parser.add_argument(
        "--allow-regress",
        action="store_true",
        help="stamp: record violations too (coordinator approval only; needs --reason)",
    )
    parser.add_argument("--reason", help="stamp --allow-regress: why; printed in the log")


def finish(
    command: str,
    *,
    label: str,
    comparison: CountComparison | None,
    new_baseline: Mapping[str, Any],
    path: Path,
    allow_regress: bool = False,
    reason: str | None = None,
    describe: Any = str,
) -> int:
    """Shared check/stamp ending. ``comparison`` is None when no baseline file exists yet."""
    failed = comparison is not None and comparison.failed
    if comparison is not None:
        for key in comparison.new:
            print(f"{label} NEW: {describe(key)}")
        for key in comparison.worse:
            print(f"{label} WORSE: {describe(key)}")
    if command == "check":
        if comparison is None:
            print(f"baseline not found: {path} (run 'make quality-stamp')")
            return 2
        if failed:
            n = len(comparison.new) + len(comparison.worse)
            print(f"{label}: {n} violation(s); fix them, do not stamp them (tools/QUALITY.md)")
            return 1
        if comparison.improved:
            print(f"{label}: {len(comparison.improved)} improved; run 'make quality-stamp'")
        print(f"{label} ok")
        return 0
    if failed and not allow_regress:
        print(f"{label}: refusing to stamp over new or worse entries (needs --allow-regress)")
        return 1
    if failed and not (reason and reason.strip()):
        print(f"{label}: --allow-regress needs --reason")
        return 2
    if failed:
        print(f"{label} WAIVED: {reason}")
    write_baseline(path, new_baseline)
    print(f"{label} baseline written: {len(new_baseline)} entries")
    return 0


def baseline_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return read_baseline(path)
    except BaselineMissingError:
        return None
