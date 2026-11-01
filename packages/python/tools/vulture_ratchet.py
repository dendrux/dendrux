"""Dead-code gate (vulture, minimum confidence 80): a new finding fails.

Intentional public API and hooks that vulture cannot see being used go in
``tools/vulture_whitelist.py`` (one bare name or ``_.attribute`` per line, as vulture's
``--make-whitelist`` writes them). Findings that are really dead code are baselined in
``tools/vulture_baseline.json`` as ``{"module:function:kind:name": count}`` (no line numbers)
until they are fixed. Every diagnostic vulture prints at this confidence counts: unused names,
unreachable code and unsatisfiable conditions. Output the gate does not understand is an
error, never silently dropped.

    python tools/vulture_ratchet.py check
    python tools/vulture_ratchet.py stamp        # after improving; refuses to add entries
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

MIN_CONFIDENCE = 80
HERE = Path(__file__).resolve().parent
BASELINE = HERE / "vulture_baseline.json"
WHITELIST = HERE / "vulture_whitelist.py"
_LOCATED = re.compile(r"^(?P<path>.+?):(?P<line>\d+): (?P<message>.+) \((?P<pct>\d+)% confidence")
_UNUSED = re.compile(r"^unused (?P<kind>[a-z ]+?) '(?P<name>.+)'$")
_UNREACHABLE = re.compile(r"^unreachable code after '(?P<name>\w+)'$")
_UNSATISFIABLE = re.compile(r"^unsatisfiable '(?P<name>\w+)' condition$")


class UnrecognisedOutputError(Exception):
    """Vulture printed something this gate does not understand; never ignore it."""


def _classify(message: str) -> tuple[str, str] | None:
    if m := _UNUSED.match(message):
        return m.group("kind").replace(" ", "_"), m.group("name")
    if m := _UNREACHABLE.match(message):
        return "unreachable_code", m.group("name")
    if m := _UNSATISFIABLE.match(message):
        return "unsatisfiable_condition", m.group("name")
    return None


def findings_from_output(
    text: str, enclosing: Callable[[str, int], str] | None = None
) -> dict[str, int]:
    """``module:function:kind:name`` -> count, for every line vulture printed."""
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        located = _LOCATED.match(line)
        classified = _classify(located.group("message")) if located else None
        if located is None or classified is None:
            raise UnrecognisedOutputError(f"unrecognised vulture output: {line.strip()}")
        path, number = located.group("path"), int(located.group("line"))
        function = enclosing(path, number) if enclosing else "<module>"
        kind, name = classified
        counts[f"{r.module_name(path)}:{function}:{kind}:{name}"] += 1
    return dict(sorted(counts.items()))


def compare(findings: Mapping[str, int], baseline: Mapping[str, int]) -> r.CountComparison:
    return r.compare_counts(findings, baseline)


def _resolver() -> Callable[[str, int], str]:
    spans: dict[str, list[tuple[int, int, str]]] = {}

    def resolve(path: str, line: int) -> str:
        if path not in spans:
            spans[path] = r.function_spans(Path(path).read_text(encoding="utf-8"))
        return r.enclosing(spans[path], line)

    return resolve


def collect_findings(src: Path = r.SRC, whitelist: Path = WHITELIST) -> dict[str, int]:
    cmd = [sys.executable, "-m", "vulture", str(src)]
    if whitelist.is_file():
        cmd.append(str(whitelist))
    cmd += ["--min-confidence", str(MIN_CONFIDENCE), "--exclude", "*/migrations/*"]
    # vulture exits 3 when it finds something; the text is what we compare.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 3):
        raise RuntimeError(f"vulture failed ({proc.returncode}): {proc.stderr.strip()}")
    return findings_from_output(proc.stdout, _resolver())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    r.add_common_args(parser)
    parser.add_argument("--src", type=Path, default=r.SRC)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--whitelist", type=Path, default=WHITELIST)
    args = parser.parse_args(argv)
    findings = collect_findings(args.src.resolve(), args.whitelist.resolve())
    old = r.baseline_or_none(args.baseline)
    return r.finish(
        args.command,
        label="dead code",
        comparison=None if old is None else compare(findings, old),
        new_baseline=findings,
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=lambda key: f"{key} (dead code; delete it or whitelist it with a reason)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
