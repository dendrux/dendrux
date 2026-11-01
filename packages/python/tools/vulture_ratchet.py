"""Dead-code gate (vulture, minimum confidence 80): a new finding fails.

Intentional public API and hooks that vulture cannot see being used go in
``tools/vulture_whitelist.py`` (one bare name or ``_.attribute`` per line, as vulture's
``--make-whitelist`` writes them). Findings that are really dead code are baselined in
``tools/vulture_baseline.json`` as ``{"module:kind:name": count}`` until they are fixed.

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
    from collections.abc import Mapping

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

MIN_CONFIDENCE = 80
HERE = Path(__file__).resolve().parent
BASELINE = HERE / "vulture_baseline.json"
WHITELIST = HERE / "vulture_whitelist.py"
_LINE = re.compile(r"^(?P<path>.+?):\d+: unused (?P<kind>[a-z ]+?) '(?P<name>.+)' \(\d+% conf")


def findings_from_output(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            kind = m.group("kind").replace(" ", "_")
            counts[f"{r.module_name(m.group('path'))}:{kind}:{m.group('name')}"] += 1
    return dict(sorted(counts.items()))


def compare(findings: Mapping[str, int], baseline: Mapping[str, int]) -> r.CountComparison:
    return r.compare_counts(findings, baseline)


def collect_findings(src: Path = r.SRC, whitelist: Path = WHITELIST) -> dict[str, int]:
    cmd = [sys.executable, "-m", "vulture", str(src)]
    if whitelist.is_file():
        cmd.append(str(whitelist))
    cmd += ["--min-confidence", str(MIN_CONFIDENCE), "--exclude", "*/migrations/*"]
    # vulture exits 3 when it finds something; the text is what we compare.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 3):
        raise RuntimeError(f"vulture failed ({proc.returncode}): {proc.stderr.strip()}")
    return findings_from_output(proc.stdout)


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
        describe=lambda key: f"{key} (unused; delete it or whitelist it with a reason)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
