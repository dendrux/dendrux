"""Fail unless a wheel carries a complete, self-consistent dashboard build.

The React dashboard ships as committed build output under
``dendrux/dashboard/static``. A rebuild produces new hashed file names, so a
forgotten ``git add`` would publish an ``index.html`` that points at assets
the wheel does not contain.

Usage: check_wheel_dashboard.py path/to/dendrux-X.Y.Z-py3-none-any.whl [static_dir]

With ``static_dir`` (the committed build directory), every file in it must also
be in the wheel, which catches lazy chunks and CSS-referenced assets that
``index.html`` does not name.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

STATIC = "dendrux/dashboard/static/"
_REF = re.compile(r"""(?:src|href)=["'](?:\./)?(assets/[^"'?#]+)""")


def problems(names: set[str], index_html: str | None) -> list[str]:
    """Human-readable problems with the dashboard files of one wheel."""
    if f"{STATIC}index.html" not in names or index_html is None:
        return [f"{STATIC}index.html is missing from the wheel."]
    found = []
    refs = _REF.findall(index_html)
    if not refs:
        found.append("index.html references no assets.")
    found.extend(
        f"index.html references {ref}, which the wheel does not contain."
        for ref in refs
        if f"{STATIC}{ref}" not in names
    )
    return found


def missing_from_wheel(names: set[str], static_dir: Path) -> list[str]:
    """Files of the committed build that the wheel lacks."""
    return [
        f"{STATIC}{path.relative_to(static_dir).as_posix()} is committed but not in the wheel."
        for path in sorted(static_dir.rglob("*"))
        if path.is_file()
        and f"{STATIC}{path.relative_to(static_dir).as_posix()}" not in names
        and path.name != ".gitkeep"
    ]


def main(argv: list[str]) -> int:
    if len(argv) not in {2, 3}:
        print(__doc__)
        return 2
    with zipfile.ZipFile(argv[1]) as wheel:
        names = set(wheel.namelist())
        has_index = f"{STATIC}index.html" in names
        index = wheel.read(f"{STATIC}index.html").decode() if has_index else None
    errors = problems(names, index)
    if len(argv) == 3:
        errors.extend(missing_from_wheel(names, Path(argv[2])))
    for error in errors:
        print(f"::error::{error}")
    if not errors:
        print("dashboard assets present and consistent.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
