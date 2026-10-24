"""Gate a pip-audit JSON report against an allowlist whose entries expire.

pip-audit can ignore an advisory forever; this gate does not. Each allowlist
entry carries a reason and an expiry date (at most 90 days out), and an entry
past its date fails the build until it is renewed with a fresh justification
or removed.

Usage: audit_gate.py audit.json allowlist.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

MAX_EXPIRY_DAYS = 90

# Packages that must appear in the report as audited. Their absence means the
# auditor looked at the wrong environment or silently skipped it.
EXPECTED_AUDITED = frozenset({"alembic", "fastapi", "httpx", "pydantic", "sqlalchemy", "typer"})
# The package under test may not be on PyPI yet (a pre-release version).
SKIP_ALLOWED = frozenset({"dendrux"})


def _norm(name: object) -> str:
    return str(name).lower().replace("_", "-")


def check_allowlist(
    allowlist: dict[str, Any], today: dt.date
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Valid, unexpired entries keyed by advisory id, plus any problems found."""
    active: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for i, entry in enumerate(allowlist.get("entries", [])):
        label = f"allowlist entry {i} ({entry.get('id', '?')})"
        if not isinstance(entry.get("id"), str) or not entry["id"].strip():
            problems.append(f"{label}: missing id.")
            continue
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            problems.append(f"{label}: missing reason.")
            continue
        try:
            expires = dt.date.fromisoformat(str(entry.get("expires")))
        except ValueError:
            problems.append(f"{label}: expires must be a YYYY-MM-DD date.")
            continue
        if expires < today:
            problems.append(f"{label}: expired on {expires}; renew with a reason or remove it.")
            continue
        if expires > today + dt.timedelta(days=MAX_EXPIRY_DAYS):
            problems.append(f"{label}: expiry {expires} is more than {MAX_EXPIRY_DAYS} days out.")
            continue
        active[entry["id"]] = entry
    return active, problems


def evaluate(
    audit: dict[str, Any],
    allowlist: dict[str, Any],
    today: dt.date,
    expected: frozenset[str] = EXPECTED_AUDITED,
) -> list[str]:
    """Every reason the build should fail; empty when the audit is clean.

    A report that is malformed, that skipped a dependency it could not audit, or
    that lacks an expected package fails: "nothing found" must mean "audited".
    """
    active, problems = check_allowlist(allowlist, today)
    dependencies = audit.get("dependencies") if isinstance(audit, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        return [*problems, "audit report has no dependencies; pip-audit did not run properly."]
    audited: set[str] = set()
    for dep in dependencies:
        if not isinstance(dep, dict) or not dep.get("name"):
            problems.append("audit report has a malformed dependency entry.")
            continue
        name = _norm(dep["name"])
        if "skip_reason" in dep:
            if name not in SKIP_ALLOWED:
                reason = str(dep["skip_reason"])[:120]
                problems.append(f"{dep['name']}: not audited ({reason}).")
            continue
        vulns = dep.get("vulns")
        if not isinstance(vulns, list):
            problems.append(f"{dep['name']}: malformed entry (no vulns list).")
            continue
        audited.add(name)
        for vuln in vulns:
            ids = {vuln.get("id", ""), *vuln.get("aliases", [])}
            if ids.isdisjoint(active):
                fixes = ", ".join(vuln.get("fix_versions", [])) or "no fix released"
                problems.append(f"{dep['name']} {dep.get('version')}: {vuln.get('id')} ({fixes}).")
    problems.extend(
        f"{name}: expected in the audit report but not audited." for name in sorted(expected - audited)
    )
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    try:
        audit = json.loads(Path(argv[1]).read_text())
        allowlist = json.loads(Path(argv[2]).read_text())
    except (OSError, ValueError) as exc:
        # A missing or unreadable report means pip-audit itself failed.
        print(f"::error::cannot read audit input: {exc}")
        return 1
    problems = evaluate(audit, allowlist, dt.datetime.now(dt.UTC).date())
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        print("no unaddressed vulnerabilities.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
