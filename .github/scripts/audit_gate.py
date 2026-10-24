"""Gate a pip-audit JSON report against an allowlist whose entries expire.

pip-audit can ignore an advisory forever; this gate does not. Each allowlist
entry carries a reason and an expiry date (at most 90 days out), and an entry
past its date fails the build until it is renewed with a fresh justification
or removed.

Usage: audit_gate.py audit.json allowlist.json inventory.json auditor_status

``inventory.json`` is ``pip list --format=json`` of the clean environment that
was audited; every installed distribution in it must appear in the audit report
with the same version (the package under test is exempt). ``auditor_status`` is
pip-audit's exit status: 0 (clean) and 1 (findings) are accepted, anything else
fails the build.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

MAX_EXPIRY_DAYS = 90

# Packages that must appear in the report as audited. Their absence means the
# auditor looked at the wrong environment or silently skipped it.
EXPECTED_AUDITED = frozenset({"alembic", "httpx", "pydantic", "sqlalchemy", "typer"})
# The package under test may not be on PyPI yet (a pre-release version).
SKIP_ALLOWED = frozenset({"dendrux"})


def _norm(name: object) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


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


def _installed(inventory: Any) -> dict[str, str] | None:
    """``{normalized name: version}`` from ``pip list --format=json``, or None if malformed."""
    if not isinstance(inventory, list) or not inventory:
        return None
    installed: dict[str, str] = {}
    for item in inventory:
        if not isinstance(item, dict) or not item.get("name") or not item.get("version"):
            return None
        installed[_norm(item["name"])] = str(item["version"])
    return installed


def evaluate(
    audit: dict[str, Any],
    allowlist: dict[str, Any],
    today: dt.date,
    inventory: Any,
    auditor_status: int,
    expected: frozenset[str] = EXPECTED_AUDITED,
) -> list[str]:
    """Every reason the build should fail; empty when the audit is clean.

    "Nothing found" must mean "everything installed was audited": the report
    has to contain every distribution of the audited environment, each with the
    installed version and a ``vulns`` list. A skipped, unversioned or missing
    distribution, a malformed report, or an unexpected auditor exit status fails.
    """
    active, problems = check_allowlist(allowlist, today)
    if auditor_status not in {0, 1}:
        problems.append(f"pip-audit exited with status {auditor_status} (expected 0 or 1).")

    installed = _installed(inventory)
    if installed is None:
        return [*problems, "installed-package inventory is empty or malformed."]
    problems.extend(
        f"{name}: expected in the audited environment but not installed."
        for name in sorted(expected - installed.keys())
    )

    dependencies = audit.get("dependencies") if isinstance(audit, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        return [*problems, "audit report has no dependencies; pip-audit did not run properly."]

    records: dict[str, dict[str, Any]] = {}
    findings = 0
    for dep in dependencies:
        if not isinstance(dep, dict) or not dep.get("name"):
            problems.append("audit report has a malformed dependency entry.")
            continue
        name = _norm(dep["name"])
        records[name] = dep
        if "skip_reason" in dep:
            if name not in SKIP_ALLOWED:
                reason = str(dep["skip_reason"])[:120]
                problems.append(f"{dep['name']}: not audited ({reason}).")
            continue
        vulns = dep.get("vulns")
        if not isinstance(vulns, list):
            problems.append(f"{dep['name']}: malformed entry (no vulns list).")
            continue
        for vuln in vulns:
            findings += 1
            ids = {vuln.get("id", ""), *vuln.get("aliases", [])}
            if ids.isdisjoint(active):
                fixes = ", ".join(vuln.get("fix_versions", [])) or "no fix released"
                problems.append(f"{dep['name']} {dep.get('version')}: {vuln.get('id')} ({fixes}).")

    for name, version in sorted(installed.items()):
        if name in SKIP_ALLOWED:
            continue
        record = records.get(name)
        if record is None:
            problems.append(f"{name} {version}: installed but missing from the audit report.")
        elif "skip_reason" in record or not isinstance(record.get("vulns"), list):
            continue  # already reported above
        elif not record.get("version"):
            problems.append(f"{name}: audit record has no version.")
        elif str(record["version"]) != version:
            problems.append(f"{name}: audited {record['version']} but {version} is installed.")

    if auditor_status == 1 and findings == 0:
        problems.append("pip-audit exited 1 but the report holds no findings (auditor error?).")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__)
        return 2
    try:
        audit = json.loads(Path(argv[1]).read_text())
        allowlist = json.loads(Path(argv[2]).read_text())
        inventory = json.loads(Path(argv[3]).read_text())
        status = int(argv[4])
    except (OSError, ValueError) as exc:
        # A missing or unreadable input means a step before the gate failed.
        print(f"::error::cannot read audit input: {exc}")
        return 1
    problems = evaluate(audit, allowlist, dt.datetime.now(dt.UTC).date(), inventory, status)
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        print("no unaddressed vulnerabilities; every installed package was audited.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
