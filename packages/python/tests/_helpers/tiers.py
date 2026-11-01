"""Test tiers: the area, ``slow`` and ``flaky`` markers, applied by path and by committed lists.

The tests are laid out by kind (``unit``, ``integration``, ...), not one directory per source
package, so the area of a test comes from its directory, then from its file name. Nothing in a
test module changes: ``tests/conftest.py`` adds the markers while collecting.

Two committed lists sit in ``tests/_helpers/tier_lists/``:

* ``slow.txt``: tests whose measured time is over the threshold (``tools/test_tiers.py slow-list``
  regenerates it from a JUnit report).
* ``flaky.txt``: quarantined tests, one reason per entry. They still run everywhere, with one
  automatic rerun (``pytest-rerunfailures``); nothing here skips a test.

An entry is ``path::name`` or ``path::name[params]``; the short form matches every parametrization.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

LISTS_DIR = Path(__file__).resolve().parent / "tier_lists"
TESTS_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = TESTS_ROOT.parent

AREAS: tuple[str, ...] = (
    "runtime",
    "stores",
    "store",
    "loops",
    "llm",
    "mcp",
    "db",
    "skills",
    "testing",
    "tools",
    "guardrails",
    "media",
    "notifiers",
    "cli",
    "strategies",
    "docs",
    "tooling",
    "live",
)

# Directories that name their own area.
_DIR_AREAS = {
    "conformance": "testing",
    "db": "db",
    "migrations": "db",
    "docs": "docs",
    "tooling": "tooling",
    "tools": "tooling",
    "live": "live",
}

# File-name rules for ``unit`` and ``integration``, first match wins. Anything left is
# ``runtime``: the run loop, resume, cancel, tenancy, budgets and the public Agent API.
_NAME_RULES: tuple[tuple[str, str], ...] = (
    (r"mcp", "mcp"),
    (r"skill", "skills"),
    (r"guardrail|detector|pii|prompt_inj|secret", "guardrails"),
    (r"media|image", "media"),
    (r"follow|notifier_pg|pending_reader|pending_visibility", "store"),
    (r"notifier|otel|observer|console_observer", "notifiers"),
    (
        r"mysql|sql_store|sqlite|schema|migration|ddl|model_column|row_oracle|settings_fixture"
        r"|statement_trim|single_connection_pool|state_store|state_value|state_aware|run_store"
        r"|crash_cut|idempotency|blob|listing_plan|bounded_subtree|pg_|test_db|t1_db|persistence"
        r"|reclaim_projection|tenant_equality|state_060|_db\b",
        "stores",
    ),
    (
        r"anthropic|openai|openrouter|provider|llm|retry|stop_reason|reasoning|cache|usage|transport"
        r"|gateway|pricing|structured|stream_response|long_provider|model_identity",
        "llm",
    ),
    (r"loops|single_call|singlecall|react|streaming|answer_phase|leased_stream|stream", "loops"),
    (r"tool", "tools"),
    (r"cli_", "cli"),
    (r"strateg", "strategies"),
    (r"store_kit|testing", "testing"),
)
_COMPILED_RULES = tuple((re.compile(pattern), area) for pattern, area in _NAME_RULES)
_DEFAULT_AREA = "runtime"


def area_for(relpath: str) -> str:
    """The area of a test file, ``relpath`` being relative to the ``tests`` directory."""
    parts = relpath.replace("\\", "/").split("/")
    if len(parts) > 1 and parts[0] in _DIR_AREAS:
        return _DIR_AREAS[parts[0]]
    name = parts[-1].removesuffix(".py")
    for pattern, area in _COMPILED_RULES:
        if pattern.search(name):
            return area
    return _DEFAULT_AREA


@dataclass(frozen=True)
class ListEntry:
    nodeid: str
    note: str


def parse_list(text: str) -> list[ListEntry]:
    """Parse a tier list: ``nodeid`` or ``nodeid | note`` per line; ``#`` starts a comment."""
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        nodeid, _, note = line.partition(" | ")
        entries.append(ListEntry(nodeid.strip(), note.strip()))
    return entries


def load_list(name: str) -> list[ListEntry]:
    path = LISTS_DIR / name
    return parse_list(path.read_text(encoding="utf-8")) if path.exists() else []


def listed(entries: list[ListEntry]) -> tuple[set[str], tuple[str, ...]]:
    """Split entries into an exact-id set and the short (all parametrizations) prefixes."""
    exact = {e.nodeid for e in entries}
    return exact, tuple(e.nodeid for e in entries if "[" not in e.nodeid)


def in_list(exact: set[str], prefixes: tuple[str, ...], nodeid: str) -> bool:
    if nodeid in exact:
        return True
    base = nodeid.partition("[")[0]
    return base in prefixes


def default_workers() -> int:
    """Worker count for a local parallel run: ``DENDRUX_TEST_WORKERS``, else cores capped at 6."""
    raw = os.environ.get("DENDRUX_TEST_WORKERS", "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(1, min(6, os.cpu_count() or 1))
