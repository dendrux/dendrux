"""Live provider tests: opt-in gate, results file, key hygiene.

These tests call real vendor APIs with keys from the environment
(ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY). They are skipped
unless DENDRUX_LIVE=1. When enabled, DENDRUX_LIVE_RESULTS must name the JSON
file the results are written to (rewritten after every test, so an interrupted
run still leaves a usable file).

Nothing here reads or prints a key value. Anything written to the results file
or raised as a failure message is scrubbed of the keys and of key-shaped
strings first.

Run:
    DENDRUX_LIVE=1 DENDRUX_LIVE_RESULTS=/path/results.json \
        pytest tests/live/test_images_live.py -v
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

LIVE_DIR = Path(__file__).resolve().parent
LIVE = os.environ.get("DENDRUX_LIVE") == "1"
RESULTS_ENV = "DENDRUX_LIVE_RESULTS"
KEY_ENVS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
_KEY_SHAPE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{12,})")

# The vendor SDKs log base64 request bodies at DEBUG ("Request options"). Keep
# them quiet no matter how the runner configures logging.
for _name in ("openai", "anthropic", "httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_results: list[dict[str, Any]] = []


def scrub(text: str) -> str:
    """Remove key values and key-shaped strings from text bound for output."""
    for env in KEY_ENVS:
        value = os.environ.get(env)
        if value and len(value) >= 8:
            text = text.replace(value, "[key]")
    return _KEY_SHAPE.sub("[key]", text)


def pytest_configure(config: pytest.Config) -> None:
    if LIVE and not os.environ.get(RESULTS_ENV):
        raise pytest.UsageError(
            f"DENDRUX_LIVE=1 needs {RESULTS_ENV}=<path> for the JSON results file"
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if LIVE:
        return
    skip = pytest.mark.skip(reason="live provider tests need DENDRUX_LIVE=1")
    for item in items:
        if LIVE_DIR in item.path.parents:
            item.add_marker(skip)


@pytest.fixture
def rec(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Per-test record the scenario fills in (usage, cache fields, notes)."""
    return {}


def _write_results() -> None:
    target = os.environ.get(RESULTS_ENV)
    if not target:
        return
    payload = {
        "written_at": datetime.now(UTC).isoformat(),
        "dendrux": _dendrux_version(),
        "results": _results,
    }
    path = Path(target)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(scrub(json.dumps(payload, indent=2, sort_keys=True, default=str)))
    tmp.replace(path)


def _dendrux_version() -> str:
    try:
        import dendrux

        return str(dendrux.__version__)
    except Exception:
        return "unknown"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()
    if not LIVE or LIVE_DIR not in item.path.parents:
        return
    if report.when == "call" or (report.when == "setup" and report.skipped):
        funcargs = getattr(item, "funcargs", {})
        rec: dict[str, Any] = dict(funcargs.get("rec") or {})
        case = funcargs.get("case")
        detail = ""
        if report.failed:
            detail = scrub(str(report.longrepr))[-1500:]
        elif report.skipped:
            longrepr = report.longrepr
            detail = scrub(str(longrepr[-1] if isinstance(longrepr, tuple) else longrepr))
        _results.append(
            {
                "provider": getattr(case, "family", None),
                "route": getattr(case, "route_id", None),
                "model": getattr(case, "model", None),
                "scenario": getattr(case, "scenario", None),
                "node": item.nodeid,
                "outcome": "pass" if report.passed else ("skip" if report.skipped else "fail"),
                "duration_s": round(report.duration, 2),
                "timestamp": datetime.now(UTC).isoformat(),
                "detail": detail,
                **{k: v for k, v in rec.items()},
            }
        )
        _write_results()
