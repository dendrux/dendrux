"""The pytest plugin: option, collection of the contracts, the kit fixture and the report."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from dendrux.testing.store_kit._declaration import (
    CLOCK,
    FACTORY,
    FAULT_INJECTOR,
    REOPEN,
    SECOND_INSTANCE,
    Declaration,
    required_fixtures_for,
)
from dendrux.testing.store_kit._kit import (
    MARKER,
    USER_PROPERTY_CATEGORY,
    USER_PROPERTY_STORE,
    CertificationFailure,
    Kit,
    check_declaration,
    fixture_or_none,
    maybe_await,
)

if TYPE_CHECKING:
    from collections.abc import Generator

CONTRACTS_DIR = Path(__file__).resolve().parent / "contracts"
PLUGIN_NAME = "dendrux.testing.store_kit"

#: Categories in report order. ``crash_recovery`` needs the optional fault injector.
CATEGORIES = (
    "declaration",
    "basic",
    "agent_flows",
    "fenced_writes",
    "retry_ids",
    "checkpointing",
    "atomic_completion",
    "recovery",
    "concurrency",
    "crash_recovery",
)
_FIXTURE_FOR = {
    "reopen": REOPEN,
    "clock": CLOCK,
    "second": SECOND_INSTANCE,
    "injector": FAULT_INJECTOR,
}


@dataclass
class _StoreReport:
    deployment: str | None = None
    declared: tuple[str, ...] = ()
    provided: frozenset[str] = frozenset()
    results: dict[str, Counter[str]] = field(default_factory=dict)
    seen: dict[str, set[str]] = field(default_factory=dict)  # category -> contract functions run
    notes: list[str] = field(default_factory=list)


@dataclass
class _Report:
    stores: dict[str, _StoreReport] = field(default_factory=dict)
    deselected: set[str] = field(default_factory=set)  # categories with a deselected contract


def contract_inventory() -> dict[str, set[str]]:
    """Every contract function the package ships, by report category."""
    import importlib
    import pkgutil

    from dendrux.testing.store_kit import contracts

    found: dict[str, set[str]] = {}
    for info in pkgutil.iter_modules(contracts.__path__):
        if not info.name.startswith("test_"):
            continue
        module = importlib.import_module(f"{contracts.__name__}.{info.name}")
        for name, function in vars(module).items():
            if not name.startswith("test_"):
                continue
            for mark in getattr(function, "pytestmark", []):
                if mark.name == MARKER:
                    found.setdefault(str(mark.args[0]), set()).add(name)
    return found


# ---------------------------------------------------------------------------
# Options and collection
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("dendrux-store-kit", "Dendrux store test kit")
    group.addoption(
        "--dendrux-store-kit",
        action="store_true",
        dest="dendrux_store_kit",
        default=False,
        help="Collect and run the Dendrux store contracts against your dendrux_* fixtures.",
    )
    parser.addini(
        "dendrux_store_kit",
        type="bool",
        default=False,
        help="Collect the Dendrux store contracts (same as --dendrux-store-kit).",
    )


def _activated(config: pytest.Config) -> bool:
    return bool(
        config.getoption("dendrux_store_kit", default=False)
        or config.getini("dendrux_store_kit")
        or PLUGIN_NAME in (config.getoption("plugins", default=None) or [])
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "dendrux_kit(category, needs=(), uses=(), format2=False): a Dendrux store contract "
        "(set by dendrux.testing.store_kit)",
    )
    config.pluginmanager.register(_Reporter(), "dendrux-store-kit-report")
    if _activated(config):
        target = str(CONTRACTS_DIR)
        if target not in config.args:
            config.args.append(target)


# ---------------------------------------------------------------------------
# The kit fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dendrux_kit_provided(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The author's fixtures that exist, resolved before the event loop starts running."""
    return {
        name: value
        for name in (FACTORY, REOPEN, SECOND_INSTANCE, CLOCK, FAULT_INJECTOR)
        if (value := fixture_or_none(request, name)) is not None
    }


@pytest_asyncio.fixture(name="dendrux_kit")
async def dendrux_kit(request: pytest.FixtureRequest, dendrux_kit_provided: dict[str, Any]) -> Kit:
    """A fresh store from the author's factory, checked against its own declaration."""
    marker = request.node.get_closest_marker(MARKER)
    if marker is None:
        raise RuntimeError("dendrux_kit is for tests marked with @contract(...)")
    category: str = marker.args[0]
    needs: tuple[str, ...] = tuple(marker.kwargs.get("needs", ()))
    uses: tuple[str, ...] = tuple(marker.kwargs.get("uses", ()))
    factory = dendrux_kit_provided.get(FACTORY)
    if factory is None:
        raise CertificationFailure(f"{FACTORY} is required but not provided")
    store = await maybe_await(factory())
    declaration = Declaration.of(store)
    props = request.node.user_properties
    props.append(
        (
            USER_PROPERTY_STORE,
            (
                declaration.label(),
                declaration.deployment,
                declaration.declared,
                tuple(sorted(dendrux_kit_provided)),
            ),
        )
    )
    props.append((USER_PROPERTY_CATEGORY, category))
    check_declaration(store, declaration)
    for capability in needs:
        if not declaration.has(capability):
            pytest.skip(f"{capability} is not declared")
    _check_fixtures(dendrux_kit_provided, declaration, uses)
    return Kit(
        dendrux_kit_provided,
        store,
        declaration,
        needs_format_2=bool(marker.kwargs.get("format2", False)),
    )


def _check_fixtures(
    provided: dict[str, Any], declaration: Declaration, uses: tuple[str, ...]
) -> None:
    obligations = declaration.required_fixtures()
    for use in uses:
        name = _FIXTURE_FOR[use]
        if use == "second" and not declaration.needs_second_instance:
            pytest.skip(f"deployment {declaration.deployment}: there is no second instance")
        if name in provided:
            continue
        if name in obligations:
            raise CertificationFailure(f"{name} is required ({obligations[name]}) but missing")
        if name == FAULT_INJECTOR:
            pytest.skip(f"{FAULT_INJECTOR} not provided: crash recovery is not certified")
        pytest.skip(f"{name} not provided and not required for this store")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, Any]:
    report = yield
    if (
        call.when == "setup"
        and call.excinfo is not None
        and isinstance(call.excinfo.value, CertificationFailure)
    ):
        report.when = "call"  # a store that does not keep its promises fails, it does not error
    return report


def _function_name(nodeid: str) -> str:
    return nodeid.split("::")[-1].split("[")[0]


class _Reporter:
    """Tallies kit results per store; one instance per pytest session (and per xdist side)."""

    def __init__(self) -> None:
        self.report = _Report()

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        for item in items:
            marker = item.get_closest_marker(MARKER)
            if marker is not None:
                self.report.deselected.add(str(marker.args[0]))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        props: dict[str, Any] = dict(report.user_properties)
        if USER_PROPERTY_STORE not in props or report.when not in ("setup", "call"):
            return
        if report.when == "setup" and report.passed:
            return  # the call phase reports the outcome
        label, deployment, declared, provided = props[USER_PROPERTY_STORE]
        category: str = props[USER_PROPERTY_CATEGORY]
        entry = self.report.stores.setdefault(label, _StoreReport())
        entry.deployment, entry.declared, entry.provided = deployment, declared, frozenset(provided)
        outcome = "skipped" if report.skipped else ("passed" if report.passed else "failed")
        entry.results.setdefault(category, Counter())[outcome] += 1
        entry.seen.setdefault(category, set()).add(_function_name(report.nodeid))
        if report.skipped:
            reason = str(report.longrepr)
            if FAULT_INJECTOR in reason:
                entry.notes.append(f"crash recovery: not certified (no {FAULT_INJECTOR} fixture)")
            elif "unverified:" in reason:
                entry.notes.append(
                    "unverified: " + reason.split("unverified:", 1)[1].strip(" ')\"")
                )

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        _write_summary(terminalreporter, self.report)


def _problems(entry: _StoreReport, report: _Report, inventory: dict[str, set[str]]) -> list[str]:
    """Reasons this store is not certified, beyond failing contracts."""
    problems = []
    missing = [
        f"{name} ({why})"
        for name, why in required_fixtures_for(entry.declared, entry.deployment).items()
        if name not in entry.provided
    ]
    if missing:
        problems.append("declared, but the fixture is missing: " + "; ".join(missing))
    if entry.deployment is None:
        problems.append("deployment is not declared")
    incomplete = [
        category
        for category in CATEGORIES
        if category in inventory
        and (category in report.deselected or inventory[category] - entry.seen.get(category, set()))
    ]
    if incomplete:
        problems.append("incomplete run, not every contract ran: " + ", ".join(incomplete))
    return problems


def _write_summary(terminalreporter: Any, report: _Report) -> None:
    if not report.stores:
        return
    inventory = contract_inventory()
    terminalreporter.section("dendrux store kit: certification summary")
    for label, entry in sorted(report.stores.items()):
        terminalreporter.write_line(label)
        failed = 0
        for category in CATEGORIES:
            counts = entry.results.get(category)
            if not counts:
                continue
            failed += counts["failed"]
            line = ", ".join(
                f"{counts[k]} {k}" for k in ("passed", "failed", "skipped") if counts[k]
            )
            terminalreporter.write_line(f"  {category:<18} {line}")
        problems = _problems(entry, report, inventory)
        notes = list(dict.fromkeys(entry.notes))
        for note in notes:
            terminalreporter.write_line(f"  {note}")
        if failed or problems:
            reasons = ([f"{failed} failing contract(s)"] if failed else []) + problems
            terminalreporter.write_line(
                "  NOT CERTIFIED: " + " | ".join(reasons), bold=True, red=True
            )
        else:
            caveat = f" ({'; '.join(notes)})" if notes else ""
            ran = [c for c in CATEGORIES if entry.results.get(c, Counter())["passed"]]
            terminalreporter.write_line(f"  CERTIFIED: {', '.join(ran)}{caveat}", bold=True)
