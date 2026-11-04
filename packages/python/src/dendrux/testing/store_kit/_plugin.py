"""The pytest plugin: collection of the contracts, the kit fixture and the report.

Loaded by ``dendrux.testing.store_kit`` only when the kit is switched on."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from _dendrux_store_kit_bootstrap import activated
from dendrux.testing.store_kit._declaration import (
    CAPABILITIES,
    CLOCK,
    FACTORY,
    FAULT_INJECTOR,
    LIVE_FACTORY,
    REOPEN,
    SECOND_INSTANCE,
    Declaration,
    required_fixtures_for,
)
from dendrux.testing.store_kit._kit import (
    MARKER,
    STORE_SKIP,
    USER_PROPERTY_CATEGORY,
    USER_PROPERTY_NEEDS,
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

#: Categories in report order. ``crash_recovery`` needs the optional fault injector;
#: ``live_updates`` needs the optional ``dendrux_live_store_factory``.
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
    "live_updates",
)
_FIXTURE_FOR = {
    "reopen": REOPEN,
    "clock": CLOCK,
    "second": SECOND_INSTANCE,
    "injector": FAULT_INJECTOR,
    "live": LIVE_FACTORY,
}


#: Capabilities whose promise the report must see exercised (``notify`` has no contracts to run).
CERTIFIABLE = tuple(name for name in CAPABILITIES if name != "notify")
_STORE_SKIP = re.compile(rf"{STORE_SKIP}\[([^\]]*)\]:\s*(.*)", re.DOTALL)
_SETUP_MESSAGE = (
    "The Dendrux store kit is switched on, but no {factory} fixture exists. Define it in your "
    "conftest.py as a fixture that returns a function making a new store over empty data; the "
    "other dendrux_* fixtures depend on what your store declares. See "
    "https://dendrux.dev/docs/guides/build-your-own-store (docs/guides/build-your-own-store.mdx)."
)


@dataclass(frozen=True)
class _StoreSkip:
    """A contract that skipped because of something the store controls."""

    function: str
    capabilities: tuple[str, ...]
    reason: str


@dataclass
class _StoreReport:
    name: str = ""
    identity: str = ""
    deployment: str | None = None
    declared: tuple[str, ...] = ()
    provided: frozenset[str] = frozenset()
    results: dict[str, Counter[str]] = field(default_factory=dict)
    seen: dict[str, set[str]] = field(default_factory=dict)  # category -> contract functions run
    notes: list[str] = field(default_factory=list)
    by_need: dict[str, Counter[str]] = field(default_factory=dict)  # capability -> outcomes
    store_skips: list[_StoreSkip] = field(default_factory=list)

    def label(self, *, qualified: bool) -> str:
        flags = ",".join(self.declared) or "none"
        name = self.identity if qualified else self.name
        return f"{name} (deployment={self.deployment}, capabilities={flags})"


@dataclass
class _Report:
    stores: dict[tuple[str, str, tuple[str, ...]], _StoreReport] = field(default_factory=dict)
    expected: dict[str, str] = field(default_factory=dict)  # contract node id -> category
    reported: set[str] = field(default_factory=set)  # node ids that produced a report


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "dendrux_kit(category, needs=(), uses=(), format2=False): a Dendrux store contract "
        "(set by dendrux.testing.store_kit)",
    )
    config.pluginmanager.register(_Reporter(), "dendrux-store-kit-report")
    if activated(config):
        target = str(CONTRACTS_DIR)
        if target not in config.args:
            config.args.append(target)


# ---------------------------------------------------------------------------
# The kit fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dendrux_kit_provided(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The author's fixtures that exist, resolved before the event loop starts running."""
    provided = {
        name: value
        for name in (FACTORY, REOPEN, SECOND_INSTANCE, CLOCK, FAULT_INJECTOR, LIVE_FACTORY)
        if (value := fixture_or_none(request, name)) is not None
    }
    if FACTORY not in provided:
        # Every contract needs the factory. Fail this one test with the full setup message and
        # stop the session (also under xdist, where pytest.exit would crash a worker) instead of
        # repeating the same failure for every contract.
        request.session.shouldstop = f"the {FACTORY} fixture is missing"
        raise CertificationFailure(_SETUP_MESSAGE.format(factory=FACTORY))
    return provided


@pytest_asyncio.fixture(name="dendrux_kit")
async def dendrux_kit(request: pytest.FixtureRequest, dendrux_kit_provided: dict[str, Any]) -> Kit:
    """A fresh store from the author's factory, checked against its own declaration."""
    marker = request.node.get_closest_marker(MARKER)
    if marker is None:
        raise RuntimeError("dendrux_kit is for tests marked with @contract(...)")
    category: str = marker.args[0]
    needs: tuple[str, ...] = tuple(marker.kwargs.get("needs", ()))
    uses: tuple[str, ...] = tuple(marker.kwargs.get("uses", ()))
    store = await maybe_await(dendrux_kit_provided[FACTORY]())
    declaration = Declaration.of(store)
    props = request.node.user_properties
    props.append(
        (
            USER_PROPERTY_STORE,
            (
                declaration.identity,
                declaration.store_name,
                declaration.deployment,
                declaration.declared,
                tuple(sorted(dendrux_kit_provided)),
            ),
        )
    )
    props.append((USER_PROPERTY_CATEGORY, category))
    props.append((USER_PROPERTY_NEEDS, needs))
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
        guards=needs,
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
        if name == LIVE_FACTORY:
            pytest.skip(f"{LIVE_FACTORY} not provided: live updates are not tested")
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

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        """Every contract variant that exists, before any ``-k``/``--deselect`` removes some."""
        for item in items:
            marker = item.get_closest_marker(MARKER)
            if marker is not None:
                self.report.expected[item.nodeid] = str(marker.args[0])

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """A worker hands its expected variants to the controller; the controller gates the exit."""
        output = getattr(session.config, "workeroutput", None)
        if output is not None:
            output["dendrux_kit_expected"] = dict(self.report.expected)
            return
        if (
            activated(session.config)
            and session.exitstatus == pytest.ExitCode.OK
            and (self._no_store_evaluated(session.config) or self._not_certified())
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _not_certified(self) -> bool:
        return any(not v.certified for v in _verdicts(self.report).values())

    def _no_store_evaluated(self, config: pytest.Config) -> bool:
        """Contracts were collected and run, yet none got far enough to evaluate a store."""
        collecting_only = bool(config.getoption("collectonly", default=False))
        return not collecting_only and bool(self.report.expected) and not self.report.stores

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        expected = (getattr(node, "workeroutput", None) or {}).get("dendrux_kit_expected")
        if expected:
            self.report.expected.update(expected)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when in ("setup", "call"):
            self.report.reported.add(report.nodeid)
        props: dict[str, Any] = dict(report.user_properties)
        if USER_PROPERTY_STORE not in props or report.when not in ("setup", "call"):
            return
        if report.when == "setup" and report.passed:
            return  # the call phase reports the outcome
        identity, name, deployment, declared, provided = props[USER_PROPERTY_STORE]
        category: str = props[USER_PROPERTY_CATEGORY]
        needs: tuple[str, ...] = tuple(props.get(USER_PROPERTY_NEEDS, ()))
        entry = self.report.stores.setdefault(
            (identity, str(deployment), tuple(declared)), _StoreReport()
        )
        entry.name, entry.identity = name, identity
        entry.deployment, entry.declared, entry.provided = deployment, declared, frozenset(provided)
        outcome = "skipped" if report.skipped else ("passed" if report.passed else "failed")
        entry.results.setdefault(category, Counter())[outcome] += 1
        function = _function_name(report.nodeid)
        entry.seen.setdefault(category, set()).add(function)
        for capability in needs:
            entry.by_need.setdefault(capability, Counter())[outcome] += 1
        if report.skipped:
            self._note_skip(entry, function, _skip_reason(report))

    @staticmethod
    def _note_skip(entry: _StoreReport, function: str, reason: str) -> None:
        by_store = _STORE_SKIP.match(reason)
        if by_store is not None:
            capabilities = tuple(c for c in by_store.group(1).split(",") if c)
            entry.store_skips.append(_StoreSkip(function, capabilities, by_store.group(2).strip()))
        elif FAULT_INJECTOR in reason:
            entry.notes.append(f"crash recovery: not certified (no {FAULT_INJECTOR} fixture)")
        elif LIVE_FACTORY in reason:
            entry.notes.append(f"live updates: not tested (no {LIVE_FACTORY} fixture)")
        elif "unverified:" in reason:
            entry.notes.append("unverified: " + reason.split("unverified:", 1)[1].strip(" ')\""))

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        config = terminalreporter.config
        if activated(config) and self._no_store_evaluated(config):
            terminalreporter.section("dendrux store kit: certification summary")
            terminalreporter.write_line(
                "  NOT CERTIFIED: incomplete run, no store evaluated: the "
                f"{FACTORY} fixture never produced a store (did it skip?)",
                bold=True,
                red=True,
            )
            return
        _write_summary(terminalreporter, self.report)


def _skip_reason(report: pytest.TestReport) -> str:
    longrepr = report.longrepr
    text = str(longrepr[2]) if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    return text.removeprefix("Skipped: ")


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
    unrun = {n: c for n, c in report.expected.items() if n not in report.reported}
    unseen = [
        category
        for category in CATEGORIES
        if category in inventory and inventory[category] - entry.seen.get(category, set())
    ]
    if unrun or unseen:
        categories = [c for c in CATEGORIES if c in set(unrun.values()) | set(unseen)]
        detail = ""
        if unrun:
            names = sorted({_function_name(n) for n in unrun})[:3]
            detail = (
                f" ({len(unrun)} of {len(report.expected)} contract variants never ran: "
                f"{', '.join(names)})"
            )
        problems.append("incomplete run, not every contract ran: " + ", ".join(categories) + detail)
    problems.extend(_unexercised(entry))
    return problems


def _unexercised(entry: _StoreReport) -> list[str]:
    """Declared capabilities whose contracts did not run, for a reason or for none."""
    problems = []
    declared = [c for c in CERTIFIABLE if c in entry.declared]
    reasons: dict[str, list[_StoreSkip]] = {}
    for skip in entry.store_skips:
        if any(c in declared for c in skip.capabilities):
            reasons.setdefault(skip.reason, []).append(skip)
    covered: set[str] = set()
    for reason, skips in reasons.items():
        affected = [c for c in declared if any(c in s.capabilities for s in skips)]
        covered.update(affected)
        problems.append(
            f"{', '.join(affected)}: {len(skips)} contract(s) skipped by the store ({reason})"
        )
    for capability in declared:
        counts = entry.by_need.get(capability)
        if capability not in covered and counts and not counts["passed"] and not counts["failed"]:
            problems.append(f"{capability}: declared, but every contract that checks it skipped")
    return problems


@dataclass
class _Verdict:
    failed: int
    problems: list[str]
    caveats: list[str]

    @property
    def certified(self) -> bool:
        return not self.failed and not self.problems


def _verdicts(report: _Report) -> dict[tuple[str, str, tuple[str, ...]], _Verdict]:
    inventory = contract_inventory()
    verdicts = {}
    for key, entry in report.stores.items():
        failed = sum(counts["failed"] for counts in entry.results.values())
        caveats = list(dict.fromkeys(entry.notes))
        for capability, reason, count in _store_skip_groups(entry):
            if capability not in entry.declared:
                caveats.append(f"{count} contract(s) skipped by the store, not tested: {reason}")
        verdicts[key] = _Verdict(failed, _problems(entry, report, inventory), caveats)
    return verdicts


def _store_skip_groups(entry: _StoreReport) -> list[tuple[str, str, int]]:
    """Contracts the store caused to skip: capability, reason and count, in report order."""
    groups: Counter[tuple[str, str]] = Counter()
    for skip in entry.store_skips:
        for capability in skip.capabilities or ("general",):
            groups[(capability, skip.reason)] += 1
    order = {name: i for i, name in enumerate((*CAPABILITIES, "general"))}
    return [
        (capability, reason, count)
        for (capability, reason), count in sorted(
            groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])
        )
    ]


def _write_summary(terminalreporter: Any, report: _Report) -> None:
    if not report.stores:
        return
    verdicts = _verdicts(report)
    names = Counter(name for name, _ in {(e.name, e.identity) for e in report.stores.values()})
    terminalreporter.section("dendrux store kit: certification summary")
    shown = sorted(
        report.stores.items(), key=lambda kv: kv[1].label(qualified=names[kv[1].name] > 1)
    )
    for key, entry in shown:
        verdict = verdicts[key]
        terminalreporter.write_line(entry.label(qualified=names[entry.name] > 1))
        for category in CATEGORIES:
            counts = entry.results.get(category)
            if not counts:
                continue
            line = ", ".join(
                f"{counts[k]} {k}" for k in ("passed", "failed", "skipped") if counts[k]
            )
            terminalreporter.write_line(f"  {category:<18} {line}")
        groups = _store_skip_groups(entry)
        if groups:
            terminalreporter.write_line("  skipped by the store (these promises were not tested):")
            for capability, reason, count in groups:
                terminalreporter.write_line(f"    {capability}: {count} contract(s): {reason}")
        notes = list(dict.fromkeys(entry.notes))
        for note in notes:
            terminalreporter.write_line(f"  {note}")
        if not verdict.certified:
            reasons = (
                [f"{verdict.failed} failing contract(s)"] if verdict.failed else []
            ) + verdict.problems
            terminalreporter.write_line(
                "  NOT CERTIFIED: " + " | ".join(reasons), bold=True, red=True
            )
        else:
            ran = ", ".join(c for c in CATEGORIES if entry.results.get(c, Counter())["passed"])
            if verdict.caveats:
                terminalreporter.write_line(
                    f"  CERTIFIED with caveats: {ran} (caveats: {'; '.join(verdict.caveats)})",
                    bold=True,
                )
            else:
                terminalreporter.write_line(f"  CERTIFIED: {ran}", bold=True)
