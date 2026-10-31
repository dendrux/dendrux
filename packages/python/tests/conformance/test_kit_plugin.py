"""The kit's own rules, checked end to end in inner pytest sessions (``pytester``).

Each test writes a small author ``conftest.py`` around the real SQLite store, runs the kit as
``pytest -p dendrux.testing.store_kit`` and reads the outcome: a declared capability whose
fixture is missing fails, an undeclared capability skips, ``deployment=None`` is not certified,
and a store that does not keep its promises fails.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest_plugins = ["pytester"]

AUTHOR = """
import pytest
import pytest_asyncio
from dataclasses import replace
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux.db.models import Base
from dendrux.db._settings import seed_settings
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

CAPS = {caps}

class Store(SQLiteStateStore):
    def __init__(self, engine):
        super().__init__(engine)
        self.capabilities = replace(self.capabilities, **CAPS)

{body}

@pytest_asyncio.fixture
async def dendrux_store_factory():
    engines = []
    async def factory():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda c: seed_settings(c, active=True))
        engines.append(engine)
        return Store(engine)
    yield factory
    for engine in engines:
        await engine.dispose()
"""
CLOCK = """
@pytest.fixture
def dendrux_clock():
    class Clock:
        def advance(self, delta): pass
    return Clock()
"""
REOPEN = """
@pytest.fixture
def dendrux_store_reopen():
    return lambda store: Store(store._engine)
"""
SECOND = """
@pytest.fixture
def dendrux_store_second_instance():
    return lambda store: Store(store._engine)
"""
BASE = {"deployment": "single_process"}
INNER = (
    "-p", "dendrux.testing.store_kit", "-p", "no:randomly", "-p", "no:cacheprovider",
    "-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function",
    "--no-cov",
)  # fmt: skip


def kit(pytester: pytest.Pytester, caps: dict[str, object], body: str = "", *extra: str, k: str):
    pytester.makeconftest(AUTHOR.format(caps=caps, body="\n".join(extra) + body))
    return pytester.runpytest(*INNER, "-k", k, "-rA")


def test_deployment_none_is_refused_and_named_in_the_summary(pytester: pytest.Pytester) -> None:
    result = kit(pytester, {"deployment": None}, k="test_deployment_is_declared")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*capabilities.deployment is not declared*"])
    result.stdout.fnmatch_lines(["*NOT CERTIFIED*"])


def test_unknown_deployment_is_not_certified(pytester: pytest.Pytester) -> None:
    result = kit(pytester, {"deployment": "cluster"}, k="test_deployment_is_declared")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*declares deployment='cluster'*"])


def test_declared_recovery_without_its_fixtures_fails(pytester: pytest.Pytester) -> None:
    result = kit(pytester, BASE, k="test_declared_capabilities_have_their_fixtures")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*dendrux_store_reopen*recovery is declared*"])
    result.stdout.fnmatch_lines(["*dendrux_clock*"])


def test_multi_server_without_a_second_instance_fails(pytester: pytest.Pytester) -> None:
    caps = {"deployment": "multi_server", "recovery": False}
    result = kit(pytester, caps, k="test_declared_capabilities_have_their_fixtures")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*dendrux_store_second_instance*multi_server*"])


def test_a_complete_declaration_passes_and_prints_a_summary(pytester: pytest.Pytester) -> None:
    result = kit(pytester, BASE, "", CLOCK, REOPEN, k="test_declared or test_deployment")
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        ["*dendrux store kit: certification summary*", "*declaration*3 passed*"]
    )


def test_a_declared_capability_whose_methods_are_missing_fails(pytester: pytest.Pytester) -> None:
    body = "_Complete = Store\nclass Store(_Complete):\n    save_trace = None\n"
    result = kit(pytester, BASE, body, CLOCK, REOPEN, k="test_deployment_is_declared")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*declares fenced_writes but save_trace lacks*"])


def test_the_plugin_collects_the_contracts_without_a_test_path(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    pytester.makeconftest(AUTHOR.format(caps=BASE, body=CLOCK + REOPEN))
    result = pytester.runpytest(*INNER, "--collect-only", "-q")
    result.stdout.fnmatch_lines(["*test_deployment_is_declared*"])


def test_without_activation_the_plugin_collects_nothing(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("pytest_plugins = ['dendrux.testing.store_kit']")
    pytester.makepyfile(test_mine=textwrap.dedent("def test_mine(): pass"))
    result = pytester.runpytest("-p", "no:randomly", "-p", "no:cacheprovider", "--no-cov", "-q")
    result.assert_outcomes(passed=1)


def test_an_undeclared_capability_skips_instead_of_failing(pytester: pytest.Pytester) -> None:
    caps = {
        "deployment": "single_process",
        "recovery": False,
        "checkpointing": False,
        "atomic_completion": False,
    }
    result = kit(pytester, caps, k="reclaim_gives_a_new_epoch or seeds_rewrites_and_conversation")
    result.assert_outcomes(skipped=2)
    assert "failed" not in result.parseoutcomes()


def test_a_declared_capability_with_a_missing_fixture_fails_its_contracts(
    pytester: pytest.Pytester,
) -> None:
    caps = {"deployment": "multi_server", "recovery": False}
    result = kit(pytester, caps, k="parallel_claims_have_exactly_one_winner")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*dendrux_store_second_instance is required*multi_server*"])


def test_without_a_fault_injector_crash_recovery_is_reported_not_certified(
    pytester: pytest.Pytester,
) -> None:
    result = kit(pytester, BASE, "", CLOCK, REOPEN, k="cancelled_write_is_not_half_applied")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*crash recovery: not certified*"])


@pytest.mark.parametrize("how", [["--dendrux-store-kit"], ["-o", "dendrux_store_kit=true"]])
def test_the_flag_and_the_ini_option_activate_a_plugin_that_was_only_loaded(
    pytester: pytest.Pytester, how: list[str]
) -> None:
    """A pytest11 entry point loads the plugin everywhere; only an opt-in collects contracts."""
    pytester.makeconftest(
        "pytest_plugins = ['dendrux.testing.store_kit']\n" + AUTHOR.format(caps=BASE, body="")
    )
    result = pytester.runpytest(
        "-p", "no:randomly", "-p", "no:cacheprovider", "--no-cov", "--collect-only", "-q", *how
    )  # fmt: skip
    assert "test_deployment_is_declared" in result.stdout.str()


def test_a_partial_run_is_never_certified(pytester: pytest.Pytester) -> None:
    """One selected basic test passes, but declared recovery has no reopen/clock fixture."""
    result = kit(pytester, BASE, k="test_create_run_round_trips_what_it_was_given")
    result.assert_outcomes(passed=1)
    text = result.stdout.str()
    assert "NOT CERTIFIED" in text, text[-800:]
    assert "incomplete run" in text
    assert "dendrux_store_reopen" in text and "dendrux_clock" in text
    assert "CERTIFIED: basic" not in text


def test_a_complete_enough_selection_still_names_what_was_not_run(
    pytester: pytest.Pytester,
) -> None:
    result = kit(
        pytester, BASE, "", CLOCK, REOPEN, k="test_create_run_round_trips_what_it_was_given"
    )
    text = result.stdout.str()
    assert "NOT CERTIFIED" in text and "incomplete run" in text
    assert "dendrux_store_reopen" not in text.split("NOT CERTIFIED")[1]


def test_without_a_clock_time_based_contracts_skip_as_unverified(
    pytester: pytest.Pytester,
) -> None:
    caps = {"deployment": "single_process", "recovery": False}
    result = kit(pytester, caps, k="sweeping_stale_running_runs_spares_fresh_ones")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*unverified: time-based sweeps and leases*"])
