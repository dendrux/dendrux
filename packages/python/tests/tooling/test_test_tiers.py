"""Test tiers: area markers by path, the slow and flaky lists, changed-file lint and CI shards."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._helpers import tiers
from tools import test_tiers as tool

PKG = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relpath", "area"),
    [
        ("db/test_060_p5_mysql_ddl.py", "db"),
        ("migrations/test_060_activate.py", "db"),
        ("docs/test_doc_snippets.py", "docs"),
        ("tooling/test_load_monitor.py", "tooling"),
        ("tools/test_import_cycles.py", "tooling"),
        ("conformance/test_matrix.py", "testing"),
        ("live/test_images_live.py", "live"),
        ("unit/test_mcp_client.py", "mcp"),
        ("integration/test_skills_v2.py", "skills"),
        ("unit/test_openai_provider.py", "llm"),
        ("integration/test_mysql_store_races.py", "stores"),
        ("unit/test_050_t1_follow.py", "store"),
        ("unit/test_streaming.py", "loops"),
        ("unit/test_guardrails.py", "guardrails"),
        ("unit/test_media_fetch.py", "media"),
        ("unit/test_cli_commands.py", "cli"),
        ("unit/test_notifier_writer.py", "notifiers"),
        ("unit/test_tool_pool.py", "tools"),
        ("unit/test_strategies.py", "strategies"),
        ("integration/test_cancellation.py", "runtime"),
    ],
)
def test_area_comes_from_the_directory_then_the_file_name(relpath: str, area: str) -> None:
    assert tiers.area_for(relpath) == area


def test_every_test_file_gets_a_registered_area() -> None:
    files = tool.test_files()
    assert len(files) > 300
    for name in files:
        assert tiers.area_for(Path(name).relative_to("tests").as_posix()) in tiers.AREAS


def test_every_area_has_at_least_one_test_file() -> None:
    used = {tiers.area_for(Path(f).relative_to("tests").as_posix()) for f in tool.test_files()}
    assert used == set(tiers.AREAS)


def test_list_parsing_keeps_the_reason_and_ignores_comments() -> None:
    entries = tiers.parse_list(
        "# header\n\ntests/a.py::test_x | too slow here\ntests/b.py::test_y[p1]\n"
    )
    assert [(e.nodeid, e.note) for e in entries] == [
        ("tests/a.py::test_x", "too slow here"),
        ("tests/b.py::test_y[p1]", ""),
    ]


def test_a_short_entry_matches_every_parametrization_and_a_full_one_only_itself() -> None:
    entries = tiers.parse_list("tests/a.py::test_x\ntests/b.py::test_y[p1]\n")
    exact, prefixes = tiers.listed(entries)
    assert tiers.in_list(exact, prefixes, "tests/a.py::test_x")
    assert tiers.in_list(exact, prefixes, "tests/a.py::test_x[sqlite-1]")
    assert tiers.in_list(exact, prefixes, "tests/b.py::test_y[p1]")
    assert not tiers.in_list(exact, prefixes, "tests/b.py::test_y[p2]")
    assert not tiers.in_list(exact, prefixes, "tests/a.py::test_xx")


def test_the_committed_lists_name_real_tests_and_every_flaky_entry_has_a_reason() -> None:
    assert tool.check_lists() == []
    flaky = tiers.load_list("flaky.txt")
    assert flaky
    assert all(entry.note for entry in flaky)


def test_check_lists_reports_a_missing_test_and_a_missing_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "flaky.txt").write_text(
        "tests/tooling/test_test_tiers.py::test_does_not_exist | why\n"
        "tests/tooling/test_test_tiers.py::test_list_parsing_keeps_the_reason_and_ignores_comments\n"
        "tests/nope.py::test_a | why\n"
    )
    monkeypatch.setattr(tiers, "LISTS_DIR", tmp_path)
    problems = tool.check_lists()
    assert any("no such test" in p for p in problems)
    assert any("a reason is required" in p for p in problems)
    assert any("no such file" in p for p in problems)


DURATIONS = """\
slowest durations
12.50s call     tests/unit/test_a.py::test_one
 1.50s setup    tests/unit/test_a.py::test_one
 0.40s call     tests/unit/test_a.py::test_two[x-1]
 6.00s call     tests/unit/test_b.py::TestK::test_three
(100 durations < 0.005s hidden.  Use -vv to show these durations.)
"""


def test_durations_are_summed_per_test_and_the_threshold_picks_the_slow_ones() -> None:
    durations = tool.parse_durations(DURATIONS)
    assert durations["tests/unit/test_a.py::test_one"] == pytest.approx(14.0)
    assert tool.slow_entries(durations, 5.0) == [
        "tests/unit/test_a.py::test_one",
        "tests/unit/test_b.py::TestK::test_three",
    ]
    assert tool.file_weights(durations) == {
        "tests/unit/test_a.py": 14.4,
        "tests/unit/test_b.py": 6.0,
    }


def test_shards_are_a_deterministic_balanced_disjoint_cover() -> None:
    files = [f"tests/unit/test_{i}.py" for i in range(40)]
    weights = {f: float(i % 7 + 1) for i, f in enumerate(files)}
    first = tool.make_shards(files, 4, weights)
    assert first == tool.make_shards(list(reversed(files)), 4, weights)
    flat = [f for shard in first for f in shard]
    assert sorted(flat) == sorted(files) and len(set(flat)) == len(files)
    loads = [sum(weights[f] for f in shard) for shard in first]
    assert max(loads) - min(loads) <= max(weights.values())


def test_a_file_without_a_recorded_weight_still_lands_in_exactly_one_shard() -> None:
    files = ["a.py", "b.py", "new.py"]
    shards = tool.make_shards(files, 2, {"a.py": 10.0, "b.py": 1.0})
    assert sorted(f for shard in shards for f in shard) == files


@pytest.mark.parametrize("count", [1, 3, 4, 7])
def test_the_real_test_files_split_into_a_disjoint_cover(count: int) -> None:
    files = tool.test_files()
    shards = tool.make_shards(files, count, tool.load_weights())
    flat = [f for shard in shards for f in shard]
    assert sorted(flat) == files and len(set(flat)) == len(files)
    assert all(shards), "no shard may be empty"


def test_default_workers_is_capped_and_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DENDRUX_TEST_WORKERS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert tiers.default_workers() == 6
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert tiers.default_workers() == 2
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    monkeypatch.setenv("DENDRUX_TEST_WORKERS", "9")
    assert tiers.default_workers() == 6  # the env var can lower the cap, never raise it
    monkeypatch.setenv("DENDRUX_TEST_WORKERS", "3")
    assert tiers.default_workers() == 3
    monkeypatch.setenv("DENDRUX_TEST_WORKERS", "junk")
    assert tiers.default_workers() == 6


def _collect(*args: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly", *args],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "DENDRUX_TEST_PG_URL"},
    ).stdout
    return [line for line in out.splitlines() if "::" in line]


def test_conftest_marks_the_area_and_the_flaky_test_by_path_and_list() -> None:
    target = "tests/integration/test_050_t7b_fix3.py"
    assert _collect(target, "-m", "area_runtime")  # no file-name rule matches: the default area
    assert not _collect(target, "-m", "area_stores")
    flaky = _collect(target, "-m", "flaky")
    assert flaky == [f"{target}::test_online_clock_migration_writer_block[postgres-300k]"]


def _run_verbose(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "DENDRUX_TEST_PG_URL"}
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--no-cov", "-p", "no:randomly", *args],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return out.stdout


def test_a_benchmark_selection_runs_in_one_process_even_with_n_given() -> None:
    target = "tests/integration/test_050_t7b_fix2.py"
    assert "[gw" not in _run_verbose("-m", "benchmark", "-n", "3", target)
    assert "[gw" in _run_verbose("-m", "not benchmark", "-n", "3", target)  # control


def test_the_makefile_exposes_the_fast_loop_and_the_merge_gate() -> None:
    makefile = (PKG / "Makefile").read_text()
    for target in ("test-fast:", "test-area:", "test-pg:", "test-kit:", "test-merge:"):
        assert f"\n{target}" in makefile
    fast = makefile.split("\ntest-fast:", 1)[1].split("\n\n", 1)[0]
    assert "--testmon" in fast and "--testmon-forceselect" in fast
    assert "--no-cov" in fast and "ruff-changed" in fast
    merge = makefile.split("\ntest-merge:", 1)[1].split("\n\n", 1)[0]
    for step in ("$(MAKE) ci", "$(MAKE) test-kit", "$(MAKE) bench"):
        assert step in merge
    assert "test-mysql" not in merge, "the local merge gate is SQLite and Postgres only"
    assert "-n 0" in makefile.split("\nbench:", 1)[1].split("\n\n", 1)[0]
    ci_line = next(line for line in makefile.splitlines() if line.startswith("ci:"))
    assert ci_line.split() == [
        "ci:",
        "lint",
        "typecheck",
        "arch",
        "test",
        "crap",
        "quality",
        "api-check",
    ]


def test_the_dev_extra_pins_the_fast_loop_and_rerun_plugins() -> None:
    import tomllib

    dev = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    assert "pytest-testmon==2.2.0" in dev
    assert "pytest-rerunfailures==16.7" in dev


def _workflow() -> dict:
    import yaml

    path = PKG.parents[1] / ".github" / "workflows" / "ci.yml"
    if not path.is_file():
        pytest.skip("repository checkout only")
    return yaml.safe_load(path.read_text())["jobs"]


def test_the_ci_test_job_runs_every_shard_and_only_once_the_per_leg_checks() -> None:
    import re

    job = _workflow()["python"]
    shards = job["strategy"]["matrix"]["shard"]
    steps = {s.get("name", ""): s for s in job["steps"]}
    count = int(re.search(r"--count (\d+)", steps["Test"]["run"]).group(1))  # type: ignore[union-attr]
    assert shards == list(range(1, count + 1))
    assert "--index ${{ matrix.shard }}" in steps["Test"]["run"]
    assert '-m "not benchmark"' in steps["Test"]["run"]
    for once in ("Lint", "Format check", "Type check", "Architecture rules"):
        assert steps[once]["if"] == "matrix.shard == 1", once
    assert steps["Build and smoke-test distributions"]["if"] == "matrix.shard == 1"
    combine = {s.get("name", ""): s for s in _workflow()["python-coverage"]["steps"]}
    run = combine["Combine coverage and apply the floor"]["run"]
    assert f"for shard in {' '.join(map(str, shards))};" in run
    assert 'test -s "cov-$shard.data"' in run


def test_the_coverage_job_covers_the_same_legs_as_the_test_job() -> None:
    jobs = _workflow()
    for key in ("python-version", "sqlalchemy-version", "exclude"):
        assert (
            jobs["python"]["strategy"]["matrix"][key]
            == jobs["python-coverage"]["strategy"]["matrix"][key]
        )


# ---- fix round: fresh checkout, benchmark-safe default, hard worker cap -----------------------


def _run_make_n(*args: str) -> str:
    out = subprocess.run(
        ["make", "-n", *args],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if not k.startswith("DENDRUX_TEST_")},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_test_fast_creates_the_testmon_data_directory_before_pytest(tmp_path: Path) -> None:
    datafile = tmp_path / "absent" / "cache" / "testmondata"
    recipe = _run_make_n("test-fast", f"TESTMON_DATAFILE={datafile}")
    assert recipe.index("prepare-testmon") < recipe.index("--testmon")

    env = {**os.environ, "TESTMON_DATAFILE": str(datafile), "PYTHONPATH": "src"}
    env.pop("DENDRUX_TEST_PG_URL", None)
    selection = ["tests/unit/test_version.py", "--testmon", "--no-cov", "-q", "-p", "no:randomly"]
    cold = subprocess.run(
        [sys.executable, "-m", "pytest", *selection],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert cold.returncode != 0, "without the directory testmon cannot open its data file"

    prepare = subprocess.run(
        [sys.executable, "tools/test_tiers.py", "prepare-testmon"],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert prepare.returncode == 0 and datafile.parent.is_dir()
    warm = subprocess.run(
        [sys.executable, "-m", "pytest", *selection],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert warm.returncode == 0, warm.stdout + warm.stderr


def _hook_option(markexpr: str, numprocesses: int | None = None) -> object:
    from types import SimpleNamespace

    option = SimpleNamespace(
        numprocesses=numprocesses, markexpr=markexpr, collectonly=False, dist="no"
    )
    return SimpleNamespace(option=option, args=["tests/integration"])


@pytest.mark.parametrize(
    ("markexpr", "numprocesses", "expected"),
    [
        ("", None, None),  # no -m: the selection can include benchmarks, so it stays serial
        ("slow", None, None),
        ("benchmark", None, 0),
        ("benchmark", 4, 0),
        ("not benchmark", None, 6),
        ("not benchmark and not slow", None, 6),
        ("not slow and not benchmark", None, 6),
        ("  not benchmark  ", None, 6),
        ("not benchmark or flaky", None, None),  # the flaky 300k benchmark would run in parallel
        ("flaky or not benchmark", None, None),
        ("not benchmark and area_runtime", None, None),
        ("not (benchmark)", None, 0),  # not recognised as "not benchmark": forced serial
        ("not benchmark", 3, 3),  # an explicit -n is respected
        ("not benchmark", 0, 0),
    ],
)
def test_default_parallelism_only_applies_to_selections_that_exclude_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
    markexpr: str,
    numprocesses: int | None,
    expected: int | None,
) -> None:
    from tests import conftest as root_conftest

    monkeypatch.setenv("DENDRUX_TEST_PG_URL", "postgresql+asyncpg://u:p@localhost/x_test")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("DENDRUX_TEST_WORKERS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 11)
    config = _hook_option(markexpr, numprocesses)
    root_conftest.pytest_cmdline_main(config)  # type: ignore[arg-type]
    assert config.option.numprocesses == expected  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "target", ["test", "test-fast", "test-area AREA=mcp", "test-pg", "test-mysql"]
)
@pytest.mark.parametrize("override", ["TEST_WORKERS=12", "TEST_WORKERS=auto", "TEST_WORKERS=100"])
def test_the_makefile_never_asks_for_more_than_six_workers(target: str, override: str) -> None:
    env_extra = {"DENDRUX_TEST_PG_URL": "postgresql+asyncpg://u:p@localhost/x_test"}
    out = subprocess.run(
        ["make", "-n", *target.split(), override],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env_extra},
    ).stdout
    import re

    counts = re.findall(r"(?<!test )-n (\S+)", out)
    assert counts, out
    assert all(c.isdigit() and 1 <= int(c) <= 6 for c in counts), counts


@pytest.mark.parametrize("cpus", [None, 1, 2, 4, 11])
def test_the_makefile_cap_also_holds_for_the_environment_and_a_lower_request(
    tmp_path: Path, cpus: int | None
) -> None:
    (tmp_path / "sitecustomize.py").write_text(f"import os\nos.cpu_count = lambda: {cpus!r}\n")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DENDRUX_TEST_")
        and k not in {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "TEST_WORKERS"}
    }
    env.update(PYTHONPATH=str(tmp_path) + os.pathsep + env.get("PYTHONPATH", ""), TEST_WORKERS="64")
    for request in ([], ["TEST_WORKERS=2"]):
        result = subprocess.run(
            ["make", "-n", "test-fast", *request],
            cwd=PKG,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        expected = min(cpus or 1, 6, 2 if request else 64)
        assert f"-n {expected} " in result.stdout, result.stdout


def test_quality_guide_says_the_fast_loop_excludes_slow_tests() -> None:
    text = (PKG / "tools" / "QUALITY.md").read_text()
    assert "never runs the tests marked `slow`" in text
    assert "`make ci` at hand-back" in text


# ---- wall-clock assertions are benchmarks, not parallel-suite tests ---------------------------

SPEED_TEST = """
import time

async def test_fast(loop):
    started = loop.time()
    await work()
    assert loop.time() - started < 0.25

def test_tripwire():
    start = time.monotonic()
    work()
    elapsed = time.monotonic() - start
    assert elapsed < 10

class TestK:
    def test_derived(self):
        t0 = time.perf_counter()
        work()
        took = time.perf_counter() - t0
        pause = took * 2
        assert pause <= 1.5

def test_not_time():
    value = len("abc")
    assert value < 0.25
"""


def test_the_scanner_finds_tight_wall_clock_bounds_and_names_the_test(tmp_path: Path) -> None:
    (tmp_path / "test_x.py").write_text(SPEED_TEST)
    found = tool.wallclock_asserts(tmp_path)
    assert [(f.test, f.bound) for f in found] == [
        ("test_x.py::test_fast", 0.25),
        ("test_x.py::TestK::test_derived", 1.5),
    ]


@pytest.mark.parametrize(
    "mutation", ["append((previous, current))", "extend([(previous, current)])"]
)
def test_scanner_traces_clock_samples_through_collections_and_comprehensions(
    tmp_path: Path, mutation: str
) -> None:
    (tmp_path / "test_samples.py").write_text(
        "def test_stalls(loop):\n"
        "    intervals = []\n"
        "    def ticker():\n"
        "        previous = loop.time()\n"
        "        current = loop.time()\n"
        f"        intervals.{mutation}\n"
        "    stalls = [min(end - start, 1.0) for start, end in intervals]\n"
        "    worst = max(stalls)\n"
        "    assert worst < 0.025\n"
        "\n"
        "def test_numeric():\n"
        "    unused_clock = time.monotonic()\n"
        "    samples = []\n"
        "    samples.append((1, 2))\n"
        "    values = [end - start for start, end in samples]\n"
        "    assert max(values) < 0.25\n"
    )
    assert [(f.test, f.bound) for f in tool.wallclock_asserts(tmp_path)] == [
        ("test_samples.py::test_stalls", 0.025)
    ]


def test_scanner_finds_the_real_derived_ticker_stall_bound() -> None:
    expected = (
        "unit/test_client_validation_scheduling.py::"
        "test_near_megabyte_validation_keeps_ticker_responsive"
    )
    found = [f for f in tool.wallclock_asserts(PKG / "tests") if f.test == expected]
    assert len(found) == 1 and found[0].bound == 0.025


def test_client_validation_speed_cases_run_only_in_the_serial_benchmark_selection() -> None:
    target = "tests/unit/test_client_validation_scheduling.py"
    name = "test_near_megabyte_validation_keeps_ticker_responsive"
    assert sum(name in line for line in _collect(target, "-m", "benchmark")) == 6
    assert not any(name in line for line in _collect(target, "-m", "not benchmark"))
    assert any(
        "test_small_validation_stays_inline" in line
        for line in _collect(target, "-m", "not benchmark")
    )


def test_every_tight_wall_clock_assertion_is_a_benchmark_or_reviewed() -> None:
    """The class of flake: a latency bound in a test that runs in the parallel suite."""
    covered = {
        e.nodeid
        for name in ("benchmark.txt", "wallclock_reviewed.txt")
        for e in tiers.load_list(name)
    }
    unlisted = [
        f"{f.test}: {f.text}"
        for f in tool.wallclock_asserts(PKG / "tests")
        if f"tests/{f.test}" not in covered and not f.marked
    ]
    assert not unlisted, (
        "a measured time is compared with a small number in a test that runs in parallel; mark it "
        "benchmark (tests/_helpers/tier_lists/benchmark.txt) or, if the bound is a generous hang "
        "tripwire, review it into wallclock_reviewed.txt:\n" + "\n".join(unlisted)
    )


def test_the_notifier_latency_test_is_selected_as_a_benchmark() -> None:
    target = "tests/integration/test_060_notifier_pg.py"
    benchmarks = _collect(target, "-m", "benchmark")
    assert any("test_n1_commit_wakes_under_50ms" in line for line in benchmarks)
    assert not any(
        "test_n1_commit_wakes_under_50ms" in line
        for line in _collect(target, "-m", "not benchmark")
    )


def test_all_four_tier_lists_are_checked_for_real_tests_and_reasons() -> None:
    assert tool.check_lists() == []
    for name in ("benchmark.txt", "wallclock_reviewed.txt"):
        entries = tiers.load_list(name)
        assert entries and all(e.note for e in entries), name


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("await asyncio.sleep(0.05)\nassert seen", True),
        ("await asyncio.sleep(3)\nassert store.count >= 2", True),
        ("if ready:\n    await asyncio.sleep(0.1)\n    assert finished", True),
        ("await asyncio.sleep(0.05)\nassert not finished", False),
        ("await asyncio.sleep(0.05)\nassert finished is False", False),
        ("await asyncio.sleep(0.05)\nassert calls == []", False),
        ("await asyncio.sleep(0.05)\nassert count == 0", False),
        ("await asyncio.sleep(0)\nassert seen", False),
        ("await asyncio.sleep(delay)\nassert seen", False),
        ("await sleep(0.05)\nassert seen", False),
        ("await asyncio.sleep(0.05)\nawait done.wait()\nassert seen", False),
    ],
)
def test_scanner_flags_sleep_immediately_before_positive_state_assertion(
    tmp_path: Path, body: str, expected: bool
) -> None:
    import textwrap

    (tmp_path / "test_sleep.py").write_text(
        "async def test_state():\n" + textwrap.indent(body, "    ") + "\n"
    )
    found = tool.wallclock_asserts(tmp_path)
    assert bool(found) is expected
    if expected:
        assert found[0].test == "test_sleep.py::test_state"
        assert "sleep" in found[0].text


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("await asyncio.wait_for(store.read(), 0.5)", True),
        ("await asyncio.wait_for(store.read(), timeout=0.5)", True),
        ("async with asyncio.timeout(0.1):\n    await store.read()", True),
        ("await store.read(timeout=0.2)", True),
        ("feed = Feed(store, max_duration_s=0.3)", True),
        ("await store.read(timeout=1)", False),
        ("await store.read(timeout=0)", False),
        ("await store.read(timeout=-1)", False),
        ("await store.read(timeout=True)", False),
        ("await store.read(timeout=delay)", False),
        ("await asyncio.wait_for(store.read(), 10)", False),
        ("feed = Feed(store, batch_size=0.2)", False),
    ],
)
def test_scanner_flags_small_deadlines_only_for_integration_io(
    tmp_path: Path, body: str, expected: bool
) -> None:
    import textwrap

    for area, fixture in (("integration", "store"), ("unit", "store"), ("integration", "fake")):
        folder = tmp_path / area / fixture
        folder.mkdir(parents=True)
        (folder / "test_deadline.py").write_text(
            f"async def test_read({fixture}):\n" + textwrap.indent(body, "    ") + "\n"
        )
    found = tool.wallclock_asserts(tmp_path)
    assert [f.test for f in found] == (
        ["integration/store/test_deadline.py::test_read"] if expected else []
    )


def test_scanner_recognizes_integration_server_setup_and_preserves_benchmark_marks(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "integration"
    folder.mkdir()
    (folder / "test_server.py").write_text(
        "@pytest.mark.benchmark\n"
        "async def test_delivery():\n"
        "    server = await asyncio.start_server(handler, '127.0.0.1', 0)\n"
        "    await asyncio.wait_for(delivered.wait(), 0.1)\n"
    )
    found = tool.wallclock_asserts(tmp_path)
    assert [(f.test, f.bound, f.marked) for f in found] == [
        ("integration/test_server.py::test_delivery", 0.1, True)
    ]
