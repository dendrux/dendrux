"""Test tiers: area markers by path, the slow and flaky lists, changed-file lint and CI shards."""

from __future__ import annotations

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
    monkeypatch.setenv("DENDRUX_TEST_WORKERS", "9")
    assert tiers.default_workers() == 9
    monkeypatch.setenv("DENDRUX_TEST_WORKERS", "junk")
    assert tiers.default_workers() == 2


def _collect(*args: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly", *args],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in __import__("os").environ.items() if k != "DENDRUX_TEST_PG_URL"},
    ).stdout
    return [line for line in out.splitlines() if "::" in line]


def test_conftest_marks_the_area_and_the_flaky_test_by_path_and_list() -> None:
    target = "tests/tooling/test_load_monitor.py"
    assert _collect(target, "-m", "area_tooling")
    assert not _collect(target, "-m", "area_runtime")
    flaky = _collect(target, "-m", "flaky")
    assert flaky == [f"{target}::test_sampler_starts_stops_and_still_records_final_sample"]


def test_the_makefile_exposes_the_fast_loop_and_the_merge_gate() -> None:
    makefile = (PKG / "Makefile").read_text()
    for target in ("test-fast:", "test-area:", "test-pg:", "test-kit:", "test-merge:"):
        assert f"\n{target}" in makefile
    fast = makefile.split("\ntest-fast:", 1)[1].split("\n\n", 1)[0]
    assert "--testmon" in fast and "--testmon-forceselect" in fast
    assert "--no-cov" in fast and "ruff-changed" in fast
    merge = makefile.split("\ntest-merge:", 1)[1].split("\n\n", 1)[0]
    for step in ("$(MAKE) ci", "$(MAKE) test-mysql", "$(MAKE) test-kit"):
        assert step in merge
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
    assert f"-eq {count}" in combine["Combine coverage and apply the floor"]["run"]


def test_the_coverage_job_covers_the_same_legs_as_the_test_job() -> None:
    jobs = _workflow()
    for key in ("python-version", "sqlalchemy-version", "exclude"):
        assert (
            jobs["python"]["strategy"]["matrix"][key]
            == jobs["python-coverage"]["strategy"]["matrix"][key]
        )
