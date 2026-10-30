"""Tests for tools/api_breaks.py: the public API break gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import api_breaks as ab

COMMITTED_ALLOWLIST = Path(__file__).resolve().parents[2] / "tools" / "api_break_allowlist.json"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


V1 = """\
def keep(a, b=1):
    return a


def drop(a):
    return a


class Thing:
    def method(self, x, y=None):
        return x
"""

V2 = """\
def keep(a, b=2):
    return a


class Thing:
    def method(self, x):
        return x
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    pkg = tmp_path / "src" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(V1)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "v1")
    _git(tmp_path, "tag", "v1")
    (pkg / "__init__.py").write_text(V2)
    _git(tmp_path, "commit", "-qam", "v2")
    return tmp_path


def _brk(kind: str, path: str, param: str | None = None) -> ab.Break:
    return ab.Break(kind=kind, path=path, param=param, detail="")


def _entry(
    kind: str,
    paths: tuple[str, ...],
    b_item: str | None = "B1",
    param: str | None = None,
    note: str = "n",
) -> ab.AllowEntry:
    return ab.AllowEntry(kind=kind, paths=paths, param=param, b_item=b_item, note=note)


KNOWN = [f"B{i}" for i in range(1, 28)]


def _allowlist_file(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "allow.json"
    path.write_text(json.dumps({"known_b_items": KNOWN, "entries": entries}))
    return path


class TestComputeBreaks:
    def test_finds_removed_objects_and_parameter_changes_against_a_git_ref(
        self, repo: Path
    ) -> None:
        found = ab.compute_breaks("demo", repo, "v1", "src")
        got = {(b.kind, b.path, b.param) for b in found}
        assert got == {
            ("parameter-changed-default", "demo.keep", "b"),
            ("object-removed", "demo.drop", None),
            ("parameter-removed", "demo.Thing.method", "y"),
        }

    def test_unchanged_package_has_no_breaks(self, repo: Path) -> None:
        assert ab.compute_breaks("demo", repo, "HEAD", "src") == []

    def test_unknown_ref_is_a_clear_error(self, repo: Path) -> None:
        with pytest.raises(ValueError, match="v9"):
            ab.compute_breaks("demo", repo, "v9", "src")


class TestAllowlist:
    def test_loads_valid_entries(self, tmp_path: Path) -> None:
        path = _allowlist_file(
            tmp_path,
            [
                {"kind": "object-removed", "paths": ["demo.drop"], "b_item": "B3", "note": "x"},
                {
                    "kind": "object-removed",
                    "paths": ["demo.gone"],
                    "b_item": None,
                    "gap": "no migration note yet",
                },
            ],
        )
        entries = ab.load_allowlist(path)
        assert [e.b_item for e in entries] == ["B3", None]
        assert entries[1].note == "no migration note yet"

    @pytest.mark.parametrize(
        "bad",
        [
            {"kind": "object-removed", "paths": ["a"], "b_item": "B99", "note": "x"},
            {"kind": "object-removed", "paths": ["a"], "b_item": "banana", "note": "x"},
            {"kind": "object-removed", "paths": ["a"], "note": "no b_item key"},
            {"kind": "object-removed", "paths": ["a"], "b_item": None},
            {"kind": "object-removed", "paths": [], "b_item": "B1", "note": "x"},
            {"paths": ["a"], "b_item": "B1", "note": "x"},
        ],
    )
    def test_invalid_entries_are_rejected(self, tmp_path: Path, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=r"entries\[0\]"):
            ab.load_allowlist(_allowlist_file(tmp_path, [bad]))

    def test_committed_allowlist_is_valid(self) -> None:
        assert ab.load_allowlist(COMMITTED_ALLOWLIST)

    def test_060_migration_rows_can_be_referenced(self) -> None:
        known = set(json.loads(COMMITTED_ALLOWLIST.read_text())["known_b_items"])
        assert {"0.6:strict-tenancy", "0.6:governance", "0.6:cache-tokens"} <= known

    def test_committed_allowlist_has_no_gaps(self) -> None:
        assert [e for e in ab.load_allowlist(COMMITTED_ALLOWLIST) if e.b_item is None] == []

    def test_050_removals_are_not_carried_into_060_allowlist(self) -> None:
        paths = {p for e in ab.load_allowlist(COMMITTED_ALLOWLIST) for p in e.paths}
        assert not {"dendrux.http", "dendrux.dashboard", "dendrux.cli.main.dashboard"} & paths


class TestClassify:
    def test_listed_break_is_mapped(self) -> None:
        breaks = [_brk("object-removed", "demo.drop")]
        result = ab.classify(breaks, [_entry("object-removed", ("demo.drop",), "B3")])
        assert len(result.mapped) == 1
        assert result.unlisted == []

    def test_unlisted_break_is_reported(self) -> None:
        result = ab.classify([_brk("object-removed", "demo.new_break")], [])
        assert [b.path for b in result.unlisted] == ["demo.new_break"]

    def test_gap_entries_pass_but_are_listed_as_gaps(self) -> None:
        breaks = [_brk("object-removed", "demo.gone")]
        result = ab.classify(breaks, [_entry("object-removed", ("demo.gone",), None)])
        assert result.unlisted == []
        assert [b.path for b, _ in result.gaps] == ["demo.gone"]

    def test_kind_and_param_must_match(self) -> None:
        breaks = [_brk("parameter-changed-default", "demo.f", "tenant_id")]
        wrong_kind = _entry("parameter-removed", ("demo.f",), param="tenant_id")
        wrong_param = _entry("parameter-changed-default", ("demo.f",), param="other")
        assert ab.classify(breaks, [wrong_kind]).unlisted
        assert ab.classify(breaks, [wrong_param]).unlisted
        right = _entry("parameter-changed-default", ("demo.f",), param="tenant_id")
        assert not ab.classify(breaks, [right]).unlisted

    def test_entry_without_param_covers_any_param(self) -> None:
        breaks = [
            _brk("parameter-removed", "demo.f", "a"),
            _brk("parameter-removed", "demo.f", "b"),
        ]
        entry = _entry("parameter-removed", ("demo.f",))
        assert not ab.classify(breaks, [entry]).unlisted

    def test_entries_that_match_nothing_are_stale(self) -> None:
        entry = _entry("object-removed", ("demo.fixed_already",))
        assert [e for e, _ in ab.classify([], [entry]).stale] == [entry]


class TestMain:
    def _run(self, repo: Path, allow: Path, *extra: str) -> int:
        return ab.main(
            [
                "--package", "demo", "--repo", str(repo), "--search-path", "src",
                "--against", "v1", "--allowlist", str(allow), *extra,
            ]
        )  # fmt: skip

    def _full_allowlist(self, tmp_path: Path, gap: bool = False) -> Path:
        b_item = None if gap else "B7"

        def entry(kind: str, path: str, item: str | None) -> dict[str, Any]:
            base: dict[str, Any] = {"kind": kind, "paths": [path], "b_item": item}
            return base | ({"gap": "no note"} if item is None else {"note": "documented"})

        return _allowlist_file(
            tmp_path,
            [
                entry("parameter-changed-default", "demo.keep", "B7"),
                entry("object-removed", "demo.drop", b_item),
                entry("parameter-removed", "demo.Thing.method", "B7"),
            ],
        )

    def test_passes_when_every_break_is_listed(self, repo: Path, tmp_path: Path) -> None:
        assert self._run(repo, self._full_allowlist(tmp_path)) == 0

    def test_fails_on_unlisted_breaks(self, repo: Path, tmp_path: Path) -> None:
        allow = _allowlist_file(tmp_path, [])
        assert self._run(repo, allow) == 1

    def test_unmapped_gaps_fail_by_default(self, repo: Path, tmp_path: Path) -> None:
        allow = self._full_allowlist(tmp_path, gap=True)
        assert self._run(repo, allow) == 1

    def test_allow_gaps_is_an_explicit_informational_mode(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        allow = self._full_allowlist(tmp_path, gap=True)
        assert self._run(repo, allow, "--allow-gaps") == 0
        assert "GAP" in capsys.readouterr().out

    def test_allow_gaps_still_fails_on_unlisted_breaks(self, repo: Path, tmp_path: Path) -> None:
        assert self._run(repo, _allowlist_file(tmp_path, []), "--allow-gaps") == 1

    def test_report_names_each_break(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(repo, self._full_allowlist(tmp_path, gap=True))
        out = capsys.readouterr().out
        assert "demo.drop" in out
        assert "B7" in out
        assert "gap" in out.lower()


def test_060_baseline_is_last_release() -> None:
    assert ab.DEFAULT_AGAINST == "v0.5.0"


def test_060_gate_rejects_unlisted_parameter_on_allowlisted_method() -> None:
    entries = ab.load_allowlist(COMMITTED_ALLOWLIST)
    unexpected = _brk("parameter-changed-default", "dendrux.agent.Agent.__init__", "tool_threads")
    assert ab.classify([unexpected], entries).unlisted == [unexpected]


def test_documentary_returns_never_preapprove_future_return_changes() -> None:
    entries = ab.load_allowlist(COMMITTED_ALLOWLIST)
    unexpected = _brk("return-changed-type", "dendrux.runtime.state.StateStore.save_trace")
    assert ab.classify([unexpected], entries).unlisted == [unexpected]


def test_documentary_inventory_does_not_hide_stale_active_entries() -> None:
    entries = ab.load_allowlist(COMMITTED_ALLOWLIST)
    assert len(ab.classify([], entries).stale) == 4


def test_060_changes_reference_060_migration_rows() -> None:
    entries = ab.load_allowlist(COMMITTED_ALLOWLIST)
    assert all(e.b_item.startswith("0.6:") for e in entries)
