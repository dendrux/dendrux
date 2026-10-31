"""The shipped SQLite reference fixtures certify the built-in store from a folder with no tests.

This is the in-repo twin of the installed-wheel proof: an inner session with no conftest and no
test files, only ``-p dendrux.testing.store_kit -p dendrux.testing.store_kit.sqlite_fixtures``.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("mode", ["memory", "file"])
def test_the_reference_fixtures_certify_sqlite_without_any_author_code(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("DENDRUX_KIT_SQLITE", mode)
    result = pytester.runpytest(
        "-p", "dendrux.testing.store_kit", "-p", "dendrux.testing.store_kit.sqlite_fixtures",
        "-p", "no:randomly", "-p", "no:cacheprovider", "--no-cov",
        "-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function",
    )  # fmt: skip
    outcomes = result.parseoutcomes()
    assert outcomes.get("failed", 0) == 0 and outcomes.get("passed", 0) >= 100
    text = result.stdout.str()
    assert "NOT CERTIFIED" not in text and "  CERTIFIED: declaration, basic" in text
    expected = "single_host" if mode == "file" else "single_process"
    result.stdout.fnmatch_lines([f"*deployment={expected}*"])
