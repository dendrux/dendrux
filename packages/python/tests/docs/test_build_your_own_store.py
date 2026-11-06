"""The "Build your own store" guide is true: the skeleton passes the kit, the hook table is exact.

The store and the ``conftest.py`` are lifted out of the guide's own text and written, unchanged,
into an empty folder. Only public imports may appear in them. The kit runs there as an author
would run it: ``pytest -p dendrux.testing.store_kit``.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from dendrux.runtime.stores.sql.base import SQLStateStore
from tests.docs._snippets import extract_snippets

REPO = Path(__file__).resolve().parents[4]
GUIDE = REPO / "docs/guides/build-your-own-store.mdx"
pytestmark = pytest.mark.skipif(not GUIDE.is_file(), reason="repository checkout only")

pytest_plugins = ["pytester"]

#: What a store author may import: the public surface only.
PUBLIC_MODULES = {
    "dendrux.store",
    "dendrux.db.models",
    "dendrux.testing.store_kit",
    "dendrux.testing.store_kit.sql",
}


def _blocks() -> dict[str, str]:
    snippets = extract_snippets(GUIDE, REPO)
    store = [s for s in snippets if "class MiniStore(SQLStateStore)" in s.code]
    conftest = [s for s in snippets if "def dendrux_store_factory" in s.code]
    assert len(store) == 1 and len(conftest) == 1, "the guide must show exactly one of each file"
    return {"my_store.py": store[0].code, "conftest.py": conftest[0].code}


def test_the_guides_skeleton_imports_only_public_modules() -> None:
    for name, code in _blocks().items():
        for module in re.findall(r"^\s*(?:from|import)\s+(dendrux[\w.]*)", code, re.MULTILINE):
            assert module in PUBLIC_MODULES or module == "dendrux", (name, module)


def test_the_guides_skeleton_and_conftest_certify_with_the_kit(pytester: pytest.Pytester) -> None:
    for name, code in _blocks().items():
        (pytester.path / name).write_text(code)
    result = pytester.runpytest(
        "-p", "dendrux.testing.store_kit", "-p", "no:cacheprovider", "-p", "no:randomly",
        "-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function",
        "--no-cov", "-q",
    )  # fmt: skip
    text = result.stdout.str()
    assert result.ret == 0, text[-3000:]
    assert "NOT CERTIFIED" not in text and "CERTIFIED with caveats" in text, text[-3000:]
    assert "MiniStore (deployment=single_host" in text


def _hook_rows() -> dict[str, str]:
    """Hook name -> the signature cell of its row in the guide's reference tables."""
    rows = {}
    for line in GUIDE.read_text().splitlines():
        match = re.match(r"^\| `((?:async )?[a-z_]+)\((.*?)\)(?: -> [^`]*)?`( \(.*?\))? \|", line)
        if match:
            rows[match.group(1).removeprefix("async ")] = line
    return rows


def test_every_hook_the_guide_documents_exists_with_the_parameters_it_lists() -> None:
    rows = _hook_rows()
    assert len(rows) >= 20, sorted(rows)
    for name, line in rows.items():
        method = getattr(SQLStateStore, name, None) or getattr(
            __import__("dendrux.runtime.stores.sql.mysql", fromlist=["x"]).MySQLStateStore, name
        )
        for parameter in inspect.signature(method).parameters:
            if parameter != "self":
                assert re.search(rf"\b{parameter}\b", line), (name, parameter)


def test_every_overridable_base_hook_is_documented() -> None:
    documented = set(_hook_rows())
    hooks = {
        "insert_if_absent_stmt", "upsert_setting", "json_get", "json_contains", "json_is_null",
        "json_patch_expr", "pause_projection", "metadata_filter_predicate",
        "idempotency_scope_predicate", "event_sequence_expr", "inserted_event_sequence",
        "lock_clause", "id_in", "child_page_query", "blob_now", "serialize_in_tx",
        "reserve_parent_writer", "connection_checks", "classify_db_error",
        "json_conversion_error", "statement_capacity", "events_for_runs_query",
    }  # fmt: skip
    assert hooks <= documented, sorted(hooks - documented)
    for name in hooks - {"statement_capacity"}:
        assert hasattr(SQLStateStore, name), name
