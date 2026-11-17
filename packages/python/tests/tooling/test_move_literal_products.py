"""Proposed regression tests for inert integer-literal products, without arithmetic folding."""

from __future__ import annotations

import hashlib

import pytest

from tools.move_check import check_move


def _proof(expression: str, manifest_mode: bool, replacement: str | None = None):
    old = f"_RECOVERY_TRACE_BYTES = {expression}\n"
    after = {
        "old": "from new import _RECOVERY_TRACE_BYTES as _RECOVERY_TRACE_BYTES\n",
        "new": f"_RECOVERY_TRACE_BYTES = {replacement or expression}\n",
    }
    evidence = None
    if manifest_mode:
        evidence = {
            "version": 1,
            "base": "a" * 40,
            "old": "old",
            "source_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "sources": {
                key: hashlib.sha256(value.encode()).hexdigest() for key, value in after.items()
            },
            "owners": {"_RECOVERY_TRACE_BYTES": "new"},
            "lexical": [],
            "hoists": [],
            "annotations": [],
            "metadata": [],
            "scaffolding": {},
            "effects": [],
            "seams": {},
            "providers": {},
        }
    return check_move(old, after, require_reexports=True, move_manifest=evidence)


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_exact_integer_literal_product_moves(manifest_mode):
    report = _proof("1024 * 1024", manifest_mode)
    assert report.ok, report.problems
    assert report.moved == {"_RECOVERY_TRACE_BYTES": "new"}


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_existing_integer_literal_still_moves(manifest_mode):
    report = _proof("1048576", manifest_mode)
    assert report.ok, report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize(
    "expression",
    [
        "bound * 1024",
        "load() * 1024",
        "obj.limit * 1024",
        "1024 * bound",
        "'x' * 1024",
        "[0] * 1024",
        "True * 1024",
        "1024.0 * 1024",
        "1024 / 0",
        "1024 // 0",
        "1024 % 0",
        "1024 ** 1024",
        "(2 * 1024) * 1024",
        "[x for x in range(2)]",
    ],
)
def test_nonliteral_or_other_arithmetic_stays_immovable(manifest_mode, expression):
    report = _proof(expression, manifest_mode)
    assert not report.ok
    assert any(problem.startswith("immovable:") for problem in report.problems)


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize("replacement", ["1024 * 1025", "1048576", "1024 * 1024 + 1"])
def test_integer_product_ast_cannot_be_changed_or_folded(manifest_mode, replacement):
    report = _proof("1024 * 1024", manifest_mode, replacement)
    assert not report.ok
    assert any(problem.startswith("changed:") for problem in report.problems)
