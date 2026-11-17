"""Metadata pins must follow all declarations of an overloaded public API."""

import ast

import pytest

from tools._move_manifest import metadata


@pytest.mark.parametrize("async_prefix", ["", "async "])
@pytest.mark.parametrize("position", ["before", "between", "after"])
def test_overloaded_public_metadata_pin_order(async_prefix, position):
    stubs = (
        "from typing import overload\n"
        f"@overload\n{async_prefix}def run(value: int) -> int: ...\n"
        f"@overload\n{async_prefix}def run(value: str) -> str: ...\n"
    )
    implementation = f"{async_prefix}def run(value): return value\n"
    pin = 'run.__module__ = "dendrux.runtime.runner"\n'
    old = ast.parse(stubs + implementation)
    source = {
        "before": pin + stubs + implementation,
        "between": stubs + pin + implementation,
        "after": stubs + implementation + pin,
    }[position]
    trees = {"new": ast.parse(source)}
    entry = {
        "root": "run",
        "module": "new",
        "attribute": "__module__",
        "value": "dendrux.runtime.runner",
    }
    if position == "after":
        metadata(old, trees, entry, "dendrux.runtime.runner")
        assert len(trees["new"].body) == 4
    else:
        with pytest.raises(ValueError, match="metadata pin must follow its definition"):
            metadata(old, trees, entry, "dendrux.runtime.runner")
