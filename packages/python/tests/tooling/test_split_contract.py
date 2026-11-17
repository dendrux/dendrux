"""One present-source contract must reject both families' facade backedges."""

from __future__ import annotations

import configparser
from pathlib import Path

import grimp
import pytest
from importlinter import configuration


def contract(graph, sources):
    configuration.configure()
    from tools.split_imports import PresentForbiddenContract

    return PresentForbiddenContract(
        "Split operation modules cannot import compatibility facades",
        {"root_packages": ["dendrux"]},
        {
            "source_modules": sources,
            "forbidden_modules": ["dendrux.runtime.runner", "dendrux.runtime.stores.sql.base"],
            "allow_indirect_imports": "True",
            "as_packages": "False",
        },
    ).check(graph, False)


def test_absent_owners_are_inactive() -> None:
    graph = grimp.ImportGraph()
    graph.add_module("dendrux.runtime.runner")
    graph.add_module("dendrux.runtime.stores.sql.base")
    assert contract(graph, ["dendrux.runtime._run_errors"]).kept


@pytest.mark.parametrize(
    "source,facade",
    [
        ("dendrux.runtime._run_errors", "dendrux.runtime.runner"),
        ("dendrux.runtime.stores.sql._sql_writes", "dendrux.runtime.stores.sql.base"),
    ],
)
def test_present_owner_cannot_import_facade(source: str, facade: str) -> None:
    graph = grimp.ImportGraph()
    for module in [source, "dendrux.runtime.runner", "dendrux.runtime.stores.sql.base"]:
        graph.add_module(module)
    graph.add_import(importer=source, imported=facade, line_number=1, line_contents="fixture")
    assert not contract(graph, [source]).kept


def test_unknown_source_is_not_silently_filtered() -> None:
    with pytest.raises(ValueError, match="unknown"):
        contract(grimp.ImportGraph(), ["dendrux.runtime._run_typo"])


def test_exact_contract_is_wired_without_weakening_existing_rules() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[2] / ".importlinter")
    section = cfg["importlinter:contract:split-no-facade-backedge"]
    assert section["type"] == "forbidden"
    assert len(section["source_modules"].split()) == 35
    assert section["as_packages"] == "False"
    assert "ignore_imports" not in section
