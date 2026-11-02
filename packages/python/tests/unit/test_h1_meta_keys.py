"""Metadata key ownership and compatibility of existing import paths."""

import ast
from pathlib import Path

import dendrux

ROOT = Path(dendrux.__file__).parent


def test_namespaced_metadata_keys_have_one_definition():
    copies = []
    for path in ROOT.rglob("*.py"):
        if path.name == "_meta_keys.py" or "migrations" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if not value.startswith("dendrux.") or any(c.isspace() for c in value):
                continue
            # Telemetry attributes and import/module names are separate vocabularies.
            if (
                path.name == "otel.py"
                or value
                in {
                    "dendrux.",
                    "dendrux.db",
                }
                or value.startswith("dendrux.testing.")
            ):
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "__module__"
                for target in parent.targets
            ):
                continue
            copies.append(f"{path.relative_to(ROOT)}:{node.lineno}: {value}")
    assert not copies, "Metadata key literals outside their home:\n" + "\n".join(copies)


def test_public_metadata_import_paths_keep_their_values():
    from dendrux import store, types
    from dendrux.runtime import _pending, persistence, replay, runner, state, submit
    from dendrux.skills import _session

    assert _pending.ERROR_INFO_META_KEY == store.ERROR_INFO_META_KEY == "dendrux.error_info"
    assert persistence.PARTS_META_KEY == replay.PARTS_KEY == "dendrux.parts"
    assert types.REPLAYED_META_KEY == "dendrux.replayed"
    assert state.PAUSE_REQUEST_META_KEY == "dendrux.pause_request"
    assert runner.STOP_REASON_META_KEY == "dendrux.stop_reason"
    assert submit.GOVERNANCE_KEY == "dendrux.governance"
    assert _session.META_KEY == "dendrux.skills"


def test_metadata_home_has_no_imports():
    path = ROOT / "_meta_keys.py"
    assert path.is_file(), "Metadata constants need an import-free home"
    tree = ast.parse(path.read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
