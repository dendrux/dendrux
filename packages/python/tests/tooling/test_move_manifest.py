"""Reviewed normalization must prove exact bindings rather than waive bodies."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.move_check import check_move


def manifest(source: str, after: dict[str, str], owners: dict[str, str], **entries: object) -> dict:
    return {
        "version": 1,
        "base": "a" * 40,
        "old": "old",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "sources": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in after.items()},
        "owners": owners,
        "lexical": [],
        "hoists": [],
        "annotations": [],
        "metadata": [],
        "scaffolding": {},
        "effects": [],
        "seams": {},
        "providers": {},
        **entries,
    }


def proof(old: str, after: dict[str, str], owners: dict[str, str], **entries: object):
    return check_move(old, after, move_manifest=manifest(old, after, owners, **entries))


def test_lexical_owner_rewrite_in_defaults_and_callbacks() -> None:
    old = "def helper(): return 1\ndef use(callback=helper): return helper, callback\n"
    after = {
        ("old"): (
            "from new import helper\nimport new as owner\ndef use(callback=owner"
            ".helper): return owner.helper, callback\n"
        ),
        "new": "def helper(): return 1\n",
    }
    report = proof(
        old,
        after,
        {"helper": "new", "use": "old"},
        lexical=[
            {"root": "use", "module": "old", "name": "helper", "target": "new.helper", "count": 2}
        ],
    )
    assert report.ok, report.problems


@pytest.mark.parametrize("replacement", ["wrong.helper", "owner.other", "helper"])
def test_wrong_or_copied_owner_is_refused(replacement: str) -> None:
    old = "def helper(): return 1\ndef use(): return helper()\n"
    after = {
        "old": "from new import helper\nimport new as owner\nimport wrong\ndef use(): return "
        + replacement
        + "()\n",
        "new": "def helper(): return 1\n",
    }
    assert not proof(
        old,
        after,
        {"helper": "new", "use": "old"},
        lexical=[
            {"root": "use", "module": "old", "name": "helper", "target": "new.helper", "count": 1}
        ],
    ).ok


@pytest.mark.parametrize("prefix", ["owner = None\n", "if False:\n import new as owner\n", ""])
def test_shadowed_rebound_or_type_only_owner_is_refused(prefix: str) -> None:
    old = "def helper(): return 1\ndef use(): return helper()\n"
    after = {
        "old": "from new import helper\n" + prefix + "def use(owner=None): return owner.helper()\n",
        "new": "def helper(): return 1\n",
    }
    assert not proof(
        old,
        after,
        {"helper": "new", "use": "old"},
        lexical=[
            {"root": "use", "module": "old", "name": "helper", "target": "new.helper", "count": 1}
        ],
    ).ok


def test_imported_seam_is_qualified_in_residual_and_moved_module() -> None:
    old = "from dependency import hook\ndef one(): return hook()\ndef two(): return hook()\n"
    after = {
        "old": "from new import one\nimport dependency\ndef two(): return dependency.hook()\n",
        "new": "import dependency\ndef one(): return dependency.hook()\n",
    }
    entries = [
        {"root": root, "module": module, "name": "hook", "target": "dependency.hook", "count": 1}
        for root, module in [("one", "new"), ("two", "old")]
    ]
    assert proof(
        old, after, {"one": "new", "two": "old"}, lexical=entries, seams={"hook": "dependency.hook"}
    ).ok
    for module, root in [("old", "two"), ("new", "one")]:
        copied = dict(after)
        copied[module] = (
            copied[module]
            .replace("import dependency", "from dependency import hook")
            .replace("dependency.hook()", "hook()")
        )
        assert not proof(
            old,
            copied,
            {"one": "new", "two": "old"},
            lexical=[e for e in entries if e["root"] != root],
            seams={"hook": "dependency.hook"},
        ).ok


def test_unlisted_body_change_and_unused_entry_fail_closed() -> None:
    old = "def f(): return 1\n"
    after = {"old": "from new import f\n", "new": old}
    assert proof(old, after, {"f": "new"}).ok
    assert not proof(old, {"old": after["old"], "new": old.replace("1", "2")}, {"f": "new"}).ok
    assert not proof(
        old,
        after,
        {"f": "new"},
        lexical=[
            {"root": "f", "module": "new", "name": "unused", "target": "x.unused", "count": 1}
        ],
    ).ok
    assert not proof(old, after, {}).ok


def test_hash_and_schema_evidence_fail_closed() -> None:
    old = "def f(): return 1\n"
    after = {"old": "from new import f\n", "new": old}
    original = manifest(old, after, {"f": "new"})
    for key, value in [
        ("source_sha256", "0" * 64),
        ("sources", {}),
        ("extra", True),
        ("version", 2),
    ]:
        bad = copy.deepcopy(original)
        bad[key] = value
        assert not check_move(old, after, move_manifest=bad).ok


def test_private_facade_alias_must_have_exact_owner() -> None:
    old = "def _f(): return 1\n"
    assert proof(old, {"old": "from new import _f\n", "new": old}, {"_f": "new"}).ok
    for alias in [
        "",
        "from wrong import _f\n",
        "from new import _f\n_f = None\n",
        "if False:\n from new import _f\n",
    ]:
        assert not proof(old, {"old": alias, "new": old}, {"_f": "new"}).ok


def test_safe_import_hoist() -> None:
    old = "def f():\n import math  # exact reason\n return math.pi\n"
    after = {
        "old": "from new import f\n",
        "new": "import math  # exact reason\ndef f():\n return math.pi\n",
    }
    entries = [
        {"root": "f", "module": "new", "statement": "import math", "comment": "# exact reason"}
    ]
    assert proof(old, after, {"f": "new"}, hoists=entries).ok
    for source in [
        "def f():\n if True:\n  import math  # exact reason\n return math.pi\n",
        "def f():\n print(1)\n import math  # exact reason\n return math.pi\n",
    ]:
        assert not proof(source, after, {"f": "new"}, hoists=entries).ok
    assert not proof(
        old,
        {"old": after["old"], "new": after["new"].replace("# exact reason", "# lost")},
        {"f": "new"},
        hoists=entries,
    ).ok


@pytest.mark.parametrize(
    "root,host",
    [
        ("_CheckedSessionFactory", "_StoreHost"),
        ("_creation_mark_proven", "_StoreHost"),
        ("_creation_metadata", "_StoreHost"),
        ("_SweepScan", "_SweepStore"),
    ],
)
def test_only_four_reviewed_store_annotations(root: str, host: str) -> None:
    old = f"def {root}(store: SQLStateStore): return store\n"
    after = {"old": f"from new import {root}\n", "new": old.replace("SQLStateStore", host)}
    edits = [{"root": root, "module": "new", "host": host, "count": 1}]
    assert proof(old, after, {root: "new"}, annotations=edits).ok
    assert not proof(old, after, {root: "new"}, annotations=[dict(edits[0], host="_Wrong")]).ok
    other = old.replace(root, "_CommitReadback")
    assert not proof(
        other,
        {"old": "from new import _CommitReadback\n", "new": other.replace("SQLStateStore", host)},
        {"_CommitReadback": "new"},
        annotations=[dict(edits[0], root="_CommitReadback")],
    ).ok


def test_metadata_is_only_original_class_or_public_runner_function() -> None:
    old = "class Error(Exception): pass\n"
    after = {"old": "from new import Error\n", "new": old + 'Error.__module__ = "old"\n'}
    entries = [{"root": "Error", "module": "new", "attribute": "__module__", "value": "old"}]
    assert proof(old, after, {"Error": "new"}, metadata=entries).ok
    assert not proof(old, after, {"Error": "new"}, metadata=[dict(entries[0], value="wrong")]).ok
    bad = {"old": after["old"], "new": old + 'Error.method.__module__ = "old"\n'}
    assert not proof(
        old, bad, {"Error": "new"}, metadata=[dict(entries[0], root="Error.method")]
    ).ok


def test_named_effects_are_not_general_statement_waivers() -> None:
    old = 'import logging\nlogger = logging.getLogger("old.literal")\n'
    after = {"old": "from new import logger\n", "new": old}
    assert proof(old, after, {"logger": "new"}, effects=["logger"]).ok
    assert not proof(
        old,
        {"old": after["old"], "new": old.replace("old.literal", "new.literal")},
        {"logger": "new"},
        effects=["logger"],
    ).ok
    assert not proof(
        "other = object()\n",
        {"old": "from new import other\n", "new": "other = object()\n"},
        {"other": "new"},
        effects=["other"],
    ).ok


def test_long_residual_or_moved_function_cannot_grow() -> None:
    old = "def f():\n" + " x = 1\n" * 81
    after = {"old": "from new import f\n", "new": old}
    assert proof(old, after, {"f": "new"}).ok
    assert not proof(
        old, {"old": after["old"], "new": old.replace("def f():\n", "def f():\n\n")}, {"f": "new"}
    ).ok
    assert not proof(old, {"old": old + "\n"}, {"f": "old"}).problems  # outside span is free
    assert not proof(old, {"old": old.replace("def f():\n", "def f():\n\n")}, {"f": "old"}).ok


def test_representative_manifest_is_tracked_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "moves"
    data = json.loads((fixture / "manifest.json").read_text())
    report = check_move(
        (fixture / "before.txt").read_text(),
        {key: (fixture / (key + ".txt")).read_text() for key in ("old", "new")},
        move_manifest=data,
    )
    assert report.ok, report.problems


def provider_sources() -> tuple[str, dict[str, str], dict[str, str]]:
    old = "class Store:\n def write(self): return 1\n def read(self): return self.write()\n"
    after = {
        ("old"): (
            "from reads import Reads\nfrom writes import Writes\nclass Store(Rea"
            "ds, Writes):\n pass\n"
        ),
        ("reads"): (
            "from writes import Writes\nclass Reads(Writes):\n def read(self): r"
            "eturn self.write()\n"
        ),
        "writes": "class Writes:\n def write(self): return 1\n",
    }
    return old, after, {"Store": "old", "Store.write": "writes", "Store.read": "reads"}


def test_real_provider_inheritance_keeps_method_proof() -> None:
    old, after, owners = provider_sources()
    report = proof(old, after, owners)
    assert report.ok, report.problems
    assert report.moved["Store.write"] == "writes"
    bad = dict(after)
    bad["writes"] = bad["writes"].replace("return 1", "return 2")
    assert not proof(old, bad, owners).ok


@pytest.mark.parametrize("change", ["wrong-binding", "unknown", "duplicate", "cycle", "c3"])
def test_provider_identity_dag_and_c3_fail_closed(change: str) -> None:
    old, after, owners = provider_sources()
    if change == "wrong-binding":
        after["reads"] = after["reads"].replace("from writes", "from wrong")
    elif change == "unknown":
        after["writes"] = after["writes"].replace("Writes:", "Writes(Unknown):")
    elif change == "duplicate":
        after["reads"] = after["reads"].replace("Reads(Writes)", "Reads(Writes, Writes)")
    elif change == "cycle":
        after["writes"] = "from reads import Reads\n" + after["writes"].replace(
            "Writes:", "Writes(Reads):"
        )
    else:
        after["old"] = after["old"].replace("Store(Reads, Writes)", "Store(Writes, Reads)")
    assert not proof(old, after, owners).ok


def test_external_provider_snapshot_preserves_own_winners() -> None:
    parent = "class Parent:\n def external(self): return 2\n"
    old = (
        "from dep import Parent\nclass Store(Parent):\n def read(self): retu"
        "rn self.write()\n def write(self): return 1\n"
    )
    after = {
        ("old"): (
            "from dep import Parent\nfrom reads import Reads\nfrom writes import"
            " Writes\nclass Store(Reads, Writes, Parent): pass\n"
        ),
        ("reads"): (
            "from dep import Parent\nclass Reads(Parent):\n def read(self): return self.write()\n"
        ),
        "writes": "class Writes:\n def write(self): return 1\n",
    }
    owners = {"Store": "old", "Store.read": "reads", "Store.write": "writes"}
    provider = {
        "dep.Parent": {
            "path": "src/dep.py",
            "sha256": hashlib.sha256(parent.encode()).hexdigest(),
            "class": parent,
        }
    }
    assert proof(old, after, owners, providers=provider).ok
    provider["dep.Parent"]["class"] = parent.replace("external", "write")
    provider["dep.Parent"]["sha256"] = hashlib.sha256(
        provider["dep.Parent"]["class"].encode()
    ).hexdigest()
    assert proof(old, after, owners, providers=provider).ok
    assert not proof(old, after, owners).ok
    after["reads"] = after["reads"].replace("from dep import Parent", "from wrong import Parent")
    assert not proof(old, after, owners, providers=provider).ok


def test_retained_super_call_keeps_original_next_winner() -> None:
    parent = "class Parent:\n def write(self): return 2\n"
    old = (
        "from dep import Parent\nclass Store(Parent):\n def write(self): ret"
        "urn 1\n def call(self): return super().write()\n"
    )
    after = {
        ("old"): (
            "from dep import Parent\nfrom writes import Writes\nclass Store(Writ"
            "es, Parent):\n def call(self): return super().write()\n"
        ),
        "writes": "class Writes:\n def write(self): return 1\n",
    }
    provider = {
        "dep.Parent": {
            "path": "src/dep.py",
            "sha256": hashlib.sha256(parent.encode()).hexdigest(),
            "class": parent,
        }
    }
    assert not proof(
        old,
        after,
        {"Store": "old", "Store.write": "writes", "Store.call": "old"},
        providers=provider,
    ).ok


def test_missing_original_metadata_pin_is_refused() -> None:
    old = "class Record: pass\n"
    assert not proof(old, {"old": "from new import Record\n", "new": old}, {"Record": "new"}).ok


def test_hoist_cannot_erase_a_parameter_or_optional_extra_boundary() -> None:
    for old in [
        "def f(math=None):\n import math\n return math.pi\n",
        "def f():\n import optional_extra\n return optional_extra.value\n",
    ]:
        stmt = "import math" if "math" in old else "import optional_extra"
        new = stmt + "\n" + old.replace(" " + stmt + "\n", "")
        assert not proof(
            old,
            {"old": "from new import f\n", "new": new},
            {"f": "new"},
            hoists=[{"root": "f", "module": "new", "statement": stmt, "comment": ""}],
        ).ok


def test_unchanged_body_cannot_import_a_different_dependency_or_logger() -> None:
    old = "from dependency import value\ndef f(): return value\n"
    after = {
        "old": "from new import f\n",
        "new": "from wrong import value\ndef f(): return value\n",
    }
    assert not proof(old, after, {"f": "new"}).ok
    old = 'import logging\nlogger = logging.getLogger("old.literal")\ndef f(): return logger\n'
    after = {
        "old": "from new import f, logger\n",
        ("new"): (
            'import logging\nlogger = logging.getLogger("old.literal")\ndef f(): return logger\n'
        ),
    }
    assert proof(old, after, {"logger": "new", "f": "new"}, effects=["logger"]).ok
    after = {
        "old": "from new import f\nfrom logs import logger\n",
        "new": "from wrong import logger\ndef f(): return logger\n",
        "logs": 'import logging\nlogger = logging.getLogger("old.literal")\n',
    }
    assert not proof(old, after, {"logger": "logs", "f": "new"}, effects=["logger"]).ok


def test_retained_skill_session_alias_is_the_only_special_import_owner() -> None:
    old = (
        "from dendrux.skills._session import current as current_skill_sess"
        "ion\ndef f(): return current_skill_session()\n"
    )
    entry = "dendrux.runtime._run_entry"
    after = {
        "old": f"import {entry} as owner\ndef f(): return owner.current_skill_session()\n",
        entry: "from dendrux.skills._session import current as current_skill_session\n",
    }
    edits = [
        {
            "root": "f",
            "module": "old",
            "name": "current_skill_session",
            "target": entry + ".current_skill_session",
            "count": 1,
        }
    ]
    assert proof(
        old,
        after,
        {"f": "old"},
        lexical=edits,
        seams={"current_skill_session": entry + ".current_skill_session"},
    ).ok
    after["old"] = old
    assert not proof(
        old, after, {"f": "old"}, seams={"current_skill_session": entry + ".current_skill_session"}
    ).ok


def test_all_281_private_runner_aliases_are_verified() -> None:
    runner = "dendrux.runtime.runner"
    old = "".join(f"def _f{i}(): return {i}\n" for i in range(281))
    after = {
        runner: "from new import _f0\n" + old.split("\n", 1)[1],
        "new": old.split("\n", 1)[0] + "\n",
    }
    owners = {f"_f{i}": runner for i in range(281)}
    owners["_f0"] = "new"
    report = check_move(
        old, after, old_key=runner, move_manifest=manifest(old, after, owners, old=runner)
    )
    assert report.ok, report.problems
    after[runner] = after[runner].replace("from new import _f0", "from wrong import _f0")
    assert not check_move(
        old, after, old_key=runner, move_manifest=manifest(old, after, owners, old=runner)
    ).ok


def test_host_scaffold_is_annotation_only_and_preserves_retained_signatures() -> None:
    old = "class Store:\n def write(self, value: int = 1) -> int: return value\n"
    host = (
        "class _StoreHost:\n _session_factory: object\n if TYPE_CHECKING:\n  "
        "def write(self, value: int = 1) -> int: ...\n"
    )
    after = {"old": old, "host": "from typing import TYPE_CHECKING\n" + host}
    owners = {"Store": "old", "Store.write": "old"}
    entries = {"_StoreHost": {"module": "host", "source": host}}
    assert proof(old, after, owners, scaffolding=entries).ok
    for bad in [
        host.replace("if TYPE_CHECKING:", "if True:"),
        host.replace("value: int", "value: object"),
        host.replace("-> int: ...", "-> int: return value"),
        host.replace("_session_factory: object", "write: object"),
    ]:
        bad_after = {"old": old, "host": "from typing import TYPE_CHECKING\n" + bad}
        assert not proof(
            old, bad_after, owners, scaffolding={"_StoreHost": {"module": "host", "source": bad}}
        ).ok


def test_transitive_provider_remains_a_proven_host() -> None:
    old, after, owners = provider_sources()
    after["old"] = after["old"].replace("Store(Reads, Writes)", "Store(Reads)")
    assert proof(old, after, owners).ok
    after["writes"] = after["writes"].replace("return 1", "return 2")
    assert not proof(old, after, owners).ok


def test_already_eager_dependency_import_can_hoist() -> None:
    old = "from dependency import other\ndef f():\n from dependency import hook\n return hook()\n"
    after = {
        "old": "from dependency import other\nfrom new import f\n",
        "new": "from dependency import hook\ndef f():\n return hook()\n",
    }
    assert proof(
        old,
        after,
        {"f": "new"},
        hoists=[
            {
                "root": "f",
                "module": "new",
                "statement": "from dependency import hook",
                "comment": "",
            }
        ],
    ).ok


def test_side_effectful_stdlib_is_not_a_safe_hoist() -> None:
    old = "def f():\n import this\n return this\n"
    after = {"old": "from new import f\n", "new": "import this\ndef f():\n return this\n"}
    assert not proof(
        old,
        after,
        {"f": "new"},
        hoists=[{"root": "f", "module": "new", "statement": "import this", "comment": ""}],
    ).ok


def test_eager_hoist_chain_and_provider_snapshots_are_hash_verified() -> None:
    from tools._move_manifest import digest, verify_provider_sources

    old = "import dependency\ndef f():\n from leaf import hook\n return hook()\n"
    source = "import leaf\n"
    step = {
        "module": "dependency",
        "path": "src/dependency.py",
        "source": source,
        "sha256": digest(source),
    }
    entry = {
        "root": "f",
        "module": "new",
        "statement": "from leaf import hook",
        "comment": "",
        "eager": [step],
    }
    after = {
        "old": "import dependency\nfrom new import f\n",
        "new": "from leaf import hook\ndef f():\n return hook()\n",
    }
    data = manifest(old, after, {"f": "new"}, hoists=[entry])
    assert check_move(old, after, move_manifest=data).ok
    verify_provider_sources(data, lambda path: source, lambda path: source)
    with pytest.raises(ValueError, match="hash"):
        verify_provider_sources(data, lambda path: source, lambda path: "changed")
    bad = copy.deepcopy(data)
    bad["hoists"][0]["eager"][0]["module"] = "wrong"
    assert not check_move(old, after, move_manifest=bad).ok
    parent = "class Parent:\n def read(self): return 1\n"
    data["hoists"] = []
    data["providers"] = {
        "dependency.Parent": {
            "path": "src/dependency.py",
            "sha256": digest(parent),
            "class": parent,
        }
    }
    verify_provider_sources(data, lambda path: parent, lambda path: parent)
    data["providers"]["dependency.Parent"]["class"] = parent.replace("return 1", "return 2")
    with pytest.raises(ValueError, match="snapshot"):
        verify_provider_sources(data, lambda path: parent, lambda path: parent)


def test_logger_name_is_pinned_to_the_original_module() -> None:
    old = "import logging\nlogger = logging.getLogger(__name__)\n"
    after = {
        "old": "from new import logger\n",
        "new": 'import logging\nlogger = logging.getLogger("old")\n',
    }
    assert proof(old, after, {"logger": "new"}, effects=["logger"]).ok
    for bad in [
        old,
        after["new"].replace('"old"', '"new"'),
        after["new"].replace("import logging", "import wrong as logging"),
    ]:
        assert not proof(
            old, {"old": after["old"], "new": bad}, {"logger": "new"}, effects=["logger"]
        ).ok
