"""Inline logger substitution is exact, declared and bound to the shared logger."""

from __future__ import annotations

import copy

import pytest

from tests.tooling.test_move_manifest import manifest
from tools.move_check import check_move

OLD = """import logging
logger = logging.getLogger(__name__)
def emit():
 import logging
 return logging.getLogger(__name__)
"""
SUPPORT = 'import logging\nlogger = logging.getLogger("old")\n'
NEW = """from support import logger
def emit():
 import logging
 return logger
"""
ENTRY = {"root": "emit", "module": "new", "count": 1}


def proof(*, old=OLD, new=NEW, support=SUPPORT, entries=(ENTRY,), field=True, owner="new"):
    after = {
        "old": "from support import logger as logger\nfrom new import emit\n",
        "support": support,
        "new": new,
    }
    extra = {"logger_calls": copy.deepcopy(list(entries))} if field else {}
    evidence = manifest(
        old, after, {"logger": "support", "emit": owner}, effects=["logger"], **extra
    )
    return check_move(old, after, move_manifest=evidence)


def test_declared_inline_call_uses_authenticated_shared_logger():
    result = proof()
    assert result.ok, result.problems
    assert result.moved == {"logger": "support", "emit": "new"}


def test_undeclared_byte_identical_logger_drift_is_refused():
    raw = NEW.replace("return logger", "return logging.getLogger(__name__)")
    assert not proof(new=raw, field=False).ok


def test_substitution_without_evidence_is_refused():
    assert not proof(field=False).ok


def test_original_module_still_allows_its_unmoved_inline_logger():
    after = {"old": OLD}
    evidence = manifest(OLD, after, {"logger": "old", "emit": "old"})
    result = check_move(OLD, after, move_manifest=evidence)
    assert result.ok, result.problems


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"root": "missing", "module": "new", "count": 1}],
        [{"root": "logger", "module": "support", "count": 1}],
        [{"root": "emit", "module": "missing", "count": 1}],
        [{"root": "emit", "module": "support", "count": 1}],
        [{"root": "emit", "module": "new", "count": 0}],
        [{"root": "emit", "module": "new", "count": 2}],
        [{"root": "emit", "module": "new", "count": True}],
        [{"root": "emit", "module": "new", "count": "1"}],
        [ENTRY, ENTRY],
        [dict(ENTRY, unknown=True)],
    ],
)
def test_unused_duplicate_or_malformed_evidence_is_refused(entries):
    assert not proof(entries=entries).ok


@pytest.mark.parametrize(
    "new",
    [
        NEW.replace("from support import logger", "from other import logger"),
        NEW.replace("from support import logger", "from support import logger as wrong"),
        NEW.replace("from support import logger", "import support\nlogger = support.logger"),
        NEW.replace("def emit():", "def emit(logger):"),
        NEW.replace("return logger", "logger = object()\n return logger"),
        NEW.replace("return logger", "return logging.getLogger(__name__)"),
        NEW.replace("return logger", 'return logging.getLogger("old")'),
        NEW.replace("return logger", "return logger.getChild('child')"),
        NEW.replace("return logger", "print('changed')\n return logger"),
        NEW.replace("return logger", "return logger, logger"),
        NEW.replace("from support import logger", "if False:\n from support import logger"),
        NEW.replace("from support import logger\n", "").replace(
            " import logging", " import logging\n from support import logger"
        ),
    ],
)
def test_destination_binding_and_other_body_changes_are_refused(new):
    assert not proof(new=new).ok


@pytest.mark.parametrize(
    "original,replacement",
    [
        ("def emit():", "def emit(__name__):"),
        ("def emit():", "def emit(logging):"),
        (" import logging\n return", " import logging\n __name__ = 'fake'\n return"),
        (" import logging\n return", " import logging\n logging = object()\n return"),
        ("logging.getLogger(__name__)\n", "logging.getLogger(__name__, extra=True)\n"),
        ("logging.getLogger(__name__)\n", "logging.getLogger(__name__, 'extra')\n"),
        ("logging.getLogger(__name__)\n", "logging.getChild(__name__)\n"),
    ],
)
def test_original_call_must_be_exact_stdlib_and_unshadowed(original, replacement):
    assert not proof(old=OLD.replace(original, replacement)).ok


def test_shared_logger_must_have_original_module_identity():
    assert not proof(support=SUPPORT.replace('"old"', '"wrong"')).ok
    assert not proof(
        old=OLD.replace(
            "logger = logging.getLogger(__name__)", 'logger = logging.getLogger("wrong")'
        ),
        support=SUPPORT.replace('"old"', '"wrong"'),
    ).ok


def test_unrelated_module_name_read_cannot_be_normalized():
    old = OLD.replace(" return logging", " print(__name__)\n return logging")
    new = NEW.replace(" return logger", ' print("old")\n return logger')
    assert not proof(old=old, new=new).ok


def test_two_calls_require_exact_count_and_both_substitutions():
    old = OLD.replace(
        "return logging.getLogger(__name__)",
        "return logging.getLogger(__name__), logging.getLogger(__name__)",
    )
    new = NEW.replace("return logger", "return logger, logger")
    result = proof(old=old, new=new, entries=[dict(ENTRY, count=2)])
    assert result.ok, result.problems
    assert not proof(old=old, new=new).ok


def test_empty_optional_field_is_not_a_general_waiver():
    after = {"old": OLD}
    evidence = manifest(OLD, after, {"logger": "old", "emit": "old"}, logger_calls=[])
    assert not check_move(OLD, after, move_manifest=evidence).ok


def test_class_method_uses_exact_per_root_evidence():
    old = OLD.replace("def emit():", "class Store:\n def emit(self):").replace(
        " import logging\n return", "  import logging\n  return"
    )
    new = NEW.replace("def emit():", "class Reads:\n def emit(self):").replace(
        " import logging\n return", "  import logging\n  return"
    )
    after = {
        "old": (
            "from support import logger as logger\nfrom new import Reads\n"
            "class Store(Reads): pass\n"
        ),
        "support": SUPPORT,
        "new": new,
    }
    data = manifest(
        old,
        after,
        {"logger": "support", "Store": "old", "Store.emit": "new"},
        effects=["logger"],
        logger_calls=[dict(ENTRY, root="Store.emit")],
    )
    result = check_move(old, after, move_manifest=data)
    assert result.ok, result.problems


@pytest.mark.parametrize("gate", ["cli", "ci"])
@pytest.mark.parametrize("damage", [None, "drop-import", "undeclared", "wrong-logger"])
def test_supported_cli_and_ci_authenticate_the_same_normalization(tmp_path, gate, damage):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from tests.tooling.test_move_ci import git, run

    root = tmp_path / "fixture"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    (root / "old.py").write_text(OLD)
    git(root, "add", ".")
    git(root, "commit", "-qm", "Fixture original")
    base = git(root, "rev-parse", "HEAD")
    current = NEW
    entries = {"logger_calls": [dict(ENTRY, module="new.py")]}
    if damage == "drop-import":
        current = current.replace(" import logging\n", "")
        entries["logger_calls"][0]["drop_import"] = {
            "statement": "import logging",
            "count": 1,
            "comment": "",
        }
    elif damage == "undeclared":
        current = current.replace("return logger", "return logging.getLogger(__name__)")
        entries = {}
    elif damage == "wrong-logger":
        current = current.replace("from support import logger", "from wrong import logger")
    after = {
        "old.py": "from support import logger as logger\nfrom new import emit\n",
        "support.py": SUPPORT,
        "new.py": current,
    }
    for name, source in after.items():
        (root / name).write_text(source)
    evidence = manifest(
        OLD, after, {"logger": "support.py", "emit": "new.py"}, effects=["logger"], **entries
    )
    evidence.update(base=base, old="old.py")
    moves = root / "tools/moves"
    moves.mkdir(parents=True)
    path = moves / "fixture.json"
    path.write_text(json.dumps(evidence))
    for suffix in ["emitted", "expanded"]:
        (moves / f"fixture-{suffix}.json").write_text(
            json.dumps({"old": {"logger": "support", "emit": "new"}})
        )
    git(root, "add", ".")
    git(root, "commit", "-qm", "Fixture move")
    if gate == "ci":
        result = run(root)
    else:
        script = Path(__file__).resolve().parents[2] / "tools/move_check.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--base",
                base,
                "--old",
                "old.py",
                "--new",
                "new.py",
                "support.py",
                "--move-manifest",
                str(path),
                "--allow-effects",
                "logger",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    if damage in (None, "drop-import"):
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "inline logger" in result.stdout + result.stderr


@pytest.mark.parametrize("declared", [False, True])
def test_conditional_stdlib_import_cannot_hide_logger_drift(declared):
    old = OLD.replace(" import logging\n return", " if True:\n  import logging\n return")
    new = NEW.replace(" import logging\n return", " if True:\n  import logging\n return")
    if not declared:
        new = new.replace("return logger", "return logging.getLogger(__name__)")
    assert not proof(old=old, new=new, field=declared).ok


def test_conditional_local_import_dominating_the_call_is_authenticated():
    old = OLD.replace(" import logging\n return", " if True:\n  import logging\n  return")
    new = NEW.replace(" import logging\n return", " if True:\n  import logging\n  return")
    result = proof(old=old, new=new)
    assert result.ok, result.problems


def test_logger_normalization_does_not_drop_the_original_import():
    assert not proof(new=NEW.replace(" import logging\n", "")).ok


def _drop_entry(**changes):
    return dict(
        ENTRY, drop_import={"statement": "import logging", "count": 1, "comment": "", **changes}
    )


def test_exact_declared_orphan_import_can_be_removed():
    result = proof(new=NEW.replace(" import logging\n", ""), entries=[_drop_entry()])
    assert result.ok, result.problems


@pytest.mark.parametrize(
    "changes",
    [
        {"count": 2},
        {"count": True},
        {"count": 0},
        {"statement": "import os"},
        {"comment": "# lost"},
        {"extra": True},
    ],
)
def test_other_import_permissions_are_refused(changes):
    assert not proof(new=NEW.replace(" import logging\n", ""), entries=[_drop_entry(**changes)]).ok


def test_import_permission_cannot_be_unused_or_cover_another_body_change():
    assert not proof(entries=[_drop_entry()]).ok
    current = NEW.replace(" import logging\n", "").replace("return logger", "return None")
    assert not proof(new=current, entries=[_drop_entry()]).ok


@pytest.mark.parametrize(
    "read",
    [
        " logging.info('read')\n",
        " def nested():\n  return logging\n",
        " value: logging.Logger\n",
        " value: 'logging.Logger'\n",
    ],
)
def test_remaining_binding_reads_prevent_import_removal(read):
    old = OLD.replace(" return logging.getLogger", read + " return logging.getLogger")
    new = NEW.replace(" import logging\n", read)
    assert not proof(old=old, new=new, entries=[_drop_entry()]).ok


@pytest.mark.parametrize("comment", ["# keep this", "# noqa: PLC0415"])
def test_import_comments_cannot_be_lost(comment):
    old = OLD.replace(" import logging\n", f" import logging  {comment}\n")
    assert not proof(old=old, new=NEW.replace(" import logging\n", ""), entries=[_drop_entry()]).ok


def test_neighbor_comments_are_preserved_exactly():
    old = OLD.replace(" import logging\n", " # before\n import logging\n # after\n")
    new = NEW.replace(" import logging\n", " # before\n # after\n")
    result = proof(old=old, new=new, entries=[_drop_entry()])
    assert result.ok, result.problems
    assert not proof(old=old, new=new.replace(" # after\n", ""), entries=[_drop_entry()]).ok
    assert not proof(old=old, new=new.replace("# after", "# changed"), entries=[_drop_entry()]).ok


def test_extra_or_unrelated_import_deletions_are_refused():
    for extra in [" import logging\n", " import os\n"]:
        old = OLD.replace(" import logging\n", " import logging\n" + extra)
        assert not proof(
            old=old, new=NEW.replace(" import logging\n", ""), entries=[_drop_entry()]
        ).ok


def test_module_import_cannot_be_removed_by_local_permission():
    old = OLD.replace(" import logging\n", "")
    assert not proof(old=old, new=NEW.replace(" import logging\n", ""), entries=[_drop_entry()]).ok


@pytest.mark.parametrize("drop", [None, 1, [], {}, {"statement": "import logging"}])
def test_malformed_import_permission_is_refused(drop):
    assert not proof(
        new=NEW.replace(" import logging\n", ""), entries=[dict(ENTRY, drop_import=drop)]
    ).ok
