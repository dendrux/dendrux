"""Assignment copies must not freeze callable patch seams in moved consumers."""

from __future__ import annotations

import hashlib

import pytest

from tools.move_check import check_move
from tools.owner_bindings import check_bindings

ENTRY = "dendrux.runtime._run_entry"
CURRENT = "dendrux.skills._session.current"
FINISHED = "dendrux.loops._helpers.record_run_finished"
TARGETS = {"current_skill_session": f"{ENTRY}.current_skill_session", "finished": FINISHED}


@pytest.mark.parametrize("module", ["dendrux.runtime.runner", "dendrux.runtime._run_tools"])
@pytest.mark.parametrize(
    ("imports", "seam", "target"),
    [
        ("from dendrux.skills._session import current", "current", CURRENT),
        ("import dendrux.skills._session as s", "s.current", CURRENT),
        ("from dendrux.loops._helpers import record_run_finished", "record_run_finished", FINISHED),
        ("from dendrux.loops import _helpers as h", "h.record_run_finished", FINISHED),
        ("import dendrux.loops._helpers", FINISHED, FINISHED),
        ("from ..loops import _helpers", "_helpers.record_run_finished", FINISHED),
    ],
    ids=["independent", "independent-attribute", "bare", "module-alias", "chain", "relative"],
)
@pytest.mark.parametrize(
    "assignment",
    [
        "copy = SEAM",
        "one = two = SEAM",
        "copy, other = SEAM, None",
        "[copy, [other]] = [SEAM, [None]]",
        "copy: object = SEAM",
    ],
    ids=["plain", "chained", "tuple", "nested-list", "annotated"],
)
def test_assignment_copy_is_refused(
    module: str, imports: str, seam: str, target: str, assignment: str
) -> None:
    source = imports + "\n" + assignment.replace("SEAM", seam) + "\n"
    problems = check_bindings(source, module, TARGETS)
    assert len(problems) == 1, problems
    assert f"owner-binding: {module}:2 copies {target}; qualify its owner" == problems[0]


@pytest.mark.parametrize(
    "body",
    [
        "def f():\n copy = current\n return copy\n",
        "def f():\n from dendrux.skills._session import current\n copy = current\n",
        "class Consumer:\n copy = current\n",
        "def f():\n def inner():\n  copy = current\n  return copy\n return inner\n",
    ],
    ids=["function", "local-import", "class", "closure"],
)
def test_independent_assignment_is_refused_in_nested_scopes(body: str) -> None:
    source = "from dendrux.skills._session import current\n" + body
    problems = check_bindings(source, "dendrux.runtime.runner", TARGETS)
    assert len(problems) == 1, problems
    assert f"copies {CURRENT}" in problems[0]


@pytest.mark.parametrize("name", ["current", "h"])
@pytest.mark.parametrize(
    "body",
    [
        "def f(NAME):\n copy = VALUE\n return copy\n",
        "def f():\n NAME = object()\n copy = VALUE\n return copy\n",
        "def f(NAME):\n copy: object = VALUE\n return copy\n",
        "def f(NAME):\n copy, other = VALUE, None\n return copy\n",
        "def f(NAME): return (lambda: VALUE)()\n",
        "def f(): return [VALUE for NAME in ()]\n",
    ],
    ids=["parameter", "local", "annotated", "tuple", "closure", "comprehension"],
)
def test_assignment_of_unrelated_shadow_passes(name: str, body: str) -> None:
    imports = (
        "from dendrux.skills._session import current\nfrom dendrux.loops import _helpers as h\n"
    )
    value = "current" if name == "current" else "h.record_run_finished"
    source = imports + body.replace("NAME", name).replace("VALUE", value)
    assert check_bindings(source, "dendrux.runtime.runner", TARGETS) == []


def test_calls_and_call_result_assignments_keep_live_owner_access() -> None:
    source = (
        "from dendrux.skills._session import current\n"
        "from dendrux.loops import _helpers as h\n"
        "def f(agent):\n"
        " session = current(agent)\n"
        " result: object = h.record_run_finished(agent)\n"
        " pair = current(agent), h.record_run_finished(agent)\n"
        " return session, result, pair\n"
    )
    assert check_bindings(source, "dendrux.runtime.runner", TARGETS) == []


def test_independent_assignment_requires_an_active_guard_target() -> None:
    source = "from dendrux.skills._session import current\ncopy = current\n"
    assert check_bindings(source, "dendrux.runtime.runner", {}) == []


@pytest.mark.parametrize("copier", ["one", "two"], ids=["moved", "residual"])
def test_move_proof_rejects_independent_assignment_copy(copier: str) -> None:
    imports = "from dendrux.skills._session import current\n"
    bodies = {
        root: f"def {root}():\n owner.current_skill_session()\n copy = current\n return copy()\n"
        if root == copier
        else f"def {root}():\n owner.current_skill_session()\n return current()\n"
        for root in ("one", "two")
    }
    old = (
        imports
        + "from dendrux.skills._session import current as current_skill_session\n"
        + (bodies["one"] + bodies["two"]).replace(
            "owner.current_skill_session", "current_skill_session"
        )
    )
    imports += f"import {ENTRY} as owner\n"
    after = {
        "old": "from new import one\n" + imports + bodies["two"],
        "new": imports + bodies["one"],
    }
    manifest = {
        "version": 1,
        "base": "a" * 40,
        "old": "old",
        "source_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "sources": {
            key: hashlib.sha256(source.encode()).hexdigest() for key, source in after.items()
        },
        "owners": {"one": "new", "two": "old"},
        "lexical": [
            {
                "root": root,
                "module": module,
                "name": "current_skill_session",
                "target": f"{ENTRY}.current_skill_session",
                "count": 1,
            }
            for root, module in [("one", "new"), ("two", "old")]
        ],
        "hoists": [],
        "annotations": [],
        "metadata": [],
        "scaffolding": {},
        "effects": [],
        "seams": {"current_skill_session": f"{ENTRY}.current_skill_session"},
        "providers": {},
    }
    report = check_move(old, after, move_manifest=manifest)
    module = "new" if copier == "one" else "old"
    line = 5 if copier == "one" else 6
    assert report.problems == [
        f"owner-binding: {module}:{line} copies {CURRENT}; qualify its owner"
    ]
