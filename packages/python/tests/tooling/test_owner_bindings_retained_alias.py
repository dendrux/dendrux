"""Keep independent skill-session imports distinct from the retained patch alias."""

import pytest

from tools.owner_bindings import check_bindings

ENTRY = "dendrux.runtime._run_entry"
TARGETS = {"current_skill_session": f"{ENTRY}.current_skill_session"}
IMPORT_CURRENT = "from dendrux.skills._session import current\n"


@pytest.mark.parametrize(
    "body",
    [
        "def f(agent): return current(agent)\n",
        "def f(current): return current\n",
        "def f():\n current = 1\n return current\n",
        "def f(): return (lambda current: current)(1)\n",
        "def f(): return [current for current in (1, 2)]\n",
    ],
    ids=["independent-call", "parameter", "local", "lambda", "comprehension"],
)
def test_retained_alias_does_not_forbid_independent_current_or_its_locals(body: str) -> None:
    assert check_bindings(IMPORT_CURRENT + body, "dendrux.runtime.runner", TARGETS) == []


def test_both_namespaces_can_be_used_together_through_their_own_bindings() -> None:
    source = (
        IMPORT_CURRENT
        + f"import {ENTRY} as owner\n"
        + "def f(agent): return current(agent), owner.current_skill_session(agent)\n"
    )
    assert check_bindings(source, "dendrux.runtime.runner", TARGETS) == []


@pytest.mark.parametrize("name", ["current_skill_session", "copied"])
@pytest.mark.parametrize("shadowed", [False, True], ids=["call", "shadowed"])
def test_copied_defining_function_still_cannot_supply_the_retained_alias(
    name: str, shadowed: bool
) -> None:
    body = f"def f({name}): return {name}\n" if shadowed else f"def f(): return {name}()\n"
    source = f"from dendrux.skills._session import current as {name}\n" + body
    problems = check_bindings(source, "dendrux.runtime.runner", TARGETS)
    assert len(problems) == 1
    assert "owner-binding:" in problems[0]


@pytest.mark.parametrize("name", ["current_skill_session", "copied", "current"])
def test_copying_the_retained_owner_attribute_is_still_forbidden(name: str) -> None:
    source = f"from {ENTRY} import current_skill_session as {name}\ndef f(): return {name}()\n"
    problems = check_bindings(source, "dendrux.runtime.runner", TARGETS)
    assert len(problems) == 1
    assert f"copies {ENTRY}.current_skill_session" in problems[0]


def test_qualified_retained_owner_access_passes() -> None:
    source = f"import {ENTRY} as owner\ndef f(): return owner.current_skill_session()\n"
    assert check_bindings(source, "dendrux.runtime.runner", TARGETS) == []


def test_retained_owner_may_keep_its_defining_import_alias() -> None:
    source = (
        "from dendrux.skills._session import current as current_skill_session\n"
        "def f(): return current_skill_session()\n"
    )
    assert check_bindings(source, ENTRY, TARGETS) == []


def test_independent_current_is_guarded_when_it_is_itself_a_registered_patch_target() -> None:
    targets = {**TARGETS, "current": "dendrux.skills._session.current"}
    source = IMPORT_CURRENT + "def f(): return current()\n"
    problems = check_bindings(source, "dendrux.runtime.runner", targets)
    assert len(problems) == 1
    assert "copies dendrux.skills._session.current" in problems[0]
