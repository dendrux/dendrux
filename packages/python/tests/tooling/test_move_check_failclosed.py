"""tools/move_check.py fails closed: each case was a reproduced false pass (first review round)."""

from __future__ import annotations

import textwrap
from typing import Any

from tools import move_check as mc

FUT = "from __future__ import annotations\n"


def src(text: str) -> str:
    return FUT + textwrap.dedent(text)


def check(old: str, after: dict[str, str], **kw: Any) -> mc.MoveReport:
    return mc.check_move(src(old), {k: src(v) for k, v in after.items()}, **kw)


def run(sources: dict[str, str]) -> dict[str, Any]:
    """Execute a module and return its namespace (fixtures only)."""
    namespace: dict[str, Any] = {}
    for text in sources.values():
        exec(compile(text, "<fixture>", "exec"), namespace)  # noqa: S102
    return namespace


# ---- statement order and multiplicity --------------------------------------------------------


def test_reordered_module_assignments_fail() -> None:
    report = check("x = 1\ny = x\n", {"old": "y = x\nx = 1\n"})
    assert any("reordered" in p for p in report.problems)


def test_reordered_class_assignments_fail() -> None:
    old = "class C:\n    x = 1\n    y = x\n"
    report = check(old, {"old": "class C:\n    y = x\n    x = 1\n"})
    assert report.problems


def test_reordered_distinct_calls_fail() -> None:
    old = "register_a()\nregister_b()\n"
    report = check(old, {"old": "register_b()\nregister_a()\n"})
    assert report.problems


def test_an_extra_same_named_method_before_the_original_fails() -> None:
    old = "class C:\n    def f(self, x=1):\n        return 1\n"
    after = (
        "class C:\n    def f(self, x=1 / 0):\n        return 99\n"
        "    def f(self, x=1):\n        return 1\n"
    )
    assert check(old, {"old": after}).problems


def test_a_repeated_name_keeps_its_count_and_order() -> None:
    old = "@property\ndef x(): ...\n@x.setter\ndef x(v): ...\n"
    assert check(old, {"old": old}).problems == []
    assert check(old, {"old": "@x.setter\ndef x(v): ...\n@property\ndef x(): ...\n"}).problems


def test_a_moved_definition_keeps_the_base_order_inside_each_module() -> None:
    old = "def a(): ...\ndef b(): ...\n"
    assert check(old, {"old": "def a(): ...\n", "new": "def b(): ...\n"}).problems == []
    assert check(old, {"old": "def b(): ...\ndef a(): ...\n"}).problems


# ---- mixins ----------------------------------------------------------------------------------

OLD_C = "class C(Base):\n    def f(self):\n        return 1\n"


def test_a_mixin_with_a_decorator_fails() -> None:
    after = {
        "old": "class C(M, Base):\n    pass\n",
        "m": "@decorate\nclass M:\n    def f(self):\n        return 1\n",
    }
    assert any("mixin M" in p for p in check(OLD_C, after).problems)


def test_a_mixin_with_a_metaclass_keyword_fails() -> None:
    after = {
        "old": "class C(M, Base):\n    pass\n",
        "m": "class M(metaclass=Evil):\n    def f(self):\n        return 1\n",
    }
    assert any("mixin M" in p for p in check(OLD_C, after).problems)


def test_a_mixin_listed_after_the_original_base_fails() -> None:
    after = {
        "old": "class C(Base, M):\n    pass\n",
        "m": "class M:\n    def f(self):\n        return 1\n",
    }
    assert any("bases" in p for p in check(OLD_C, after).problems)


def test_a_mixin_listed_first_keeps_method_resolution_at_runtime() -> None:
    base = "class Base:\n    def f(self):\n        return 10\n"
    before = run({"all": base + OLD_C})
    after = run(
        {
            "m": "class M:\n    def f(self):\n        return 1\n",
            "old": base + "class C(M, Base):\n    pass\n",
        }
    )
    assert before["C"]().f() == after["C"]().f() == 1
    report = check(
        OLD_C,
        {
            "old": "class C(M, Base):\n    pass\n",
            "m": "class M:\n    def f(self):\n        return 1\n",
        },
    )
    assert report.problems == []


def test_same_named_mixins_in_two_modules_are_ambiguous() -> None:
    after = {
        "old": "class C(M):\n    pass\n",
        "a": "class M:\n    def f(self):\n        return 99\n",
        "b": "class M:\n    def f(self):\n        return 1\n",
    }
    assert any("ambiguous" in p for p in check(OLD_C.replace("(Base)", ""), after).problems)


def test_a_mixin_with_an_unlisted_base_fails_and_a_listed_one_passes() -> None:
    after = {
        "old": "class C(M):\n    pass\n",
        "m": "class M(Attrs):\n    def f(self):\n        return 1\n",
        "attrs": "class Attrs:\n    pass\n",
    }
    old = "class C:\n    def f(self):\n        return 1\n"
    assert check(old, after).problems
    assert check(old, after, allow_new=("Attrs",)).problems == []


def test_a_non_method_member_cannot_move_into_a_mixin() -> None:
    after = {"old": "class C(M):\n    pass\n", "m": "class M:\n    size = 3\n"}
    assert check("class C:\n    size = 3\n", after).problems


# ---- exports ---------------------------------------------------------------------------------


def test_deleting_a_dunder_all_entry_fails() -> None:
    old = 'def f(): ...\n__all__ = ["f"]\n'
    after = {"old": "__all__ = []\n", "new": "def f(): ...\n"}
    assert check(old, after).problems


def test_changing_the_dunder_all_expression_fails() -> None:
    old = '__all__ = register("old")\n'
    assert check(old, {"old": '__all__ = register("new")\n'}).problems
    assert check(old, {"old": old}).problems == []


def test_a_name_in_dunder_all_must_stay_bound() -> None:
    old = 'def f(): ...\n__all__ = ["f"]\n'
    after = {"old": '__all__ = ["f"]\n', "new": "def f(): ...\n"}
    assert any("export" in p for p in check(old, after).problems)
    kept = {"old": 'from new import f\n__all__ = ["f"]\n', "new": "def f(): ...\n"}
    assert check(old, kept).problems == []


# ---- executable guards -----------------------------------------------------------------------


def test_a_changed_import_guard_condition_fails() -> None:
    old = 'if record("old"):\n    import os\n'
    assert check(old, {"old": 'if record("new"):\n    import os\n'}).problems
    assert check(old, {"old": ""}).problems


def test_a_try_import_guard_is_compared() -> None:
    old = "try:\n    import fast\nexcept ImportError:\n    fast = None\n"
    assert check(old, {"old": "import fast\n"}).problems


def test_type_checking_imports_need_explanations_and_other_guards_stay_strict() -> None:
    old = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import a\n"
    report = check(
        old, {"old": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import b\n"}
    )
    assert any("extra-import" in problem and "import b" in problem for problem in report.problems)
    assert check(
        old, {"old": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    X = 1\n"}
    ).problems


# ---- re-exports are real runtime bindings from the right module ------------------------------

MOVED_F = {"new": "def f(): ...\n"}


def reexport(old_after: str) -> mc.MoveReport:
    return check("def f(): ...\n", {"old": old_after, **MOVED_F}, require_reexports=True)


def test_a_module_level_import_from_the_new_module_is_a_re_export() -> None:
    assert reexport("from new import f\n").problems == []
    assert reexport("from new import f as f\n").problems == []


def test_a_type_checking_only_import_is_not_a_re_export() -> None:
    report = reexport(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from new import f\n"
    )
    assert any("re-export" in p for p in report.problems)


def test_a_function_local_import_is_not_a_re_export() -> None:
    local = "def keep():\n    from new import f\n"
    report = check("def f(): ...\n" + local, {"old": local, **MOVED_F}, require_reexports=True)
    assert any("re-export" in p for p in report.problems)


def test_an_import_from_the_wrong_module_is_not_a_re_export() -> None:
    assert any("elsewhere" in p for p in reexport("from other import f\n").problems)


def test_a_relative_import_cannot_be_verified() -> None:
    assert any("relative" in p for p in reexport("from .new import f\n").problems)


def test_a_conditional_import_is_not_proved() -> None:
    assert reexport("if flag:\n    from new import f\n").problems


# ---- other fail-closed rules ------------------------------------------------------------------


def test_from_future_imports_must_match_in_every_module() -> None:
    old = "def f(): ...\n"
    report = mc.check_move(src(old), {"old": src(""), "new": "def f(): ...\n"})
    assert any("__future__" in p for p in report.problems)


def test_a_call_statement_may_not_move() -> None:
    old = "register()\ndef f(): ...\n"
    report = check(old, {"old": "def f(): ...\n", "new": "register()\n"})
    assert any("immovable" in p or "extra" in p for p in report.problems)


def test_an_assignment_with_a_call_moves_only_when_allowed() -> None:
    old = "logger = get_logger()\n"
    after = {"old": "", "new": old}
    assert check(old, after).problems
    assert check(old, after, allow_effects=("logger",)).problems == []
    assert check("LIMIT = 10\n", {"old": "", "new": "LIMIT = 10\n"}).problems == []


def test_a_stay_in_place_statement_is_compared_even_without_moves() -> None:
    assert check("register(1)\n", {"old": "register(2)\n"}).problems
