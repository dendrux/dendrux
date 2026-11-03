"""tools/semgrep_ratchet.py: custom Semgrep rules, ratcheted by rule + module + function."""

from __future__ import annotations

import os
import textwrap
from typing import TYPE_CHECKING

import pytest

from tools import semgrep_ratchet as sr

if TYPE_CHECKING:
    from pathlib import Path

SOURCE = textwrap.dedent(
    """\
    class Store:
        async def append(self, session):
            await session.commit()

        def helper(self):
            def inner():
                return 1
            return inner

    TOP = 1
    """
)


def test_enclosing_function_is_the_innermost_qualified_name() -> None:
    tree = sr.function_spans(SOURCE)
    assert sr.enclosing(tree, 3) == "Store.append"
    assert sr.enclosing(tree, 6) == "Store.helper.<locals>.inner"
    assert sr.enclosing(tree, 8) == "Store.helper"
    assert sr.enclosing(tree, 10) == "<module>"


def test_scope_uses_include_and_exclude_prefixes_under_the_package() -> None:
    meta = {"include": ["runtime/stores/sql/"], "exclude": ["runtime/stores/sql/mysql.py"]}
    assert sr.in_scope("src/dendrux/runtime/stores/sql/base.py", meta)
    assert not sr.in_scope("src/dendrux/runtime/stores/sql/mysql.py", meta)
    assert not sr.in_scope("src/dendrux/db/session.py", meta)
    assert sr.in_scope("src/dendrux/anything.py", {})
    assert not sr.in_scope("src/dendrux/types.py", {"exclude": ["types.py"]})


def test_keys_count_matches_per_rule_module_and_function(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    (src / "runtime").mkdir(parents=True)
    (src / "runtime" / "m.py").write_text(SOURCE)
    results = [
        {
            "check_id": "semgrep.some-rule",
            "path": str(src / "runtime" / "m.py"),
            "start": {"line": 3},
            "extra": {"metadata": {}},
        },
        {
            "check_id": "semgrep.some-rule",
            "path": str(src / "runtime" / "m.py"),
            "start": {"line": 3},
            "extra": {"metadata": {}},
        },
        {
            "check_id": "x.other",
            "path": str(src / "runtime" / "m.py"),
            "start": {"line": 10},
            "extra": {"metadata": {}},
        },
    ]
    assert sr.keys_from_results(results) == {
        "other|dendrux.runtime.m|<module>": 1,
        "some-rule|dendrux.runtime.m|Store.append": 2,
    }


def test_a_new_match_fails_and_a_known_one_passes() -> None:
    known = {"r|dendrux.a|f": 1}
    assert not sr.compare(known, known).failed
    assert sr.compare({**known, "r|dendrux.a|g": 1}, known).new == ["r|dendrux.a|g"]
    assert sr.compare({"r|dendrux.a|f": 2}, known).worse == ["r|dendrux.a|f"]


def test_without_semgrep_the_gate_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMGREP", raising=False)
    monkeypatch.setattr(sr.shutil, "which", lambda name: None)
    with pytest.raises(sr.SemgrepMissingError, match="pip install semgrep=="):
        sr.find_semgrep()


# --- end to end, only where semgrep is installed (CI quality job, or SEMGREP=... locally) ---

HAVE_SEMGREP = False
try:
    sr.find_semgrep()
    HAVE_SEMGREP = True
except sr.SemgrepMissingError:
    pass


@pytest.mark.skipif(not HAVE_SEMGREP, reason="semgrep not installed (set SEMGREP=/path/to/semgrep)")
def test_the_rules_pass_their_own_fixtures() -> None:
    assert sr.main(["test"]) == 0


@pytest.mark.skipif(not HAVE_SEMGREP, reason="semgrep not installed (set SEMGREP=/path/to/semgrep)")
def test_end_to_end_flags_a_new_commit_outside_write_tx(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux" / "runtime" / "stores" / "sql"
    src.mkdir(parents=True)
    for parent in (src, src.parent, src.parent.parent, src.parent.parent.parent):
        (parent / "__init__.py").write_text("")
    mod = src / "base.py"
    mod.write_text("class S:\n    async def _write_tx(self, s):\n        await s.commit()\n")
    baseline = tmp_path / "base.json"
    common = ["--src", str(tmp_path / "src" / "dendrux"), "--baseline", str(baseline)]
    assert sr.main(["stamp", *common]) == 0
    assert sr.main(["check", *common]) == 0
    mod.write_text(mod.read_text() + "    async def other(self, s):\n        await s.commit()\n")
    assert sr.main(["check", *common]) == 1
    assert os.path.exists(baseline)


ALLOW = "dendrux.runtime.stores.sql.base:SQLStateStore._write_tx"
STORE_SOURCE = """\
class SQLStateStore:
    async def _write_tx(self, session):
        await session.commit()

        def helper():
            session.commit()

    async def append(self, session):
        await session.commit()

    async def other(self, session):
        async def _write_tx(s):
            await s.commit()


class OtherStore:
    async def _write_tx(self, session):
        await session.commit()


async def _write_tx(session):
    await session.commit()
"""
# line numbers of each commit in STORE_SOURCE
IN_WRITE_TX, IN_HELPER, IN_APPEND, IN_NESTED, IN_OTHER_CLASS, IN_FUNCTION = 3, 6, 9, 13, 18, 22


def _commit_hit(path: Path, line: int) -> dict:
    metadata = {"include": ["runtime/stores/sql/"], "allow_in": [ALLOW]}
    return {
        "check_id": "x.session-commit-outside-write-tx",
        "path": str(path),
        "start": {"line": line},
        "extra": {"metadata": metadata},
    }


def test_the_commit_exemption_covers_only_sqlstatestore_write_tx(tmp_path: Path) -> None:
    path = tmp_path / "src" / "dendrux" / "runtime" / "stores" / "sql" / "base.py"
    path.parent.mkdir(parents=True)
    path.write_text(STORE_SOURCE)
    lines = [IN_WRITE_TX, IN_HELPER, IN_APPEND, IN_NESTED, IN_OTHER_CLASS, IN_FUNCTION]
    keys = sr.keys_from_results([_commit_hit(path, n) for n in lines])
    rule = "session-commit-outside-write-tx|dendrux.runtime.stores.sql.base|"
    assert keys == {
        rule + "OtherStore._write_tx": 1,
        rule + "SQLStateStore.append": 1,
        rule + "SQLStateStore.other.<locals>._write_tx": 1,
        rule + "_write_tx": 1,
    }


def test_an_exemption_names_a_module_so_a_same_named_class_elsewhere_is_not_exempt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "src" / "dendrux" / "runtime" / "stores" / "sql" / "mysql.py"
    path.parent.mkdir(parents=True)
    path.write_text(STORE_SOURCE)
    keys = sr.keys_from_results([_commit_hit(path, IN_WRITE_TX)])
    rule = "session-commit-outside-write-tx|dendrux.runtime.stores.sql.mysql|"
    assert keys == {rule + "SQLStateStore._write_tx": 1}


def _scratch_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "src" / "dendrux"
    for rel, text in {"__init__.py": "", **files}.items():
        target = src / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return src


SKIP_NO_SEMGREP = pytest.mark.skipif(
    not HAVE_SEMGREP, reason="semgrep not installed (set SEMGREP=/path/to/semgrep)"
)


def _clean_then_bad(tmp_path: Path, clean: str, bad: str, where: str = "runtime/m.py") -> int:
    """Stamp a tree with ``clean`` code, swap in ``bad``, and return the check's exit code."""
    src = _scratch_tree(tmp_path, {where: clean})
    baseline = tmp_path / "base.json"
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert sr.main(["stamp", *args]) == 0
    assert sr.main(["check", *args]) == 0
    (src / where).write_text(bad)
    return sr.main(["check", *args])


@SKIP_NO_SEMGREP
def test_a_nosemgrep_comment_does_not_hide_a_match(tmp_path: Path) -> None:
    clean = "async def f(session, store):\n    await store.write(session)\n"
    bad = (
        "async def f(session, store):\n"
        "    # nosemgrep: session-commit-outside-write-tx\n"
        "    await session.commit()\n"
    )
    assert _clean_then_bad(tmp_path, clean, bad, "runtime/stores/sql/m.py") == 1


@SKIP_NO_SEMGREP
@pytest.mark.parametrize("kwargs", ["**{}", "**extra", "**unrelated_options()", "**image_kwargs"])
def test_double_star_kwargs_do_not_stand_in_for_tenant_id(tmp_path: Path, kwargs: str) -> None:
    clean = "async def f(extra):\n    await _process_tool_calls(step=1, tenant_id=None)\n"
    bad = f"async def f(extra):\n    await _process_tool_calls(step=1, {kwargs})\n"
    assert _clean_then_bad(tmp_path, clean, bad) == 1


@SKIP_NO_SEMGREP
def test_an_explicit_tenant_id_next_to_double_star_kwargs_is_fine(tmp_path: Path) -> None:
    clean = "async def f(extra):\n    await _process_tool_calls(step=1, tenant_id=None)\n"
    good = "async def f(extra):\n    await _process_tool_calls(step=1, tenant_id=None, **extra)\n"
    assert _clean_then_bad(tmp_path, clean, good) == 0


EXC_PREFIX = (
    "def f(logger, record, run_id):\n    try:\n        work()\n    except Exception as exc:\n"
)


@SKIP_NO_SEMGREP
@pytest.mark.parametrize(
    "statement",
    [
        'logger.error("failed", extra={"error_info": str(exc)})',
        'logger.error("failed", extra={"detail": [f"{exc}"]})',
        "record(error_info={'text': str(exc)})",
        "error_info = str(exc)",
        "payload = {'error_info': f'bad {exc}'}",
    ],
)
def test_nested_exception_text_in_logs_and_error_info_is_found(
    tmp_path: Path, statement: str
) -> None:
    clean = EXC_PREFIX + "        logger.error('failed', extra={'kind': type(exc).__name__})\n"
    bad = EXC_PREFIX + f"        {statement}\n"
    assert _clean_then_bad(tmp_path, clean, bad) == 1


@SKIP_NO_SEMGREP
def test_safe_exception_reporting_is_not_flagged(tmp_path: Path) -> None:
    clean = EXC_PREFIX + "        pass\n"
    safe = (
        EXC_PREFIX
        + "        logger.error('failed', extra={'kind': type(exc).__name__, 'run': str(run_id)})\n"
        + "        error_info = type(exc).__name__\n"
        + "        payload = {'error_info': 'provider_error'}\n"
        + "        text = str(exc)\n"
    )
    assert _clean_then_bad(tmp_path, clean, safe) == 0


# --- shared-vocabulary rules: allow_in is exact, and path scope keeps definitions out.
# Each test builds one scratch tree and runs Semgrep once. ---

_FUNCTION = "async def {name}(store, run_id, code, name, text):\n    {body}\n\n"
_METHOD = "class {cls}:\n    async def {name}(self, store, run_id):\n        {body}\n\n"
_FAIL = "await store.finalize_run(run_id, status=RunStatus.ERROR.value)"
_REBUILD = "await store.finalize_with_event(run_id, status=RunStatus.ERROR.value, event_type=EV)"
_CANCEL = "await store.finalize_run(run_id, status=RunStatus.CANCELLED.value)"
_STARTED = "await _emit_event(store, run_id, StoredEventType.RUN_STARTED.value, {})"
_FINGERPRINT = "fp = compute_idempotency_fingerprint(name, text)"
_OUTPUT_NAME = 'fp = f"{name.__module__}.{name.__qualname__}"'
_LOSS = 'return code.startswith("08")'

# (file, template, class, allowed function, body); the class is empty for a module function
ALLOWED_COPIES = [
    ("runtime/runner.py", _FUNCTION, "", "_persist_run_failure", _FAIL),
    ("runtime/_reclaim.py", _FUNCTION, "", "_handle_rebuild_failure", _REBUILD),
    ("runtime/runner.py", _FUNCTION, "", "_persist_run_cancellation", _CANCEL),
    ("runtime/runner.py", _FUNCTION, "", "_finalize_loop_cancelled", _CANCEL),
    ("runtime/runner.py", _FUNCTION, "", "_pause_with_event", _CANCEL),
    ("agent.py", _METHOD, "Agent", "_cancel_one", _CANCEL),
    ("runtime/runner.py", _FUNCTION, "", "_start_new_run", _STARTED),
    ("runtime/runner.py", _FUNCTION, "", "_new_run_metadata", _FINGERPRINT),
    ("runtime/runner.py", _FUNCTION, "", "_new_run_metadata", _OUTPUT_NAME),
    ("runtime/runner.py", _FUNCTION, "", "_idempotent_hit_after_image_error", _FINGERPRINT),
    ("runtime/stores/_errors.py", _FUNCTION, "", "_is_connection_loss_code", _LOSS),
]


@SKIP_NO_SEMGREP
def test_an_allowed_shape_passes_only_in_its_own_function(tmp_path: Path) -> None:
    files: dict[str, str] = {}
    expected: set[tuple[str, str]] = set()
    for where, template, cls, name, body in ALLOWED_COPIES:
        module = "dendrux." + where.removesuffix(".py").replace("/", ".")
        # The allowed place first, then the same code in the wrong place (another name and,
        # for a method, another class): only the wrong places may be reported.
        wrong = [("Other", name), (cls, "somewhere_else")] if cls else [("", "somewhere_else")]
        for place_cls, place_name in [(cls, name), *wrong]:
            files[where] = files.get(where, "") + template.format(
                cls=place_cls, name=place_name, body=body
            )
        expected.update((module, f"{c}.{n}" if c else n) for c, n in wrong)
    keys = sr.collect_keys(_scratch_tree(tmp_path, files))
    assert {tuple(key.split("|")[1:]) for key in keys} == expected


@SKIP_NO_SEMGREP
def test_definition_sites_and_historical_migrations_are_out_of_scope(tmp_path: Path) -> None:
    definitions = {
        "_lifecycle.py": (
            'RUN_STARTED = "run.started"\nSTALE = "stale_running"\n'
            'CHECKPOINTED = {"storage_format": 2}\n'
        ),
        "_codes.py": 'STALE = "stale_running"\nHELP = "Start with ./dendrux.db or set the URL."\n'
        'LOST = ("57P01", "57P02")\n',
        "_meta_keys.py": 'PARTS = "dendrux.parts"\n',
        "db/session.py": 'DEFAULT = "sqlite+aiosqlite:///./dendrux.db"\n',
        "db/migrations/versions/old.py": (
            'EVENT = "run.started"\nREASON = {"reason": "cancel_requested"}\n'
        ),
        "runtime/stores/sql/base.py": (
            "async def f(store, run_id):\n"
            "    await store.finalize_run(run_id, status=RunStatus.CANCELLED.value)\n"
        ),
        "guardrails/_protocol.py": 'GID = f"{kind.__module__}.{kind.__qualname__}"\n',
        "m.py": (
            'PendingPause.__module__ = "dendrux.store"\n'
            "def f(code):\n    return code == '53300' or code.startswith('23')\n"
        ),
    }
    assert sr.collect_keys(_scratch_tree(tmp_path, definitions)) == {}


@SKIP_NO_SEMGREP
def test_a_copy_of_a_shared_value_outside_its_module_is_flagged(tmp_path: Path) -> None:
    copies = {
        "event": 'EVENT = "run.started"',
        "reason": 'REASON = {"reason": "stale_running"}',
        "storage": "FORMAT_OK = record.storage_format == 2",
        "meta": 'PARTS = meta["dendrux.parts"]',
        "url": 'URL = "sqlite+aiosqlite:///./dendrux.db"',
        "loss": 'LOST = code in ("57P01", "57P03")',
    }
    source = "".join(
        f"def {name}(record, code, meta):\n    {line}\n" for name, line in copies.items()
    )
    keys = sr.collect_keys(_scratch_tree(tmp_path, {"runtime/m.py": source}))
    assert {key.split("|")[0] for key in keys} == {
        "stored-event-name-literal",
        "lifecycle-reason-literal",
        "storage-format-literal",
        "meta-key-literal",
        "default-sqlite-url-literal",
        "connection-loss-code-copy",
    }
    assert {key.rsplit("|", 1)[1] for key in keys} == set(copies)
