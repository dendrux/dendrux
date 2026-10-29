"""Coded logging sites must use the closed registry emission boundary."""

import ast
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "src/dendrux"
MIGRATED = ("llm", "runtime", "store", "mcp", "loops", "rest")


def bypasses(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "critical",
            "log",
        }:
            continue
        coded_extra = any(
            kw.arg == "extra"
            and isinstance(kw.value, ast.Dict)
            and any(isinstance(key, ast.Constant) and key.value == "code" for key in kw.value.keys)
            for kw in node.keywords
        )
        coded_message = any(
            isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and re.search(r"\bcode=|\bmcp_discovery_%", part.value)
            for arg in node.args
            for part in ast.walk(arg)
        )
        if coded_extra or coded_message:
            found.append(node.lineno)
    return found


@pytest.mark.parametrize("package", MIGRATED)
def test_coded_sites_use_log_code(package):
    paths = (
        sorted(
            p
            for p in SOURCE.rglob("*.py")
            if p.name != "_codes.py" and p.relative_to(SOURCE).parts[0] not in MIGRATED
        )
        if package == "rest"
        else sorted((SOURCE / package).rglob("*.py"))
    )
    assert paths
    violations = [
        (str(p.relative_to(SOURCE)), line) for p in paths for line in bypasses(p.read_text())
    ]
    assert not violations, violations


@pytest.mark.parametrize(
    "source",
    [
        'logger.warning("event", extra={"code": "usage_invalid"})',
        'logger.warning("code=usage_invalid")',
        'logger.warning("code=%s", code)',
        'logger.warning("mcp_discovery_%s", reason)',
    ],
)
def test_scan_rejects_bypasses(source):
    assert bypasses(source) == [1]


def test_scan_accepts_registry_emission():
    assert not bypasses("_codes.log_code(logger, _codes.USAGE_INVALID)")


def test_provider_diagnostic_uses_closed_fields(caplog):
    from dendrux.errors import ProviderError
    from dendrux.llm._resilience import log_provider_error

    log_provider_error(
        ProviderError(provider="PRIVATE_SENTINEL", model="PRIVATE_SENTINEL", category="unknown"),
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    )
    (record,) = caplog.records
    assert record.kind == "provider"
    assert record.provider == "custom"
    assert record.category == "unknown"
    assert "PRIVATE_SENTINEL" not in repr(record.__dict__)


def assigned_strings(tree, names):
    """Read code-valued assignments, excluding condition strings in ternaries."""
    definitions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            definitions.setdefault(node.target.id, []).append(node.value)

    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, (ast.Tuple, ast.List)):
            continue
        if not isinstance(node.iter, ast.Name):
            continue
        for sequence in definitions.get(node.iter.id, []):
            if not isinstance(sequence, (ast.Tuple, ast.List)):
                continue
            for item in sequence.elts:
                if not isinstance(item, (ast.Tuple, ast.List)):
                    continue
                for target, value in zip(node.target.elts, item.elts, strict=True):
                    if isinstance(target, ast.Name):
                        definitions.setdefault(target.id, []).append(value)

    def values(node, seen=frozenset()):
        if isinstance(node, ast.Constant):
            return {node.value} if isinstance(node.value, str) else set()
        if isinstance(node, ast.Name) and node.id not in seen:
            return set().union(
                *(values(item, seen | {node.id}) for item in definitions.get(node.id, []))
            )
        if isinstance(node, ast.IfExp):
            return values(node.body, seen) | values(node.orelse, seen)
        if isinstance(node, ast.BoolOp):
            return set().union(*(values(item, seen) for item in node.values))
        return set()

    return set().union(*(values(ast.Name(id=name)) for name in names))


def dynamic_registry_vocabularies():
    import inspect
    from typing import get_args

    from dendrux.errors import ProviderErrorCategory, RunNotReclaimableError
    from dendrux.llm._provider_codes import PROVIDER_CODES, SELF_RAISED_CODES, map_provider_code
    from dendrux.mcp._errors import DiscoveryLimitReason

    trees = {
        str(path.relative_to(SOURCE)): ast.parse(path.read_text()) for path in SOURCE.rglob("*.py")
    }
    # Fail closed if a new module introduces a dynamic lookup that needs its own vocabulary.
    sites = {
        path
        for path, tree in trees.items()
        if any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "REGISTRY"
            for node in ast.walk(tree)
        )
    }
    assert sites == {
        "llm/_resilience.py",
        "runtime/_reclaim.py",
        "runtime/state.py",
        "runtime/persistence.py",
        "loops/react.py",
        "mcp/_client.py",
    }
    vocabularies = {
        "provider categories": set(get_args(ProviderErrorCategory)),
        "provider codes": set(SELF_RAISED_CODES)
        | {"provider_unknown"}
        | {
            map_provider_code(provider, raw)
            for provider, table in PROVIDER_CODES.items()
            for raw in table
        },
        "reclaim reasons": set(re.findall(r"``([a-z_]+)``", inspect.getdoc(RunNotReclaimableError)))
        - {"reason"},
        "MCP discovery": {"mcp_discovery_" + reason for reason in get_args(DiscoveryLimitReason)},
    }
    # Check every concrete reclaim refusal too: adding a reason without updating
    # the public vocabulary must not evade registry coverage.
    for tree in trees.values():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RunNotReclaimableError"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
            ):
                vocabularies["reclaim reasons"].add(node.args[1].value)
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "reclaim_run"
            ):
                vocabularies["reclaim reasons"] |= assigned_strings(node, {"reason"})
    for path in (
        "loops/react.py",
        "runtime/persistence.py",
        "runtime/_reclaim.py",
        "runtime/state.py",
    ):
        for function in ast.walk(trees[path]):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Stop at the closest function containing a lookup; outer methods may
            # contain unrelated reason assignments in their other nested helpers.
            body = ast.Module(body=function.body, type_ignores=[])
            direct = [
                node
                for node in ast.walk(body)
                if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "REGISTRY"
            ]
            if not direct or any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and any(
                    isinstance(child, ast.Subscript)
                    and isinstance(child.value, ast.Attribute)
                    and child.value.attr == "REGISTRY"
                    for child in ast.walk(node)
                )
                for node in function.body
            ):
                continue
            names = {"code", "reason", "legacy_reason"} | {
                part.id
                for node in direct
                for part in ast.walk(node.slice)
                if isinstance(part, ast.Name)
            }
            codes = assigned_strings(body, names)
            codes |= {
                part.value
                for node in direct
                for part in ast.walk(node.slice)
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            }
            if path == "runtime/state.py":
                # The sweep projects this storage failure reason to legacy_run_swept.
                codes.discard("stale_running_legacy")
            vocabularies[f"{path}:{function.name}"] = codes
    return vocabularies


def test_dynamic_registry_vocabularies_are_registered():
    from dendrux._codes import REGISTRY

    vocabularies = dynamic_registry_vocabularies()
    assert vocabularies["provider categories"]
    assert {
        "timeout_not_started",
        "timeout_outcome_unknown",
        "cancelled",
        "outcome_unknown",
    } <= set().union(*vocabularies.values())
    assert {
        "legacy_run_swept",
        "stale_running",
        "never_started",
        "reclaim_limit",
        "cancel_requested",
    } <= set().union(*vocabularies.values())
    missing = {
        name: sorted(values - REGISTRY.keys())
        for name, values in vocabularies.items()
        if values - REGISTRY.keys()
    }
    assert not missing, missing
