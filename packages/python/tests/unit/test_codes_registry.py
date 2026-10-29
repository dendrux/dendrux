"""The registry covers literal and finite dynamic emissions, with closed fields."""

import ast
import importlib
import importlib.util
import logging
import re
from pathlib import Path

import pytest

from dendrux.errors import PROVIDER_ERROR_CATEGORIES

SOURCE = Path(__file__).parents[2] / "src/dendrux"
RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def registry_module():
    assert importlib.util.find_spec("dendrux._codes"), "stable log registry is missing"
    return importlib.import_module("dendrux._codes")


def literal_codes(source):
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    constants = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value.value, str)
    }
    found = set()

    def strings(value, bindings, seen=frozenset()):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return {value.value}
        if isinstance(value, ast.Name) and value.id not in seen:
            return set().union(
                *(strings(v, bindings, seen | {value.id}) for v in bindings.get(value.id, []))
            )
        if isinstance(value, ast.IfExp):
            return strings(value.body, bindings, seen) | strings(value.orelse, bindings, seen)
        if isinstance(value, ast.BoolOp):
            return set().union(*(strings(v, bindings, seen) for v in value.values))
        return set()

    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in {"warning", "error", "info", "debug", "exception", "critical"}:
            continue
        scope = call
        while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parents[scope]
        bindings = {key: [ast.Constant(value)] for key, value in constants.items()}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, []).append(node.value)
        if (
            call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            message = call.args[0].value
            match = re.search(r"\bcode=%s", message)
            if match:
                index = len(re.findall(r"%(?:[sdfr])", message[: match.start()])) + 1
                if index < len(call.args):
                    found.update(strings(call.args[index], bindings))
        for node in ast.walk(call):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.update(re.findall(r"\bcode=([a-z][a-z0-9_.]+)", node.value))
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "code":
                        found.update(strings(value, bindings))
                        for part in ast.walk(value):
                            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                                found.add(part.value)
                            elif isinstance(part, ast.Name) and part.id in constants:
                                found.add(constants[part.id])
    return found


def test_d1_every_emitted_literal_is_registered():
    registry = registry_module().REGISTRY
    emitted = set().union(*(literal_codes(p.read_text()) for p in SOURCE.rglob("*.py")))
    assert len(emitted) >= 60
    assert emitted <= registry.keys(), sorted(emitted - registry.keys())


def test_d1_literal_scan_rejects_an_unregistered_code():
    emitted = literal_codes('logger.warning("code=unregistered_test_code")')
    assert emitted - registry_module().REGISTRY.keys() == {"unregistered_test_code"}


@pytest.mark.parametrize(
    "source",
    [
        'NEW_CODE = "unregistered_test_code"\nlogger.warning("event", extra={"code": NEW_CODE})',
        'code = "unregistered_test_code"\nlogger.warning("run_id=%s code=%s", run_id, code)',
        'code = "unregistered_test_code" if started else "cancelled"\n'
        'logger.warning("code=%s", code)',
    ],
)
def test_d1_scan_follows_constant_and_dynamic_literal_bindings(source):
    assert literal_codes(source) - registry_module().REGISTRY.keys() == {"unregistered_test_code"}


def test_d1_dynamic_emissions_are_registered():
    registry = registry_module().REGISTRY
    dynamic = {
        "outcome_unknown",
        "cancelled",
        "timeout_not_started",
        "timeout_outcome_unknown",
        "size_limit",
        "invalid_tool_schema",
        "migration_lock_timeout",
        "input_size_limit",
        "submission_missing",
        "persistence_error",
        "stale_running",
        "cancel_requested",
    } | {
        f"mcp_discovery_{reason}"
        for reason in (
            "repeated_cursor",
            "page_cap",
            "deadline",
            "tool_cap",
            "byte_cap",
            "cursor_length",
        )
    }
    assert dynamic <= registry.keys()


@pytest.mark.parametrize("category", PROVIDER_ERROR_CATEGORIES)
def test_provider_log_category_fallback_is_registered(category, caplog):
    from dendrux.errors import ProviderError
    from dendrux.llm._resilience import log_provider_error

    log_provider_error(ProviderError(category=category), RUN_ID)
    (record,) = caplog.records
    assert record.code == category
    assert registry_module().REGISTRY[record.code].level == record.levelno


def test_d1_registry_declares_every_existing_structured_field():
    registry = registry_module().REGISTRY
    missing = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        constants = "\n".join(
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        )
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr not in {"warning", "error", "info", "debug", "exception"}:
                continue
            for keyword in call.keywords:
                if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                    continue
                keys = {
                    key.value
                    for key in keyword.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                for name in literal_codes(constants + "\n" + ast.unparse(call)):
                    if name in registry and keys - registry[name].fields:
                        missing.append(
                            (
                                str(path.relative_to(SOURCE)),
                                name,
                                sorted(keys - registry[name].fields),
                            )
                        )
    assert not missing, missing


def test_log_code_emits_base_and_declared_fields(caplog):
    module = registry_module()
    code = module.REGISTRY["lease_renew_failed"]
    with caplog.at_level(logging.DEBUG):
        module.log_code(logging.getLogger("dendrux.test"), code, run_id=RUN_ID, attempt=2)
    (record,) = caplog.records
    assert (record.code, record.kind, record.run_id, record.attempt) == (
        code.name,
        code.kind,
        RUN_ID,
        2,
    )
    assert record.levelno == code.level


@pytest.mark.parametrize(
    "fields",
    [
        {"payload": "SECRET_SENTINEL"},
        {"attempt": "SECRET_SENTINEL"},
        {"attempt": True},
        {"attempt": float("inf")},
        {"run_id": "SECRET_SENTINEL"},
        {"tenant_id": "not an identifier"},
        {"category": "SECRET_SENTINEL"},
        {"call_id": "SECRET_SENTINEL"},
    ],
)
def test_invalid_fields_raise_without_echoing_values(fields):
    module = registry_module()
    with pytest.raises(ValueError) as error:
        module.log_code(
            logging.getLogger("dendrux.test"), module.REGISTRY["lease_renew_failed"], **fields
        )
    assert "SECRET_SENTINEL" not in str(error.value)


def test_production_drops_bad_fields_with_one_safe_diagnostic(monkeypatch, caplog):
    module = registry_module()
    monkeypatch.setattr(module, "_STRICT_FIELDS", False)
    module.log_code(
        logging.getLogger("dendrux.test"),
        module.REGISTRY["lease_renew_failed"],
        run_id=RUN_ID,
        attempt="SECRET_SENTINEL",
        payload="SECRET_SENTINEL",
    )
    assert [r.code for r in caplog.records] == ["log_field_dropped", "lease_renew_failed"]
    assert caplog.records[0].count == 2
    assert "attempt" not in caplog.records[1].__dict__
    assert "SECRET_SENTINEL" not in caplog.text
    assert "SECRET_SENTINEL" not in repr([r.__dict__ for r in caplog.records])


def test_registry_metadata_is_complete_and_immutable():
    module = registry_module()
    assert all(
        c.name == name
        and c.fires_when
        and c.do
        and c.kind
        and {"run_id", "tenant_id", "code", "kind"} <= c.fields
        for name, c in module.REGISTRY.items()
    )
    with pytest.raises(TypeError):
        module.REGISTRY["injected"] = module.REGISTRY["run_fenced"]


def test_declared_enum_numeric_and_identifier_values(caplog):
    module = registry_module()
    logger = logging.getLogger("dendrux.test")
    module.log_code(
        logger,
        module.REGISTRY["sweep_cap_reached"],
        run_id=RUN_ID,
        tenant_id="acme",
        category="stale_running",
        cap=10,
        inspected=10,
    )
    (record,) = caplog.records
    assert record.category == "stale_running" and record.tenant_id == "acme"
    module.log_code(
        logger,
        module.REGISTRY["stream_size_limit"],
        provider="openai",
        category="rejected",
        status=400,
        timeout_reason="call_deadline",
        retryable=False,
    )
    assert caplog.records[-1].retryable is False
    assert module._valid_field("call_id", RUN_ID)
    assert not module._valid_field("run_id", "8" + RUN_ID[1:])
    assert not module._valid_field("run_id", RUN_ID.lower())
    with pytest.raises(ValueError):
        module.log_code(logger, module.REGISTRY["sweep_cap_reached"], category="PRIVATE_SENTINEL")
    with pytest.raises(ValueError):
        module.log_code(
            logger, module.Code("unregistered", 30, "runtime", frozenset(), "when", "do")
        )


def test_field_enums_cover_existing_public_closed_types():
    from typing import get_args

    from dendrux.errors import ProviderErrorCategory, TimeoutReason
    from dendrux.runtime._entry_checks import EntryPoint

    module = registry_module()
    assert set(get_args(ProviderErrorCategory)) <= module._ENUMS["category"]
    assert set(get_args(TimeoutReason)) == module._ENUMS["timeout_reason"]
    assert set(get_args(EntryPoint)) <= module._ENUMS["entry"]
    assert module._valid_field("reason", "governance_changed")
