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
        {"tenant_id": "PRIVATE_TENANT_SENTINEL\n"},
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
    monkeypatch.delenv("DENDRUX_STRICT_LOG_FIELDS", raising=False)
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


@pytest.mark.parametrize("strict", [None, "0", "1"])
def test_strict_fields_are_explicit_and_not_inferred_from_pytest(monkeypatch, caplog, strict):
    module = registry_module()
    if strict is None:
        monkeypatch.delenv("DENDRUX_STRICT_LOG_FIELDS", raising=False)
    else:
        monkeypatch.setenv("DENDRUX_STRICT_LOG_FIELDS", strict)
    logger = logging.getLogger("dendrux.test")
    if strict == "1":
        with pytest.raises(ValueError):
            module.log_code(logger, module.LEASE_RENEW_FAILED, attempt="PRIVATE_SENTINEL")
        assert not caplog.records
    else:
        module.log_code(logger, module.LEASE_RENEW_FAILED, attempt="PRIVATE_SENTINEL")
        assert [r.code for r in caplog.records] == ["log_field_dropped", "lease_renew_failed"]
        assert "PRIVATE_SENTINEL" not in repr([r.__dict__ for r in caplog.records])


def test_custom_provider_is_a_fixed_token_in_strict_mode(monkeypatch, caplog):
    module = registry_module()
    monkeypatch.setenv("DENDRUX_STRICT_LOG_FIELDS", "1")
    module.log_code(
        logging.getLogger("dendrux.test"),
        module.PROVIDER_UNKNOWN,
        provider="PRIVATE_PROVIDER_SENTINEL",
    )
    (record,) = caplog.records
    assert record.provider == "custom"
    assert "PRIVATE_PROVIDER_SENTINEL" not in repr(record.__dict__)


@pytest.mark.parametrize("tenant", ["acme.com", "org:7", "ops@example.test", "équipe", "x" * 256])
def test_runtime_tenant_identifiers_are_preserved(tenant, caplog):
    module = registry_module()
    module.log_code(
        logging.getLogger("dendrux.test"),
        module.LEASE_RENEW_FAILED,
        run_id=RUN_ID,
        tenant_id=tenant,
        attempt=3,
    )
    (record,) = caplog.records
    assert record.tenant_id == tenant


@pytest.mark.parametrize("tenant", ["", "x" * 257, "tenant\n", "tenant\x7f", "tenant\x85", 7])
def test_invalid_tenant_drops_only_that_field(monkeypatch, caplog, tenant):
    module = registry_module()
    monkeypatch.delenv("DENDRUX_STRICT_LOG_FIELDS", raising=False)
    module.log_code(
        logging.getLogger("dendrux.test"),
        module.LEASE_RENEW_FAILED,
        run_id=RUN_ID,
        tenant_id=tenant,
        attempt=3,
    )
    diagnostic, record = caplog.records
    assert diagnostic.code == "log_field_dropped" and diagnostic.count == 1
    assert "tenant_id" not in record.__dict__
    assert record.attempt == 3 and record.run_id == diagnostic.run_id == RUN_ID


@pytest.mark.parametrize(
    "name,value",
    [
        ("call_id", "toolu_01.A:b-c"),
        ("tool_call_id", "call_abc"),
        ("call_id", "a" * 128),
        ("trace_id", "a" * 32),
        ("span_id", "b" * 16),
        ("run_id", RUN_ID),
        ("event_id", RUN_ID),
        ("row_id", RUN_ID),
        ("pause_id", RUN_ID),
    ],
)
def test_identifiers_validate_their_issuer_shape(name, value):
    assert registry_module()._valid_field(name, value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("call_id", "a" * 129),
        ("tool_call_id", "has space"),
        ("call_id", "private\n"),
        ("trace_id", RUN_ID),
        ("trace_id", "g" * 32),
        ("trace_id", "a" * 31),
        ("span_id", "a" * 32),
        ("row_id", "call_abc"),
        ("event_id", "a" * 32),
    ],
)
def test_identifiers_reject_other_issuer_shapes(name, value):
    assert not registry_module()._valid_field(name, value)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("name,value", [("limit", 5), ("status", 429)])
def test_valid_but_undeclared_fields_are_rejected(monkeypatch, caplog, strict, name, value):
    module = registry_module()
    assert module._valid_field(name, value)
    assert any(name in code.fields for code in module.REGISTRY.values())
    assert name not in module.LEASE_RENEW_FAILED.fields
    monkeypatch.setenv("DENDRUX_STRICT_LOG_FIELDS", "1" if strict else "0")
    logger = logging.getLogger("dendrux.test")
    if strict:
        with pytest.raises(ValueError):
            module.log_code(logger, module.LEASE_RENEW_FAILED, attempt=3, **{name: value})
        assert not caplog.records
    else:
        module.log_code(logger, module.LEASE_RENEW_FAILED, attempt=3, **{name: value})
        diagnostic, record = caplog.records
        assert diagnostic.code == "log_field_dropped" and diagnostic.count == 1
        assert name not in record.__dict__ and record.attempt == 3


def test_registry_membership_uses_constant_time_identity(monkeypatch, caplog):
    module = registry_module()

    def no_equality(self, other):
        raise AssertionError("registry membership must not compare metadata")

    monkeypatch.setattr(module.Code, "__eq__", no_equality)
    module.log_code(logging.getLogger("dendrux.test"), module.LEASE_RENEW_FAILED, attempt=3)
    assert caplog.records[-1].attempt == 3


def test_post_rebase_recovery_dynamic_codes_are_registered():
    names = {
        "legacy_run",
        "not_running",
        "lease_fresh",
        "already_reclaimed",
        "agent_mismatch",
        "skill_mismatch",
        "loop_unsupported",
        "recovery_call_missing",
        "recovery_state_missing",
        "recovery_state_invalid",
        "never_started",
        "run_finalized",
        "run_not_found",
        "tool_replayed",
    }
    assert names <= registry_module().REGISTRY.keys()
