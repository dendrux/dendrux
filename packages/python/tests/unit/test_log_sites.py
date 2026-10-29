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
