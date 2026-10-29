"""Closed provider projections preserve classification without copying vendor codes."""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

from dendrux.errors import ProviderError
from dendrux.types import ErrorInfo

SENTINEL = "PRIVATE_VENDOR_SENTINEL_7921"


@pytest.mark.parametrize("raw", [None, "", "bad code", "bad\n", "x" * 65])
def test_missing_and_invalid_codes_remain_absent(raw):
    mapper = mapper_module().map_provider_code
    assert mapper("openai", raw) is None
    error = ProviderError(provider="openai", provider_code=raw)
    assert error.provider_code is None
    assert error.safe_provider_code is None
    assert error.provider_code_raw is None
    assert ErrorInfo.from_error(error).provider_code is None
    assert error.safe_message.endswith("code n/a)")


def test_import_dendrux_alone_installs_closed_mapping():
    import subprocess
    import sys

    script = """
import dendrux, sys
assert 'dendrux.llm._provider_codes' in sys.modules, 'import dendrux must install the closed mapper'
errors = sys.modules['dendrux.errors']
mapping = sys.modules['dendrux.llm._provider_codes']
assert errors._provider_code_mapper is mapping.map_provider_code
error = dendrux.ProviderError(provider='custom', provider_code='PRIVATE_SENTINEL')
assert error.provider_code == 'provider_unknown'
assert dendrux.ProviderError().provider_code is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["UnsupportedModalityError", "ImageRejectedError"])
@pytest.mark.parametrize(
    "raw,expected",
    [
        (SENTINEL, "provider_unknown"),
        (None, None),
        ("invalid text", None),
        ("invalid_image", "invalid_image"),
    ],
)
def test_image_error_codes_use_closed_hook(name, raw, expected):
    import copy
    import pickle

    from dendrux import errors

    cls = getattr(errors, name)
    error = cls(provider="openai", model="gpt-test", provider_code=raw)
    assert error.provider_code == expected
    assert copy.copy(error).provider_code == expected
    assert pickle.loads(pickle.dumps(error)).provider_code == expected
    assert SENTINEL not in str(error) + repr(error)


@pytest.mark.parametrize("name", ["UnsupportedModalityError", "ImageRejectedError"])
def test_image_error_codes_call_the_shared_hook(monkeypatch, name):
    from dendrux import errors

    seen = []

    def mapper(provider, raw):
        seen.append((provider, raw))
        return "provider_unknown"

    monkeypatch.setattr(errors, "_provider_code_mapper", mapper)
    error = getattr(errors, name)(provider="custom", model="m", provider_code=SENTINEL)
    assert error.provider_code == "provider_unknown"
    assert seen == [("custom", SENTINEL)]


@pytest.mark.parametrize(
    "provider,raw,mapped",
    [
        ("custom", SENTINEL, "provider_unknown"),
        ("openrouter", "429", "rate_limit_exceeded"),
        ("openai", "rate_limit_exceeded", "rate_limit_exceeded"),
        ("custom", None, None),
        ("custom", "empty_completion", "empty_completion"),
    ],
)
def test_conformance_projects_only_provider_code_channels(provider, raw, mapped):
    import copy

    from tests.conformance import harness

    def snapshot(code):
        message = f"{provider} rate_limit (HTTP 429, code {code or 'n/a'})"
        info = {
            "provider": provider,
            "category": "rate_limit",
            "retryable": True,
            "provider_code": code,
            "safe_message": message,
            "status_code": 429,
        }
        return {
            "runs": [{"record": {"error": message}}],
            "steps": [{"result": {"error_info": info}}],
            "rows": {"run_events": [{"data": {"error_info": info, "error": message}}]},
        }

    expected, actual = snapshot(raw), snapshot(mapped)
    original = copy.deepcopy(expected)
    assert harness.diff(expected, actual) is None
    assert expected == original
    actual["steps"][0]["result"]["error_info"]["status_code"] = 500
    assert harness.diff(expected, actual) is not None


@pytest.mark.parametrize("field", ["answer", "content", "code", "other", "output", "calls"])
def test_conformance_mapping_does_not_mask_other_values(field):
    from tests.conformance import harness

    raw_message = f"custom rate_limit (HTTP 429, code {SENTINEL})"
    mapped_message = "custom rate_limit (HTTP 429, code provider_unknown)"
    old, new = raw_message, mapped_message
    if field in {"output", "calls"}:
        old, new = {"provider_code": SENTINEL}, {"provider_code": "provider_unknown"}
    assert harness.diff({field: old}, {field: new}) is not None
    assert (
        harness.diff(
            {"error": "arbitrary " + raw_message}, {"error": "arbitrary " + mapped_message}
        )
        is not None
    )


def test_conformance_projects_stream_error_slot_without_changing_text():
    from tests.conformance import harness

    raw = f"custom rate_limit (HTTP 429, code {SENTINEL})"
    mapped = "custom rate_limit (HTTP 429, code provider_unknown)"
    old = {"stream_events": [["run_error", raw, None, None, None, None, raw, None, None, None]]}
    new = {"stream_events": [["run_error", raw, None, None, None, None, mapped, None, None, None]]}
    assert harness.diff(old, new) is None
    new["stream_events"][0][1] = mapped
    assert harness.diff(old, new) is not None


def test_conformance_projects_loaded_golden_before_canonical_comparison(tmp_path):
    import json

    from tests.conformance import harness

    path = tmp_path / "golden.json"
    raw = f"custom rate_limit (HTTP 429, code {SENTINEL})"
    path.write_text(json.dumps({"error": raw}))
    expected = {"error": "custom rate_limit (HTTP 429, code provider_unknown)"}
    assert harness.o.dumps_canonical(harness.read_json(path)) == harness.o.dumps_canonical(expected)
    assert json.loads(path.read_text()) == {"error": raw}


def mapper_module():
    assert importlib.util.find_spec("dendrux.llm._provider_codes"), (
        "closed provider table is missing"
    )
    return importlib.import_module("dendrux.llm._provider_codes")


@pytest.mark.parametrize(
    "provider", ["anthropic", "openai", "openai_responses", "openrouter", "custom", None]
)
@pytest.mark.parametrize("raw", [SENTINEL, None, "", "INVALID_TEXT with spaces"])
def test_unknown_is_closed_at_every_safe_exception_projection(provider, raw):
    error = ProviderError(provider=provider, provider_code=raw)
    expected = "provider_unknown" if raw == SENTINEL else None
    assert error.provider_code == expected
    assert error.safe_provider_code == expected
    assert ErrorInfo.from_error(error).provider_code == expected
    assert SENTINEL not in repr(error)
    assert SENTINEL not in str(error)
    assert SENTINEL not in error.safe_message
    assert SENTINEL not in repr(ErrorInfo.from_error(error).to_dict())
    assert error.provider_code_raw == (raw if raw == SENTINEL else None)


@pytest.mark.parametrize(
    "provider,raw,expected",
    [
        ("anthropic", "overloaded_error", "overloaded_error"),
        ("openai", "rate_limit_exceeded", "rate_limit_exceeded"),
        ("openai_responses", "context_length_exceeded", "context_length_exceeded"),
        ("openrouter", "429", "rate_limit_exceeded"),
        ("openrouter", "402", "insufficient_credits"),
        ("anthropic", "429", "provider_unknown"),
        ("custom", "invalid_api_key", "provider_unknown"),
    ],
)
def test_per_provider_mapping(provider, raw, expected):
    assert mapper_module().map_provider_code(provider, raw) == expected


def test_closed_outputs_are_idempotent_and_registered():
    module = mapper_module()
    registry = importlib.import_module("dendrux._codes").REGISTRY
    for provider, table in module.PROVIDER_CODES.items():
        for raw, mapped in table.items():
            assert module.map_provider_code(provider, raw) == mapped
            assert module.map_provider_code(provider, mapped) == mapped
            assert mapped in registry
    for provider in [*module.PROVIDER_CODES, "custom"]:
        for raw in module.SELF_RAISED_CODES:
            assert module.map_provider_code(provider, raw) == raw
            assert raw in registry
        assert module.map_provider_code(provider, "provider_unknown") == "provider_unknown"


@pytest.mark.parametrize(
    "provider", ["anthropic", "openai", "openai_responses", "openrouter", "custom", None]
)
@pytest.mark.parametrize("code", ["response_size_limit", "response_encoding_unsupported"])
def test_transport_self_raised_codes_are_stable_and_registered(provider, code):
    module = mapper_module()
    assert code in module.SELF_RAISED_CODES
    assert module.map_provider_code(provider, code) == code
    registry = importlib.import_module("dendrux._codes").REGISTRY
    assert code in registry


def self_raised_literals(source):
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = set()

    def literals(value, bindings, seen=frozenset()):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return {value.value}
        if isinstance(value, ast.Name) and value.id not in seen:
            return set().union(
                *(literals(v, bindings, seen | {value.id}) for v in bindings.get(value.id, []))
            )
        if isinstance(value, ast.IfExp):
            return literals(value.body, bindings, seen) | literals(value.orelse, bindings, seen)
        if isinstance(value, ast.BoolOp):
            return set().union(*(literals(v, bindings, seen) for v in value.values))
        return set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        values = [kw.value for kw in node.keywords if kw.arg == "provider_code"]
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_reject":
            values += node.args[:1]
        if isinstance(node.func, ast.Name) and node.func.id in {
            "content_filter_error",
            "refusal_error",
        }:
            values += [kw.value for kw in node.keywords if kw.arg == "code"]
        scope = node
        while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parents[scope]
        bindings = {}
        for statement in ast.walk(scope):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, []).append(statement.value)
        for value in values:
            found.update(literals(value, bindings))
    return found


def test_new_self_raised_provider_codes_require_registration():
    module = mapper_module()
    root = Path(__file__).parents[2] / "src/dendrux/llm"
    found = set().union(*(self_raised_literals(p.read_text()) for p in root.rglob("*.py")))
    assert {"empty_completion", "stream_size_limit", "stream_tool_call_limit"} <= found
    assert {"response_size_limit", "response_encoding_unsupported"} <= found
    assert found <= module.SELF_RAISED_CODES, sorted(found - module.SELF_RAISED_CODES)


def test_self_raised_scan_follows_local_variable_codes():
    source = (
        "def fail(streaming):\n"
        '    code = "new_unknown_code" if streaming else "response_size_limit"\n'
        "    raise ProviderResponseError(provider_code=code)\n"
    )
    found = self_raised_literals(source)
    assert found == {"new_unknown_code", "response_size_limit"}
    assert found - mapper_module().SELF_RAISED_CODES == {"new_unknown_code"}


def test_classification_still_reads_raw_vendor_fields():
    from dendrux.llm._resilience import error_from_fields

    error = error_from_fields(
        provider="openai", model="gpt-test", status_code=429, code=SENTINEL, error_type=None
    )
    assert isinstance(error, ProviderError)
    assert error.provider_code == "provider_unknown"
    assert error.category == "rate_limit" and error.retryable


def test_mcp_transport_detail_does_not_copy_raw_provider_code(caplog):
    import logging

    from dendrux.mcp import MCPSource
    from dendrux.mcp._client import MCPClientAdapter

    error = ProviderError(provider="openai", provider_code=SENTINEL)
    adapter = MCPClientAdapter(MCPSource.http("test", "https://mcp.example.invalid"))
    with caplog.at_level(logging.DEBUG):
        projected = adapter._safe_error("connect", error)
    assert (
        SENTINEL not in str(projected) + repr(projected) + projected.transport_detail + caplog.text
    )


@pytest.mark.xfail(
    strict=True, reason="Unowned ProviderError constructor preserves vendor message in args"
)
def test_vendor_message_is_absent_from_exception_str_and_repr():
    error = ProviderError(provider="openai", message=SENTINEL)
    assert SENTINEL not in str(error) + repr(error)


def test_vendor_message_is_absent_from_mcp_transport_detail_and_debug_log(caplog):
    import logging

    from dendrux.mcp import MCPSource
    from dendrux.mcp._client import MCPClientAdapter

    adapter = MCPClientAdapter(MCPSource.http("test", "https://mcp.example.invalid"))
    with caplog.at_level(logging.DEBUG):
        error = adapter._safe_error("connect", RuntimeError(SENTINEL))
    assert SENTINEL not in error.transport_detail + caplog.text


def test_anthropic_timeout_code_keeps_resolution_and_is_registered():
    module = mapper_module()
    assert module.map_provider_code("anthropic", "timeout_error") == "timeout_error"
    from dendrux._codes import REGISTRY

    assert REGISTRY["timeout_error"].kind == "provider"


@pytest.mark.parametrize("clone", ["copy", "deepcopy", "pickle"])
def test_context_length_copy_keeps_private_raw_code(clone):
    import copy
    import pickle

    from dendrux.errors import ContextLengthExceededError

    error = ContextLengthExceededError(
        "request exceeds context", provider="openai", provider_code=SENTINEL
    )
    restored = (
        pickle.loads(pickle.dumps(error)) if clone == "pickle" else getattr(copy, clone)(error)
    )
    assert restored.provider_code_raw == SENTINEL
    assert restored.provider_code == restored.safe_provider_code == "provider_unknown"
    assert SENTINEL not in restored.safe_message
