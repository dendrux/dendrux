"""The committed operator reference is exactly the registry's generated view."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from dendrux._codes import REGISTRY

ROOT = Path(__file__).parents[4]
SCRIPT = ROOT / "packages/python/tools/gen_log_codes.py"
DOCUMENT = ROOT / "docs/reference/log-codes.mdx"


def generator():
    assert SCRIPT.exists(), "registry documentation generator is missing"
    spec = importlib.util.spec_from_file_location("gen_log_codes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_log_codes_match_generated_output():
    generated = generator().render()
    assert DOCUMENT.exists(), "generated log-code reference is missing"
    assert DOCUMENT.read_text() == generated


def test_reference_has_every_code_kind_and_operator_guidance():
    generated = generator().render()
    assert "What to alert on" in generated
    assert "Fires when" in generated and "Do" in generated
    assert "<" not in generated and "{" not in generated and "}" not in generated
    for code in REGISTRY.values():
        assert f"| `{code.name}` |" in generated
        assert code.fires_when in generated and code.do in generated
    for kind in {code.kind for code in REGISTRY.values()}:
        assert f"## {'MCP' if kind == 'mcp' else kind.replace('_', ' ').title()}" in generated


def test_generator_cli_writes_requested_file(tmp_path):
    module = generator()
    output = tmp_path / "nested/log-codes.mdx"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == module.render()


def test_reference_describes_registry_emission_and_strict_opt_in():
    generated = generator().render()
    assert "Coded diagnostics use `log_code`" in generated
    assert "`cascade_truncated`" in generated.split("## What to alert on", 1)[1]
    assert "DENDRUX_STRICT_LOG_FIELDS=1" in generated
    assert "Unregistered code objects always raise" in generated
    assert "## MCP" in generated and "## Mcp" not in generated
