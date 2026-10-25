"""The OpenTelemetry docs page must not leave a process-global tracer provider behind.

`trace.set_tracer_provider` is one-shot per process. If the docs run first, the notifier
tests could not install their own provider and 17 of them read an empty exporter. Each
order runs in a fresh interpreter so the process-global state starts clean.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2]
DOCS_PAGE = "tests/docs/test_doc_snippets.py::test_page_snippets[docs/recipes/opentelemetry.mdx]"
NOTIFIER_TESTS = "tests/unit/test_otel_notifier.py"


def _run(*targets: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COV_", "COVERAGE_")) and k != "PYTEST_ADDOPTS"
    }
    env |= {"PYTHONPATH": str(PKG / "src"), "PYTHON_DOTENV_DISABLED": "1"}
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", *targets,
            "-q", "-p", "no:randomly", "-p", "no:cacheprovider", "-p", "no:xdist", "--no-cov",
        ],
        cwd=PKG,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )  # fmt: skip


@pytest.mark.parametrize(
    "order",
    [(DOCS_PAGE, NOTIFIER_TESTS), (NOTIFIER_TESTS, DOCS_PAGE)],
    ids=["docs-first", "tests-first"],
)
def test_otel_docs_and_notifier_tests_pass_in_either_order(
    order: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # `make ci` and CI can pass options through PYTEST_ADDOPTS; the nested run must ignore them.
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:cacheprovider --randomly-seed=7")
    result = _run(*order)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert " failed" not in result.stdout
