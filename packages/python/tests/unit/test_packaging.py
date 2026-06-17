"""Packaging guards — keep the dependency declaration honest.

These read pyproject.toml directly (the source of truth) rather than
installed metadata, which is stale under editable installs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def test_httpx_is_a_base_dependency() -> None:
    """httpx is imported at module load (llm/_retry_telemetry.py) on the core
    import path, so it must be a base dependency — not only pulled transitively
    via a provider extra. Otherwise `import dendrux` fails for installs without
    a provider SDK (e.g. a read-only RunStore + make_read_router consumer).
    """
    base_deps = _pyproject()["project"]["dependencies"]
    assert any(dep.replace(" ", "").lower().startswith("httpx") for dep in base_deps), (
        f"httpx must be a base dependency; found base deps: {base_deps}"
    )
