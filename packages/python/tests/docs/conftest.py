"""The sandbox every docs snippet runs in."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.docs._runner import provider_mocks

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def docs_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A scratch cwd, dummy keys, and a mock model in place of every real provider.

    Each mock checks its constructor arguments against the real provider's signature, so
    a snippet cannot pass with an option the real class rejects.
    """
    monkeypatch.chdir(tmp_path)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, "docs-test-key")
    monkeypatch.delenv("DENDRUX_DATABASE_URL", raising=False)
    for module, name, mock in provider_mocks():
        monkeypatch.setattr(module, name, mock)
    yield
    from dendrux.db.session import reset_engine

    asyncio.run(reset_engine())
