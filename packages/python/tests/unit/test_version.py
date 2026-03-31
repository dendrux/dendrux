"""Verify package imports and version."""

from dendrux import __version__


def test_version_exists() -> None:
    assert __version__ == "0.1.0a1"
