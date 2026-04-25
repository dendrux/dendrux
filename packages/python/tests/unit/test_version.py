"""Verify package imports and version."""

from dendrux import __version__


def test_version_exists() -> None:
    """The installed package must have a real version, not the dev fallback."""
    assert __version__
    assert __version__ != "0.0.0+dev"
