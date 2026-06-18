"""Tests for the package version resolver."""

from __future__ import annotations

from importlib.metadata import version

from bodhi_rag._version import get_version


def test_get_version_matches_installed_metadata() -> None:
    """
    The resolver should return the same version as importlib.metadata.

    This is the single-source-of-truth assertion: the API server's
    FastAPI app version, the /health response body, and the installed
    distribution all share the same string.
    """
    assert get_version() == version("bodhi-rag")


def test_get_version_is_stable() -> None:
    """The resolver is memoized; the second call returns the same object."""
    assert get_version() == get_version()
