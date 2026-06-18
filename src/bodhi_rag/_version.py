"""
Package version resolution.

Reads the version from the installed distribution via
`importlib.metadata`, falling back to a regex parse of `pyproject.toml`
when the package is not installed (e.g. in a source checkout used
for local testing without `pip install -e .`).

This is the single source of truth for the runtime version string
used by the API server's OpenAPI metadata and the `/health`
response body, so both always agree with the actual built artifact
(`dist/bodhi_rag-X.Y.Z-py3-none-any.whl`).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the current `bodhi-rag` package version as a string."""
    try:
        return version("bodhi-rag")
    except Exception as exc:  # noqa: BLE001 - intentional broad catch; see fallback below
        # `importlib.metadata.version("bodhi-rag")` can fail in source-checkout contexts
        # where the package is not installed (e.g. `uv run pytest` without `pip install -e .`).
        # We fall through to the `pyproject.toml` regex below; the failure is observable
        # at DEBUG level for anyone investigating a misconfigured environment.
        logging.getLogger(__name__).debug(
            "importlib.metadata.version('bodhi-rag') failed; falling back to pyproject.toml: %s",
            exc,
        )

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "0.0.0+unknown"

    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        text,
        re.MULTILINE,
    )
    if not match:
        return "0.0.0+unknown"
    return match.group(1)
