"""
Smoke tests for `bodhi_rag.application`'s public surface.

Catches the class of bug where a name is advertised in `__all__`
but cannot be imported (because the underlying module is broken
or has been removed).
"""

from __future__ import annotations

import importlib


def test_all_advertised_names_resolve() -> None:
    """Every name in `bodhi_rag.application.__all__` must be importable."""
    app = importlib.import_module("bodhi_rag.application")

    for name in app.__all__:
        assert hasattr(app, name), f"`bodhi_rag.application.{name}` is advertised but not present"


def test_indexing_entry_point_is_the_facade() -> None:
    """
    `BhodiApplication.index_document` is the canonical indexing entry point.

    After the removal of `IndexDocumentsUseCase` (a broken unused use
    case), indexing flows through the facade, not through a separate
    use case class.
    """
    facade = importlib.import_module("bodhi_rag.application.facade")
    assert hasattr(facade.BhodiApplication, "index_document")
