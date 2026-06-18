"""Unit tests for the in-memory vector store adapter."""

from __future__ import annotations

import pytest

from bodhi_rag.application.config import VectorStoreConfig
from bodhi_rag.domain.exceptions import DocumentNotFoundError
from bodhi_rag.domain.value_objects import DocumentId
from bodhi_rag.infrastructure.vector_store.in_memory import MockVectorStoreAdapter


@pytest.mark.asyncio
async def test_delete_raises_not_found_for_missing_document() -> None:
    """Deleting a missing document should raise DocumentNotFoundError."""
    adapter = MockVectorStoreAdapter(VectorStoreConfig(provider="in_memory"))

    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        await adapter.delete(DocumentId())
