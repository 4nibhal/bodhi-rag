"""
Integration tests for Chroma vector store adapter.

Requires chromadb package to be installed.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING

import pytest

from bodhi_rag.application.config import VectorStoreConfig
from bodhi_rag.domain.entities import Chunk
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.infrastructure.vector_store.chroma import ChromaVectorStoreAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def temp_dir() -> Iterator[pathlib.Path]:
    """Create a temporary directory for Chroma persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def chroma_adapter(temp_dir: pathlib.Path) -> ChromaVectorStoreAdapter:
    """Create a Chroma adapter with temporary directory."""
    config = VectorStoreConfig(
        provider="chroma",
        persist_directory=temp_dir,
        collection_name="test_collection",
    )
    return ChromaVectorStoreAdapter(config)


@pytest.mark.asyncio
async def test_add_and_search(chroma_adapter: ChromaVectorStoreAdapter) -> None:
    """Test adding chunks and searching."""
    doc_id = DocumentId()
    chunk_id = ChunkId(document_id=doc_id, chunk_index=0)

    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        content="This is a test document about machine learning.",
        chunk_index=0,
        total_chunks=1,
    )

    embedding = [0.1] * 128

    await chroma_adapter.add([chunk], [embedding])

    results = await chroma_adapter.search(embedding, top_k=1)

    assert len(results) == 1
    assert results[0].text == "This is a test document about machine learning."
    assert results[0].document_id == doc_id


@pytest.mark.asyncio
async def test_delete(chroma_adapter: ChromaVectorStoreAdapter) -> None:
    """Test deleting a document."""
    doc_id = DocumentId()
    chunk_id = ChunkId(document_id=doc_id, chunk_index=0)

    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        content="Document to delete.",
        chunk_index=0,
        total_chunks=1,
    )

    embedding = [0.1] * 128

    await chroma_adapter.add([chunk], [embedding])

    results = await chroma_adapter.search(embedding, top_k=1)
    assert len(results) == 1

    await chroma_adapter.delete(doc_id)

    results = await chroma_adapter.search(embedding, top_k=1)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_persist(chroma_adapter: ChromaVectorStoreAdapter) -> None:
    """Test that persist doesn't error."""
    await chroma_adapter.persist()


@pytest.mark.asyncio
async def test_multiple_chunks(chroma_adapter: ChromaVectorStoreAdapter) -> None:
    """Test adding and searching multiple chunks."""
    doc_id = DocumentId()

    chunks: list[Chunk] = []
    embeddings: list[list[float]] = []
    for i in range(3):
        chunk_id = ChunkId(document_id=doc_id, chunk_index=i)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content=f"Chunk {i} content",
            chunk_index=i,
            total_chunks=3,
        )
        chunks.append(chunk)
        embeddings.append([0.1 + i * 0.01] * 128)

    await chroma_adapter.add(chunks, embeddings)

    results = await chroma_adapter.search(embeddings[0], top_k=3)

    assert len(results) == 3
