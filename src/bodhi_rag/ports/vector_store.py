"""
Vector store port definition.

Defines the contract for vector storage and retrieval adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from bodhi_rag.domain.entities import Chunk
    from bodhi_rag.domain.value_objects import DocumentId


class RetrievedDocument(Protocol):
    """
    A document retrieved from the vector store.

    This is a protocol to avoid coupling to specific implementations.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class VectorStorePort(Protocol):
    """
    Protocol for vector storage and retrieval.

    Adapters implementing this port handle persisting chunks
    with their embeddings and searching for similar chunks.
    """

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of chunks to store.
            embeddings: List of embedding vectors corresponding to chunks.

        Raises:
            VectorStoreError: If storage fails.

        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """
        Search for chunks most similar to the query embedding.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieved documents ranked by similarity.

        Raises:
            VectorStoreError: If search fails.

        """
        ...

    async def delete(self, document_id: DocumentId) -> None:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: The document ID to delete.

        Raises:
            VectorStoreError: If deletion fails.

        """
        ...

    async def persist(self) -> None:
        """
        Persist any pending changes to durable storage.

        Raises:
            VectorStoreError: If persistence fails.

        """
        ...
