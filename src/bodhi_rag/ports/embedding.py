"""
Embedding port definition.

Defines the contract for embedding generation adapters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPort(Protocol):
    """
    Protocol for embedding generation.

    Adapters implementing this port must provide async methods
    for embedding document chunks and query strings.
    """

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If embedding generation fails.

        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string into a vector.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector for the query.

        Raises:
            EmbeddingError: If embedding generation fails.

        """
        ...

    async def dimensions(self) -> int:
        """
        Return the dimensionality of embedding vectors.

        Returns:
            Number of dimensions in each embedding vector.

        """
        ...
