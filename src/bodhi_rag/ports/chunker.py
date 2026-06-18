"""
Chunking port definition.

Defines the contract for text chunking adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from bodhi_rag.domain.entities import Chunk


class ChunkerPort(Protocol):
    """
    Protocol for text chunking strategies.

    Adapters implementing this port handle splitting
    text into smaller, indexable chunks.
    """

    async def chunk(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Raw text to chunk.
            chunk_size: Target chunk size (provider-dependent, e.g., tokens or characters).
            overlap: Overlap between consecutive chunks.

        Returns:
            List of document chunks.

        Raises:
            ChunkingError: If chunking fails.

        """
        ...

    @property
    def default_chunk_size(self) -> int:
        """Return the default chunk size for this chunker."""
        ...

    @property
    def default_overlap(self) -> int:
        """Return the default overlap for this chunker."""
        ...
