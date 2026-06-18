"""
Recursive character text splitter chunker adapter.

Splits text by attempting separators in order of preference,
recursively falling back to smaller separators for pieces
that still exceed the target chunk size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from bodhi_rag.domain.entities import Chunk
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bodhi_rag.application.config import ChunkerConfig


class RecursiveChunkerAdapter:
    """
    Recursive character text splitter.

    Attempts to split by the largest natural separator first
    (paragraphs, then lines, then sentences, then clauses, then words).
    If a resulting piece is still larger than chunk_size, it recursively
    applies the next smaller separator.
    """

    DEFAULT_CHUNK_SIZE: ClassVar[int] = 512
    DEFAULT_OVERLAP: ClassVar[int] = 64

    # Separators in order of preference (largest boundary first)
    SEPARATORS: ClassVar[tuple[str, ...]] = ("\n\n", "\n", ". ", ", ", " ")

    def __init__(self, config: ChunkerConfig) -> None:
        self._config = config
        self._chunk_size = config.chunk_size or self.DEFAULT_CHUNK_SIZE
        self._overlap = config.overlap if config.overlap is not None else self.DEFAULT_OVERLAP

    def _split_recursively(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text by separators until pieces fit in chunk_size."""
        if not text:
            return []

        # Base case: text already fits
        if len(text) <= self._chunk_size:
            return [text]

        # No more separators: force split by fixed size
        if not separators:
            return [
                text[i : i + self._chunk_size]
                for i in range(0, len(text), self._chunk_size)
            ]

        separator = separators[0]
        remaining = separators[1:]
        parts = text.split(separator)

        result: list[str] = []
        for raw_part in parts:
            part = raw_part.strip()
            if not part:
                continue
            if len(part) > self._chunk_size:
                result.extend(self._split_recursively(part, remaining))
            else:
                result.append(part)

        return result

    def _merge_splits(self, splits: list[str], chunk_size: int) -> list[str]:
        """Merge consecutive splits into chunks of approximately chunk_size."""
        if not splits:
            return []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # If a single split exceeds chunk_size, add it as its own chunk
            if split_len > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(split)
                continue

            # Check if adding this split would exceed chunk_size
            separator_len = 1 if current_chunk else 0
            projected = current_length + separator_len + split_len

            if projected > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(split)
            current_length += separator_len + split_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _apply_overlap(
        self, chunks: list[str], chunk_size: int, overlap: int,
    ) -> list[str]:
        """Apply overlap between consecutive chunks."""
        if not chunks or overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = result[-1]
            current = chunks[i]
            overlap_text = prev[-overlap:] if len(prev) >= overlap else prev

            combined = overlap_text + current

            # Ensure we don't exceed chunk_size; trim from current if needed
            if len(combined) > chunk_size:
                available = chunk_size - len(overlap_text)
                if available > 0:
                    combined = overlap_text + current[:available]
                else:
                    combined = overlap_text[:chunk_size]

            result.append(combined)

        return result

    @traced("chunker.recursive.chunk")
    async def chunk(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[Chunk]:
        """Split text into chunks using recursive character splitting."""
        size = chunk_size if chunk_size is not None else self._chunk_size
        ov = overlap if overlap is not None else self._overlap

        if size <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)
        if ov < 0:
            msg = "overlap must be non-negative"
            raise ValueError(msg)
        if ov >= size:
            msg = "overlap must be less than chunk_size"
            raise ValueError(msg)

        stripped = text.strip() if text else ""
        if not stripped:
            return []

        # Short text: single chunk
        if len(stripped) <= size:
            doc_id = DocumentId()
            chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
            return [
                Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=stripped,
                    chunk_index=0,
                    total_chunks=1,
                ),
            ]

        # Recursive split → merge → apply overlap
        splits = self._split_recursively(stripped, list(self.SEPARATORS))
        chunks_text = self._merge_splits(splits, size)

        if ov > 0:
            chunks_text = self._apply_overlap(chunks_text, size, ov)

        total_chunks = len(chunks_text)
        if total_chunks == 0:
            return []

        doc_id = DocumentId()
        chunks: list[Chunk] = []

        for index, chunk_text in enumerate(chunks_text):
            chunk_id = ChunkId(document_id=doc_id, chunk_index=index)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=chunk_text,
                    chunk_index=index,
                    total_chunks=total_chunks,
                ),
            )

        return chunks

    @property
    def default_chunk_size(self) -> int:
        """Return the default chunk size."""
        return self._chunk_size

    @property
    def default_overlap(self) -> int:
        """Return the default overlap."""
        return self._overlap
