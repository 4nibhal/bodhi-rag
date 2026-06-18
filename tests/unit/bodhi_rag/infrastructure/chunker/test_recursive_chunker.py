"""Unit tests for RecursiveChunkerAdapter."""

from __future__ import annotations

import pytest

from bodhi_rag.application.config import ChunkerConfig
from bodhi_rag.domain.entities import Chunk
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.infrastructure.chunker.recursive import RecursiveChunkerAdapter


class TestRecursiveChunkerAdapter:
    """Test suite for recursive character text splitting chunker."""

    @pytest.fixture
    def default_adapter(self) -> RecursiveChunkerAdapter:
        """Adapter with default config."""
        config = ChunkerConfig(provider="recursive")
        return RecursiveChunkerAdapter(config)

    @pytest.fixture
    def small_chunk_adapter(self) -> RecursiveChunkerAdapter:
        """Adapter with small chunk_size for easier testing."""
        config = ChunkerConfig(provider="recursive", chunk_size=50, overlap=10)
        return RecursiveChunkerAdapter(config)

    @pytest.mark.asyncio
    async def test_simple_text_splits_into_approximate_chunks(
        self,
        small_chunk_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Simple text is divided into chunks of approximately chunk_size."""
        text = (
            "This is the first sentence of the document. "
            "Here is another sentence that adds more content. "
            "A third sentence follows right after that one. "
            "Finally, we have a fourth sentence to end this paragraph."
        )

        chunks = await small_chunk_adapter.chunk(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 50

    @pytest.mark.asyncio
    async def test_separator_preference_order(
        self,
        small_chunk_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Separators are respected in order: paragraphs, lines, sentences, etc."""
        # Text with paragraph breaks - should prefer splitting at \n\n.
        text = (
            "First paragraph with some content here.\n\n"
            "Second paragraph with more content here.\n\n"
            "Third paragraph with even more content here.\n\n"
            "Fourth paragraph to finish the document here."
        )

        chunks = await small_chunk_adapter.chunk(text)

        # With chunk_size=50, paragraphs are > 50 chars so they'll be
        # split further by sentence (". ") - but the key assertion is
        # that we never merge across paragraph boundaries blindly.
        # We verify chunks don't contain "\n\n" (paragraphs were split first).
        for chunk in chunks:
            assert "\n\n" not in chunk.content

    @pytest.mark.asyncio
    async def test_overlap_between_consecutive_chunks(
        self,
        small_chunk_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Overlap is present between consecutive chunks."""
        text = (
            "Sentence one has some words. "
            "Sentence two has more words. "
            "Sentence three has extra words. "
            "Sentence four has final words."
        )

        chunks = await small_chunk_adapter.chunk(text)

        assert len(chunks) >= 2
        for i in range(1, len(chunks)):
            prev = chunks[i - 1].content
            curr = chunks[i].content
            assert curr.startswith(prev[-10:])

    @pytest.mark.asyncio
    async def test_short_text_returns_single_chunk(
        self,
        default_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Text shorter than chunk_size returns a single chunk."""
        text = "Short text."

        chunks = await default_adapter.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    @pytest.mark.asyncio
    async def test_text_without_natural_separators(
        self,
        small_chunk_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Text without natural separators falls back to forced fixed-size chunks."""
        text = "a" * 200

        chunks = await small_chunk_adapter.chunk(text)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.content) <= 50
        for chunk in chunks:
            assert set(chunk.content) == {"a"}

    @pytest.mark.asyncio
    async def test_overlap_greater_or_equal_chunk_size_raises(
        self,
        default_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Overlap >= chunk_size must raise ValueError."""
        text = "Some text here."

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            await default_adapter.chunk(text, chunk_size=100, overlap=100)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            await default_adapter.chunk(text, chunk_size=100, overlap=150)

    @pytest.mark.asyncio
    async def test_chunk_objects_have_correct_ids(
        self,
        small_chunk_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Chunk objects have properly formed ChunkId and DocumentId."""
        text = (
            "First sentence is here. "
            "Second sentence is here. "
            "Third sentence is here. "
            "Fourth sentence is here."
        )

        chunks = await small_chunk_adapter.chunk(text)

        assert len(chunks) >= 1
        doc_id = chunks[0].document_id
        assert isinstance(doc_id, DocumentId)

        for index, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.document_id == doc_id
            assert isinstance(chunk.id, ChunkId)
            assert chunk.id.document_id == doc_id
            assert chunk.id.chunk_index == index
            assert chunk.chunk_index == index
            assert chunk.total_chunks == len(chunks)

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(
        self,
        default_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Empty or whitespace-only text returns an empty list."""
        assert await default_adapter.chunk("") == []
        assert await default_adapter.chunk("   ") == []
        assert await default_adapter.chunk("\n\n\n") == []

    @pytest.mark.asyncio
    async def test_negative_chunk_size_raises(
        self,
        default_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Negative or zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            await default_adapter.chunk("text", chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            await default_adapter.chunk("text", chunk_size=-5)

    @pytest.mark.asyncio
    async def test_negative_overlap_raises(
        self,
        default_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            await default_adapter.chunk("text", chunk_size=100, overlap=-1)

    @pytest.mark.asyncio
    async def test_default_properties(
        self,
        default_adapter: RecursiveChunkerAdapter,
    ) -> None:
        """Default chunk size and default overlap return configured values."""
        assert default_adapter.default_chunk_size == 512
        assert default_adapter.default_overlap == 64

    @pytest.mark.asyncio
    async def test_custom_config_properties(self) -> None:
        """Adapter respects custom config values for defaults."""
        config = ChunkerConfig(provider="recursive", chunk_size=128, overlap=16)
        adapter = RecursiveChunkerAdapter(config)

        assert adapter.default_chunk_size == 128
        assert adapter.default_overlap == 16
