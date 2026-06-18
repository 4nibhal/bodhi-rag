"""Contract tests for Protocol-based ports and their adapters."""

from __future__ import annotations

from io import BytesIO
from types import FunctionType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pytest

from bodhi_rag.application.config import (
    ChunkerConfig,
    ConversationConfig,
    DocumentParserConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)
from bodhi_rag.conversation.infrastructure.volatile import (
    VolatileConversationMemoryAdapter,
)
from bodhi_rag.conversation.ports.memory import ConversationMemoryPort
from bodhi_rag.domain.entities import Chunk, ConversationTurn
from bodhi_rag.domain.value_objects import ChunkId, ConversationId, DocumentId
from bodhi_rag.infrastructure.chunker.fixed_size import FixedSizeChunkerAdapter
from bodhi_rag.infrastructure.document_parser.mock import MockDocumentParserAdapter
from bodhi_rag.infrastructure.embedding.mock import MockEmbeddingAdapter
from bodhi_rag.infrastructure.llm.mock import MockLLMAdapter
from bodhi_rag.infrastructure.vector_store.in_memory import MockVectorStoreAdapter
from bodhi_rag.ports.chunker import ChunkerPort
from bodhi_rag.ports.document_parser import DocumentParserPort
from bodhi_rag.ports.embedding import EmbeddingPort
from bodhi_rag.ports.llm import LLMPort
from bodhi_rag.ports.vector_store import VectorStorePort

if TYPE_CHECKING:
    from pathlib import Path


def _check_protocol_methods(obj: object, protocol: type[Protocol]) -> list[str]:
    protocol_members = []
    for name, value in protocol.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, property | FunctionType):
            protocol_members.append(name)

    return [name for name in protocol_members if not hasattr(obj, name)]


@runtime_checkable
class RuntimeEmbeddingPort(EmbeddingPort, Protocol):
    pass


@runtime_checkable
class RuntimeVectorStorePort(VectorStorePort, Protocol):
    pass


@runtime_checkable
class RuntimeDocumentParserPort(DocumentParserPort, Protocol):
    pass


@runtime_checkable
class RuntimeChunkerPort(ChunkerPort, Protocol):
    pass


@runtime_checkable
class RuntimeLLMPort(LLMPort, Protocol):
    pass


@runtime_checkable
class RuntimeConversationMemoryPort(ConversationMemoryPort, Protocol):
    pass


class TestEmbeddingPortContract:
    """Verify MockEmbeddingAdapter fulfills EmbeddingPort."""

    @pytest.fixture
    def adapter(self) -> MockEmbeddingAdapter:
        config = EmbeddingConfig(provider="mock")
        return MockEmbeddingAdapter(config)

    def test_has_required_methods(self, adapter: MockEmbeddingAdapter) -> None:
        """Adapter has all methods required by EmbeddingPort."""
        missing = _check_protocol_methods(adapter, EmbeddingPort)
        assert not missing, f"Missing methods: {missing}"

    def test_runtime_checkable(self, adapter: MockEmbeddingAdapter) -> None:
        """Adapter passes runtime Protocol check."""
        assert isinstance(adapter, RuntimeEmbeddingPort)

    @pytest.mark.asyncio
    async def test_embed_documents(self, adapter: MockEmbeddingAdapter) -> None:
        """embed_documents returns list of vectors."""
        embeddings = await adapter.embed_documents(["hello", "world"])
        assert len(embeddings) == 2
        assert all(isinstance(vec, list) for vec in embeddings)
        assert all(isinstance(x, float) for vec in embeddings for x in vec)

    @pytest.mark.asyncio
    async def test_embed_query(self, adapter: MockEmbeddingAdapter) -> None:
        """embed_query returns vector."""
        embedding = await adapter.embed_query("query")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_dimensions(self, adapter: MockEmbeddingAdapter) -> None:
        """Dimensions returns int."""
        dims = await adapter.dimensions()
        assert isinstance(dims, int)
        assert dims > 0


class TestVectorStorePortContract:
    """Verify MockVectorStoreAdapter fulfills VectorStorePort."""

    @pytest.fixture
    def adapter(self) -> MockVectorStoreAdapter:
        config = VectorStoreConfig(provider="in_memory")
        return MockVectorStoreAdapter(config)

    def test_has_required_methods(self, adapter: MockVectorStoreAdapter) -> None:
        """Adapter has all methods required by VectorStorePort."""
        missing = _check_protocol_methods(adapter, VectorStorePort)
        assert not missing, f"Missing methods: {missing}"

    def test_runtime_checkable(self, adapter: MockVectorStoreAdapter) -> None:
        """Adapter passes runtime Protocol check."""
        assert isinstance(adapter, RuntimeVectorStorePort)

    @pytest.mark.asyncio
    async def test_add_and_search(self, adapter: MockVectorStoreAdapter) -> None:
        """Can add chunks and search."""
        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content="Test content",
            chunk_index=0,
            total_chunks=1,
        )

        await adapter.add([chunk], [[0.1, 0.2, 0.3]])
        results = await adapter.search([0.1, 0.2, 0.3], top_k=1)

        assert len(results) == 1
        assert results[0].text == "Test content"


class TestChunkerPortContract:
    """Verify FixedSizeChunkerAdapter fulfills ChunkerPort."""

    @pytest.fixture
    def adapter(self) -> FixedSizeChunkerAdapter:
        config = ChunkerConfig(provider="fixed_size", chunk_size=100)
        return FixedSizeChunkerAdapter(config)

    def test_has_required_methods(self, adapter: FixedSizeChunkerAdapter) -> None:
        """Adapter has all methods required by ChunkerPort."""
        missing = _check_protocol_methods(adapter, ChunkerPort)
        assert not missing, f"Missing methods: {missing}"

    def test_default_chunk_size(self, adapter: FixedSizeChunkerAdapter) -> None:
        """Default chunk_size property."""
        assert isinstance(adapter.default_chunk_size, int)
        assert adapter.default_chunk_size == 100

    def test_default_overlap(self, adapter: FixedSizeChunkerAdapter) -> None:
        """Default overlap property."""
        assert isinstance(adapter.default_overlap, int)


class TestLLMPortContract:
    """Verify MockLLMAdapter fulfills LLMPort."""

    @pytest.fixture
    def adapter(self) -> MockLLMAdapter:
        config = LLMConfig(provider="mock")
        return MockLLMAdapter(config)

    def test_has_required_methods(self, adapter: MockLLMAdapter) -> None:
        """Adapter has all methods required by LLMPort."""
        missing = _check_protocol_methods(adapter, LLMPort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_generate(self, adapter: MockLLMAdapter) -> None:
        """Generate returns str from message-oriented input."""
        result = await adapter.generate([
            {"role": "user", "content": "Hello?"},
        ])
        assert isinstance(result, str)


class TestConversationMemoryPortContract:
    """Verify VolatileConversationMemoryAdapter fulfills ConversationMemoryPort."""

    @pytest.fixture
    def adapter(self) -> VolatileConversationMemoryAdapter:
        config = ConversationConfig(provider="volatile")
        return VolatileConversationMemoryAdapter(config)

    def test_has_required_methods(
        self,
        adapter: VolatileConversationMemoryAdapter,
    ) -> None:
        """Adapter has all methods required by ConversationMemoryPort."""
        missing = _check_protocol_methods(adapter, ConversationMemoryPort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_add_and_get_history(
        self,
        adapter: VolatileConversationMemoryAdapter,
    ) -> None:
        """Can add turn and retrieve history."""
        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hello",
            assistant_message="Hi there",
        )

        await adapter.add(conv_id, turn)
        history = await adapter.get_history(conv_id)

        assert len(history) == 1
        assert history[0].user_message == "Hello"

    @pytest.mark.asyncio
    async def test_clear(self, adapter: VolatileConversationMemoryAdapter) -> None:
        """Can clear conversation history."""
        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hello",
            assistant_message="Hi",
        )

        await adapter.add(conv_id, turn)
        await adapter.clear(conv_id)
        history = await adapter.get_history(conv_id)

        assert len(history) == 0


class TestDocumentParserPortContract:
    """Verify MockDocumentParserAdapter fulfills DocumentParserPort."""

    @pytest.fixture
    def adapter(self) -> MockDocumentParserAdapter:
        config = DocumentParserConfig(provider="mock")
        return MockDocumentParserAdapter(config)

    def test_has_required_methods(self, adapter: MockDocumentParserAdapter) -> None:
        """Adapter has all methods required by DocumentParserPort."""
        missing = _check_protocol_methods(adapter, DocumentParserPort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_parse_bytes(self, adapter: MockDocumentParserAdapter) -> None:
        """Can parse bytes source."""
        document = await adapter.parse(b"hello world")
        assert document.metadata["source"] == "unknown"
        assert "mock content" in document.text

    @pytest.mark.asyncio
    async def test_parse_path(
        self,
        adapter: MockDocumentParserAdapter,
        tmp_path: Path,
    ) -> None:
        """Can parse file path source."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        document = await adapter.parse(file_path)
        assert document.metadata["source"] == str(file_path)
        assert str(file_path) in document.text

    @pytest.mark.asyncio
    async def test_parse_stream(self, adapter: MockDocumentParserAdapter) -> None:
        """Can parse binary stream."""
        stream = BytesIO(b"hello world")
        document = await adapter.parse(stream)
        assert document.metadata["source"] == "unknown"
        assert "mock content" in document.text
