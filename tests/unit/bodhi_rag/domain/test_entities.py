"""Tests for domain entities."""

import pytest

from bodhi_rag.domain.entities import Answer, Chunk, ConversationTurn, Document, Query
from bodhi_rag.domain.value_objects import ChunkId, Citation, ConversationId, DocumentId


class TestDocument:
    def test_create_document_with_text(self) -> None:
        doc = Document(id=DocumentId(), text="Hello world")
        assert doc.text == "Hello world"
        assert isinstance(doc.id, DocumentId)

    def test_document_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="Document text cannot be empty"):
            Document(id=DocumentId(), text="")

    def test_document_with_metadata(self) -> None:
        doc = Document(id=DocumentId(), text="Content", metadata={"source": "test.pdf"})
        assert doc.metadata["source"] == "test.pdf"


class TestQuery:
    def test_create_query(self) -> None:
        query = Query(text="What is this?")
        assert query.text == "What is this?"
        assert query.conversation_id is None

    def test_query_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            Query(text="")


class TestChunk:
    def test_create_chunk(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content="Chunk content",
            chunk_index=0,
            total_chunks=3,
        )
        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0

    def test_chunk_validates_chunk_index(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        with pytest.raises(ValueError, match="chunk_index must be non-negative"):
            Chunk(
                id=chunk_id,
                document_id=doc_id,
                content="Content",
                chunk_index=-1,
                total_chunks=3,
            )


class TestAnswer:
    def test_create_answer_with_citations(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        citation = Citation(
            chunk_id=chunk_id,
            text="Source text",
            source_document="doc.pdf",
        )
        answer = Answer(
            text="The answer is yes",
            citations=(citation,),
        )
        assert answer.text == "The answer is yes"
        assert len(answer.citations) == 1


class TestConversationTurn:
    def test_create_conversation_turn(self) -> None:
        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hello",
            assistant_message="Hi there",
            turn_index=0,
        )
        assert turn.user_message == "Hello"
        assert turn.assistant_message == "Hi there"
