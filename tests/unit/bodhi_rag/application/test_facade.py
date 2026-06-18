"""Unit tests for the bodhi-rag application facade."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from bodhi_rag._version import get_version
from bodhi_rag.application.facade import BhodiApplication
from bodhi_rag.application.models import (
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
)
from bodhi_rag.domain.entities import Chunk, Document, RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.indexing.application.index import IndexDocumentUseCase


def _build_app(
    *,
    index_document: AsyncMock | None = None,
    delete_document: AsyncMock | None = None,
    retrieve_query: AsyncMock | None = None,
    synthesize_answer: AsyncMock | None = None,
    conversation_memory: AsyncMock | None = None,
) -> BhodiApplication:
    def _make_passthrough_retrieve() -> AsyncMock:
        mock = AsyncMock()

        async def _execute(_question: str, _top_k: int) -> list[RetrievedDocument]:
            return []

        mock.execute.side_effect = _execute
        return mock

    def _make_passthrough_synthesize() -> AsyncMock:
        mock = AsyncMock()
        mock.execute.return_value = "Answer"
        return mock

    def _make_passthrough_conversation_memory() -> AsyncMock:
        mock = AsyncMock()
        mock.get_history.return_value = []
        return mock

    def _make_passthrough_index() -> AsyncMock:
        mock = AsyncMock()

        async def _execute(_request: IndexDocumentRequest) -> IndexDocumentResponse:
            return IndexDocumentResponse(
                document_id="00000000-0000-0000-0000-000000000000",
                chunk_count=1,
            )

        mock.execute.side_effect = _execute
        return mock

    return BhodiApplication(
        index_document=index_document or _make_passthrough_index(),
        delete_document=delete_document or AsyncMock(),
        retrieve_query=retrieve_query or _make_passthrough_retrieve(),
        synthesize_answer=synthesize_answer or _make_passthrough_synthesize(),
        conversation_memory=conversation_memory or _make_passthrough_conversation_memory(),
    )


@pytest.mark.asyncio
async def test_index_document_preserves_document_identity_and_merges_metadata() -> None:
    """Indexing should preserve parser identity and authoritative provenance."""
    parser_doc_id = DocumentId("11111111-1111-1111-1111-111111111111")
    chunker_doc_id = DocumentId("22222222-2222-2222-2222-222222222222")
    parser_document = Document(
        id=parser_doc_id,
        text="alpha beta gamma delta",
        metadata={
            "source": "doc.pdf",
            "filename": "doc.pdf",
            "title": "Parser Title",
            "custom": "parser",
        },
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    parser = AsyncMock()
    parser.parse.return_value = parser_document

    chunker = AsyncMock()
    chunker.chunk.return_value = [
        Chunk(
            id=ChunkId(document_id=chunker_doc_id, chunk_index=0),
            document_id=chunker_doc_id,
            content="alpha beta",
            chunk_index=0,
            total_chunks=2,
            metadata={"page": 3, "section": "intro"},
        ),
        Chunk(
            id=ChunkId(document_id=chunker_doc_id, chunk_index=1),
            document_id=chunker_doc_id,
            content="gamma delta",
            chunk_index=1,
            total_chunks=2,
            metadata={"page": 4},
        ),
    ]

    embedding = AsyncMock()
    embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    vector_store = AsyncMock()

    index_use_case = IndexDocumentUseCase(
        document_parser=parser,
        chunker=chunker,
        embedding=embedding,
        vector_store=vector_store,
    )

    app = _build_app(index_document=index_use_case)

    response = await app.index_document(
        IndexDocumentRequest(
            source="ignored.pdf",
            metadata={
                "source": "user-supplied-source",
                "title": "User Title",
                "custom": "user",
                "category": "policy",
            },
        ),
    )

    assert response.document_id == str(parser_doc_id)
    assert response.chunk_count == 2

    stored_chunks, stored_embeddings = vector_store.add.await_args.args
    assert stored_embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert [chunk.document_id for chunk in stored_chunks] == [parser_doc_id, parser_doc_id]
    assert [str(chunk.id) for chunk in stored_chunks] == [
        f"{parser_doc_id}:0",
        f"{parser_doc_id}:1",
    ]
    assert all(chunk.total_chunks == 2 for chunk in stored_chunks)
    assert stored_chunks[0].metadata["source"] == "doc.pdf"
    assert stored_chunks[0].metadata["filename"] == "doc.pdf"
    assert stored_chunks[0].metadata["title"] == "Parser Title"
    assert stored_chunks[0].metadata["user_source"] == "user-supplied-source"
    assert stored_chunks[0].metadata["user_title"] == "User Title"
    assert stored_chunks[0].metadata["custom"] == "user"
    assert stored_chunks[0].metadata["category"] == "policy"
    assert stored_chunks[0].metadata["page"] == 3
    assert stored_chunks[0].metadata["section"] == "intro"


@pytest.mark.asyncio
async def test_query_uses_human_provenance_in_citations() -> None:
    """Query citations should prefer human-friendly provenance metadata."""
    document_id = DocumentId("33333333-3333-3333-3333-333333333333")
    retrieved = RetrievedDocument(
        chunk_id=ChunkId(document_id=document_id, chunk_index=0),
        document_id=document_id,
        text="Relevant excerpt from the document.",
        score=0.9,
        metadata={"filename": "doc.pdf", "page": "7"},
    )

    retrieve_query = AsyncMock()
    retrieve_query.execute.return_value = [retrieved]
    synthesize_answer = AsyncMock()
    synthesize_answer.execute.return_value = "Answer"

    app = _build_app(
        retrieve_query=retrieve_query,
        synthesize_answer=synthesize_answer,
    )

    response = await app.query(QueryRequest(question="What happened?", top_k=1))

    assert response.answer_text == "Answer"
    assert response.citations[0].source_document == "doc.pdf"
    assert response.citations[0].page == 7


@pytest.mark.asyncio
async def test_health_check_is_non_invasive() -> None:
    """Health checks should not invoke external adapters."""
    app = _build_app()

    health = await app.health_check()

    assert health.status == "healthy"
    assert health.version == get_version()
    assert health.services == {"embedding": True, "vector_store": True, "llm": True}


@pytest.mark.asyncio
async def test_query_overfetches_then_reranks_then_trims_to_top_k() -> None:
    """retrieve_query.execute() and synthesize_answer.execute() are called with the right inputs."""
    doc_id = DocumentId("44444444-4444-4444-4444-444444444444")
    reranked = [
        RetrievedDocument(
            chunk_id=ChunkId(document_id=doc_id, chunk_index=5),
            document_id=doc_id,
            text="chunk 5",
            score=0.9,
            metadata={"idx": 5},
        ),
        RetrievedDocument(
            chunk_id=ChunkId(document_id=doc_id, chunk_index=2),
            document_id=doc_id,
            text="chunk 2",
            score=0.5,
            metadata={"idx": 2},
        ),
    ]

    retrieve_query = AsyncMock()
    retrieve_query.execute.return_value = reranked
    synthesize_answer = AsyncMock()
    synthesize_answer.execute.return_value = "promoted answer"

    app = _build_app(
        retrieve_query=retrieve_query,
        synthesize_answer=synthesize_answer,
    )

    response = await app.query(QueryRequest(question="q", top_k=2))

    retrieve_query.execute.assert_awaited_once_with("q", 2)

    args, kwargs = synthesize_answer.execute.await_args
    assert args[0] == "q"
    assert args[1] == reranked
    assert kwargs["temperature"] == 0.7

    assert response.answer_text == "promoted answer"
    assert [citation.text for citation in response.citations] == ["chunk 5", "chunk 2"]


@pytest.mark.asyncio
async def test_query_passes_temperature_from_request() -> None:
    """The temperature field in QueryRequest reaches the synthesize_answer use case."""
    retrieve_query = AsyncMock()
    retrieve_query.execute.return_value = []
    synthesize_answer = AsyncMock()
    synthesize_answer.execute.return_value = "answer"

    app = _build_app(
        retrieve_query=retrieve_query,
        synthesize_answer=synthesize_answer,
    )

    await app.query(QueryRequest(question="q", top_k=3, temperature=0.42))

    _args, kwargs = synthesize_answer.execute.await_args
    assert kwargs["temperature"] == 0.42


@pytest.mark.asyncio
async def test_query_propagates_empty_retrieval() -> None:
    """Empty retrieval (cold start, no docs) produces an answer with no citations."""
    retrieve_query = AsyncMock()
    retrieve_query.execute.return_value = []
    synthesize_answer = AsyncMock()
    synthesize_answer.execute.return_value = "no-docs answer"

    app = _build_app(
        retrieve_query=retrieve_query,
        synthesize_answer=synthesize_answer,
    )

    response = await app.query(QueryRequest(question="q", top_k=5))

    assert response.answer_text == "no-docs answer"
    assert response.citations == []
