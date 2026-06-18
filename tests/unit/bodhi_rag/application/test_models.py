"""Tests for application request/response models."""

import pytest
from pydantic import ValidationError

from bodhi_rag._version import get_version
from bodhi_rag.application.models import (
    CitationResponse,
    HealthStatus,
    IndexDocumentRequest,
    QueryRequest,
    QueryResponse,
)


class TestIndexDocumentRequest:
    def test_create_minimal(self) -> None:
        request = IndexDocumentRequest(source="document.pdf")
        assert request.source == "document.pdf"
        assert request.metadata == {}

    def test_create_with_all_fields(self) -> None:
        request = IndexDocumentRequest(
            source="doc.pdf",
            metadata={"author": "test"},
            chunk_size=512,
            overlap=64,
        )
        assert request.chunk_size == 512


class TestQueryRequest:
    def test_defaults(self) -> None:
        request = QueryRequest(question="What is this?")
        assert request.question == "What is this?"
        assert request.top_k == 5
        assert request.temperature == 0.7
        assert request.conversation_id is None

    def test_top_k_validation(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=0)  # must be >= 1

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="test", temperature=-0.1)


class TestQueryResponse:
    def test_create_with_citations(self) -> None:
        citation = CitationResponse(
            chunk_id="abc:0",
            text="Source text",
            source_document="doc.pdf",
        )
        response = QueryResponse(
            answer_text="The answer",
            citations=[citation],
            conversation_id="conv-123",
        )
        assert response.answer_text == "The answer"
        assert len(response.citations) == 1


class TestHealthStatus:
    def test_create(self) -> None:
        status = HealthStatus(status="healthy", version=get_version())
        assert status.status == "healthy"
        assert status.version == get_version()
