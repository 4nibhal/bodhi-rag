"""Unit tests for bodhi-rag API routes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bodhi_rag._version import get_version
from bodhi_rag.application.models import (
    HealthStatus,
    IndexDocumentResponse,
    QueryResponse,
)
from bodhi_rag.domain.exceptions import DocumentNotFoundError
from bodhi_rag.interfaces.api import app as api_app_module
from bodhi_rag.interfaces.api.app import create_app

ClientFactory = Callable[[Path | None], AbstractContextManager[TestClient]]


def _reset_rate_limit_state() -> None:
    api_app_module._rate_limit_requests.clear()  # noqa: SLF001


@pytest.fixture
def mock_bodhi_rag_app() -> AsyncMock:
    """Provide a mocked BhodiApplication."""
    mock = AsyncMock()
    mock.health_check = AsyncMock(
        return_value=HealthStatus(status="healthy", version=get_version()),
    )
    mock.index_document = AsyncMock(
        return_value=IndexDocumentResponse(document_id="doc-123", chunk_count=5),
    )
    mock.query = AsyncMock(
        return_value=QueryResponse(
            answer_text="Test answer",
            citations=[],
            conversation_id=None,
        ),
    )
    mock.delete_document = AsyncMock(return_value=None)
    mock.get_conversation_history = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def client_factory(
    mock_bodhi_rag_app: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> ClientFactory:
    """Create TestClient instances with configurable API source roots."""

    @contextmanager
    def _factory(source_root: Path | None) -> Iterator[TestClient]:
        _reset_rate_limit_state()

        if source_root is None:
            monkeypatch.delenv("BODHI_API_SOURCE_ROOT", raising=False)
        else:
            monkeypatch.setenv("BODHI_API_SOURCE_ROOT", str(source_root))

        with patch("bodhi_rag.interfaces.api.app.Container") as mock_container_cls:
            mock_container = mock_container_cls.return_value
            mock_container.build.return_value = mock_bodhi_rag_app
            app = create_app()
            with TestClient(app) as test_client:
                yield test_client

    return _factory


@pytest.fixture
def client(client_factory: ClientFactory, tmp_path: Path) -> Iterator[TestClient]:
    """Create a default TestClient with an allowed source root."""
    (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4\n%test\n")
    with client_factory(tmp_path) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health check should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == get_version()


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_returns_200(self, client: TestClient) -> None:
        """Query endpoint should return answer."""
        payload = {"question": "What is this?", "top_k": 3}
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["answer_text"] == "Test answer"

    def test_query_invalid_body_returns_422(self, client: TestClient) -> None:
        """Invalid body should trigger validation error."""
        response = client.post("/query", json={})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    """Tests for POST /documents and DELETE /documents/{id}."""

    def test_index_document_returns_201(self, client: TestClient) -> None:
        """Indexing a document should return 201."""
        payload = {"source": "test.pdf", "metadata": {"key": "value"}}
        response = client.post("/documents", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["chunk_count"] == 5

    def test_index_document_resolves_source_within_root(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Indexing should pass a validated, resolved Path to the app layer."""
        payload = {"source": "test.pdf", "metadata": {"key": "value"}}
        response = client.post("/documents", json=payload)

        assert response.status_code == 201
        request_arg = mock_bodhi_rag_app.index_document.await_args.args[0]
        assert request_arg.source == (tmp_path / "test.pdf").resolve()

    def test_index_document_rejects_local_paths_when_root_unset(
        self,
        client_factory: ClientFactory,
        mock_bodhi_rag_app: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Local file indexing via API should require an explicit root."""
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4\n%test\n")

        with client_factory(None) as client:
            response = client.post("/documents", json={"source": str(tmp_path / "test.pdf")})

        assert response.status_code == 400
        assert "BODHI_API_SOURCE_ROOT" in response.json()["detail"]
        mock_bodhi_rag_app.index_document.assert_not_called()

    def test_index_document_rejects_paths_outside_root(
        self,
        client_factory: ClientFactory,
        mock_bodhi_rag_app: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """API indexing should reject sources outside the configured root."""
        source_root = tmp_path / "allowed"
        source_root.mkdir()
        outside_file = tmp_path / "outside.pdf"
        outside_file.write_bytes(b"%PDF-1.4\n%test\n")

        with client_factory(source_root) as client:
            response = client.post("/documents", json={"source": str(outside_file)})

        assert response.status_code == 400
        assert "within BODHI_API_SOURCE_ROOT" in response.json()["detail"]
        mock_bodhi_rag_app.index_document.assert_not_called()

    def test_index_document_rejects_disallowed_suffix(
        self,
        client_factory: ClientFactory,
        mock_bodhi_rag_app: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """API indexing should reject unsupported local file types."""
        (tmp_path / "test.py").write_text("print('nope')", encoding="utf-8")

        with client_factory(tmp_path) as client:
            response = client.post("/documents", json={"source": "test.py"})

        assert response.status_code == 400
        assert ".py" not in response.json()["detail"]
        assert "one of" in response.json()["detail"]
        mock_bodhi_rag_app.index_document.assert_not_called()

    def test_index_document_internal_error_is_sanitized(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
    ) -> None:
        """Unexpected indexing errors should not leak to API clients."""
        mock_bodhi_rag_app.index_document.side_effect = RuntimeError("secret details")

        response = client.post("/documents", json={"source": "test.pdf"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"

    def test_delete_document_returns_200(self, client: TestClient) -> None:
        """Deleting a document should return confirmation."""
        response = client.delete("/documents/12345678-1234-1234-1234-123456789abc")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["document_id"] == "12345678-1234-1234-1234-123456789abc"

    def test_delete_document_not_found_returns_404(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
    ) -> None:
        """Deleting a missing document should return 404."""
        mock_bodhi_rag_app.delete_document.side_effect = DocumentNotFoundError(
            "12345678-1234-1234-1234-123456789abc",
        )
        response = client.delete("/documents/12345678-1234-1234-1234-123456789abc")
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"


class TestConversationsEndpoint:
    """Tests for GET /conversations/{id}."""

    def test_get_conversation_returns_200(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
    ) -> None:
        """Getting conversation history should return turns."""
        turn = AsyncMock()
        turn.user_message = "Hello"
        turn.assistant_message = "Hi there"
        turn.turn_index = 0
        mock_bodhi_rag_app.get_conversation_history.return_value = [turn]

        response = client.get("/conversations/12345678-1234-1234-1234-123456789abc")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "12345678-1234-1234-1234-123456789abc"
        assert len(data["turns"]) == 1
        assert data["turns"][0]["user_message"] == "Hello"


class TestQueryRouteErrors:
    """Tests for query route error handling."""

    def test_query_internal_error_is_sanitized(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
    ) -> None:
        """Unexpected query errors should not leak to API clients."""
        mock_bodhi_rag_app.query.side_effect = RuntimeError("secret details")

        response = client.post("/query", json={"question": "What is this?", "top_k": 3})

        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"
