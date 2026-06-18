"""Unit tests for rate limiting middleware."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bodhi_rag._version import get_version
from bodhi_rag.application.models import (
    HealthStatus,
    IndexDocumentResponse,
    QueryResponse,
)
from bodhi_rag.interfaces.api import app as api_app_module
from bodhi_rag.interfaces.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator


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
def client(mock_bodhi_rag_app: AsyncMock) -> Iterator[TestClient]:
    """Create a TestClient with a mocked BhodiApplication."""
    _reset_rate_limit_state()
    with patch("bodhi_rag.interfaces.api.app.Container") as mock_container_cls:
        mock_container = mock_container_cls.return_value
        mock_container.build.return_value = mock_bodhi_rag_app
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


class TestRateLimiting:
    """Tests for in-memory rate limiter (100 req/min per IP)."""

    def test_health_not_rate_limited(self, client: TestClient) -> None:
        """Health endpoint should be exempt from rate limiting."""
        for _ in range(105):
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limit_returns_429(self, client: TestClient) -> None:
        """After 100 non-health requests, should return 429."""
        payload = {"question": "What is this?", "top_k": 3}
        for _ in range(100):
            response = client.post("/query", json=payload)
            assert response.status_code == 200

        response = client.post("/query", json=payload)
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_rate_limit_resets_after_window(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rate limit should reset after old timestamps expire."""
        payload = {"question": "What is this?", "top_k": 3}
        for _ in range(100):
            response = client.post("/query", json=payload)
            assert response.status_code == 200

        response = client.post("/query", json=payload)
        assert response.status_code == 429

        current_time = time.time()
        monkeypatch.setattr(api_app_module.time, "time", lambda: current_time + 120)

        response = client.post("/query", json=payload)
        assert response.status_code == 200
