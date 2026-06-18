"""Unit tests for health check endpoint with real services map."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bodhi_rag._version import get_version
from bodhi_rag.application.models import HealthStatus
from bodhi_rag.interfaces.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def mock_bodhi_rag_app() -> AsyncMock:
    """Provide a mocked BhodiApplication."""
    return AsyncMock()


@pytest.fixture
def client(mock_bodhi_rag_app: AsyncMock) -> Iterator[TestClient]:
    """Create a TestClient with a mocked BhodiApplication."""
    with patch("bodhi_rag.interfaces.api.app.Container") as mock_container_cls:
        mock_container = mock_container_cls.return_value
        mock_container.build.return_value = mock_bodhi_rag_app
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Tests for GET /health with real health_check behavior."""

    def test_health_returns_200_with_services(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
    ) -> None:
        """Health check should return healthy status with services map."""
        mock_bodhi_rag_app.health_check = AsyncMock(
            return_value=HealthStatus(
                status="healthy",
                version=get_version(),
                services={"embedding": True, "vector_store": True, "llm": True},
            ),
        )
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == get_version()
        assert data["services"] == {
            "embedding": True,
            "vector_store": True,
            "llm": True,
        }

    def test_health_degraded_returns_503(
        self,
        client: TestClient,
        mock_bodhi_rag_app: AsyncMock,
    ) -> None:
        """Degraded health check should return 503."""
        mock_bodhi_rag_app.health_check = AsyncMock(
            return_value=HealthStatus(
                status="degraded",
                version=get_version(),
                services={"embedding": False, "vector_store": True, "llm": True},
            ),
        )
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["embedding"] is False
