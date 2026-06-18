"""Unit tests for OpenAI embeddings adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bodhi_rag.application.config import EmbeddingConfig
from bodhi_rag.infrastructure.embedding.openai import OpenAIEmbeddingsAdapter


def _embedding_response(*values: float) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=list(values))],
    )


class TestOpenAIEmbeddingsAdapter:
    @pytest.fixture
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            extra={"api_key": "FAKE_API_KEY_FOR_TESTS"},
        )

    @pytest.fixture
    def adapter(self, config: EmbeddingConfig) -> OpenAIEmbeddingsAdapter:
        return OpenAIEmbeddingsAdapter(config)

    @pytest.mark.asyncio
    async def test_embed_query_uses_api_key_from_config_extra_when_env_missing(
        self,
        adapter: OpenAIEmbeddingsAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(
            return_value=_embedding_response(0.1, 0.2),
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_openai:
            result = await adapter.embed_query("hello")

        assert result == [0.1, 0.2]
        assert mock_openai.call_args.kwargs["api_key"] == "FAKE_API_KEY_FOR_TESTS"

    @pytest.mark.asyncio
    async def test_embed_query_uses_injected_api_key_even_if_env_changes(
        self,
        adapter: OpenAIEmbeddingsAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "FROM_ENV")
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(
            return_value=_embedding_response(0.1, 0.2),
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_openai:
            await adapter.embed_query("hello")

        assert mock_openai.call_args.kwargs["api_key"] == "FAKE_API_KEY_FOR_TESTS"

    @pytest.mark.asyncio
    async def test_embed_query_raises_when_no_api_key_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter = OpenAIEmbeddingsAdapter(EmbeddingConfig(provider="openai"))

        with pytest.raises(
            ValueError,
            match="injected api_key",
        ):
            await adapter.embed_query("hello")


    @pytest.mark.asyncio
    async def test_embed_query_passes_base_url_when_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter = OpenAIEmbeddingsAdapter(
            EmbeddingConfig(
                provider="openai",
                extra={
                    "api_key": "FAKE_API_KEY_FOR_TESTS",
                    "base_url": "https://embeddings.example.invalid/v1",
                },
            ),
        )
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(
            return_value=_embedding_response(0.1, 0.2),
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_openai:
            await adapter.embed_query("hello")

        assert mock_openai.call_args.kwargs == {
            "api_key": "FAKE_API_KEY_FOR_TESTS",
            "base_url": "https://embeddings.example.invalid/v1",
        }
