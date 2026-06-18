"""Tests for application config."""

import pytest
from pydantic import ValidationError

from bodhi_rag.application.config import (
    BhodiConfig,
    ChunkerConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)


class TestEmbeddingConfig:
    def test_create_with_defaults(self) -> None:
        config = EmbeddingConfig(provider="openai")
        assert config.provider == "openai"
        assert config.model is None
        assert config.dimensions is None
        assert config.batch_size == 100

    def test_create_with_custom_values(self) -> None:
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            batch_size=50,
        )
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536

    def test_batch_size_validation(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(provider="openai", batch_size=0)  # must be >= 1


class TestLLMConfig:
    def test_temperature_bounds(self) -> None:
        config = LLMConfig(provider="openai", temperature=0.5)
        assert config.temperature == 0.5

    def test_temperature_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="openai", temperature=-0.1)

    def test_temperature_rejects_above_2(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="openai", temperature=2.5)


class TestChunkerConfig:
    def test_create_with_defaults(self) -> None:
        config = ChunkerConfig(provider="recursive")
        assert config.provider == "recursive"
        assert config.chunk_size is None

    def test_create_with_custom_chunk_size(self) -> None:
        config = ChunkerConfig(provider="fixed_size", chunk_size=512, overlap=64)
        assert config.chunk_size == 512
        assert config.overlap == 64


class TestBhodiConfig:
    def test_default_config(self) -> None:
        config = BhodiConfig()
        assert config.embedding.provider == "openai"
        assert config.vector_store.provider == "chroma"
        assert config.chunker.provider == "recursive"

    def test_custom_providers(self) -> None:
        config = BhodiConfig(
            embedding=EmbeddingConfig(provider="mock"),
            vector_store=VectorStoreConfig(provider="in_memory"),
        )
        assert config.embedding.provider == "mock"
        assert config.vector_store.provider == "in_memory"
