"""Unit tests for the dependency injection container."""

from __future__ import annotations

import pytest

from bodhi_rag.application.config import (
    BhodiConfig,
    ChunkerConfig,
    ConversationConfig,
    DocumentParserConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)
from bodhi_rag.infrastructure.container import Container


def _make_config(
    *,
    embedding: str = "mock",
    vector_store: str = "in_memory",
    llm: str = "mock",
) -> BhodiConfig:
    return BhodiConfig(
        embedding=EmbeddingConfig(provider=embedding, dimensions=8),
        vector_store=VectorStoreConfig(provider=vector_store),
        chunker=ChunkerConfig(provider="fixed_size", chunk_size=64),
        parser=DocumentParserConfig(provider="mock"),
        llm=LLMConfig(provider=llm),
        conversation=ConversationConfig(provider="volatile"),
    )


def test_build_raises_for_unknown_embedding_provider() -> None:
    """Unknown embedding providers should fail fast."""
    container = Container(_make_config(embedding="mystery"))

    with pytest.raises(ValueError, match="Unknown embedding provider: mystery"):
        container.build()


def test_build_raises_for_unknown_vector_store_provider() -> None:
    """Unknown vector store providers should fail fast."""
    container = Container(_make_config(vector_store="mystery"))

    with pytest.raises(ValueError, match="Unknown vector_store provider: mystery"):
        container.build()


def test_container_injects_openai_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI credentials are resolved in the container, not inside adapters."""
    monkeypatch.setenv("OPENAI_API_KEY", "FROM_ENV")
    config = _make_config(embedding="openai")
    container = Container(config)

    adapter = object.__getattribute__(container, "_create_embedding_adapter")()

    assert object.__getattribute__(adapter, "_config").extra["api_key"] == "FROM_ENV"


def test_container_raises_when_openai_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI adapters should fail fast when no credential is resolvable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(embedding="openai")
    container = Container(config)

    with pytest.raises(ValueError, match="OpenAI credentials required"):
        object.__getattribute__(container, "_create_embedding_adapter")()


@pytest.mark.parametrize("provider", ["openai-compatible", "openrouter", "lmstudio", "vllm"])
def test_container_resolves_openai_compatible_embedding_aliases(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible embedding aliases should reuse the OpenAI adapter."""
    monkeypatch.setenv("OPENAI_API_KEY", "FROM_ENV")
    container = Container(_make_config(embedding=provider))

    adapter = object.__getattribute__(container, "_create_embedding_adapter")()

    assert adapter.__class__.__name__ == "OpenAIEmbeddingsAdapter"
    assert object.__getattribute__(adapter, "_config").extra["api_key"] == "FROM_ENV"


@pytest.mark.parametrize("provider", ["openai-compatible", "openrouter", "lm_studio", "vllm"])
def test_container_resolves_openai_compatible_llm_aliases(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible LLM aliases should reuse the OpenAI adapter."""
    monkeypatch.setenv("OPENAI_API_KEY", "FROM_ENV")
    container = Container(_make_config(llm=provider))

    adapter = object.__getattribute__(container, "_create_llm_adapter")()

    assert adapter.__class__.__name__ == "OpenAILLMAdapter"
    assert object.__getattribute__(adapter, "_config").extra["api_key"] == "FROM_ENV"
