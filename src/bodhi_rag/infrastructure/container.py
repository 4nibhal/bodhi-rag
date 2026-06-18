"""
Dependency injection container for new architecture.

Wires ports to adapter implementations based on configuration.
This is the new Container that works with the Protocol-based ports.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final, TypeVar, cast

from bodhi_rag.application.config import BhodiConfig, EmbeddingConfig, LLMConfig
from bodhi_rag.conversation.ports.memory import ConversationMemoryPort
from bodhi_rag.ports.chunker import ChunkerPort
from bodhi_rag.ports.document_parser import DocumentParserPort
from bodhi_rag.ports.embedding import EmbeddingPort
from bodhi_rag.ports.llm import LLMPort
from bodhi_rag.ports.reranker import RerankerPort
from bodhi_rag.ports.vector_store import VectorStorePort

if TYPE_CHECKING:
    from pathlib import Path

    from bodhi_rag.application.facade import BhodiApplication


OpenAIConfigT = TypeVar("OpenAIConfigT", EmbeddingConfig, LLMConfig)


OPENAI_COMPATIBLE_PROVIDER_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "lmstudio",
        "lm_studio",
        "openrouter",
        "vllm",
    },
)


def _resolve_openai_compatible_provider_family(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in OPENAI_COMPATIBLE_PROVIDER_ALIASES:
        return "openai"
    return normalized


class Container:
    """
    Dependency injection container for bodhi-rag architecture.

    Builds adapter instances based on configuration and wires
    them to the BhodiApplication facade.
    """

    def __init__(self, config: BhodiConfig) -> None:
        self._config = config
        self._adapters: dict[type, object] = {}

    @classmethod
    def build_from(
        cls,
        config: BhodiConfig | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> Container:
        """
        Build a `Container` from a config, an optional config path, or both.

        When `config` is provided, it is used as-is. When `config_path` is
        provided, `load_bodhi_config(config_path=...)` is called and the
        result is used. When both are provided, the explicit `config` wins
        and `config_path` is documented in a debug log (it is not
        re-loaded). Callers should prefer one path or the other.
        """
        from bodhi_rag.application.config_loader import load_bodhi_config

        if config is None:
            if config_path is None:
                config = load_bodhi_config()
            else:
                config = load_bodhi_config(config_path=config_path)
        return cls(config)

    def build(self) -> BhodiApplication:
        """Build and return a fully wired BhodiApplication."""
        from bodhi_rag.answering.application.synthesize import (
            SynthesizeAnswerUseCase,
        )
        from bodhi_rag.application.facade import BhodiApplication
        from bodhi_rag.conversation.application.memory import (
            ConversationMemoryUseCase,
        )
        from bodhi_rag.indexing.application.delete import (
            DeleteDocumentUseCase,
        )
        from bodhi_rag.indexing.application.index import (
            IndexDocumentUseCase,
        )
        from bodhi_rag.retrieval.application.retrieve import (
            RetrieveQueryUseCase,
        )

        document_parser = cast("DocumentParserPort", self._get_adapter(DocumentParserPort))
        chunker = cast("ChunkerPort", self._get_adapter(ChunkerPort))
        embedding = cast("EmbeddingPort", self._get_adapter(EmbeddingPort))
        vector_store = cast("VectorStorePort", self._get_adapter(VectorStorePort))
        reranker = cast("RerankerPort", self._get_adapter(RerankerPort))
        llm = cast("LLMPort", self._get_adapter(LLMPort))
        conversation_memory = cast(
            "ConversationMemoryPort",
            self._get_adapter(ConversationMemoryPort),
        )

        return BhodiApplication(
            index_document=IndexDocumentUseCase(
                document_parser=document_parser,
                chunker=chunker,
                embedding=embedding,
                vector_store=vector_store,
            ),
            delete_document=DeleteDocumentUseCase(
                vector_store=vector_store,
            ),
            retrieve_query=RetrieveQueryUseCase(
                embedding=embedding,
                vector_store=vector_store,
                reranker=reranker,
            ),
            synthesize_answer=SynthesizeAnswerUseCase(
                llm=llm,
            ),
            conversation_memory=ConversationMemoryUseCase(conversation_memory),
        )

    def _get_adapter(self, port_type: type[object]) -> object:
        """Get or create adapter for given port type."""
        if port_type not in self._adapters:
            factory_name = f"_create_{port_type.__name__.replace('Port', '').lower()}_adapter"
            factory = getattr(self, factory_name, None)
            if factory is None:
                msg = f"No factory for adapter type: {port_type.__name__}"
                raise ValueError(msg)
            self._adapters[port_type] = factory()
        return self._adapters[port_type]

    def _create_embedding_adapter(self) -> object:
        """Create embedding adapter based on config."""
        from bodhi_rag.infrastructure.embedding.mock import MockEmbeddingAdapter

        provider = _resolve_openai_compatible_provider_family(
            self._config.embedding.provider,
        )

        if provider == "mock":
            return MockEmbeddingAdapter(self._config.embedding)

        if provider == "openai":
            from bodhi_rag.infrastructure.embedding.openai import (
                OpenAIEmbeddingsAdapter,
            )

            return OpenAIEmbeddingsAdapter(
                self._with_openai_api_key(self._config.embedding),
            )

        msg = f"Unknown embedding provider: {provider}"
        raise ValueError(msg)

    def _create_vectorstore_adapter(self) -> object:
        """Create vector store adapter based on config."""
        from bodhi_rag.infrastructure.vector_store.in_memory import (
            MockVectorStoreAdapter,
        )

        provider = self._config.vector_store.provider.lower()

        if provider == "in_memory":
            return MockVectorStoreAdapter(self._config.vector_store)

        if provider == "chroma":
            from bodhi_rag.infrastructure.vector_store.chroma import (
                ChromaVectorStoreAdapter,
            )

            return ChromaVectorStoreAdapter(self._config.vector_store)

        msg = f"Unknown vector_store provider: {provider}"
        raise ValueError(msg)

    def _create_chunker_adapter(self) -> object:
        """Create chunker adapter based on config."""
        from bodhi_rag.infrastructure.chunker.fixed_size import (
            FixedSizeChunkerAdapter,
        )
        from bodhi_rag.infrastructure.chunker.recursive import (
            RecursiveChunkerAdapter,
        )

        provider = self._config.chunker.provider.lower()

        if provider == "fixed_size":
            return FixedSizeChunkerAdapter(self._config.chunker)

        if provider == "recursive":
            return RecursiveChunkerAdapter(self._config.chunker)

        msg = f"Unknown chunker provider: {provider}"
        raise ValueError(msg)

    def _create_documentparser_adapter(self) -> object:
        """Create document parser adapter based on config."""
        from bodhi_rag.infrastructure.document_parser.mock import (
            MockDocumentParserAdapter,
        )

        provider = self._config.parser.provider.lower()

        if provider == "mock":
            return MockDocumentParserAdapter(self._config.parser)

        if provider == "pypdf":
            from bodhi_rag.infrastructure.document_parser.pypdf import (
                PyPDFDocumentParserAdapter,
            )

            return PyPDFDocumentParserAdapter(self._config.parser)

        msg = f"Unknown parser provider: {provider}"
        raise ValueError(msg)

    def _create_llm_adapter(self) -> object:
        """Create LLM adapter based on config."""
        from bodhi_rag.infrastructure.llm.mock import MockLLMAdapter

        provider = _resolve_openai_compatible_provider_family(self._config.llm.provider)

        if provider == "mock":
            return MockLLMAdapter(self._config.llm)

        if provider == "ollama":
            from bodhi_rag.infrastructure.llm.ollama import OllamaLLMAdapter

            return OllamaLLMAdapter(self._config.llm)

        if provider == "openai":
            from bodhi_rag.infrastructure.llm.openai import OpenAILLMAdapter

            return OpenAILLMAdapter(self._with_openai_api_key(self._config.llm))

        msg = f"Unknown llm provider: {provider}"
        raise ValueError(msg)

    def _create_conversationmemory_adapter(self) -> object:
        """Create conversation memory adapter based on config."""
        from bodhi_rag.conversation.infrastructure.volatile import (
            VolatileConversationMemoryAdapter,
        )

        provider = self._config.conversation.provider.lower()

        if provider == "volatile":
            return VolatileConversationMemoryAdapter(self._config.conversation)

        msg = f"Unknown conversation provider: {provider}"
        raise ValueError(msg)

    def _create_reranker_adapter(self) -> object:
        """Create reranker adapter based on config."""
        from bodhi_rag.infrastructure.reranker.cross_encoder import (
            CrossEncoderReranker,
        )
        from bodhi_rag.infrastructure.reranker.noop import NoOpReranker

        provider = self._config.reranker.provider.lower()

        if provider == "noop":
            return NoOpReranker(self._config.reranker)

        if provider == "cross_encoder":
            return CrossEncoderReranker(self._config.reranker)

        msg = f"Unknown reranker provider: {provider}"
        raise ValueError(msg)

    def _with_openai_api_key(self, config: OpenAIConfigT) -> OpenAIConfigT:
        """Inject a resolved OpenAI API key into adapter config."""
        extra = getattr(config, "extra", {})
        api_key = extra.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            provider = getattr(config, "provider", "openai")
            msg = (
                f"OpenAI credentials required for provider '{provider}'. "
                "Set OPENAI_API_KEY or provide extra.api_key in config."
            )
            raise ValueError(msg)
        updated = config.model_copy(update={"extra": {**extra, "api_key": api_key}})
        return cast("OpenAIConfigT", updated)
