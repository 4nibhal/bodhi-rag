"""
OpenAI embeddings adapter.

Generates embeddings using OpenAI's API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from bodhi_rag.infrastructure.tracing import traced

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from bodhi_rag.application.config import EmbeddingConfig


class OpenAIEmbeddingsAdapter:
    """
    OpenAI embeddings adapter.

    Uses OpenAI's text-embedding-3 models for generating embeddings.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._client: AsyncOpenAI | None = None
        self._model = config.model or self.DEFAULT_MODEL
        self._dimensions = config.dimensions or self.DEFAULT_DIMENSIONS

    async def _ensure_client(self) -> None:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = self._config.extra.get("api_key")
            base_url = self._config.extra.get("base_url")
            if not api_key:
                msg = "OpenAI embedding adapter requires an injected api_key"
                raise ValueError(msg)

            if isinstance(base_url, str) and base_url:
                self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            else:
                self._client = AsyncOpenAI(api_key=api_key)

    @traced("openai.embed_documents")
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using OpenAI."""
        await self._ensure_client()

        client = self._client
        if client is None:
            msg = "OpenAI embedding client not initialized"
            raise RuntimeError(msg)

        response = await client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )

        return [cast("list[float]", item.embedding) for item in response.data]

    @traced("openai.embed_query")
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query using OpenAI."""
        await self._ensure_client()

        client = self._client
        if client is None:
            msg = "OpenAI embedding client not initialized"
            raise RuntimeError(msg)

        response = await client.embeddings.create(
            model=self._model,
            input=[text],
            dimensions=self._dimensions,
        )

        return cast("list[float]", response.data[0].embedding)

    async def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions
