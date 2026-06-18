"""
RetrieveQueryUseCase.

Encapsulates the full retrieval pipeline: embed the question,
overfetch from the vector store, rerank, and return the final
candidate set. The answering bounded context consumes the
output; the application facade delegates here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.domain.entities import RetrievedDocument
    from bodhi_rag.ports.embedding import EmbeddingPort
    from bodhi_rag.ports.reranker import RerankerPort
    from bodhi_rag.ports.vector_store import VectorStorePort


class RetrieveQueryUseCase:
    """
    Application-layer entry point for retrieval.

    The orchestration order is:

    1. Embed the question with the configured embedding provider.
    2. Overfetch from the vector store: ask for
       `top_k * reranker.overfetch_factor` candidates so the
       reranker has room to reorder without losing high-relevance
       items that the raw vector similarity ranked just below the
       cutoff.
    3. Rerank and trim to `top_k` via the configured reranker.
    4. Return the final relevance-ordered list of chunks.
    """

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        reranker: RerankerPort,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._reranker = reranker

    async def execute(
        self,
        question: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        """
        Run the retrieval pipeline for `question`.

        Args:
            question: The user query.
            top_k: Number of chunks to return.

        Returns:
            Reranked and trimmed list of `RetrievedDocument` chunks.

        """
        query_embedding = await self._embedding.embed_query(question)

        # Defensive max with top_k: a zero overfetch_factor must
        # never produce an under-fetch that returns fewer candidates
        # than the caller asked for.
        overfetch_top_k = max(
            top_k * self._reranker.overfetch_factor,
            top_k,
        )

        candidates = await self._vector_store.search(
            query_embedding,
            overfetch_top_k,
        )

        return await self._reranker.rerank(
            question,
            candidates,
            top_k=top_k,
        )
