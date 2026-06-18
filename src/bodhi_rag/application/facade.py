"""bodhi-rag application facade - orchestrates use cases via protocol ports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bodhi_rag._version import get_version
from bodhi_rag.application.models import (
    CitationResponse,
    HealthStatus,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
)

if TYPE_CHECKING:
    from bodhi_rag.answering.application.synthesize import SynthesizeAnswerUseCase
    from bodhi_rag.conversation.application.memory import ConversationMemoryUseCase
    from bodhi_rag.domain.entities import ConversationTurn, RetrievedDocument
    from bodhi_rag.domain.value_objects import ConversationId, DocumentId
    from bodhi_rag.indexing.application.delete import DeleteDocumentUseCase
    from bodhi_rag.indexing.application.index import IndexDocumentUseCase
    from bodhi_rag.retrieval.application.retrieve import RetrieveQueryUseCase


def _citation_source_document(retrieved_document: RetrievedDocument) -> str:
    filename = retrieved_document.metadata.get("filename")
    if isinstance(filename, str) and filename:
        return filename

    source = retrieved_document.metadata.get("source")
    if isinstance(source, str) and source and source not in {"bytes", "stream"}:
        source_name = Path(source).name
        if source_name:
            return source_name

    return str(retrieved_document.document_id)


def _citation_page(metadata: dict[str, Any]) -> int | None:
    page = metadata.get("page")
    if page is None or isinstance(page, bool):
        return None

    try:
        return int(page)
    except (TypeError, ValueError):
        return None


class BhodiApplication:
    def __init__(
        self,
        *,
        # Use cases for every public operation. F5-C: facade is now
        # a pure orchestrator over use cases; no ports are injected
        # directly. The composition root (Container) wires the
        # underlying ports and the use cases.
        index_document: IndexDocumentUseCase,
        delete_document: DeleteDocumentUseCase,
        retrieve_query: RetrieveQueryUseCase,
        synthesize_answer: SynthesizeAnswerUseCase,
        conversation_memory: ConversationMemoryUseCase,
    ) -> None:
        self._index_document = index_document
        self._delete_document = delete_document
        self._retrieve_query = retrieve_query
        self._synthesize_answer = synthesize_answer
        self._conversation_memory = conversation_memory

    async def index_document(
        self, request: IndexDocumentRequest,
    ) -> IndexDocumentResponse:
        # Indexing pipeline (parse -> chunk -> embed -> add) lives in
        # the indexing bounded context as IndexDocumentUseCase.
        return await self._index_document.execute(request)

    async def query(self, request: QueryRequest) -> QueryResponse:
        # Retrieval pipeline lives in the retrieval bounded context:
        # embed -> vector store (overfetch) -> rerank -> trim to top_k.
        retrieved = await self._retrieve_query.execute(
            request.question,
            request.top_k,
        )

        # Answer synthesis lives in the answering bounded context:
        # LLM call with the reranked contexts.
        answer_text = await self._synthesize_answer.execute(
            request.question,
            retrieved,
            temperature=request.temperature,
        )

        citations = [
            CitationResponse(
                chunk_id=str(doc.chunk_id),
                text=doc.text[:200],
                source_document=_citation_source_document(doc),
                page=_citation_page(doc.metadata),
            )
            for doc in retrieved
        ]

        return QueryResponse(
            answer_text=answer_text,
            citations=citations,
            conversation_id=request.conversation_id,
        )

    async def health_check(self) -> HealthStatus:
        # F5-C: health check reports the use cases that are wired, not
        # the underlying ports. If a use case is None, the app is
        # misconfigured. The mapping below preserves the public
        # service names (embedding / vector_store / llm) so the API
        # contract is unchanged.
        services = {
            "embedding": self._index_document is not None,
            "vector_store": self._index_document is not None,
            "llm": self._synthesize_answer is not None,
        }

        status = "healthy" if all(services.values()) else "degraded"
        return HealthStatus(
            status=status,
            version=get_version(),
            services=services,
        )

    async def delete_document(self, document_id: DocumentId) -> None:
        """Delete a document and all its chunks."""
        await self._delete_document.execute(document_id)

    async def get_conversation_history(
        self,
        conversation_id: ConversationId,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Get conversation history for a given conversation ID."""
        return await self._conversation_memory.get_history(conversation_id, limit)
