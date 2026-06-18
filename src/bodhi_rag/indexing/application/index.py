"""
IndexDocumentUseCase.

Orchestrates the full indexing pipeline: parse the source into
a Document, chunk its text, embed the chunks, and add the
(chunks, embeddings) pair to the vector store. Returns an
IndexDocumentResponse with the document_id and chunk_count.

The use case holds the cross-context ports it depends on
(parser, chunker, embedding, vector_store). Cross-cutting concerns
(telemetry spans `indexing.parse/chunk/embed/store`, structured
logging, error mapping) belong here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bodhi_rag.application.models import IndexDocumentResponse
from bodhi_rag.domain.entities import Chunk, Document
from bodhi_rag.domain.value_objects import ChunkId

if TYPE_CHECKING:
    from bodhi_rag.application.models import IndexDocumentRequest
    from bodhi_rag.ports.chunker import ChunkerPort
    from bodhi_rag.ports.document_parser import DocumentParserPort
    from bodhi_rag.ports.embedding import EmbeddingPort
    from bodhi_rag.ports.vector_store import VectorStorePort


class IndexDocumentUseCase:
    """
    Application-layer entry point for indexing a single document.

    The pipeline order is:
        1. Parse the source into a Document (parser).
        2. Chunk the document text (chunker).
        3. Embed the chunks (embedding).
        4. Persist (chunks, embeddings) into the vector store.

    Chunk indices are rebound so they are 0..N-1 and the document
    ID is consistent across the chunks (one DocumentId, N ChunkIds
    of the form `<doc_id>:<index>`).
    """

    def __init__(
        self,
        *,
        document_parser: DocumentParserPort,
        chunker: ChunkerPort,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._document_parser = document_parser
        self._chunker = chunker
        self._embedding = embedding
        self._vector_store = vector_store

    async def execute(
        self,
        request: IndexDocumentRequest,
    ) -> IndexDocumentResponse:
        """Run the full indexing pipeline for `request.source`."""
        source = Path(request.source) if isinstance(request.source, str) else request.source
        parsed_document = await self._document_parser.parse(source)

        # Merge user-supplied metadata with the parser's authoritative
        # metadata. Keys that would shadow reserved provenance fields
        # (source, filename, etc.) are renamed with a `user_` prefix
        # so the document's canonical provenance stays unambiguous.
        document = _build_merged_document(parsed_document, request.metadata)

        chunks = await self._chunker.chunk(
            document.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )

        # Re-index chunks to 0..N-1 under a single document_id.
        rebound_chunks = _rebind_chunks(document, chunks)

        embeddings = await self._embedding.embed_documents(
            [chunk.content for chunk in rebound_chunks],
        )

        await self._vector_store.add(rebound_chunks, embeddings)

        return _build_index_response(document, rebound_chunks)


# Reserved provenance keys whose names must not be shadowed by
# user-supplied metadata. Mirrors the set in `application/facade.py`
# prior to F5-C; it lives here now because the metadata merge is
# part of the indexing pipeline's responsibility.
_RESERVED_PROVENANCE_KEYS = frozenset(
    {
        "source",
        "source_path",
        "filename",
        "file_type",
        "page_count",
        "author",
        "title",
        "subject",
    },
)


def _build_merged_document(parsed_document: Document, user_metadata: dict | None) -> Document:
    """Build the indexed Document with user metadata merged in."""
    merged = dict(parsed_document.metadata)
    for key, value in (user_metadata or {}).items():
        if key in _RESERVED_PROVENANCE_KEYS:
            merged[f"user_{key}"] = value
            continue
        merged[key] = value
    return Document(
        id=parsed_document.id,
        text=parsed_document.text,
        metadata=merged,
        indexed_at=parsed_document.indexed_at,
    )


def _rebind_chunks(document: Document, chunks: list) -> list:
    """Reindex chunks to 0..N-1 under a single document_id, merging metadata."""
    total_chunks = len(chunks)
    rebound = []
    for index, chunk in enumerate(chunks):
        rebound.append(
            Chunk(
                id=ChunkId(document_id=document.id, chunk_index=index),
                document_id=document.id,
                content=chunk.content,
                chunk_index=index,
                total_chunks=total_chunks,
                metadata={**document.metadata, **chunk.metadata},
            ),
        )
    return rebound


def _build_index_response(document: Document, rebound_chunks: list) -> IndexDocumentResponse:
    """Build the public IndexDocumentResponse from the indexed state."""
    return IndexDocumentResponse(
        document_id=str(document.id),
        chunk_count=len(rebound_chunks),
    )
