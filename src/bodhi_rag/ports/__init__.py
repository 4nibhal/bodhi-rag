"""
Ports package.

Contains Protocol definitions for the cross-context adapter ports.
Bounded-context-owned ports (e.g. ConversationMemoryPort) live in
their context's `ports/` module and are imported from there, not
from this top-level `__init__`.
"""

from bodhi_rag.ports.chunker import ChunkerPort
from bodhi_rag.ports.document_parser import DocumentParserPort
from bodhi_rag.ports.embedding import EmbeddingPort
from bodhi_rag.ports.llm import LLMMessage, LLMPort, LLMRole
from bodhi_rag.ports.reranker import RerankerPort
from bodhi_rag.ports.vector_store import VectorStorePort

__all__ = [
    "ChunkerPort",
    "DocumentParserPort",
    # Lower-level adapter ports
    "EmbeddingPort",
    "LLMMessage",
    "LLMPort",
    "LLMRole",
    "RerankerPort",
    "VectorStorePort",
]
