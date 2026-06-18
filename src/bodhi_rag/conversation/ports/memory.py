"""
Conversation memory port.

The Protocol that the application use case depends on. Adapters in
`conversation.infrastructure` satisfy this Protocol; the use case
in `conversation.application` calls into it.

Re-exported from the top-level `bodhi_rag.ports.conversation_memory`
for backward compatibility with code that imported the port from
the canonical (cross-context) location. New code should import
directly from `bodhi_rag.conversation.ports.memory`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from bodhi_rag.domain.entities import ConversationTurn
    from bodhi_rag.domain.value_objects import ConversationId


class ConversationMemoryPort(Protocol):
    """
    Protocol for conversation memory storage.

    Adapters implementing this port handle persisting
    and retrieving conversation history.
    """

    async def add(
        self,
        conversation_id: ConversationId,
        turn: ConversationTurn,
    ) -> None:
        """Add a conversation turn to history."""
        ...

    async def get_history(
        self,
        conversation_id: ConversationId,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Get conversation history (most recent first when limit is set)."""
        ...

    async def clear(self, conversation_id: ConversationId) -> None:
        """Clear conversation history."""
        ...
