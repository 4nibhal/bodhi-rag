"""
ConversationMemoryUseCase.

Thin orchestrator over the `ConversationMemoryPort`. Exists so the
facade and other bounded contexts depend on a stable application-layer
abstraction; the port can change (e.g. add ACL) without rippling
into callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.conversation.ports.memory import ConversationMemoryPort
    from bodhi_rag.domain.entities import ConversationTurn
    from bodhi_rag.domain.value_objects import ConversationId


class ConversationMemoryUseCase:
    """
    Application-layer entry point for conversation memory operations.

    Currently a direct delegation to the port. The use case shape
    is the right place to add cross-cutting concerns (telemetry spans,
    structured logging, conversation-level policies) without leaking
    them into the adapter or the port.
    """

    def __init__(self, port: ConversationMemoryPort) -> None:
        self._port = port

    async def add(
        self,
        conversation_id: ConversationId,
        turn: ConversationTurn,
    ) -> None:
        """Append a turn to the conversation history."""
        await self._port.add(conversation_id, turn)

    async def get_history(
        self,
        conversation_id: ConversationId,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Return the most recent `limit` turns (or all if `limit` is None)."""
        return await self._port.get_history(conversation_id, limit)

    async def clear(self, conversation_id: ConversationId) -> None:
        """Drop the conversation history entirely."""
        await self._port.clear(conversation_id)
