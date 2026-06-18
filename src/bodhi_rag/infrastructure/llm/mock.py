"""
Mock LLM adapter for testing.

Returns deterministic responses without network calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bodhi_rag.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bodhi_rag.application.config import LLMConfig
    from bodhi_rag.ports.llm import LLMMessage


class MockLLMAdapter:
    """Fake LLM adapter that returns deterministic responses."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @traced("mock.llm.generate")
    async def generate(
        self,
        messages: list[LLMMessage],
        **_kwargs: str | float,
    ) -> str:
        """Return a mock response that includes the last user message."""
        user_messages = [
            message["content"]
            for message in messages
            if message["role"] == "user"
        ]
        if not user_messages:
            return "This is a mock response."
        return f"Mock answer: {user_messages[-1]}"
