"""
LLM port definition.

Defines the contract for message-oriented language model adapters.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

LLMRole = Literal["system", "user", "assistant"]


class LLMMessage(TypedDict):
    """Single model-native message passed to an LLM adapter."""

    role: LLMRole
    content: str


class LLMPort(Protocol):
    """Protocol for message-oriented language model generation."""

    async def generate(
        self,
        messages: list[LLMMessage],
        **kwargs: str | float,
    ) -> str:
        """
        Generate text from a sequence of messages.

        Args:
            messages: Ordered chat messages for the target provider.
            **kwargs: Provider-specific generation parameters.

        Returns:
            Generated text response.

        Raises:
            LLMError: If generation fails.

        """
        ...
