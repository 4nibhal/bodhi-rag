"""
OpenAI LLM adapter.

Generates text using OpenAI's chat completions API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from bodhi_rag.domain.exceptions import LLMError
from bodhi_rag.infrastructure.tracing import traced

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

    from bodhi_rag.application.config import LLMConfig
    from bodhi_rag.ports.llm import LLMMessage


class OpenAILLMAdapter:
    """OpenAI adapter for LLM text generation."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: AsyncOpenAI | None = None
        self._model = config.model or self.DEFAULT_MODEL

    async def _ensure_client(self) -> None:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = self._config.extra.get("api_key")
            base_url = self._config.extra.get("base_url")
            if not api_key:
                msg = "OpenAI LLM adapter requires an injected api_key"
                raise ValueError(msg)

            if isinstance(base_url, str) and base_url:
                self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            else:
                self._client = AsyncOpenAI(api_key=api_key)

    @traced("openai.llm.generate")
    async def generate(
        self,
        messages: list[LLMMessage],
        **kwargs: str | float,
    ) -> str:
        """Generate text from message-oriented input using OpenAI."""
        await self._ensure_client()

        temperature = float(kwargs.get("temperature", self._config.temperature))
        max_tokens = int(kwargs.get("max_tokens", self._config.max_tokens or 2048))
        openai_messages = [
            cast(
                "ChatCompletionMessageParam",
                {"role": message["role"], "content": message["content"]},
            )
            for message in messages
        ]

        client = self._client
        if client is None:
            msg = "OpenAI client not initialized"
            raise RuntimeError(msg)

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            msg = "generate"
            raise LLMError(msg, str(exc)) from exc

        content = response.choices[0].message.content
        if content is None:
            return ""
        return cast("str", content)
