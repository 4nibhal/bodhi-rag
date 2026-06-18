"""
Ollama LLM adapter.

Generates text using Ollama local LLM API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bodhi_rag.infrastructure.tracing import traced

if TYPE_CHECKING:
    import httpx

    from bodhi_rag.application.config import LLMConfig
    from bodhi_rag.ports.llm import LLMMessage


def _render_messages_as_prompt(messages: list[LLMMessage]) -> str:
    sections = [
        f"{message['role'].capitalize()}: {message['content']}"
        for message in messages
    ]
    return "\n\n".join(sections)


class OllamaLLMAdapter:
    """Ollama adapter for local LLM generation."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._model = config.model or self.DEFAULT_MODEL
        self._base_url = config.extra.get("base_url", self.DEFAULT_BASE_URL)

    async def _ensure_client(self) -> None:
        """Lazy initialization of Ollama client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=60.0,
            )

    @traced("ollama.generate")
    async def generate(
        self,
        messages: list[LLMMessage],
        **kwargs: str | float,
    ) -> str:
        """Generate text from message-oriented input using Ollama."""
        await self._ensure_client()

        temperature = float(kwargs.get("temperature", 0.7))
        max_tokens = int(kwargs.get("max_tokens", 2048))

        client = self._client
        if client is None:
            msg = "Ollama client not initialized"
            raise RuntimeError(msg)

        response = await client.post(
            "/api/generate",
            json={
                "model": self._model,
                "prompt": _render_messages_as_prompt(messages),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            msg = "Unexpected Ollama response payload"
            raise TypeError(msg)

        result = data.get("response", "")
        if not isinstance(result, str):
            msg = "Unexpected Ollama response body"
            raise TypeError(msg)
        return result
