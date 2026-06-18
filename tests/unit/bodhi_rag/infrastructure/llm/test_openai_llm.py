"""Unit tests for OpenAI LLM adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bodhi_rag.application.config import LLMConfig
from bodhi_rag.domain.exceptions import LLMError
from bodhi_rag.infrastructure.llm.openai import OpenAILLMAdapter


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me a joke"},
    ]


class TestOpenAILLMAdapter:
    """Test suite for OpenAILLMAdapter."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        """Default LLM config for testing."""
        return LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            extra={"api_key": "FAKE_API_KEY_FOR_TESTS"},
        )

    @pytest.fixture
    def adapter(self, config: LLMConfig) -> OpenAILLMAdapter:
        """Adapter instance for testing."""
        return OpenAILLMAdapter(config)

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Mock OpenAI chat completion response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Generated response"
        return response

    @pytest.mark.asyncio
    async def test_generate_calls_openai_with_correct_params(
        self,
        adapter: OpenAILLMAdapter,
        mock_openai_response: MagicMock,
    ) -> None:
        """generate() forwards message payloads to OpenAI."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        messages = _messages()

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await adapter.generate(
                messages,
                temperature=0.5,
                max_tokens=100,
            )

        mock_client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=100,
        )
        assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_wraps_openai_exception_in_llm_error(
        self,
        adapter: OpenAILLMAdapter,
    ) -> None:
        """generate() wraps OpenAI exceptions in LLMError."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API rate limit exceeded"),
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client), pytest.raises(
            LLMError,
        ) as exc_info:
            await adapter.generate(_messages())

        assert "API rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.operation == "generate"

    def test_default_model_when_config_model_is_none(self) -> None:
        """Adapter uses the default model when config model is not set."""
        config = LLMConfig(provider="openai")
        adapter = OpenAILLMAdapter(config)
        assert object.__getattribute__(adapter, "_model") == OpenAILLMAdapter.DEFAULT_MODEL

    def test_model_from_config(self) -> None:
        """Adapter uses the configured model when provided."""
        config = LLMConfig(provider="openai", model="gpt-4o")
        adapter = OpenAILLMAdapter(config)
        assert object.__getattribute__(adapter, "_model") == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_uses_config_defaults(
        self,
        adapter: OpenAILLMAdapter,
        config: LLMConfig,
        mock_openai_response: MagicMock,
    ) -> None:
        """generate() falls back to config defaults when kwargs are not provided."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            await adapter.generate(_messages())

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == config.temperature
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_generate_uses_api_key_from_config_extra_when_env_missing(
        self,
        adapter: OpenAILLMAdapter,
        mock_openai_response: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Config-provided credentials work without relying on process env."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_openai:
            await adapter.generate(_messages())

        assert mock_openai.call_args.kwargs["api_key"] == "FAKE_API_KEY_FOR_TESTS"

    @pytest.mark.asyncio
    async def test_generate_uses_injected_api_key_even_if_env_changes(
        self,
        adapter: OpenAILLMAdapter,
        mock_openai_response: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adapter credentials come from injected config, not live env reads."""
        monkeypatch.setenv("OPENAI_API_KEY", "FROM_ENV")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_openai:
            await adapter.generate(_messages())

        assert mock_openai.call_args.kwargs["api_key"] == "FAKE_API_KEY_FOR_TESTS"


    @pytest.mark.asyncio
    async def test_generate_passes_base_url_when_configured(
        self,
        mock_openai_response: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OpenAI-compatible endpoints should receive an explicit base URL."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter = OpenAILLMAdapter(
            LLMConfig(
                provider="openai",
                extra={
                    "api_key": "FAKE_API_KEY_FOR_TESTS",
                    "base_url": "https://llm.example.invalid/v1",
                },
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_openai:
            await adapter.generate(_messages())

        assert mock_openai.call_args.kwargs == {
            "api_key": "FAKE_API_KEY_FOR_TESTS",
            "base_url": "https://llm.example.invalid/v1",
        }
