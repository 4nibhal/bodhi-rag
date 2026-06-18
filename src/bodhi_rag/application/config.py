"""
Configuration schema for bodhi-rag platform.

All runtime configuration is driven through these Pydantic models.
No hardcoded values for models, temperatures, chunk sizes, or paths.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Mapping


class ConfigError(ValueError):
    """
    Raised when configuration cannot be loaded or validated.

    Subclasses ValueError so callers that catch the broad type still work.
    Error messages must include enough context to debug the issue: file
    path (when TOML), field name, and the layer where the problem was
    detected (CLI / env / TOML / defaults).
    """


class EmbeddingConfig(BaseModel):
    """
    Embedding provider configuration.

    Provider-specific settings go in `extra`.
    """

    provider: str = Field(
        description="Embedding provider name (e.g., 'openai', 'local', 'mock')",
    )
    model: str | None = Field(
        default=None, description="Model name (provider-specific)",
    )
    dimensions: int | None = Field(
        default=None, description="Embedding dimensions (provider-specific)",
    )
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch size for embedding generation",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )


class VectorStoreConfig(BaseModel):
    """Vector store provider configuration."""

    provider: str = Field(
        description="Vector store provider name (e.g., 'chroma', 'qdrant', 'in_memory')",
    )
    persist_directory: Path | None = Field(
        default=None, description="Directory for persistent storage",
    )
    collection_name: str = Field(default="bodhi-rag", description="Collection name")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )


class LLMConfig(BaseModel):
    """LLM provider configuration for generation."""

    provider: str = Field(
        description="LLM provider name (e.g., 'openai', 'anthropic', 'ollama', 'mock')",
    )
    model: str | None = Field(
        default=None, description="Model name (provider-specific)",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens to generate",
    )
    context_window: int | None = Field(
        default=None, ge=1, description="Context window size (provider-specific)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )


class ChunkerConfig(BaseModel):
    """Text chunking configuration."""

    provider: str = Field(
        description="Chunking strategy (e.g., 'fixed_size', 'recursive', 'semantic')",
    )
    chunk_size: int | None = Field(
        default=None, ge=1, description="Target chunk size (model/provider-dependent)",
    )
    overlap: int | None = Field(
        default=None, ge=0, description="Overlap between chunks",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )


class DocumentParserConfig(BaseModel):
    """Document parsing configuration."""

    provider: str = Field(description="Parser provider (e.g., 'pypdf', 'unstructured')")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )


class ConversationConfig(BaseModel):
    """Conversation memory configuration."""

    provider: str = Field(
        description="Memory provider (e.g., 'volatile', 'persistent')",
    )
    max_history: int | None = Field(
        default=None, ge=1, description="Maximum turns to retain per conversation",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )


class RerankerConfig(BaseModel):
    """
    Reranker configuration.

    Wave 1 establishes the schema and the no-hardcoded-defaults policy:
    when `provider == "cross_encoder"`, `model` is required and the
    adapter is responsible for failing fast at construction time if it
    is missing. The actual `CrossEncoderReranker` adapter lands in
    Wave 3a; the contract enforced here is that *no* reranker adapter
    in this project is ever constructed without an explicit model name
    chosen by the user via `bodhi.toml`, env, or CLI.

    `overfetch_factor` and `batch_size` are operational tunables with
    acceptable defaults; they are not model selections, so a default
    is documented and expected to be overridable.
    """

    provider: Literal["noop", "cross_encoder"] = Field(
        default="noop",
        description="Reranker provider: 'noop' (default) or 'cross_encoder' (opt-in).",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model identifier (e.g. a sentence-transformers cross-encoder name). "
            "REQUIRED when provider='cross_encoder'; the adapter raises ConfigError "
            "if it is missing. NEVER hardcoded in adapter code."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Optional override of the post-rerank top_k.",
    )
    overfetch_factor: int = Field(
        default=4,
        ge=1,
        le=64,
        description=(
            "Multiplier on top_k for the pre-rerank vector-store search. "
            "Skipped when reranker is NoOpReranker."
        ),
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for the cross-encoder scoring call.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config",
    )

    @model_validator(mode="after")
    def _validate_cross_encoder_model(self) -> RerankerConfig:
        if self.provider == "cross_encoder" and (
            self.model is None or self.model.strip() == ""
        ):
            msg = (
                "RerankerConfig.model is required when provider is "
                '"cross_encoder". Set BODHI_RERANKER_MODEL or define '
                "[reranker] model in bodhi.toml."
            )
            raise ConfigError(msg)
        return self


class TelemetryConfig(BaseModel):
    """OpenTelemetry configuration."""

    enabled: bool = Field(default=True, description="Enable telemetry spans")
    service_name: str = Field(default="bodhi-rag", description="Service name for traces")
    exporter: str = Field(
        default="console", description="Exporter type ('console', 'otlp', 'none')",
    )
    otlp_endpoint: str | None = Field(
        default=None, description="OTLP collector endpoint",
    )


class ApiConfig(BaseModel):
    """API transport configuration."""

    host: str = Field(default="127.0.0.1", description="API bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="API bind port")
    source_root: Path | None = Field(
        default=None,
        description="Optional local-source root enforced by the API indexing endpoint",
    )


def _config_section_map() -> dict[str, tuple[type[BaseModel], dict[str, str]]]:
    return {
        "parser": (DocumentParserConfig, {"BODHI_PARSER_PROVIDER": "provider"}),
        "chunker": (
            ChunkerConfig,
            {
                "BODHI_CHUNKER_PROVIDER": "provider",
                "BODHI_CHUNKER_CHUNK_SIZE": "chunk_size",
                "BODHI_CHUNKER_OVERLAP": "overlap",
            },
        ),
        "embedding": (
            EmbeddingConfig,
            {
                "BODHI_EMBEDDING_PROVIDER": "provider",
                "BODHI_EMBEDDING_MODEL": "model",
            },
        ),
        "vector_store": (
            VectorStoreConfig,
            {
                "BODHI_VECTOR_STORE_PROVIDER": "provider",
                "BODHI_INDEX_PERSIST_DIRECTORY": "persist_directory",
            },
        ),
        "llm": (
            LLMConfig,
            {
                "BODHI_LLM_PROVIDER": "provider",
                "BODHI_LLM_MODEL": "model",
            },
        ),
        "conversation": (
            ConversationConfig,
            {"BODHI_CONVERSATION_PROVIDER": "provider"},
        ),
        "reranker": (
            RerankerConfig,
            {
                "BODHI_RERANKER_PROVIDER": "provider",
                "BODHI_RERANKER_MODEL": "model",
            },
        ),
        "telemetry": (TelemetryConfig, {}),
        "api": (
            ApiConfig,
            {
                "BODHI_API_HOST": "host",
                "BODHI_API_PORT": "port",
                "BODHI_API_SOURCE_ROOT": "source_root",
            },
        ),
    }


def _resolve_section_kwargs(
    data: Mapping[str, Any], env: Mapping[str, str] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for section, (model_cls, env_map) in _config_section_map().items():
        section_data: dict[str, Any] = {}
        raw_section = data.get(section)
        if isinstance(raw_section, dict):
            section_data.update(raw_section)
        if env is not None:
            for env_key, field_name in env_map.items():
                if env_key in env:
                    section_data[field_name] = env[env_key]
        if not section_data:
            continue
        try:
            kwargs[section] = model_cls.model_validate(section_data)
        except Exception as exc:
            msg = f"Invalid [{section}] configuration: {exc}"
            raise ConfigError(msg) from exc
    return kwargs


class BhodiConfig(BaseModel):
    """
    Root configuration for bodhi-rag platform.

    All values are optional with sensible defaults to allow partial overrides.
    """

    parser: DocumentParserConfig = Field(
        default_factory=lambda: DocumentParserConfig(provider="pypdf"),
    )
    chunker: ChunkerConfig = Field(
        default_factory=lambda: ChunkerConfig(provider="recursive"),
    )
    embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(provider="openai"),
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=lambda: VectorStoreConfig(provider="chroma"),
    )
    llm: LLMConfig = Field(
        default_factory=lambda: LLMConfig(provider="openai"),
    )
    conversation: ConversationConfig = Field(
        default_factory=lambda: ConversationConfig(provider="volatile"),
    )
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig,
    )
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields in config files without failing
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BhodiConfig:
        """
        Build a `BhodiConfig` from environment variables only.

        The TOML layer is skipped. The `env` mapping defaults to `os.environ`
        but tests can pass a synthetic mapping. Unknown env vars are ignored;
        only the documented `BODHI_*` (and the API-layer `BODHI_API_*`)
        variables are consumed.
        """
        env_data = env if env is not None else os.environ
        return cls(**_resolve_section_kwargs(cls().model_dump(mode="python"), env_data))

    @classmethod
    def from_toml(
        cls, path: str | Path, *, env: Mapping[str, str] | None = None,
    ) -> BhodiConfig:
        """
        Build a `BhodiConfig` from a TOML file.

        The TOML file is parsed with `tomllib`; nested sections are validated
        against the corresponding Pydantic sub-config (`[embedding]` ->
        `EmbeddingConfig`, etc.). Unknown sections / keys are ignored, matching
        the `extra="ignore"` policy on the root model. Missing required
        fields raise `ConfigError` (e.g. `[reranker] provider = "cross_encoder"`
        with no `model`).

        The optional `env` overlay is applied AFTER the TOML layer: any
        `BODHI_*` env var overrides the corresponding TOML field at the
        per-sub-config dict level. This keeps the precedence
        TOML < env < CLI when `load_bodhi_config` is used, and matches the
        documented 12-factor behaviour.
        """
        path = Path(path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            msg = f"TOML config file not found: {path}"
            raise ConfigError(msg) from exc
        except OSError as exc:
            msg = f"Could not read TOML config file {path}: {exc}"
            raise ConfigError(msg) from exc

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Malformed TOML in {path}: {exc}"
            raise ConfigError(msg) from exc

        merged_data = cls().model_dump(mode="python")
        merged_data.update({key: value for key, value in data.items() if isinstance(value, dict)})
        try:
            return cls(**_resolve_section_kwargs(merged_data, env))
        except ConfigError as exc:
            msg = f"Invalid config data loaded from {path}: {exc}"
            raise ConfigError(msg) from exc
