"""
Tests for the TOML config loader (`application/config_loader.py`).

These tests are hermetic: each test gets its own tmp directory and a
fresh empty env mapping so it cannot leak state into other tests.
The tests cover the four-layer precedence (CLI > env > TOML > defaults)
plus the malformed-TOML, missing-file, and missing-required-field
error paths.
"""

from __future__ import annotations

import os
import textwrap
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from bodhi_rag.application.config import ApiConfig, BhodiConfig, ConfigError, RerankerConfig
from bodhi_rag.application.config_loader import load_bodhi_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def empty_env() -> Iterator[dict[str, str]]:
    """
    Yield a fresh empty env mapping, restoring os.environ on teardown.

    The loader reads `BODHI_CONFIG_PATH` from the env; isolating tests
    from the real process environment keeps them deterministic.
    """
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith("BODHI_"):
            del os.environ[key]
    yield {}
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def tmp_toml(tmp_path: Path) -> Path:
    """Return a path inside tmp_path; the file does not exist yet."""
    return tmp_path / "bodhi.toml"


class TestLoadFromTomlOnly:
    def test_load_from_toml_only(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """A TOML file populates the config when env is empty and no CLI overrides."""
        tmp_toml.write_text(
            textwrap.dedent(
                """
                [embedding]
                provider = "openai"
                model = "text-embedding-3-small"

                [vector_store]
                provider = "in_memory"

                [chunker]
                provider = "fixed_size"
                chunk_size = 256
                overlap = 32
                """,
            ).strip(),
            encoding="utf-8",
        )
        config = load_bodhi_config(env=empty_env, config_path=tmp_toml)
        assert isinstance(config, BhodiConfig)
        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.vector_store.provider == "in_memory"
        assert config.chunker.provider == "fixed_size"
        assert config.chunker.chunk_size == 256
        assert config.chunker.overlap == 32


class TestEnvOverridesToml:
    def test_env_overrides_toml(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """`BODHI_EMBEDDING_MODEL` in env beats the TOML value."""
        _ = empty_env
        tmp_toml.write_text(
            textwrap.dedent(
                """
                [embedding]
                provider = "openai"
                model = "x"
                """,
            ).strip(),
            encoding="utf-8",
        )
        env = {"BODHI_EMBEDDING_MODEL": "y"}
        config = load_bodhi_config(env=env, config_path=tmp_toml)
        assert config.embedding.model == "y"
        assert config.embedding.provider == "openai"

    def test_env_overrides_api_transport_fields(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """API transport config should resolve from env through the central loader."""
        _ = empty_env
        tmp_toml.write_text(
            textwrap.dedent(
                """
                [api]
                host = "0.0.0.0"
                port = 7000
                """,
            ).strip(),
            encoding="utf-8",
        )
        env = {
            "BODHI_API_HOST": "127.0.0.1",
            "BODHI_API_PORT": "9000",
        }
        config = load_bodhi_config(env=env, config_path=tmp_toml)
        assert config.api == ApiConfig(host="127.0.0.1", port=9000)


class TestCliOverridesEnv:
    def test_cli_overrides_env(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """CLI overrides beat env."""
        _ = empty_env
        env = {"BODHI_EMBEDDING_MODEL": "from-env"}
        cli = {"embedding": {"model": "from-cli"}}
        config = load_bodhi_config(
            env=env,
            config_path=tmp_toml,
            cli_overrides=cli,
        )
        assert config.embedding.model == "from-cli"


class TestPrecedenceCharacterization:
    def test_cli_env_toml_layers_merge_without_losing_lower_priority_fields(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """CLI beats env beats TOML while preserving unspecified lower-layer fields."""
        _ = empty_env
        tmp_toml.write_text(
            textwrap.dedent(
                """
                [embedding]
                provider = "openai"
                model = "from-toml"
                batch_size = 25

                [embedding.extra]
                api_key = "from-toml-extra"
                endpoint = "https://example.invalid"
                """,
            ).strip(),
            encoding="utf-8",
        )

        config = load_bodhi_config(
            env={"BODHI_EMBEDDING_MODEL": "from-env"},
            config_path=tmp_toml,
            cli_overrides={"embedding": {"batch_size": 50}},
        )

        assert config.embedding.provider == "openai"
        assert config.embedding.model == "from-env"
        assert config.embedding.batch_size == 50
        assert config.embedding.extra == {
            "api_key": "from-toml-extra",
            "endpoint": "https://example.invalid",
        }


class TestMissingTomlSilent:
    def test_missing_toml_silent(
        self,
        tmp_path: Path,
        empty_env: dict[str, str],
    ) -> None:
        """A missing `config_path` is silently skipped (falls through to env/defaults)."""
        missing = tmp_path / "does_not_exist.toml"
        config = load_bodhi_config(env=empty_env, config_path=missing)
        assert isinstance(config, BhodiConfig)
        assert config.embedding.provider == "openai"
        assert config.vector_store.provider == "chroma"

    def test_missing_default_bodhi_toml_silent(
        self,
        tmp_path: Path,
        empty_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no path is provided and ./bodhi.toml is absent, the loader skips silently."""
        monkeypatch.chdir(tmp_path)
        config = load_bodhi_config(env=empty_env)
        assert isinstance(config, BhodiConfig)
        assert config.reranker.provider == "noop"


class TestMalformedTomlRaisesConfigError:
    def test_malformed_toml_raises_config_error(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """A syntactically broken TOML file raises `ConfigError` mentioning the path."""
        tmp_toml.write_text("this is = not valid TOML [[[", encoding="utf-8")
        with pytest.raises(ConfigError) as excinfo:
            load_bodhi_config(env=empty_env, config_path=tmp_toml)
        assert str(tmp_toml) in str(excinfo.value)


class TestMissingRequiredFieldRaisesConfigError:
    def test_cross_encoder_without_model_raises(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """
        [reranker] provider = 'cross_encoder' without model raises ConfigError.

        The validator on `RerankerConfig` runs in Wave 1 to establish the
        no-hardcoded-defaults contract, even though the
        `CrossEncoderReranker` adapter itself lands in Wave 3a.
        """
        tmp_toml.write_text(
            textwrap.dedent(
                """
                [reranker]
                provider = "cross_encoder"
                """,
            ).strip(),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            load_bodhi_config(env=empty_env, config_path=tmp_toml)
        msg = str(excinfo.value)
        assert "RerankerConfig.model" in msg
        assert "cross_encoder" in msg

    def test_noop_reranker_does_not_require_model(
        self,
        tmp_toml: Path,
        empty_env: dict[str, str],
    ) -> None:
        """[reranker] provider = 'noop' (default) does NOT require a model."""
        tmp_toml.write_text(
            textwrap.dedent(
                """
                [reranker]
                provider = "noop"
                """,
            ).strip(),
            encoding="utf-8",
        )
        config = load_bodhi_config(env=empty_env, config_path=tmp_toml)
        assert config.reranker.provider == "noop"
        assert config.reranker.model is None

    def test_reranker_model_validator_direct(self) -> None:
        """
        Direct construction with provider='cross_encoder' and model=None raises ConfigError.

        Pydantic v2 wraps the inner `ConfigError` in a `ValidationError`; both
        are subclasses of `ValueError`. The test accepts either form so the
        validator contract is robust to future Pydantic changes.
        """
        with pytest.raises((ConfigError, ValidationError)) as excinfo:
            RerankerConfig(provider="cross_encoder", model=None)
        assert "RerankerConfig.model" in str(excinfo.value)

    def test_reranker_model_validator_blank_string_raises(self) -> None:
        """An empty / whitespace model is also rejected."""
        with pytest.raises((ConfigError, ValidationError)):
            RerankerConfig(provider="cross_encoder", model="   ")


class TestConfigPathResolution:
    def test_explicit_path_above_env(
        self,
        tmp_toml: Path,
        tmp_path: Path,
        empty_env: dict[str, str],
    ) -> None:
        """The explicit `config_path` wins over `BODHI_CONFIG_PATH` in env."""
        _ = empty_env
        explicit = tmp_toml
        explicit.write_text(
            textwrap.dedent(
                """
                [embedding]
                provider = "openai"
                model = "from-explicit"
                """,
            ).strip(),
            encoding="utf-8",
        )
        other = tmp_path / "other.toml"
        other.write_text(
            textwrap.dedent(
                """
                [embedding]
                provider = "openai"
                model = "from-env-path"
                """,
            ).strip(),
            encoding="utf-8",
        )
        env = {"BODHI_CONFIG_PATH": str(other)}
        config = load_bodhi_config(env=env, config_path=explicit)
        assert config.embedding.model == "from-explicit"

    def test_bodhi_config_path_env_when_no_explicit(
        self,
        tmp_path: Path,
        empty_env: dict[str, str],
    ) -> None:
        """When no explicit `config_path` is given, `BODHI_CONFIG_PATH` is honored."""
        _ = empty_env
        target = tmp_path / "config.toml"
        target.write_text(
            textwrap.dedent(
                """
                [embedding]
                provider = "openai"
                model = "from-env"
                """,
            ).strip(),
            encoding="utf-8",
        )
        config = load_bodhi_config(env={"BODHI_CONFIG_PATH": str(target)})
        assert config.embedding.model == "from-env"


class TestDefaultsApplied:
    def test_defaults_applied_when_no_toml_no_env(
        self,
        empty_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When no TOML and no env, the built-in defaults populate the config."""
        monkeypatch.chdir(tmp_path)
        config = load_bodhi_config(env=empty_env)
        assert config.parser.provider == "pypdf"
        assert config.chunker.provider == "recursive"
        assert config.embedding.provider == "openai"
        assert config.vector_store.provider == "chroma"
        assert config.llm.provider == "openai"
        assert config.conversation.provider == "volatile"
        assert config.reranker.provider == "noop"
        assert config.telemetry.enabled is True
        assert config.telemetry.exporter == "console"
        assert config.api.host == "127.0.0.1"
        assert config.api.port == 8000
