"""
Configuration loader for bodhi-rag.

Implements the 12-factor precedence:
    CLI overrides > env vars > TOML file > built-in defaults

The single public entry point is `load_bodhi_config(...)`. It composes
the layers, never persists state, and never mutates the caller's
environment. All errors are reported as `ConfigError` (subclass of
`ValueError`) with enough context (file path / field / layer) to
debug from a single error message.

Resolution order for the TOML file path:
    1. explicit `config_path` kwarg
    2. `BODHI_CONFIG_PATH` env var
    3. `./bodhi.toml` (relative to CWD)
    4. skip file layer silently

`tomllib` is stdlib in Python 3.11+; no new runtime dep.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from bodhi_rag.application.config import (
    BhodiConfig,
    ChunkerConfig,
    ConfigError,
    ConversationConfig,
    DocumentParserConfig,
    EmbeddingConfig,
    LLMConfig,
    RerankerConfig,
    TelemetryConfig,
    VectorStoreConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["load_bodhi_config"]


ConfigSectionModel = type[BaseModel]


def _resolve_config_path(
    config_path: str | Path | None, env: Mapping[str, str] | None,
) -> Path | None:
    """
    Resolve the TOML config path per the documented precedence.

    Returns None when the file layer is to be skipped silently. A missing
    TOML file is ALWAYS silent (per spec: "skip file layer silently"),
    regardless of whether the path came from the explicit kwarg, the
    `BODHI_CONFIG_PATH` env var, or the `./bodhi.toml` default. A malformed
    TOML file, by contrast, is reported as `ConfigError` from
    `BhodiConfig.from_toml` with the file path and the syntax issue.
    """
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path).expanduser())
    env_path = (env or {}).get("BODHI_CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path("./bodhi.toml"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _apply_cli_overrides(
    base: BhodiConfig, cli_overrides: Mapping[str, Any] | None,
) -> BhodiConfig:
    """
    Apply CLI overrides on top of the base config.

    The CLI override mapping uses the same dotted-path semantics that
    the rest of the config surface uses: keys are top-level sections
    (`embedding`, `vector_store`, `reranker`, ...) and the value is a
    dict of field overrides. Example:

        cli_overrides = {"reranker": {"provider": "cross_encoder"}}

    For each overridden section, the loader merges the CLI fields on top
    of the existing sub-config (preserving any field not mentioned in
    the CLI dict) and re-validates the resulting sub-config. A CLI
    override that is structurally invalid raises `ConfigError` with the
    field name in the message.
    """
    if not cli_overrides:
        return base

    # Map of section name -> sub-config class, mirroring BhodiConfig.from_toml.
    section_classes: dict[str, ConfigSectionModel] = {}
    for field_name in BhodiConfig.model_fields:
        existing = getattr(base, field_name, None)
        if existing is not None:
            section_classes[field_name] = type(existing)
    # Make sure all known sub-configs are present even if base is partial.
    section_classes.setdefault("parser", DocumentParserConfig)
    section_classes.setdefault("chunker", ChunkerConfig)
    section_classes.setdefault("embedding", EmbeddingConfig)
    section_classes.setdefault("vector_store", VectorStoreConfig)
    section_classes.setdefault("llm", LLMConfig)
    section_classes.setdefault("conversation", ConversationConfig)
    section_classes.setdefault("reranker", RerankerConfig)
    section_classes.setdefault("telemetry", TelemetryConfig)

    new_kwargs: dict[str, Any] = {}
    for section, fields in cli_overrides.items():
        if not isinstance(fields, dict):
            msg = f"CLI override for [{section}] must be a dict, got {type(fields).__name__}"
            raise ConfigError(msg)
        existing = getattr(base, section, None)
        merged: dict[str, Any] = {}
        if existing is not None and hasattr(existing, "model_dump"):
            merged.update(existing.model_dump())
        else:
            # No existing sub-config — fall back to the section class.
            cls = section_classes.get(section)
            if cls is not None and hasattr(cls, "model_fields"):
                merged.update(dict.fromkeys(cls.model_fields))
        merged.update(fields)
        cls = section_classes.get(section)
        if cls is None:
            msg = f"Unknown config section in CLI overrides: {section!r}"
            raise ConfigError(msg)
        try:
            new_kwargs[section] = cls.model_validate(merged)
        except Exception as exc:
            msg = f"Invalid CLI override for [{section}]: {exc}"
            raise ConfigError(msg) from exc

    try:
        return base.model_copy(update=new_kwargs)
    except Exception as exc:
        msg = f"Invalid CLI overrides: {exc}"
        raise ConfigError(msg) from exc


def load_bodhi_config(
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> BhodiConfig:
    """
    Load a `BhodiConfig` from CLI + env + TOML + defaults.

    Precedence (highest priority first):
        1. `cli_overrides`  — explicit kwargs from the CLI parser
        2. env vars         — `BODHI_*` and `BODHI_API_*` (Mapping or os.environ)
        3. TOML file        — see `_resolve_config_path` for the path search
        4. built-in defaults — Pydantic `default_factory` lambdas

    `env` defaults to None, in which case the current process `os.environ`
    is used. Tests can pass a synthetic mapping for hermetic behaviour.

    `config_path` resolution:
        explicit kwarg -> `BODHI_CONFIG_PATH` env var -> `./bodhi.toml` -> skip

    A missing TOML file at the explicit kwarg is reported as `ConfigError`.
    A missing `./bodhi.toml` is silent (skipped). A malformed TOML is
    reported with the file path and the syntax issue. A missing required
    field (e.g. `[reranker] provider = "cross_encoder"` without `model`)
    is reported as `ConfigError` from the Pydantic validator on the
    matching sub-config.
    """
    resolved = _resolve_config_path(config_path, env)

    if resolved is not None:
        try:
            base = BhodiConfig.from_toml(resolved, env=env)
        except ConfigError:
            raise
        except Exception as exc:
            msg = f"Could not load TOML config {resolved}: {exc}"
            raise ConfigError(msg) from exc
    else:
        # No TOML layer — start from env-only (defaults are baked into
        # the default_factory lambdas).
        base = BhodiConfig.from_env(env=env)

    return _apply_cli_overrides(base, cli_overrides)


def parse_toml_text(text: str) -> dict[str, Any]:
    """
    Parse a TOML string with `tomllib` and return a dict.

    Exposed for tests and for callers that need the raw parsed data
    (e.g. to print effective config in a `bodhi-rag config` subcommand,
    which is out of scope for Wave 1).

    Errors are rewrapped as `ConfigError` for layer-consistent reporting.
    """
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Malformed TOML: {exc}"
        raise ConfigError(msg) from exc
