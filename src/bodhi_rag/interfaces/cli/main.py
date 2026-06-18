"""Main CLI entry point for bodhi-rag."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import httpx

from bodhi_rag.application.config_loader import load_bodhi_config

if TYPE_CHECKING:
    from pathlib import Path

    from bodhi_rag.application.config import BhodiConfig


def _load_config_or_exit(config_path: str | Path | None) -> BhodiConfig:
    """
    Load the config via `load_bodhi_config` and exit on `ConfigError`.

    Returns the loaded `BhodiConfig`. On error, exits with code 2.
    """
    try:
        return load_bodhi_config(config_path=config_path)
    except Exception:  # noqa: BLE001 - top-level CLI error path
        sys.exit(2)


def _health_command(config: BhodiConfig) -> int:
    """
    Probe the live bodhi-rag-api /health endpoint and propagate its state.

    Exit codes:
        0 - the API is healthy (HTTP 200, status=healthy)
        2 - the API is degraded (HTTP 503, status=degraded, or
            HTTP 200 with status != "healthy")
        1 - the API is unreachable, refused the connection, or
            returned an unexpected status

    The CLI is a client of the API, not a parallel process that
    re-instantiates adapters. It uses the resolved API config to
    locate the API process.
    """
    url = f"http://{config.api.host}:{config.api.port}/health"

    try:
        resp = httpx.get(url, timeout=2.0)
    except httpx.RequestError:
        return 1

    try:
        data = resp.json()
    except ValueError:
        return 1

    status_value = data.get("status", "unknown")
    data.get("version", "unknown")
    data.get("services", {})

    if resp.status_code == 200 and status_value == "healthy":
        return 0

    if status_value == "degraded" or resp.status_code == 503:
        return 2

    return 1


def main() -> int:
    """Run the bodhi-rag CLI and dispatch the subcommand."""
    parser = argparse.ArgumentParser(
        description="bodhi-rag - Production-ready RAG framework",
    )
    # Top-level config flag. The flag is parsed before subcommand dispatch
    # so the loaded config is available to every subcommand via `load_bodhi_config`.
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to a bodhi.toml config file. Overrides BODHI_CONFIG_PATH "
            "and ./bodhi.toml. See docs/configuration.md for the schema."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index subcommand
    index_parser = subparsers.add_parser(
        "index",
        help="Index documents for querying",
    )
    index_parser.add_argument("source", type=str, help="Path to document or directory")

    # Query subcommand
    query_parser = subparsers.add_parser(
        "query",
        help="Query indexed documents",
    )
    query_parser.add_argument("question", type=str, help="Question to ask")

    # Health subcommand
    subparsers.add_parser("health", help="Probe the live API /health endpoint")

    args = parser.parse_args()

    # Load the TOML config (if any) up front; subcommands that need a
    # `BhodiConfig` will use it, others (like `health`) ignore it.
    config = _load_config_or_exit(args.config)

    if args.command == "index":
        from bodhi_rag.interfaces.cli.indexing import main as index_main

        index_main([args.source], config=config)
        return 0
    if args.command == "query":
        from bodhi_rag.interfaces.cli.query import main as query_main

        query_main([args.question], config=config)
        return 0
    if args.command == "health":
        return _health_command(config)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
