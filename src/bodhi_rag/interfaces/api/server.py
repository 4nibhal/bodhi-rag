from __future__ import annotations

import argparse

import uvicorn

from bodhi_rag.application.config import ConfigError
from bodhi_rag.application.config_loader import load_bodhi_config
from bodhi_rag.interfaces.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bodhi RAG API server (FastAPI + Uvicorn)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: 8000)",
    )
    args = parser.parse_args()

    cli_overrides = {
        "api": {
            key: value
            for key, value in (("host", args.host), ("port", args.port))
            if value is not None
        },
    }

    try:
        config = load_bodhi_config(cli_overrides=cli_overrides)
    except ConfigError as exc:
        msg = f"Config error: {exc}"
        raise SystemExit(msg) from exc

    uvicorn.run(
        create_app(config),
        host=config.api.host,
        port=config.api.port,
    )
