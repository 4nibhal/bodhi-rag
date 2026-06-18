"""CLI command for indexing documents."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING, TextIO

from bodhi_rag.application.config_loader import load_bodhi_config
from bodhi_rag.application.models import IndexDocumentRequest
from bodhi_rag.infrastructure.container import Container

if TYPE_CHECKING:
    from bodhi_rag.application.config import BhodiConfig


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for index command."""
    parser = argparse.ArgumentParser(
        description="Index documents for later querying.",
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to document or directory to index.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Target chunk size (default from config).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap between chunks (default from config).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Additional metadata as JSON string.",
    )
    return parser


async def run_index(
    source: str,
    config: BhodiConfig | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: dict | None = None,
) -> str:
    """Run index operation and return result string."""
    resolved_config = config or load_bodhi_config()
    container = Container(resolved_config)
    app = container.build()

    request = IndexDocumentRequest(
        source=source,
        metadata=metadata or {},
        chunk_size=chunk_size,
        overlap=overlap,
    )

    try:
        response = await app.index_document(request)
        return (  # noqa: TRY300  # return inside try: index_document can raise; we want the except clause to catch it
            f"Indexed {response.chunk_count} chunks from {source}. "
            f"Document ID: {response.document_id}"
        )
    except Exception as exc:  # noqa: BLE001  # CLI: any failure formats as a user-friendly error string
        return f"Error indexing {source}: {exc}"


def main(
    argv: list[str] | None = None,
    *,
    config: BhodiConfig | None = None,
    stdout: TextIO | None = None,
) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    output = stdout or sys.stdout

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        import json

        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            sys.exit(1)

    result = asyncio.run(
        run_index(
            source=args.source,
            config=config,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            metadata=metadata,
        ),
    )
    print(result, file=output)


if __name__ == "__main__":
    main()
