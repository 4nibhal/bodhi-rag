"""CLI command for querying documents."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING, TextIO

from bodhi_rag.application.config_loader import load_bodhi_config
from bodhi_rag.application.models import QueryRequest
from bodhi_rag.infrastructure.container import Container

if TYPE_CHECKING:
    from bodhi_rag.application.config import BhodiConfig


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for query command."""
    parser = argparse.ArgumentParser(
        description="Query indexed documents.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask.",
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        default=None,
        help="Conversation ID for continuity.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7).",
    )
    return parser


async def run_query(
    question: str,
    config: BhodiConfig | None = None,
    conversation_id: str | None = None,
    top_k: int = 5,
    temperature: float = 0.7,
) -> str:
    """Run query operation and return result string."""
    resolved_config = config or load_bodhi_config()
    container = Container(resolved_config)
    app = container.build()

    request = QueryRequest(
        question=question,
        conversation_id=conversation_id,
        top_k=top_k,
        temperature=temperature,
    )

    try:
        response = await app.query(request)
        lines = [
            f"Answer: {response.answer_text}",
            "",
            "Citations:",
        ]
        for i, citation in enumerate(response.citations, 1):
            lines.append(f"  [{i}] {citation.text[:100]}... (source: {citation.source_document})")

        if response.conversation_id:
            lines.append(f"\nConversation ID: {response.conversation_id}")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001  # CLI: any failure formats as a user-friendly error string
        return f"Error querying: {exc}"


def main(
    argv: list[str] | None = None,
    *,
    config: BhodiConfig | None = None,
    stdout: TextIO | None = None,
) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    output = stdout or sys.stdout

    result = asyncio.run(
        run_query(
            question=args.question,
            config=config,
            conversation_id=args.conversation_id,
            top_k=args.top_k,
            temperature=args.temperature,
        ),
    )
    print(result, file=output)


if __name__ == "__main__":
    main()
