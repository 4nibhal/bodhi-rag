"""
Document parser port definition.

Defines the contract for document parsing adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from bodhi_rag.domain.entities import Document


class DocumentParserPort(Protocol):
    """
    Protocol for document parsing.

    Adapters implementing this port handle reading documents
    from various sources and extracting their content.
    """

    async def parse(
        self,
        source: Path | bytes | BinaryIO,
    ) -> Document:
        """
        Parse a document from a source.

        Args:
            source: Path to file, raw bytes, or file-like object.

        Returns:
            Parsed document with text and metadata.

        Raises:
            InvalidDocumentError: If document cannot be parsed.

        """
        ...

    async def extract_text(
        self,
        source: Path | bytes | BinaryIO,
    ) -> str:
        """
        Extract raw text from a document.

        Args:
            source: Path to file, raw bytes, or file-like object.

        Returns:
            Extracted text content.

        Raises:
            InvalidDocumentError: If text extraction fails.

        """
        ...

    async def extract_metadata(
        self,
        source: Path | bytes | BinaryIO,
    ) -> dict:
        """
        Extract metadata from a document.

        Args:
            source: Path to file, raw bytes, or file-like object.

        Returns:
            Dictionary of metadata (e.g., page count, author, title).

        Raises:
            InvalidDocumentError: If metadata extraction fails.

        """
        ...
