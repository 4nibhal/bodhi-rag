"""Unit tests for bodhi-rag CLI entry points."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bodhi_rag.application.config import ApiConfig, BhodiConfig
from bodhi_rag.interfaces.cli.indexing import main as index_main
from bodhi_rag.interfaces.cli.main import _health_command
from bodhi_rag.interfaces.cli.main import main as cli_main
from bodhi_rag.interfaces.cli.query import main as query_main


class TestHealthCommand:
    """Tests for the live-API health probe."""

    def test_returns_0_when_api_healthy(self) -> None:
        with patch("bodhi_rag.interfaces.cli.main.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "status": "healthy",
                    "version": "0.1.0",
                    "services": {
                        "embedding": True,
                        "vector_store": True,
                        "llm": True,
                    },
                },
            )
            assert _health_command(BhodiConfig(api=ApiConfig())) == 0

    def test_returns_2_when_api_degraded(self) -> None:
        with patch("bodhi_rag.interfaces.cli.main.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=503,
                json=lambda: {
                    "status": "degraded",
                    "version": "0.1.0",
                    "services": {
                        "embedding": True,
                        "vector_store": False,
                        "llm": True,
                    },
                },
            )
            assert _health_command(BhodiConfig(api=ApiConfig())) == 2

    def test_returns_1_when_api_unreachable(self) -> None:
        with patch("bodhi_rag.interfaces.cli.main.httpx.get") as mock_get:
            mock_get.side_effect = httpx.RequestError(
                "connection refused",
                request=MagicMock(),
            )
            assert _health_command(BhodiConfig(api=ApiConfig())) == 1


class TestCliMain:
    """Tests for the main CLI dispatcher."""

    def test_index_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """bodhi-rag index --help should show usage."""
        with pytest.raises(SystemExit) as exc_info, patch.object(
            sys,
            "argv",
            ["bodhi-rag", "index", "--help"],
        ):
            cli_main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "index" in captured.out

    def test_query_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """bodhi-rag query --help should show usage."""
        with pytest.raises(SystemExit) as exc_info, patch.object(
            sys,
            "argv",
            ["bodhi-rag", "query", "--help"],
        ):
            cli_main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "query" in captured.out

    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Running bodhi-rag without arguments should print help and return 1."""
        with patch.object(sys, "argv", ["bodhi-rag"]):
            assert cli_main() == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out


class TestCliIndexing:
    """Tests for the indexing CLI command."""

    @patch("bodhi_rag.interfaces.cli.indexing.run_index")
    def test_index_document(
        self,
        mock_run_index: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """bodhi-rag-index should delegate to run_index."""
        mock_run_index.return_value = "Indexed 5 chunks from test.pdf"
        index_main(["test.pdf"], config=BhodiConfig())
        captured = capsys.readouterr()
        assert "Indexed 5 chunks from test.pdf" in captured.out
        mock_run_index.assert_called_once()

    @patch("bodhi_rag.interfaces.cli.indexing.run_index")
    def test_index_with_options(self, mock_run_index: MagicMock) -> None:
        """bodhi-rag-index should pass chunk-size and overlap."""
        mock_run_index.return_value = "done"
        index_main(
            ["test.pdf", "--chunk-size", "500", "--overlap", "50"],
            config=BhodiConfig(),
        )
        mock_run_index.assert_called_once_with(
            source="test.pdf",
            config=BhodiConfig(),
            chunk_size=500,
            overlap=50,
            metadata=None,
        )


class TestCliQuery:
    """Tests for the query CLI command."""

    @patch("bodhi_rag.interfaces.cli.query.run_query")
    def test_query_question(
        self,
        mock_run_query: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """bodhi-rag query should delegate to run_query."""
        mock_run_query.return_value = "Answer: 42"
        query_main(["What is the answer?"], config=BhodiConfig())
        captured = capsys.readouterr()
        assert "Answer: 42" in captured.out
        mock_run_query.assert_called_once()

    @patch("bodhi_rag.interfaces.cli.query.run_query")
    def test_query_with_conversation_id(self, mock_run_query: MagicMock) -> None:
        """bodhi-rag query should pass conversation-id."""
        mock_run_query.return_value = "Answer: yes"
        query_main(["Is this true?", "--conversation-id", "abc123"], config=BhodiConfig())
        mock_run_query.assert_called_once_with(
            question="Is this true?",
            config=BhodiConfig(),
            conversation_id="abc123",
            top_k=5,
            temperature=0.7,
        )
