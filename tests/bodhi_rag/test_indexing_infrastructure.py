from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from bodhi_rag.indexing import infrastructure


class InfrastructureCharacterizationTest(TestCase):
    def test_directory_loading_uses_pdf_and_text_globs(self) -> None:
        calls: list[tuple[str, str, object]] = []
        pdf_loader_class = object()
        text_loader_class = object()
        documents_by_glob = {
            infrastructure.PDF_GLOB: [SimpleNamespace(page_content="pdf")],
            infrastructure.TEXT_GLOB: [SimpleNamespace(page_content="text")],
        }

        def fake_directory_loader(
            directory_path: str,
            *,
            glob: str,
            loader_cls: object,
        ) -> SimpleNamespace:
            calls.append((directory_path, glob, loader_cls))
            return SimpleNamespace(load=lambda: list(documents_by_glob[glob]))

        with patch.object(
            infrastructure,
            "_directory_loader_class",
            return_value=fake_directory_loader,
        ), patch.object(
            infrastructure,
            "_pdf_loader_class",
            return_value=pdf_loader_class,
        ), patch.object(
            infrastructure,
            "_text_loader_class",
            return_value=text_loader_class,
        ):
            documents = infrastructure.load_documents_from_directory("/docs")

        assert [doc.page_content for doc in documents] == ["pdf", "text"]
        assert calls == [
            ("/docs", infrastructure.PDF_GLOB, pdf_loader_class),
            ("/docs", infrastructure.TEXT_GLOB, text_loader_class),
        ]

    def test_file_loading_uses_pdf_loader_for_pdf_paths(self) -> None:
        pdf_path = "/workspace/source.PDF"
        text_path = "/workspace/source.txt"
        pdf_loader = SimpleNamespace(load=lambda: ["pdf-doc"])
        text_loader = SimpleNamespace(load=lambda: ["text-doc"])
        pdf_loader_factory = Mock(return_value=pdf_loader)
        text_loader_factory = Mock(return_value=text_loader)

        with patch.object(
            infrastructure,
            "_pdf_loader_class",
            return_value=pdf_loader_factory,
        ) as pdf_patch, patch.object(
            infrastructure,
            "_text_loader_class",
            return_value=text_loader_factory,
        ) as text_patch:
            pdf_documents = infrastructure.load_documents_from_file(pdf_path)
            text_documents = infrastructure.load_documents_from_file(text_path)

        assert pdf_documents == ["pdf-doc"]
        assert text_documents == ["text-doc"]
        pdf_patch.assert_called_once_with()
        text_patch.assert_called_once_with()
        pdf_loader_factory.assert_called_once_with(pdf_path)
        text_loader_factory.assert_called_once_with(text_path)
