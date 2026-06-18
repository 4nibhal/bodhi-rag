from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from bodhi_rag.indexing.ports import DocumentLoader
    from bodhi_rag.indexing.settings import IndexingSettings

TEXT_GLOB = "**/*.{txt,md,py,js,ts,cpp,java,go,rs}"
PDF_GLOB = "**/*.pdf"


def _directory_loader_class() -> Any:
    from langchain_community.document_loaders import (  # type: ignore[import-not-found]
        DirectoryLoader,
    )

    return DirectoryLoader


def _pdf_loader_class() -> Any:
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import-not-found]

    return PyPDFLoader


def _text_loader_class() -> Any:
    from langchain_community.document_loaders import TextLoader  # type: ignore[import-not-found]

    return TextLoader


def _splitter_class() -> Any:
    from langchain.text_splitter import (  # type: ignore[import-not-found]
        RecursiveCharacterTextSplitter,
    )

    return RecursiveCharacterTextSplitter


def _vectorstore_class() -> Any:
    from langchain_chroma import Chroma  # type: ignore[import-not-found]

    return Chroma


def build_directory_loaders(directory_path: str) -> list[DocumentLoader]:
    directory_loader_class = _directory_loader_class()
    pdf_loader_class = _pdf_loader_class()
    text_loader_class = _text_loader_class()
    return [
        directory_loader_class(
            directory_path,
            glob=PDF_GLOB,
            loader_cls=pdf_loader_class,
        ),
        directory_loader_class(
            directory_path,
            glob=TEXT_GLOB,
            loader_cls=text_loader_class,
        ),
    ]


def load_documents_from_directory(directory_path: str) -> list[Any]:
    documents: list[Any] = []
    for loader in build_directory_loaders(directory_path):
        documents.extend(loader.load())
    return documents


def build_file_loader(file_path: str) -> DocumentLoader:
    if file_path.lower().endswith(".pdf"):
        return cast("DocumentLoader", _pdf_loader_class()(file_path))
    return cast("DocumentLoader", _text_loader_class()(file_path))


def load_documents_from_file(file_path: str) -> list[Any]:
    return build_file_loader(file_path).load()


def build_splitter(settings: IndexingSettings) -> Any:
    return _splitter_class()(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def build_vectorstore(settings: IndexingSettings, embeddings: Any) -> Any:
    return _vectorstore_class()(
        embedding_function=embeddings,
        persist_directory=str(settings.persist_directory),
    )


def build_retriever(vectorstore: Any, settings: IndexingSettings) -> Any:
    return vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})
