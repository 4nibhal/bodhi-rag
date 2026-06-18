from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bodhi_rag.application.config_loader import load_bodhi_config

if TYPE_CHECKING:
    from bodhi_rag.application.config import BhodiConfig

DEFAULT_PERSIST_DIRECTORY_NAME = "chroma_db"


@dataclass(frozen=True, slots=True)
class IndexingSettings:
    persist_directory: Path
    retriever_k: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @classmethod
    def from_bhodi_config(
        cls,
        config: BhodiConfig,
        *,
        cwd: Path | None = None,
    ) -> IndexingSettings:
        """Build compatibility indexing settings from the central config."""
        base_directory = cwd or Path.cwd()
        persist_directory = (
            config.vector_store.persist_directory
            or base_directory / DEFAULT_PERSIST_DIRECTORY_NAME
        )
        return cls(persist_directory=Path(persist_directory))

    @classmethod
    def from_environment(cls, cwd: Path | None = None) -> IndexingSettings:
        """Compatibility shim delegating to the central config loader."""
        return cls.from_bhodi_config(load_bodhi_config(env=os.environ), cwd=cwd)
