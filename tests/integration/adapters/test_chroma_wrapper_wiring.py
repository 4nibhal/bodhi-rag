"""
Integration test for ChromaVectorStoreAdapter wrapper wiring.

Verify `ChromaVectorStoreAdapter` actually wraps the chromadb
collection in `SafeChromaCollection`.

The unit tests in `tests/unit/.../test_safe_chroma_collection.py` verify
the wrapper's behavior in isolation. This integration test verifies
that the adapter (`chroma.py`) is wired through the wrapper, not
bypassing it by holding a raw `chromadb.Collection` directly.

If a future refactor changes `_create_client_and_collection` to return
the raw collection (forgetting the wrapper), the security perimeter
is silently broken. This test catches that regression at PR time.
"""

from __future__ import annotations

import pytest

from bodhi_rag.application.config import VectorStoreConfig
from bodhi_rag.infrastructure.vector_store.chroma import ChromaVectorStoreAdapter
from bodhi_rag.infrastructure.vector_store.safe_chroma_collection import (
    SafeChromaCollection,
)


@pytest.fixture
def adapter(tmp_path: pytest.TempPathFactory | object) -> ChromaVectorStoreAdapter:
    """Build an adapter with a tmp persist directory."""
    config = VectorStoreConfig(
        provider="chroma",
        persist_directory=str(tmp_path / "chroma"),
    )
    return ChromaVectorStoreAdapter(config)


@pytest.mark.asyncio
async def test_adapter_collection_is_wrapped(adapter: ChromaVectorStoreAdapter) -> None:
    """After lazy init, the adapter collection must be a safe wrapper."""
    ensure_client = object.__getattribute__(adapter, "_ensure_client")
    await ensure_client()
    collection = object.__getattribute__(adapter, "_collection")
    assert collection is not None
    assert isinstance(collection, SafeChromaCollection), (
        f"adapter._collection is {type(collection).__name__}, "
        "expected SafeChromaCollection. The security perimeter is not "
        "wired correctly in chroma.py."
    )


@pytest.mark.asyncio
async def test_adapter_rejects_none_embeddings_at_wired_layer(
    adapter: ChromaVectorStoreAdapter,
) -> None:
    """
    Calling adapter.add with embeddings=None must raise the wrapper's ValueError.

    The adapter's `add` method signature has `embeddings: list[list[float]]`,
    so a strict type checker would reject `None`. But the wrapper also
    rejects `None` at runtime as defense in depth. This test asserts the
    runtime check is in the call path.
    """
    ensure_client = object.__getattribute__(adapter, "_ensure_client")
    await ensure_client()

    with pytest.raises(ValueError, match="must be pre-computed"):
        # type: ignore[arg-type]  -- intentionally passing None to test the runtime check
        await adapter.add(chunks=[], embeddings=None)
