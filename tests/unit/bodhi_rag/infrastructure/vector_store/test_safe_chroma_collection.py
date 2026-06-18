"""Unit tests for the SafeChromaCollection defensive wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bodhi_rag.infrastructure.vector_store.safe_chroma_collection import (
    SafeChromaCollection,
)


@pytest.fixture
def raw_collection() -> MagicMock:
    return MagicMock(name="raw_chroma_collection")


@pytest.fixture
def safe(raw_collection: MagicMock) -> SafeChromaCollection:
    return SafeChromaCollection(raw_collection)


class TestAddEnforcesPrecomputedEmbeddings:
    def test_passes_through_when_embeddings_provided(
        self,
        safe: SafeChromaCollection,
        raw_collection: MagicMock,
    ) -> None:
        safe.add(
            ids=["1"],
            embeddings=[[0.1, 0.2]],
            documents=["hello"],
            metadatas=[{"k": "v"}],
        )
        raw_collection.add.assert_called_once_with(
            ids=["1"],
            embeddings=[[0.1, 0.2]],
            documents=["hello"],
            metadatas=[{"k": "v"}],
        )

    def test_rejects_none_embeddings(self, safe: SafeChromaCollection) -> None:
        with pytest.raises(ValueError, match="must be pre-computed"):
            safe.add(
                ids=["1"],
                embeddings=None,
                documents=["hello"],
                metadatas=[{}],
            )

    def test_rejects_missing_embeddings_kwarg(self, safe: SafeChromaCollection) -> None:
        with pytest.raises(TypeError):
            safe.add(  # type: ignore[call-arg]
                ids=["1"],
                documents=["hello"],
                metadatas=[{}],
            )


class TestQueryEnforcesPrecomputedEmbeddings:
    def test_passes_through_when_query_embeddings_provided(
        self,
        safe: SafeChromaCollection,
        raw_collection: MagicMock,
    ) -> None:
        safe.query(query_embeddings=[[0.1, 0.2]], n_results=5)
        raw_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=5,
        )

    def test_rejects_none_query_embeddings(self, safe: SafeChromaCollection) -> None:
        with pytest.raises(ValueError, match="must be pre-computed"):
            safe.query(query_embeddings=None, n_results=5)

    def test_rejects_missing_query_embeddings_kwarg(self, safe: SafeChromaCollection) -> None:
        with pytest.raises(TypeError):
            safe.query(n_results=5)  # type: ignore[call-arg]


class TestGetAndDeleteAreTransparent:
    def test_get_passes_through(
        self,
        safe: SafeChromaCollection,
        raw_collection: MagicMock,
    ) -> None:
        safe.get(where={"k": "v"})
        raw_collection.get.assert_called_once_with(where={"k": "v"})

    def test_delete_passes_through(
        self,
        safe: SafeChromaCollection,
        raw_collection: MagicMock,
    ) -> None:
        safe.delete(ids=["1", "2"])
        raw_collection.delete.assert_called_once_with(ids=["1", "2"])


class TestOnlyAllowedMethodsAreExposed:
    @pytest.mark.parametrize(
        "name",
        [
            "update",
            "upsert",
            "peek",
            "modify",
            "_embed_search_string_queries",
            "random_method",
        ],
    )
    def test_unknown_method_raises_attribute_error(
        self,
        safe: SafeChromaCollection,
        name: str,
    ) -> None:
        with pytest.raises(AttributeError, match="does not expose"):
            getattr(safe, name)()

    def test_allowed_methods_constant_is_audited_set(self) -> None:
        """
        The allowlist must not change without security review.

        Each method in `_ALLOWED_METHODS` was individually audited for
        its interaction with the chromadb embedding deserialization
        sink at `chromadb/api/collection_configuration.py:73-91`. The
        sink is reachable only from `CollectionCommon._embed`, which
        is only called when the caller omits pre-computed embeddings.

        - `add`, `update`, `upsert`: reach the sink if `embeddings` is
          None or absent. `add` is in the allowlist with a runtime
          check; `update` and `upsert` are deliberately excluded to
          keep the surface minimal.
        - `query`: reaches the sink if `query_embeddings` is None or
          absent. In the allowlist with a runtime check.
        - `get`, `delete`: do not call `_embed`. In the allowlist
          without an embedding check (they cannot reach the sink).
        - `peek`, `modify`, `_embed_search_string_queries`, etc.: not
          used by bodhi-rag. Excluded to keep the perimeter small.

        Adding a new method requires: (1) this test must be updated
        with the new method, (2) the wrapper must validate any
        embedding kwargs the new method accepts, (3) a security
        review of the new method's interaction with the sink.
        """
        assert frozenset({"add", "query", "get", "delete"}) == type.__getattribute__(
            SafeChromaCollection,
            "_ALLOWED_METHODS",
        )
