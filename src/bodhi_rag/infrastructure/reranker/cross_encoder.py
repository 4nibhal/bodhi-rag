"""
Cross-encoder reranker adapter.

Uses a sentence-transformers CrossEncoder model to score each
(query, chunk) pair and reorder the input by descending relevance.
The model name is read from `RerankerConfig.model`; this adapter
never hardcodes a default (Wave 1 policy).

Dependencies:
- This adapter requires the optional `cross-encoder` extra, which
  installs `sentence-transformers`. If the package is not installed,
  `__init__` raises `ConfigError` with an actionable message.
- The CrossEncoder model itself is loaded lazily on first `rerank()`
  call to honor the no-import-time-side-effects policy in root
  AGENTS.md.
"""

from __future__ import annotations

from typing import Protocol, cast

from bodhi_rag.application.config import ConfigError, RerankerConfig
from bodhi_rag.domain.entities import RetrievedDocument


class _CrossEncoderProtocol(Protocol):
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]: ...


_OPTIONAL_EXTRA_HINT = (
    "Install the cross-encoder optional extra: "
    "`uv tool install bodhi-rag[cross-encoder]` (or "
    "`uv add sentence-transformers` in development)."
)


class CrossEncoderReranker:
    """
    Reranker backed by a sentence-transformers CrossEncoder.

    Each `rerank()` call scores every (query, chunk.text) pair with
    the configured model, then returns the input chunks reordered by
    descending score (and trimmed to `top_k` if set). Scores on the
    returned chunks are the cross-encoder logits, not the original
    vector-store similarity.
    """

    def __init__(self, config: RerankerConfig) -> None:
        # The model-validator in RerankerConfig has already enforced
        # that `config.model` is set and non-empty when
        # `config.provider == "cross_encoder"`. We re-check here
        # defensively so the adapter fails loudly if a future
        # constructor bypasses the config path.
        if not config.model or not config.model.strip():
            msg = (
                "CrossEncoderReranker requires a non-empty model name. "
                "Set RerankerConfig.model or BODHI_RERANKER_MODEL."
            )
            raise ConfigError(msg)

        self._model_name = config.model.strip()
        self._batch_size = config.batch_size
        self._overfetch_factor = config.overfetch_factor
        self._encoder: _CrossEncoderProtocol | None = None

    @property
    def overfetch_factor(self) -> int:
        """Return the configured overfetch factor for the query pipeline."""
        return self._overfetch_factor

    def _ensure_encoder(self) -> _CrossEncoderProtocol:
        """Lazily import and instantiate the CrossEncoder on first use."""
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
            except ImportError as exc:
                msg = (
                    "CrossEncoderReranker requires the `sentence-transformers` "
                    "package. " + _OPTIONAL_EXTRA_HINT
                )
                raise ConfigError(msg) from exc
            self._encoder = cast("_CrossEncoderProtocol", CrossEncoder(self._model_name))
        return self._encoder

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Rerank chunks by cross-encoder relevance.

        Args:
            query: The user query to score chunks against.
            chunks: Candidate chunks (typically from the vector store).
            top_k: Optional cap on the number of returned chunks.

        Returns:
            Chunks in descending cross-encoder score order.

        """
        if not chunks:
            return []

        encoder = self._ensure_encoder()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = encoder.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        scored: list[tuple[float, RetrievedDocument]] = list(zip(scores, chunks, strict=True))
        scored.sort(key=lambda item: item[0], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return [
            RetrievedDocument(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(score),
                metadata=chunk.metadata,
            )
            for score, chunk in scored
        ]


__all__ = ["CrossEncoderReranker"]
