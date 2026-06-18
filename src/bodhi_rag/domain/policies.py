"""
Domain policies for bodhi-rag platform.

These encapsulate business rules extracted from DefaultRetrievalCollaborator
and other service logic. Policies are stateless and contain pure business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """
    Policy governing document retrieval behavior.

    Encapsulates rules for reranking, summarization thresholds,
    and truncation decisions.
    """

    reranker_max_length: int = 512
    document_summary_token_limit: int = 300
    reranker_score_threshold: float | None = None

    def should_summarize(self, token_count: int) -> bool:
        """Determine if a document should be summarized based on token count."""
        return token_count > self.document_summary_token_limit

    def get_reranker_max_length(self) -> int:
        """Get the max length parameter for reranker calls."""
        return self.reranker_max_length


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """
    Policy governing answer generation behavior.

    Encapsulates rules for role mapping, prompt truncation,
    and summary thresholds.
    """

    role_mapping: dict[str, str] | None = None
    prompt_summary_token_limit: int = 1200
    raw_summary_char_limit: int = 2500
    summarizer_max_length: int = 1500
    summarizer_min_length: int = 500

    def __post_init__(self) -> None:
        if self.role_mapping is None:
            object.__setattr__(
                self,
                "role_mapping",
                {
                    "question": "user",
                    "human": "user",
                    "user": "user",
                    "answer": "assistant",
                    "assistant": "assistant",
                    "ai": "assistant",
                },
            )

    def map_role(self, role: str) -> str:
        """Map a legacy role to a model-native role."""
        mapping = self.role_mapping or {}
        return mapping.get(role, "user")

    def should_summarize_prompt(self, token_count: int) -> bool:
        """Determine if the prompt should be summarized based on token count."""
        return token_count > self.prompt_summary_token_limit

    def should_summarize_text(self, char_count: int) -> bool:
        """Determine if text should be summarized based on character count."""
        return char_count < self.raw_summary_char_limit


@dataclass(frozen=True, slots=True)
class ContextAssemblyPolicy:
    """
    Policy governing how context is assembled from retrieved documents.

    Encapsulates token budget rules for context window management.
    """

    context_token_limit: int = 2000
    document_separator: str = "\n"

    def compute_available_tokens(
        self,
        total_used: int,
        max_tokens: int | None = None,
    ) -> int:
        """Compute remaining tokens available after accounting for used tokens."""
        effective_max = (
            max_tokens if max_tokens is not None else self.context_token_limit
        )
        return max(0, effective_max - total_used)

    def should_truncate(self, token_count: int) -> bool:
        """Determine if content should be truncated based on token count."""
        return token_count > self.context_token_limit


@dataclass(frozen=True, slots=True)
class IndexingTarget:
    """Filesystem facts supplied by an adapter for indexing policy checks."""

    path: str
    exists: bool
    is_dir: bool
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.exists and self.is_dir:
            msg = "Non-existent indexing targets cannot be directories"
            raise ValueError(msg)
        if self.is_dir and self.size_bytes is not None:
            msg = "Directory indexing targets must not include a file size"
            raise ValueError(msg)
        if self.size_bytes is not None and self.size_bytes < 0:
            msg = "Indexing target size must be non-negative"
            raise ValueError(msg)

    @property
    def is_absolute(self) -> bool:
        """Return whether the declared path is absolute."""
        return PurePath(self.path).is_absolute()

    @property
    def suffix(self) -> str:
        """Return the normalized file extension for the declared path."""
        return PurePath(self.path).suffix.lower()

    @property
    def is_file(self) -> bool:
        """Return whether the supplied facts describe a file."""
        return self.exists and not self.is_dir


@dataclass(frozen=True, slots=True)
class IndexingPolicy:
    """
    Policy governing document indexing behavior.

    Encapsulates rules for path validation and indexing constraints.
    """

    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".doc", ".docx")
    max_file_size_mb: int = 100
    require_absolute_path: bool = True

    def is_valid_target(self, target: IndexingTarget) -> bool:
        """
        Validate that supplied indexing facts meet policy requirements.

        For directories, only checks existence and absolute path requirement.
        For files, also checks the allowed extensions.
        """
        if self.require_absolute_path and not target.is_absolute:
            return False
        if not target.exists:
            return False
        if target.is_dir:
            return True
        return target.suffix in self.allowed_extensions

    def validate_file_size(self, target: IndexingTarget) -> bool:
        """Check if a file is within the allowed size limit."""
        if not target.is_file:
            return True
        if target.size_bytes is None:
            msg = "File indexing targets must include size_bytes"
            raise ValueError(msg)
        size_mb = target.size_bytes / (1024 * 1024)
        return size_mb <= self.max_file_size_mb
