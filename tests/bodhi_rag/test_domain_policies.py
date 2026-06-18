"""Tests for domain policies."""

from __future__ import annotations

import pytest

from bodhi_rag.domain import (
    ContextAssemblyPolicy,
    GenerationPolicy,
    IndexingPolicy,
    IndexingTarget,
    RetrievalPolicy,
)
from bodhi_rag.domain.exceptions import PolicyViolationError
from bodhi_rag.domain.services import IndexingDomainService

ABS_FILE = "/workspace/test.txt"
ABS_ALT_FILE = "/workspace/test.exe"
ABS_DIR = "/workspace/docs"
REL_FILE = "relative/path/file.txt"


def test_retrieval_policy_defaults() -> None:
    policy = RetrievalPolicy()
    assert policy.reranker_max_length == 512
    assert policy.document_summary_token_limit == 300


def test_should_summarize() -> None:
    policy = RetrievalPolicy(document_summary_token_limit=100)
    assert not policy.should_summarize(50)
    assert not policy.should_summarize(100)
    assert policy.should_summarize(101)


def test_retrieval_policy_is_frozen() -> None:
    policy = RetrievalPolicy()
    with pytest.raises(AttributeError):
        policy.reranker_max_length = 1024


def test_generation_policy_defaults() -> None:
    policy = GenerationPolicy()
    assert policy.prompt_summary_token_limit == 1200
    assert policy.raw_summary_char_limit == 2500
    assert policy.role_mapping is not None


def test_map_role_user() -> None:
    policy = GenerationPolicy()
    assert policy.map_role("question") == "user"
    assert policy.map_role("human") == "user"
    assert policy.map_role("user") == "user"


def test_map_role_assistant() -> None:
    policy = GenerationPolicy()
    assert policy.map_role("answer") == "assistant"
    assert policy.map_role("assistant") == "assistant"
    assert policy.map_role("ai") == "assistant"


def test_map_role_unknown_defaults_to_user() -> None:
    assert GenerationPolicy().map_role("unknown") == "user"


def test_should_summarize_prompt() -> None:
    policy = GenerationPolicy(prompt_summary_token_limit=100)
    assert not policy.should_summarize_prompt(50)
    assert not policy.should_summarize_prompt(100)
    assert policy.should_summarize_prompt(101)


def test_should_summarize_text_by_char_count() -> None:
    policy = GenerationPolicy(raw_summary_char_limit=1000)
    assert policy.should_summarize_text(500)
    assert not policy.should_summarize_text(1000)
    assert not policy.should_summarize_text(1001)


def test_generation_policy_is_frozen() -> None:
    policy = GenerationPolicy()
    with pytest.raises(AttributeError):
        policy.prompt_summary_token_limit = 2000


def test_context_assembly_policy_defaults() -> None:
    policy = ContextAssemblyPolicy()
    assert policy.context_token_limit == 2000
    assert policy.document_separator == "\n"


def test_compute_available_tokens() -> None:
    policy = ContextAssemblyPolicy(context_token_limit=1000)
    assert policy.compute_available_tokens(0) == 1000
    assert policy.compute_available_tokens(300) == 700
    assert policy.compute_available_tokens(1000) == 0
    assert policy.compute_available_tokens(1500) == 0


def test_compute_available_tokens_custom_max() -> None:
    policy = ContextAssemblyPolicy(context_token_limit=2000)
    assert policy.compute_available_tokens(500, max_tokens=800) == 300


def test_should_truncate() -> None:
    policy = ContextAssemblyPolicy(context_token_limit=1000)
    assert not policy.should_truncate(500)
    assert not policy.should_truncate(1000)
    assert policy.should_truncate(1001)


def test_context_assembly_policy_is_frozen() -> None:
    policy = ContextAssemblyPolicy()
    with pytest.raises(AttributeError):
        policy.context_token_limit = 5000


def test_indexing_policy_defaults() -> None:
    policy = IndexingPolicy()
    assert ".txt" in policy.allowed_extensions
    assert ".md" in policy.allowed_extensions
    assert policy.max_file_size_mb == 100
    assert policy.require_absolute_path


def test_is_valid_target_with_valid_extension() -> None:
    policy = IndexingPolicy(allowed_extensions=(".txt", ".md"))
    target = IndexingTarget(path=ABS_FILE, exists=True, is_dir=False, size_bytes=10)
    assert policy.is_valid_target(target)


def test_is_valid_target_with_invalid_extension() -> None:
    policy = IndexingPolicy(allowed_extensions=(".txt", ".md"))
    target = IndexingTarget(path=ABS_ALT_FILE, exists=True, is_dir=False, size_bytes=10)
    assert not policy.is_valid_target(target)


def test_is_valid_target_nonexistent() -> None:
    policy = IndexingPolicy()
    target = IndexingTarget(path="/missing/file.txt", exists=False, is_dir=False)
    assert not policy.is_valid_target(target)


def test_is_valid_target_relative_when_required() -> None:
    policy = IndexingPolicy(require_absolute_path=True)
    target = IndexingTarget(path=REL_FILE, exists=True, is_dir=False, size_bytes=10)
    assert not policy.is_valid_target(target)


def test_is_valid_target_relative_when_not_required() -> None:
    policy = IndexingPolicy(require_absolute_path=False)
    target = IndexingTarget(path=REL_FILE, exists=True, is_dir=False, size_bytes=10)
    assert policy.is_valid_target(target)


def test_validate_file_size_within_limit() -> None:
    policy = IndexingPolicy(max_file_size_mb=100)
    target = IndexingTarget(path=ABS_FILE, exists=True, is_dir=False, size_bytes=1024)
    assert policy.validate_file_size(target)


def test_validate_file_size_exceeds_limit() -> None:
    policy = IndexingPolicy(max_file_size_mb=1)
    target = IndexingTarget(
        path=ABS_FILE,
        exists=True,
        is_dir=False,
        size_bytes=2 * 1024 * 1024,
    )
    assert not policy.validate_file_size(target)


def test_validate_file_size_directory() -> None:
    policy = IndexingPolicy(max_file_size_mb=1)
    target = IndexingTarget(path=ABS_DIR, exists=True, is_dir=True)
    assert policy.validate_file_size(target)


def test_validate_file_size_requires_file_metadata() -> None:
    policy = IndexingPolicy(max_file_size_mb=1)
    target = IndexingTarget(path=ABS_FILE, exists=True, is_dir=False)
    with pytest.raises(ValueError, match="size_bytes"):
        policy.validate_file_size(target)


def test_indexing_target_rejects_invalid_directory_metadata() -> None:
    with pytest.raises(ValueError, match="must not include a file size"):
        IndexingTarget(path=ABS_DIR, exists=True, is_dir=True, size_bytes=10)


def test_indexing_policy_is_frozen() -> None:
    policy = IndexingPolicy()
    with pytest.raises(AttributeError):
        policy.max_file_size_mb = 200


def test_validate_index_request_raises_for_invalid_target() -> None:
    service = IndexingDomainService(IndexingPolicy(allowed_extensions=(".txt",)))
    target = IndexingTarget(path=ABS_ALT_FILE, exists=True, is_dir=False, size_bytes=10)

    with pytest.raises(PolicyViolationError):
        service.validate_index_request(target)


def test_validate_index_request_accepts_valid_target() -> None:
    service = IndexingDomainService(IndexingPolicy(allowed_extensions=(".txt",)))
    target = IndexingTarget(path=ABS_FILE, exists=True, is_dir=False, size_bytes=10)

    service.validate_index_request(target)
