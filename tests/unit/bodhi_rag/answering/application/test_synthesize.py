"""Unit tests for answer synthesis orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from bodhi_rag.answering.application.synthesize import (
    SYSTEM_MESSAGE,
    SynthesizeAnswerUseCase,
)
from bodhi_rag.domain.entities import RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, DocumentId


@pytest.mark.asyncio
async def test_execute_builds_message_oriented_request_with_context() -> None:
    """Answer synthesis assembles context in the application layer."""
    llm = AsyncMock()
    llm.generate.return_value = "Grounded answer"
    use_case = SynthesizeAnswerUseCase(llm)

    document_id = DocumentId("33333333-3333-3333-3333-333333333333")
    contexts = [
        RetrievedDocument(
            chunk_id=ChunkId(document_id=document_id, chunk_index=0),
            document_id=document_id,
            text="bodhi-rag is a document processing platform.",
            score=0.95,
        ),
    ]

    result = await use_case.execute(
        "What is bodhi-rag?",
        contexts,
        temperature=0.3,
    )

    llm.generate.assert_awaited_once()
    messages = llm.generate.await_args.args[0]
    kwargs = llm.generate.await_args.kwargs

    assert result == "Grounded answer"
    assert messages == [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                "Context:\n[Document 1]\n"
                "bodhi-rag is a document processing platform.\n\n"
                "Question: What is bodhi-rag?"
            ),
        },
    ]
    assert kwargs == {"temperature": 0.3}


@pytest.mark.asyncio
async def test_execute_marks_missing_context_explicitly() -> None:
    """Answer synthesis sends an explicit empty-context marker."""
    llm = AsyncMock()
    llm.generate.return_value = "I do not know."
    use_case = SynthesizeAnswerUseCase(llm)

    await use_case.execute("What is bodhi-rag?", [], temperature=0.7)

    messages = llm.generate.await_args.args[0]
    assert messages[1]["content"].startswith("Context:\n[No retrieved context]")
