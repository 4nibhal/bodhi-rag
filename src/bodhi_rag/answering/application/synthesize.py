"""
SynthesizeAnswerUseCase.

Takes the question and the retrieved context, assembles a message-oriented
request, and returns the synthesized answer text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.ports.llm import LLMMessage, LLMPort
    from bodhi_rag.ports.vector_store import RetrievedDocument


SYSTEM_MESSAGE = (
    "Answer the user's question using only the provided context. "
    "If the context does not contain the answer, say that you do not know."
)


def _build_messages(
    question: str,
    contexts: list[RetrievedDocument],
) -> list[LLMMessage]:
    context_parts = [
        f"[Document {index}]\n{document.text}"
        for index, document in enumerate(contexts, start=1)
    ]
    context_block = "\n\n".join(context_parts) or "[No retrieved context]"
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        },
    ]


class SynthesizeAnswerUseCase:
    """Application-layer entry point for synthesizing the final answer."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def execute(
        self,
        question: str,
        contexts: list[RetrievedDocument],
        *,
        temperature: float,
    ) -> str:
        """
        Synthesize the natural-language answer for `question` grounded in `contexts`.

        Args:
            question: The user query.
            contexts: Chunks already reranked and trimmed to top_k.
            temperature: Sampling temperature passed through to the LLM.

        Returns:
            The synthesized answer text.

        """
        return await self._llm.generate(
            _build_messages(question, contexts),
            temperature=temperature,
        )
