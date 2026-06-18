"""
Answer synthesis bounded context.

Owns the responsibility of producing the final natural-language
response that the user sees. The application layer assembles
message-oriented LLM requests so grounding behavior can evolve
without leaking prompt construction into provider adapters.

Hexagonal layout:

    answering/
    ├── domain/        (empty: Answer entity will live here when the
    │                   context grows its own invariants)
    ├── application/synthesize.py  (SynthesizeAnswerUseCase)
    ├── ports/         (empty: the context uses the cross-context
    │                   LLMPort rather than defining a context-local
    │                   port; if a context-local port emerges it
    │                   belongs here)
    └── infrastructure/ (empty: no context-local adapters; the
                         LLM adapters live in infrastructure/llm/)
"""
