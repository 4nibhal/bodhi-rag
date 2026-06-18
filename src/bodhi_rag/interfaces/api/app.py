"""
FastAPI application factory.

Creates and configures the Bodhi RAG API server.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from fastapi.responses import Response

    from bodhi_rag.application.config import BhodiConfig
    from bodhi_rag.application.facade import BhodiApplication

from bodhi_rag._version import get_version
from bodhi_rag.application.config_loader import load_bodhi_config
from bodhi_rag.infrastructure.container import Container

if TYPE_CHECKING:
    from fastapi.responses import Response

    from bodhi_rag.application.facade import BhodiApplication

API_ALLOWED_SOURCE_SUFFIXES = frozenset({".pdf", ".txt", ".md", ".rst"})


@dataclass(frozen=True, slots=True)
class ApiSourcePolicy:
    root: Path | None
    allowed_suffixes: frozenset[str] = API_ALLOWED_SOURCE_SUFFIXES


def _load_api_source_policy(config: BhodiConfig) -> ApiSourcePolicy:
    configured_root = config.api.source_root
    if configured_root is None:
        return ApiSourcePolicy(root=None)
    return ApiSourcePolicy(root=Path(configured_root).expanduser().resolve())


# Module-level state
_state: dict[str, Any] = {"app": None, "bodhi_rag_app": None, "source_policy": None}

# Rate limiting state
_rate_limit_requests: dict[str, list[float]] = {}
_rate_limit_lock = asyncio.Lock()
RATE_LIMIT_MAX = 100
RATE_LIMIT_WINDOW = 60  # seconds


async def _rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Apply a per-IP rate limit; skip the /health endpoint."""
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    async with _rate_limit_lock:
        timestamps = _rate_limit_requests.get(client_ip, [])
        # Clean old timestamps
        timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            _rate_limit_requests[client_ip] = timestamps
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        timestamps.append(now)
        _rate_limit_requests[client_ip] = timestamps

    return await call_next(request)


def create_app(config: BhodiConfig | None = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional BhodiConfig. If not provided, uses defaults.

    Returns:
        Configured FastAPI application.

    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Initialize and tear down the application state."""
        cfg = config or load_bodhi_config()
        container = Container(cfg)
        _state["bodhi_rag_app"] = container.build()
        _state["source_policy"] = _load_api_source_policy(cfg)
        try:
            yield
        finally:
            _state["bodhi_rag_app"] = None
            _state["source_policy"] = None

    app = FastAPI(
        title="Bodhi RAG API",
        description="Production-ready RAG framework with clean hexagonal architecture",
        version=get_version(),
        lifespan=lifespan,
    )

    # Register middleware
    app.middleware("http")(_rate_limit_middleware)

    # Include routers
    from bodhi_rag.interfaces.api.routes import health, indexing, query

    app.include_router(health.router, tags=["health"])
    app.include_router(indexing.router, tags=["documents"])
    app.include_router(query.router, tags=["query"])

    # Global exception handler for domain errors
    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc)
                if not isinstance(exc, Exception)
                else "Internal server error",
            },
        )

    _state["app"] = app
    return app


def get_bodhi_rag_app() -> BhodiApplication:
    """Get the BhodiApplication instance."""
    app = _state.get("bodhi_rag_app")
    if app is None:
        msg = "Application not initialized"
        raise RuntimeError(msg)
    return cast("BhodiApplication", app)


def get_api_source_policy() -> ApiSourcePolicy:
    """Get the API-local source policy."""
    policy = _state.get("source_policy")
    if policy is None:
        msg = "Application not initialized"
        raise RuntimeError(msg)
    return cast("ApiSourcePolicy", policy)
