"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.vector_store import ensure_index
from app.api import health, ingest, webhooks

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    setup_logging()
    settings = get_settings()
    _log.info("app_starting", demo_mode=settings.demo_mode)

    # Ensure OpenSearch index exists on startup (skip in demo mode)
    if not settings.demo_mode:
        try:
            ensure_index()
        except Exception:
            _log.exception("opensearch_index_init_failed")
    else:
        _log.info("demo_mode_active", msg="Skipping OpenSearch init — using demo stubs")

    yield

    _log.info("app_shutting_down")


app = FastAPI(
    title="RAG Support System",
    description="Retrieval-Augmented Generation for Zendesk support tickets",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Routers ──────────────────────────────────────────────
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(webhooks.router)


# ── Global exception handler ────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    _log.exception("unhandled_exception", path=request.url.path, method=request.method)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Request logging middleware ───────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    _log.info("http_request_start", method=request.method, path=request.url.path)
    response = await call_next(request)
    _log.info("http_request_end", method=request.method, path=request.url.path, status=response.status_code)
    return response
