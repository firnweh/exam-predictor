"""
FastAPI Application — PRAJNA Intelligence Layer API
====================================================
Production-grade API with:
- Dependency injection for all services
- Request ID tracking
- Structured logging
- Error handling with typed error responses
- OpenAPI docs at /docs
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import Settings, get_settings
from services.api.routers import insights, reports, predictions, copilot, data_bridge
from services.api.routers.qbank import router as qbank_router
from services.api.routers.mistakes import router as mistakes_router

logger = logging.getLogger(__name__)


# ── Application factory ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info(f"PRAJNA Intelligence Layer starting — env={settings.environment}")

    # Warm up the prediction adapter
    from services.api.deps import get_prediction_adapter
    adapter = get_prediction_adapter()
    logger.info("Prediction adapter ready")

    # Initialize RAG (non-blocking — indexing happens separately via script)
    if settings.rag_enabled:
        from services.api.deps import get_rag_retriever
        retriever = get_rag_retriever()
        logger.info(f"RAG retriever initialized: {settings.chroma_host}")

    logger.info("PRAJNA Intelligence Layer ready ✓")
    yield

    logger.info("PRAJNA Intelligence Layer shutting down")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title       = "PRAJNA Exam Intelligence API",
        description = "SLM-powered exam topic intelligence layer on top of the PRAJNA prediction engine.",
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        lifespan    = lifespan,
    )

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Request ID middleware ──
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        t_start = time.time()
        response: Response = await call_next(request)
        latency_ms = round((time.time() - t_start) * 1000)
        response.headers["X-Request-ID"]  = request_id
        response.headers["X-Latency-Ms"]  = str(latency_ms)
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} latency={latency_ms}ms rid={request_id}"
        )
        return response

    # ── Exception handlers ──
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "success":    False,
                "error":      "validation_error",
                "message":    str(exc),
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success":  False,
                "error":    "internal_error",
                "message":  "An internal error occurred. Check server logs.",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # ── Routers ──
    app.include_router(insights.router,     prefix="/api/v1/insights",     tags=["Insights"])
    app.include_router(reports.router,      prefix="/api/v1/reports",      tags=["Reports"])
    app.include_router(predictions.router,  prefix="/api/v1/predictions",  tags=["Predictions"])
    app.include_router(copilot.router,      prefix="/api/v1/copilot",      tags=["Copilot"])
    app.include_router(data_bridge.router,  prefix="/api/v1/data",         tags=["Data Bridge"])
    app.include_router(qbank_router)
    app.include_router(mistakes_router)

    # ── Health ──
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "service": "prajna-intelligence", "version": "1.0.0"}

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service": "PRAJNA Exam Intelligence Layer",
            "version": "1.0.0",
            "docs":    "/docs",
            "endpoints": [
                "/api/v1/insights",
                "/api/v1/reports",
                "/api/v1/predictions",
                "/api/v1/copilot",
            ],
        }

    return app


app = create_app()
