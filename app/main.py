"""Composition root: wires the layers together and exposes the FastAPI app.

Layering:
    app.api            routes + Pydantic schemas (HTTP only)
    app.services       generation business logic
    app.core           config, structured logging, errors, metrics
    app.infrastructure vLLM engine wrapper (lazy import — this module
                       still imports cleanly without vLLM installed)

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Or:        python -m app
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.schemas import ErrorResponse
from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.logging import configure_logging, get_logger
from app.infrastructure.vllm_engine import ModelEngine
from app.services.generation import GenerationService

logger = get_logger(__name__)


async def _service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Domain error -> clean JSON error body, no traceback."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={"code": exc.code, "message": exc.message}
        ).model_dump(),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: never leak a traceback or internals to the client."""
    logger.exception("unhandled error", extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={"code": "internal_error", "message": "Internal server error"}
        ).model_dump(),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app, wiring config -> engine -> service -> routes."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    engine = ModelEngine(model_name=settings.model_name)
    service = GenerationService(engine=engine, settings=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.load_model_on_startup:
            try:
                engine.load()
            except Exception as exc:
                # Stay up degraded (/health reports it) rather than crash:
                # a failed model load must not take down the whole service.
                logger.error("model load failed at startup", extra={"error": str(exc)})
        yield

    app = FastAPI(
        title="LLM Serving with vLLM",
        description=(
            "Production-style LLM inference service: FastAPI + vLLM, "
            "with Prometheus metrics."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.generation_service = service
    app.add_exception_handler(ServiceError, _service_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, log_level="info"
    )
