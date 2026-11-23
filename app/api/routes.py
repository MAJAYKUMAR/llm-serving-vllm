"""HTTP routes: /generate, /health, /metrics, and /.

Routes resolve the GenerationService from ``app.state`` (wired up by
the composition root in app.main) and delegate everything to it.
Domain errors are translated to clean JSON error bodies by the
exception handlers in app.main — no tracebacks ever leak to clients.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from prometheus_client import generate_latest
from starlette.responses import PlainTextResponse

from app.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.services.generation import GenerationService

router = APIRouter()


def _service(request: Request) -> GenerationService:
    return request.app.state.generation_service


@router.get("/", response_model=dict)
def root(request: Request) -> dict:
    service = _service(request)
    return {
        "service": "llm-serving-vllm",
        "model": service.engine.model_name,
        "model_loaded": service.engine.loaded,
        "endpoints": ["/health", "/generate", "/metrics"],
    }


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return _service(request).health()


@router.post("/generate", response_model=GenerateResponse)
def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    return _service(request).generate(body)


@router.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type="text/plain")
