"""Generation service: the business logic behind /generate and /health.

Owns request defaults (from Settings), timing, metrics, and structured
logging. Raises the domain errors from app.core.errors; the API layer
turns those into HTTP responses.
"""

from __future__ import annotations

import time

from app.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.core.config import Settings
from app.core.errors import GenerationError, ModelNotLoadedError
from app.core.logging import get_logger
from app.core.metrics import GENERATION_LATENCY, REQUESTS_TOTAL, TOKENS_GENERATED
from app.infrastructure.vllm_engine import GenerationResult, ModelEngine

logger = get_logger(__name__)


class GenerationService:
    """Coordinates the model engine for generation and health checks."""

    def __init__(self, engine: ModelEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings

    def health(self) -> HealthResponse:
        """Current service health, driven by whether the model is loaded."""
        return HealthResponse(
            status="ok" if self.engine.loaded else "degraded",
            model=self.engine.model_name,
            model_loaded=self.engine.loaded,
        )

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a completion for a validated request."""
        max_tokens = request.max_tokens or self.settings.max_tokens_default
        logger.info(
            "generation started",
            extra={
                "model": self.engine.model_name,
                "max_tokens": max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        )
        start = time.monotonic()
        try:
            result: GenerationResult = self.engine.generate(
                prompt=request.prompt,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except ModelNotLoadedError:
            REQUESTS_TOTAL.labels(endpoint="/generate", status="unavailable").inc()
            logger.warning("generation rejected: model not loaded")
            raise
        except GenerationError:
            REQUESTS_TOTAL.labels(endpoint="/generate", status="error").inc()
            logger.error("generation failed", extra={"model": self.engine.model_name})
            raise
        except Exception as exc:  # any unexpected engine failure -> clean 500
            REQUESTS_TOTAL.labels(endpoint="/generate", status="error").inc()
            logger.error(
                "generation failed unexpectedly",
                extra={"model": self.engine.model_name},
            )
            raise GenerationError(f"Generation failed: {exc}") from exc
        GENERATION_LATENCY.observe(time.monotonic() - start)
        REQUESTS_TOTAL.labels(endpoint="/generate", status="success").inc()
        TOKENS_GENERATED.inc(result.generated_tokens)
        logger.info(
            "generation completed",
            extra={
                "model": self.engine.model_name,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "latency_seconds": result.latency_seconds,
            },
        )
        return GenerateResponse(
            text=result.text,
            model=self.engine.model_name,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            latency_seconds=result.latency_seconds,
        )
