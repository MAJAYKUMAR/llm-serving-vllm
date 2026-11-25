"""Generation service: the business logic behind /generate and /health.

Owns request defaults (from Settings), timing, metrics, and structured
logging. Raises the domain errors from app.core.errors; the API layer
turns those into HTTP responses.

Concurrency is bounded by a semaphore sized by
``Settings.max_concurrent_requests``: when every slot is busy the
service fails fast with ServerBusyError (503) instead of queueing
requests against the GPU until it runs out of memory.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar

from app.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.core.config import Settings
from app.core.errors import GenerationError, ModelNotLoadedError, ServerBusyError
from app.core.logging import get_logger
from app.core.metrics import GENERATION_LATENCY, REQUESTS_TOTAL, TOKENS_GENERATED
from app.infrastructure.vllm_engine import GenerationResult, ModelEngine

logger = get_logger(__name__)

# Request id for the request currently being handled, set by the
# middleware in app.main. Included in log lines so a slow request can
# be traced end-to-end; None outside a request (e.g. direct service
# calls in tests).
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _log_extra(**fields: object) -> dict:
    """Extra log fields, plus the request id when one is set."""
    request_id = request_id_var.get()
    if request_id is not None:
        fields["request_id"] = request_id
    return fields


class GenerationService:
    """Coordinates the model engine for generation and health checks."""

    def __init__(self, engine: ModelEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings
        self._inflight = threading.Semaphore(settings.max_concurrent_requests)

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
        if not self._inflight.acquire(blocking=False):
            REQUESTS_TOTAL.labels(endpoint="/generate", status="busy").inc()
            logger.warning(
                "generation rejected: server busy",
                extra=_log_extra(model=self.engine.model_name),
            )
            raise ServerBusyError("Server is busy, please try again later.")
        try:
            logger.info(
                "generation started",
                extra=_log_extra(
                    model=self.engine.model_name,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
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
                logger.warning(
                    "generation rejected: model not loaded",
                    extra=_log_extra(model=self.engine.model_name),
                )
                raise
            except GenerationError:
                REQUESTS_TOTAL.labels(endpoint="/generate", status="error").inc()
                logger.error(
                    "generation failed",
                    extra=_log_extra(model=self.engine.model_name),
                )
                raise
            except Exception as exc:  # any unexpected engine failure -> clean 500
                REQUESTS_TOTAL.labels(endpoint="/generate", status="error").inc()
                logger.error(
                    "generation failed unexpectedly",
                    extra=_log_extra(model=self.engine.model_name),
                )
                raise GenerationError(f"Generation failed: {exc}") from exc
            GENERATION_LATENCY.observe(time.monotonic() - start)
            REQUESTS_TOTAL.labels(endpoint="/generate", status="success").inc()
            TOKENS_GENERATED.inc(result.generated_tokens)
            logger.info(
                "generation completed",
                extra=_log_extra(
                    model=self.engine.model_name,
                    prompt_tokens=result.prompt_tokens,
                    generated_tokens=result.generated_tokens,
                    latency_seconds=result.latency_seconds,
                ),
            )
            return GenerateResponse(
                text=result.text,
                model=self.engine.model_name,
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
                latency_seconds=result.latency_seconds,
            )
        finally:
            self._inflight.release()
