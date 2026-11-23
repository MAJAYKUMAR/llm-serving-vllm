"""vLLM model engine wrapper.

The vLLM import is intentionally lazy: importing this module must work
without vLLM installed, so the API surface can be tested on a CPU-only
machine with the light dependencies. The model is only loaded when
:meth:`ModelEngine.load` is called.

The engine speaks plain dataclasses, not API schemas — the service
layer owns the translation to HTTP request/response models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.errors import GenerationError, ModelNotLoadedError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    """Raw result of one engine generation."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float


class ModelEngine:
    """Thin wrapper around a vLLM LLM instance."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._llm = None
        self._sampling_params_cls = None
        self.loaded = False

    def load(self) -> None:
        """Load the model with vLLM. Raises if vLLM is unavailable."""
        from vllm import LLM, SamplingParams  # noqa: PLC0415  (lazy import)

        logger.info("loading model", extra={"model": self.model_name})
        self._sampling_params_cls = SamplingParams
        self._llm = LLM(model=self.model_name)
        self.loaded = True
        logger.info("model loaded", extra={"model": self.model_name})

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        """Generate a completion. Raises ModelNotLoadedError / GenerationError."""
        if not self.loaded or self._llm is None:
            raise ModelNotLoadedError(
                "Model is not loaded. Set LOAD_MODEL_ON_STARTUP=1 or call "
                "the engine's load() before generating."
            )
        sampling_params = self._sampling_params_cls(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        start = time.monotonic()
        try:
            outputs = self._llm.generate([prompt], sampling_params)
        except Exception as exc:  # engine failure -> clean 500, no traceback
            raise GenerationError(f"Generation failed: {exc}") from exc
        latency = time.monotonic() - start
        output = outputs[0]
        return GenerationResult(
            text=output.outputs[0].text,
            prompt_tokens=len(output.prompt_token_ids),
            generated_tokens=len(output.outputs[0].token_ids),
            latency_seconds=round(latency, 3),
        )
